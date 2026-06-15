#!/usr/bin/env python3
"""Search probability-space foreground gates against MSLNet-style metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cathaction.data.task1 import load_task1_mask
from cathaction.training.task1_baseline import (
    _autocast_context,
    _path_for_json,
    _resolve_path,
    _safe_filename,
    primary_logits_for_label_mode,
    select_device,
    write_json,
)
from scripts.task1.evaluate_tta_ensemble import (
    ModelSpec,
    _apply_tta_image,
    _build_loader,
    _invert_tta_probability,
    _load_model_spec,
    _normalize_tta_modes,
    _normalized_weights,
    _predict_model_probability,
    _validate_matching_batches,
)
from scripts.task1.evaluate_tta_ensemble_original_space import (
    _probability_to_original_space,
    _validate_original_space_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        nargs=4,
        metavar=("NAME", "CONFIG", "CHECKPOINT", "WEIGHT"),
        required=True,
        help="Model spec. Repeat for ensembles.",
    )
    parser.add_argument(
        "--tta",
        choices=("none", "hflip", "vflip", "hvflip"),
        action="append",
        default=None,
        help="TTA transform to include. Repeatable. Defaults to none.",
    )
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--preset", choices=("quick", "expanded"), default="quick")
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=None,
        help="Restrict to candidate IDs from the selected preset. Baseline is added automatically.",
    )
    parser.add_argument("--write-best-predictions", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device(args.device or "auto")
    model_specs = [_load_model_spec(raw, device=device) for raw in args.model]
    _validate_original_space_specs(model_specs)

    repo_root = model_specs[0].repo_root
    output_dir = _resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _resolve_path(args.eval_manifest, repo_root)
    domain_filter = set(str(value) for value in args.domain) if args.domain else None
    eval_manifest = _prepare_manifest(
        source_manifest,
        output_dir=output_dir,
        domain_filter=domain_filter,
        max_samples=args.max_samples,
    )
    batch_size = args.batch_size or min(
        int(spec.config["training"]["batch_size"]) for spec in model_specs
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else min(int(spec.config["training"].get("num_workers", 0)) for spec in model_specs)
    )
    loaders = [
        _build_loader(
            spec,
            eval_manifest=eval_manifest,
            max_samples=None,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for spec in model_specs
    ]
    reference_records = {
        str(record["sample_id"]): record
        for record in loaders[0].dataset.records  # type: ignore[attr-defined]
    }
    tta_modes = _normalize_tta_modes(args.tta)
    weights = _normalized_weights(model_specs)
    candidates = _select_candidates(_candidate_grid(args.preset), args.candidate_id)

    rows, best_candidate = _evaluate_candidates(
        model_specs,
        loaders,
        reference_records=reference_records,
        candidates=candidates,
        weights=weights,
        tta_modes=tta_modes,
        device=device,
    )
    metrics_csv = output_dir / "candidate_metrics.csv"
    _write_csv(metrics_csv, rows)

    result: dict[str, Any] = {
        "metric_style": "fast MSLNet-style probability gate search",
        "search_note": (
            "Uses exact binary Dice/IoU and distance-to-target precision r3; "
            "recall r3 is computed via radius-3 binary dilation. Full AHD/F1 r0-r4 "
            "must be recomputed for promoted candidates."
        ),
        "models": [
            {
                "name": spec.name,
                "config_path": _path_for_json(spec.config_path, repo_root),
                "checkpoint_path": _path_for_json(spec.checkpoint_path, repo_root),
                "weight": spec.weight,
                "normalized_weight": weights[index],
                "resize_mode": str(spec.config["data"].get("resize_mode", "direct")),
                "image_size": spec.config["data"]["image_size"],
            }
            for index, spec in enumerate(model_specs)
        ],
        "tta_modes": tta_modes,
        "source_eval_manifest": _path_for_json(source_manifest, repo_root),
        "eval_manifest": _path_for_json(eval_manifest, repo_root),
        "domain_filter": sorted(domain_filter) if domain_filter else None,
        "samples": len(reference_records),
        "candidates": len(candidates),
        "preset": args.preset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(device),
        "baseline": _public_row(next(row for row in rows if row["candidate_id"] == "baseline")),
        "best": _public_row(best_candidate),
        "top": [_public_row(row) for row in rows[: int(args.top_k)]],
        "candidate_metrics_csv": _path_for_json(metrics_csv, repo_root),
    }

    if args.write_best_predictions:
        best_dir = output_dir / "best_predictions"
        best_manifest = _write_candidate_predictions(
            model_specs,
            loaders,
            reference_records=reference_records,
            candidate=best_candidate,
            weights=weights,
            tta_modes=tta_modes,
            output_dir=best_dir,
            device=device,
        )
        result["best_prediction_manifest"] = _path_for_json(best_manifest, repo_root)

    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


def _prepare_manifest(
    source_manifest: Path,
    *,
    output_dir: Path,
    domain_filter: set[str] | None,
    max_samples: int | None,
) -> Path:
    if domain_filter is None and max_samples is None:
        return source_manifest
    rows = _read_csv(source_manifest)
    if domain_filter is not None:
        rows = [row for row in rows if str(row.get("domain", "")) in domain_filter]
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    if not rows:
        raise ValueError("No manifest rows remain after filtering.")
    filtered_manifest = output_dir / "filtered_eval_manifest.csv"
    _write_csv(filtered_manifest, rows)
    return filtered_manifest


def _candidate_grid(preset: str) -> list[dict[str, Any]]:
    domains = ["all", "phantom"]
    candidates: list[dict[str, Any]] = [_candidate("baseline", "all", "baseline", None)]
    if preset == "quick":
        thresholds = {
            "foreground_prob": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
            "max_foreground_prob": [0.40, 0.45, 0.50, 0.55, 0.60],
            "foreground_margin": [0.10, 0.20, 0.30, 0.40, 0.50],
            "max_margin": [0.02, 0.05, 0.10, 0.15, 0.20],
            "toolness_prob": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        }
    elif preset == "expanded":
        thresholds = {
            "foreground_prob": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
            "max_foreground_prob": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            "foreground_margin": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60],
            "max_margin": [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "toolness_prob": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        }
    else:
        raise ValueError(f"Unsupported preset: {preset}")
    for domain in domains:
        for gate, values in thresholds.items():
            for threshold in values:
                candidates.append(
                    _candidate(
                        f"{domain}_{gate}_t{_threshold_id(threshold)}",
                        domain,
                        gate,
                        threshold,
                    )
                )
    return candidates


def _candidate(
    candidate_id: str,
    domain: str,
    gate: str,
    threshold: float | None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "domain": domain,
        "gate": gate,
        "threshold": None if threshold is None else float(threshold),
    }


def _select_candidates(
    candidates: list[dict[str, Any]],
    candidate_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not candidate_ids:
        return candidates
    requested = {"baseline", *(str(candidate_id) for candidate_id in candidate_ids)}
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    missing = sorted(requested.difference(by_id))
    if missing:
        available = ", ".join(sorted(by_id))
        raise ValueError(f"Unknown candidate ID(s): {missing}. Available candidates: {available}")
    return [candidate for candidate in candidates if str(candidate["candidate_id"]) in requested]


def _evaluate_candidates(
    model_specs: list[ModelSpec],
    loaders: list[DataLoader[dict[str, Any]]],
    *,
    reference_records: dict[str, dict[str, str]],
    candidates: list[dict[str, Any]],
    weights: list[float],
    tta_modes: list[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_root = model_specs[0].repo_root
    states = {candidate["candidate_id"]: _new_state() for candidate in candidates}
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    disk3 = _disk_structure(3)

    with torch.no_grad():
        for batches in tqdm(
            zip(*loaders),
            total=len(loaders[0]),
            desc="prob-gate-search",
            leave=False,
        ):
            reference_batch = batches[0]
            _validate_matching_batches(batches)
            sample_ids = [str(value) for value in reference_batch["sample_id"]]
            batch_probabilities = [
                _predict_model_probability(
                    spec,
                    batch["image"].to(device),
                    tta_modes=tta_modes,
                    target_shape=tuple(int(value) for value in batch["image"].shape[-2:]),
                    device=device,
                )
                for spec, batch in zip(model_specs, batches)
            ]
            batch_toolness_probabilities = [
                _predict_toolness_probability(
                    spec,
                    batch["image"].to(device),
                    tta_modes=tta_modes,
                    target_shape=tuple(int(value) for value in batch["image"].shape[-2:]),
                    device=device,
                )
                for spec, batch in zip(model_specs, batches)
            ]

            for index, sample_id in enumerate(sample_ids):
                record = reference_records[sample_id]
                target = load_task1_mask(
                    _resolve_path(record["mask_path"], repo_root),
                    record["mask_encoding"],
                ).astype(np.uint8)
                original_shape = tuple(int(value) for value in target.shape[-2:])
                probability = _ensemble_probability(
                    model_specs,
                    batch_probabilities,
                    weights=weights,
                    index=index,
                    original_shape=original_shape,
                )
                toolness_probability = _toolness_original_probability(
                    model_specs,
                    batch_toolness_probabilities,
                    index=index,
                    original_shape=original_shape,
                )
                target_fg = target > 0
                target_count = int(target_fg.sum())
                distance_to_target = ndimage.distance_transform_edt(~target_fg)
                for candidate in candidates:
                    prediction = _apply_probability_gate(
                        probability,
                        candidate,
                        domain=str(record.get("domain", "")),
                        toolness_probability=toolness_probability,
                    )
                    metrics = _fast_binary_metrics(
                        prediction > 0,
                        target_fg,
                        target_count=target_count,
                        distance_to_target=distance_to_target,
                        disk3=disk3,
                    )
                    _append_state(
                        states[candidate["candidate_id"]],
                        sample={
                            "domain": str(record.get("domain", "")),
                            "collection": str(record.get("collection", "")),
                        },
                        **metrics,
                    )

    rows = [_summarize_state(candidate_by_id[candidate_id], state) for candidate_id, state in states.items()]
    baseline = next(row for row in rows if row["candidate_id"] == "baseline")
    for row in rows:
        row["delta_dice"] = float(row["dice"] - baseline["dice"])
        row["delta_iou"] = float(row["iou"] - baseline["iou"])
        row["delta_precision_r3"] = float(row["precision_r3"] - baseline["precision_r3"])
        row["delta_recall_r3"] = float(row["recall_r3"] - baseline["recall_r3"])
        row["delta_f1_r3"] = float(row["f1_r3"] - baseline["f1_r3"])
    rows.sort(key=lambda row: (float(row["f1_r3"]), float(row["dice"])), reverse=True)
    return rows, rows[0]


def _write_candidate_predictions(
    model_specs: list[ModelSpec],
    loaders: list[DataLoader[dict[str, Any]]],
    *,
    reference_records: dict[str, dict[str, str]],
    candidate: dict[str, Any],
    weights: list[float],
    tta_modes: list[str],
    output_dir: Path,
    device: torch.device,
) -> Path:
    repo_root = model_specs[0].repo_root
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batches in tqdm(
            zip(*loaders),
            total=len(loaders[0]),
            desc="write-prob-gate-best",
            leave=False,
        ):
            reference_batch = batches[0]
            _validate_matching_batches(batches)
            sample_ids = [str(value) for value in reference_batch["sample_id"]]
            batch_probabilities = [
                _predict_model_probability(
                    spec,
                    batch["image"].to(device),
                    tta_modes=tta_modes,
                    target_shape=tuple(int(value) for value in batch["image"].shape[-2:]),
                    device=device,
                )
                for spec, batch in zip(model_specs, batches)
            ]
            batch_toolness_probabilities = [
                _predict_toolness_probability(
                    spec,
                    batch["image"].to(device),
                    tta_modes=tta_modes,
                    target_shape=tuple(int(value) for value in batch["image"].shape[-2:]),
                    device=device,
                )
                for spec, batch in zip(model_specs, batches)
            ]
            for index, sample_id in enumerate(sample_ids):
                record = reference_records[sample_id]
                target = load_task1_mask(
                    _resolve_path(record["mask_path"], repo_root),
                    record["mask_encoding"],
                ).astype(np.uint8)
                probability = _ensemble_probability(
                    model_specs,
                    batch_probabilities,
                    weights=weights,
                    index=index,
                    original_shape=tuple(int(value) for value in target.shape[-2:]),
                )
                toolness_probability = _toolness_original_probability(
                    model_specs,
                    batch_toolness_probabilities,
                    index=index,
                    original_shape=tuple(int(value) for value in target.shape[-2:]),
                )
                prediction = _apply_probability_gate(
                    probability,
                    candidate,
                    domain=str(record.get("domain", "")),
                    toolness_probability=toolness_probability,
                )
                prediction_path = output_dir / f"{_safe_filename(sample_id)}.png"
                Image.fromarray(prediction).save(prediction_path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "collection": str(record.get("collection", "")),
                        "domain": str(record.get("domain", "")),
                        "prediction_path": _path_for_json(prediction_path, repo_root),
                        "height": str(int(prediction.shape[0])),
                        "width": str(int(prediction.shape[1])),
                    }
                )
    manifest_path = output_dir / "predictions.csv"
    _write_csv(manifest_path, rows)
    return manifest_path


def _predict_toolness_probability(
    spec: ModelSpec,
    image: torch.Tensor,
    *,
    tta_modes: list[str],
    target_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor | None:
    if not _has_toolness_channel(spec):
        return None
    probabilities: list[torch.Tensor] = []
    for mode in tta_modes:
        transformed = _apply_tta_image(image, mode)
        with _autocast_context(device=device, use_amp=spec.use_amp):
            logits = spec.model(transformed)
        if logits.ndim != 4 or logits.shape[1] < 4:
            return None
        # The first three channels are the multiclass head; channel 3 is the
        # binary tool-vs-background auxiliary logit used during Stage 7 training.
        _ = primary_logits_for_label_mode(logits, str(spec.config["data"]["label_mode"]))
        toolness = torch.sigmoid(logits[:, 3:4].float())
        probabilities.append(_invert_tta_probability(toolness, mode))
    probability = torch.stack(probabilities, dim=0).mean(dim=0)
    if tuple(probability.shape[-2:]) != target_shape:
        probability = F.interpolate(probability, size=target_shape, mode="bilinear", align_corners=False)
    return probability


def _has_toolness_channel(spec: ModelSpec) -> bool:
    model_cfg = spec.config.get("model", {})
    if bool(model_cfg.get("toolness_auxiliary", False)) or bool(model_cfg.get("auxiliary_toolness", False)):
        return True
    return int(model_cfg.get("out_channels", 0)) >= 4


def _toolness_original_probability(
    model_specs: list[ModelSpec],
    batch_toolness_probabilities: list[torch.Tensor | None],
    *,
    index: int,
    original_shape: tuple[int, int],
) -> torch.Tensor | None:
    probabilities: list[torch.Tensor] = []
    for spec, probability_batch in zip(model_specs, batch_toolness_probabilities):
        if probability_batch is None:
            continue
        probabilities.append(
            _probability_to_original_space(
                probability_batch[index],
                original_shape=original_shape,
                resize_mode=str(spec.config["data"].get("resize_mode", "direct")),
            )
        )
    if not probabilities:
        return None
    return torch.stack(probabilities, dim=0).mean(dim=0)


def _ensemble_probability(
    model_specs: list[ModelSpec],
    batch_probabilities: list[torch.Tensor],
    *,
    weights: list[float],
    index: int,
    original_shape: tuple[int, int],
) -> torch.Tensor:
    probability_sum: torch.Tensor | None = None
    for spec, probability_batch, weight in zip(model_specs, batch_probabilities, weights):
        original_probability = _probability_to_original_space(
            probability_batch[index],
            original_shape=original_shape,
            resize_mode=str(spec.config["data"].get("resize_mode", "direct")),
        )
        weighted_probability = original_probability * float(weight)
        probability_sum = weighted_probability if probability_sum is None else probability_sum + weighted_probability
    if probability_sum is None:
        raise RuntimeError("No model probabilities were produced.")
    return probability_sum


def _apply_probability_gate(
    probability: torch.Tensor,
    candidate: dict[str, Any],
    *,
    domain: str,
    toolness_probability: torch.Tensor | None,
) -> np.ndarray:
    prediction = torch.argmax(probability, dim=0).cpu().numpy().astype(np.uint8)
    if candidate["gate"] == "baseline":
        return prediction
    if str(candidate["domain"]) == "phantom" and str(domain) != "phantom":
        return prediction

    prob = probability.detach().cpu()
    bg = prob[0].numpy()
    fg1 = prob[1].numpy()
    fg2 = prob[2].numpy()
    fg = fg1 + fg2
    max_fg = np.maximum(fg1, fg2)
    gate = str(candidate["gate"])
    if gate == "foreground_prob":
        score = fg
    elif gate == "max_foreground_prob":
        score = max_fg
    elif gate == "foreground_margin":
        score = fg - bg
    elif gate == "max_margin":
        score = max_fg - bg
    elif gate == "toolness_prob":
        if toolness_probability is None:
            raise ValueError("toolness_prob candidate requested, but no toolness auxiliary probability is available.")
        score = toolness_probability.detach().cpu().squeeze(0).numpy()
    else:
        raise ValueError(f"Unsupported probability gate: {gate}")

    threshold = float(candidate["threshold"])
    output = prediction.copy()
    output[(output > 0) & (score < threshold)] = 0
    return output


def _fast_binary_metrics(
    pred_fg: np.ndarray,
    target_fg: np.ndarray,
    *,
    target_count: int,
    distance_to_target: np.ndarray,
    disk3: np.ndarray,
) -> dict[str, float]:
    pred_count = int(pred_fg.sum())
    intersection = int(np.logical_and(pred_fg, target_fg).sum())
    union = int(np.logical_or(pred_fg, target_fg).sum())
    dice = 1.0 if pred_count + target_count == 0 else float(2.0 * intersection / (pred_count + target_count))
    iou = 1.0 if union == 0 else float(intersection / union)
    if pred_count == 0 and target_count == 0:
        precision_r3 = 1.0
        recall_r3 = 1.0
    elif pred_count == 0 or target_count == 0:
        precision_r3 = 0.0
        recall_r3 = 0.0
    else:
        precision_r3 = float(np.count_nonzero(distance_to_target[pred_fg] <= 3) / pred_count)
        dilated_prediction = ndimage.binary_dilation(pred_fg, structure=disk3)
        recall_r3 = float(np.logical_and(target_fg, dilated_prediction).sum() / target_count)
    return {
        "dice": dice,
        "iou": iou,
        "precision_r3": precision_r3,
        "recall_r3": recall_r3,
        "f1_r3": _f1(precision_r3, recall_r3),
    }


def _new_state() -> dict[str, Any]:
    return {
        "dice": [],
        "iou": [],
        "precision_r3": [],
        "recall_r3": [],
        "f1_r3": [],
        "by_domain": defaultdict(lambda: defaultdict(list)),
    }


def _append_state(state: dict[str, Any], *, sample: dict[str, str], **metrics: float) -> None:
    for key, value in metrics.items():
        state[key].append(value)
        state["by_domain"][str(sample["domain"])][key].append(value)


def _summarize_state(candidate: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    for key in ["dice", "iou", "precision_r3", "recall_r3", "f1_r3"]:
        result[key] = _mean(state[key])
    for domain, metrics in sorted(state["by_domain"].items()):
        for key in ["dice", "iou", "precision_r3", "recall_r3", "f1_r3"]:
            result[f"{domain}_{key}"] = _mean(metrics[key])
        result[f"{domain}_samples"] = len(metrics["dice"])
    result["num_samples"] = len(state["dice"])
    return result


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate_id",
        "domain",
        "gate",
        "threshold",
        "dice",
        "delta_dice",
        "iou",
        "delta_iou",
        "precision_r3",
        "delta_precision_r3",
        "recall_r3",
        "delta_recall_r3",
        "f1_r3",
        "delta_f1_r3",
        "animal_f1_r3",
        "phantom_f1_r3",
    ]
    return {key: row.get(key) for key in keys if key in row}


def _disk_structure(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _threshold_id(value: float) -> str:
    return str(value).replace(".", "p")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
