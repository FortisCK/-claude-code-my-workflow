#!/usr/bin/env python3
"""Evaluate multiple Task 1 ensemble weight vectors in one inference pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cathaction.data.task1 import load_task1_mask
from cathaction.metrics.segmentation import (
    dice_score,
    iou_score,
    per_class_dice,
    per_class_iou,
    pixel_accuracy,
)
from cathaction.training.task1_baseline import (
    _append_sample_metrics,
    _mean_or_nan,
    _new_metric_store,
    _path_for_json,
    _per_label_means,
    _resolve_path,
    _summarize_metric_groups,
    metric_labels,
    select_device,
    write_json,
)
from scripts.task1.evaluate_tta_ensemble import (
    ModelSpec,
    _build_loader,
    _load_model_spec,
    _normalize_tta_modes,
    _predict_model_probability,
    _validate_matching_batches,
)
from scripts.task1.evaluate_tta_ensemble_original_space import (
    _probability_to_original_space,
    _validate_original_space_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate many Task 1 ensemble weight vectors after mapping every "
            "model prediction back to original mask space. Model inference is "
            "shared across all candidates."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=4,
        metavar=("NAME", "CONFIG", "CHECKPOINT", "WEIGHT"),
        required=True,
        help=(
            "Model spec. The WEIGHT field is retained for metadata only; "
            "candidate weights are provided separately."
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME:W1,W2,...",
        help="Candidate weight vector. Repeatable. Length must match --model count.",
    )
    parser.add_argument(
        "--tta",
        choices=("none", "hflip", "vflip", "hvflip"),
        action="append",
        default=None,
        help="TTA transform to include. Repeatable. Defaults to none.",
    )
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device(args.device or "auto")
    model_specs = [_load_model_spec(raw, device=device) for raw in args.model]
    _validate_original_space_specs(model_specs)
    candidates = _parse_candidates(args.candidate, expected_length=len(model_specs))

    repo_root = model_specs[0].repo_root
    eval_manifest = (
        _resolve_path(args.eval_manifest, repo_root)
        if args.eval_manifest is not None
        else _resolve_path(model_specs[0].config["paths"]["eval_manifest"], repo_root)
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
            max_samples=args.max_samples,
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
    label_mode = str(model_specs[0].config["data"]["label_mode"])

    metrics_by_candidate = _evaluate_weight_grid(
        model_specs,
        loaders,
        reference_records=reference_records,
        candidates=candidates,
        tta_modes=tta_modes,
        device=device,
        label_mode=label_mode,
    )

    result = {
        "models": [
            {
                "name": spec.name,
                "config_path": _path_for_json(spec.config_path, repo_root),
                "checkpoint_path": _path_for_json(spec.checkpoint_path, repo_root),
                "metadata_weight": spec.weight,
                "resize_mode": str(spec.config["data"].get("resize_mode", "direct")),
                "image_size": spec.config["data"]["image_size"],
            }
            for spec in model_specs
        ],
        "tta_modes": tta_modes,
        "metric_space": "original_mask",
        "eval_manifest": _path_for_json(eval_manifest, repo_root),
        "eval_samples": next(iter(metrics_by_candidate.values()))["num_samples"],
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(device),
        "candidates": [
            {
                "name": candidate.name,
                "raw_weights": candidate.raw_weights,
                "normalized_weights": candidate.normalized_weights,
                "metrics": metrics_by_candidate[candidate.name],
            }
            for candidate in candidates
        ],
    }
    result["ranked_candidates"] = sorted(
        (
            {
                "name": item["name"],
                "dice": item["metrics"]["dice"],
                "label_1": item["metrics"]["per_class_dice"]["label_1"],
                "label_2": item["metrics"]["per_class_dice"]["label_2"],
                "animal": item["metrics"]["by_domain"].get("animal", {}).get("dice"),
                "phantom": item["metrics"]["by_domain"].get("phantom", {}).get("dice"),
            }
            for item in result["candidates"]
        ),
        key=lambda row: row["dice"],
        reverse=True,
    )

    output_json = _resolve_path(args.output_json, repo_root)
    write_json(output_json, result)
    print(json.dumps(result, indent=2))
    return 0


class Candidate:
    def __init__(self, name: str, raw_weights: list[float]) -> None:
        self.name = name
        self.raw_weights = raw_weights
        total = sum(raw_weights)
        if total <= 0:
            raise ValueError(f"Candidate {name!r} must have positive total weight.")
        self.normalized_weights = [float(weight) / total for weight in raw_weights]


def _parse_candidates(values: list[str], *, expected_length: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for value in values:
        if ":" not in value:
            raise ValueError(
                f"Candidate must use NAME:W1,W2,... format, got {value!r}"
            )
        name, weights_text = value.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Candidate name is empty in {value!r}")
        if name in seen:
            raise ValueError(f"Duplicate candidate name: {name!r}")
        seen.add(name)
        weights = [float(part) for part in weights_text.split(",") if part.strip()]
        if len(weights) != expected_length:
            raise ValueError(
                f"Candidate {name!r} has {len(weights)} weights; expected {expected_length}."
            )
        if any(weight < 0 for weight in weights):
            raise ValueError(f"Candidate {name!r} has negative weights: {weights}")
        candidates.append(Candidate(name, weights))
    return candidates


def _new_candidate_store(labels: list[int]) -> dict[str, Any]:
    return {
        "dice_scores": [],
        "iou_scores": [],
        "accuracy_scores": [],
        "per_label_dice": {label: [] for label in labels},
        "per_label_iou": {label: [] for label in labels},
        "domain_metrics": {},
        "collection_metrics": {},
        "num_batches": 0,
    }


def _evaluate_weight_grid(
    model_specs: list[ModelSpec],
    loaders: list[DataLoader[dict[str, Any]]],
    *,
    reference_records: dict[str, dict[str, str]],
    candidates: list[Candidate],
    tta_modes: list[str],
    device: torch.device,
    label_mode: str,
) -> dict[str, dict[str, Any]]:
    repo_root = model_specs[0].repo_root
    labels = metric_labels(label_mode)
    stores = {candidate.name: _new_candidate_store(labels) for candidate in candidates}

    with torch.no_grad():
        for batches in tqdm(
            zip(*loaders),
            total=len(loaders[0]),
            desc="evaluate-weight-grid",
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
            for store in stores.values():
                store["num_batches"] += 1

            for index, sample_id in enumerate(sample_ids):
                record = reference_records[sample_id]
                target_mask = load_task1_mask(
                    _resolve_path(record["mask_path"], repo_root),
                    record["mask_encoding"],
                ).astype(np.uint8)
                original_shape = tuple(int(value) for value in target_mask.shape[-2:])
                original_probabilities = [
                    _probability_to_original_space(
                        probability_batch[index],
                        original_shape=original_shape,
                        resize_mode=str(spec.config["data"].get("resize_mode", "direct")),
                    )
                    for spec, probability_batch in zip(model_specs, batch_probabilities)
                ]

                for candidate in candidates:
                    probability_sum: torch.Tensor | None = None
                    for probability, weight in zip(
                        original_probabilities,
                        candidate.normalized_weights,
                    ):
                        weighted_probability = probability * float(weight)
                        probability_sum = (
                            weighted_probability
                            if probability_sum is None
                            else probability_sum + weighted_probability
                        )
                    if probability_sum is None:
                        raise RuntimeError("No model probabilities were produced.")
                    pred_mask = torch.argmax(probability_sum, dim=0).cpu().numpy().astype(np.uint8)
                    _append_grid_metrics(
                        stores[candidate.name],
                        pred_mask,
                        target_mask,
                        labels=labels,
                        domain=str(record.get("domain", "unknown")),
                        collection=str(record.get("collection", "unknown")),
                    )

    return {name: _summarize_candidate_store(store) for name, store in stores.items()}


def _append_grid_metrics(
    store: dict[str, Any],
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    labels: list[int],
    domain: str,
    collection: str,
) -> None:
    store["dice_scores"].append(dice_score(pred_mask, target_mask, labels=labels))
    store["iou_scores"].append(iou_score(pred_mask, target_mask, labels=labels))
    store["accuracy_scores"].append(pixel_accuracy(pred_mask, target_mask))
    for label, score in per_class_dice(pred_mask, target_mask, labels=labels).items():
        store["per_label_dice"][label].append(score)
    for label, score in per_class_iou(pred_mask, target_mask, labels=labels).items():
        store["per_label_iou"][label].append(score)
    _append_sample_metrics(
        store["domain_metrics"].setdefault(domain, _new_metric_store(labels)),
        pred_mask,
        target_mask,
        labels,
    )
    _append_sample_metrics(
        store["collection_metrics"].setdefault(collection, _new_metric_store(labels)),
        pred_mask,
        target_mask,
        labels,
    )


def _summarize_candidate_store(store: dict[str, Any]) -> dict[str, Any]:
    return {
        "dice": _mean_or_nan(store["dice_scores"]),
        "iou": _mean_or_nan(store["iou_scores"]),
        "miou": _mean_or_nan(store["iou_scores"]),
        "per_class_dice": _per_label_means(store["per_label_dice"]),
        "per_class_iou": _per_label_means(store["per_label_iou"]),
        "pixel_accuracy": _mean_or_nan(store["accuracy_scores"]),
        "num_batches": int(store["num_batches"]),
        "num_samples": len(store["dice_scores"]),
        "by_domain": _summarize_metric_groups(store["domain_metrics"]),
        "by_collection": _summarize_metric_groups(store["collection_metrics"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
