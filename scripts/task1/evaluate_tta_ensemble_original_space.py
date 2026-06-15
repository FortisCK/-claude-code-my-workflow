#!/usr/bin/env python3
"""Evaluate Task 1 TTA/ensemble checkpoints in original mask space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
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
    _normalized_weights,
    _predict_model_probability,
    _validate_matching_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Task 1 checkpoints with flip TTA and probability ensembling "
            "after mapping every model prediction back to original mask space."
        )
    )
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
    weights = _normalized_weights(model_specs)
    label_mode = str(model_specs[0].config["data"]["label_mode"])

    metrics = _evaluate_original_space(
        model_specs,
        loaders,
        reference_records=reference_records,
        weights=weights,
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
                "weight": spec.weight,
                "normalized_weight": weights[index],
                "resize_mode": str(spec.config["data"].get("resize_mode", "direct")),
                "image_size": spec.config["data"]["image_size"],
            }
            for index, spec in enumerate(model_specs)
        ],
        "tta_modes": tta_modes,
        "metric_space": "original_mask",
        "eval_manifest": _path_for_json(eval_manifest, repo_root),
        "eval_samples": metrics["num_samples"],
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(device),
        "metrics": metrics,
    }
    output_json = _resolve_path(args.output_json, repo_root)
    write_json(output_json, result)
    print(json.dumps(result, indent=2))
    return 0


def _validate_original_space_specs(model_specs: list[ModelSpec]) -> None:
    if not model_specs:
        raise ValueError("At least one --model spec is required.")
    label_modes = {str(spec.config["data"]["label_mode"]) for spec in model_specs}
    if label_modes != {"multiclass_012"}:
        raise ValueError(f"Only multiclass_012 is supported, got {sorted(label_modes)}")
    repo_roots = {spec.repo_root for spec in model_specs}
    if len(repo_roots) != 1:
        raise ValueError(f"All models must share a repo_root, got {sorted(str(p) for p in repo_roots)}")
    for spec in model_specs:
        if spec.weight < 0:
            raise ValueError(f"Model weight must be non-negative for {spec.name}, got {spec.weight}")
        resize_mode = str(spec.config["data"].get("resize_mode", "direct"))
        if resize_mode not in {"direct", "aspect_pad", "letterbox"}:
            raise ValueError(f"Unsupported resize_mode for {spec.name}: {resize_mode!r}")


def _evaluate_original_space(
    model_specs: list[ModelSpec],
    loaders: list[DataLoader[dict[str, Any]]],
    *,
    reference_records: dict[str, dict[str, str]],
    weights: list[float],
    tta_modes: list[str],
    device: torch.device,
    label_mode: str,
) -> dict[str, Any]:
    repo_root = model_specs[0].repo_root
    labels = metric_labels(label_mode)
    dice_scores: list[float] = []
    iou_scores: list[float] = []
    accuracy_scores: list[float] = []
    per_label_dice: dict[int, list[float]] = {label: [] for label in labels}
    per_label_iou: dict[int, list[float]] = {label: [] for label in labels}
    domain_metrics: dict[str, dict[str, Any]] = {}
    collection_metrics: dict[str, dict[str, Any]] = {}
    num_batches = 0

    with torch.no_grad():
        for batches in tqdm(
            zip(*loaders),
            total=len(loaders[0]),
            desc="evaluate-original",
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
            num_batches += 1

            for index, sample_id in enumerate(sample_ids):
                record = reference_records[sample_id]
                target_mask = load_task1_mask(
                    _resolve_path(record["mask_path"], repo_root),
                    record["mask_encoding"],
                ).astype(np.uint8)
                original_shape = tuple(int(value) for value in target_mask.shape[-2:])
                probability_sum: torch.Tensor | None = None

                for spec, probability_batch, weight in zip(
                    model_specs,
                    batch_probabilities,
                    weights,
                ):
                    original_probability = _probability_to_original_space(
                        probability_batch[index],
                        original_shape=original_shape,
                        resize_mode=str(spec.config["data"].get("resize_mode", "direct")),
                    )
                    weighted_probability = original_probability * float(weight)
                    probability_sum = (
                        weighted_probability
                        if probability_sum is None
                        else probability_sum + weighted_probability
                    )

                if probability_sum is None:
                    raise RuntimeError("No model probabilities were produced.")
                pred_mask = torch.argmax(probability_sum, dim=0).cpu().numpy().astype(np.uint8)

                dice_scores.append(dice_score(pred_mask, target_mask, labels=labels))
                iou_scores.append(iou_score(pred_mask, target_mask, labels=labels))
                accuracy_scores.append(pixel_accuracy(pred_mask, target_mask))
                for label, score in per_class_dice(pred_mask, target_mask, labels=labels).items():
                    per_label_dice[label].append(score)
                for label, score in per_class_iou(pred_mask, target_mask, labels=labels).items():
                    per_label_iou[label].append(score)

                domain = str(record.get("domain", "unknown"))
                collection = str(record.get("collection", "unknown"))
                _append_sample_metrics(
                    domain_metrics.setdefault(domain, _new_metric_store(labels)),
                    pred_mask,
                    target_mask,
                    labels,
                )
                _append_sample_metrics(
                    collection_metrics.setdefault(collection, _new_metric_store(labels)),
                    pred_mask,
                    target_mask,
                    labels,
                )

    return {
        "dice": _mean_or_nan(dice_scores),
        "iou": _mean_or_nan(iou_scores),
        "miou": _mean_or_nan(iou_scores),
        "per_class_dice": _per_label_means(per_label_dice),
        "per_class_iou": _per_label_means(per_label_iou),
        "pixel_accuracy": _mean_or_nan(accuracy_scores),
        "num_batches": num_batches,
        "num_samples": len(dice_scores),
        "by_domain": _summarize_metric_groups(domain_metrics),
        "by_collection": _summarize_metric_groups(collection_metrics),
    }


def _probability_to_original_space(
    probability: torch.Tensor,
    *,
    original_shape: tuple[int, int],
    resize_mode: str,
) -> torch.Tensor:
    if probability.ndim != 3:
        raise ValueError(f"Expected C,H,W probability tensor, got {tuple(probability.shape)}")
    if resize_mode == "direct":
        return _resize_probability(probability, original_shape)
    if resize_mode in {"aspect_pad", "letterbox"}:
        canvas_height, canvas_width = (int(value) for value in probability.shape[-2:])
        original_height, original_width = original_shape
        scale = min(
            canvas_width / float(original_width),
            canvas_height / float(original_height),
        )
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        left = (canvas_width - resized_width) // 2
        top = (canvas_height - resized_height) // 2
        cropped = probability[:, top : top + resized_height, left : left + resized_width]
        return _resize_probability(cropped, original_shape)
    raise ValueError(f"Unsupported resize_mode: {resize_mode!r}")


def _resize_probability(
    probability: torch.Tensor,
    output_shape: tuple[int, int],
) -> torch.Tensor:
    return F.interpolate(
        probability.unsqueeze(0),
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


if __name__ == "__main__":
    raise SystemExit(main())
