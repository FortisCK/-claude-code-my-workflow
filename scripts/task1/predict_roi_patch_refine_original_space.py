#!/usr/bin/env python3
"""Refine original-space Task 1 predictions with coarse-model ROI patch inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cathaction.training.task1_baseline import (
    _image_normalization_tensors,
    _image_size_tuple,
    _image_to_tensor,
    _normalize_image_tensor,
    _path_for_json,
    _resolve_path,
    _safe_filename,
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
from scripts.task1.evaluate_tta_ensemble_original_space import (
    _probability_to_original_space,
    _validate_original_space_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write original-space predictions by fusing a full-frame coarse model or "
            "ensemble with ROI-local patch-refiner model predictions."
        )
    )
    parser.add_argument(
        "--coarse-model",
        action="append",
        nargs=4,
        metavar=("NAME", "CONFIG", "CHECKPOINT", "WEIGHT"),
        required=True,
        help="Full-frame model spec. Repeat for ensembles.",
    )
    parser.add_argument(
        "--patch-model",
        action="append",
        nargs=4,
        metavar=("NAME", "CONFIG", "CHECKPOINT", "WEIGHT"),
        required=True,
        help="Patch-refiner model spec. Repeat for patch ensembles.",
    )
    parser.add_argument(
        "--tta",
        choices=("none", "hflip", "vflip", "hvflip"),
        action="append",
        default=None,
        help="TTA transform to include for both coarse and patch models. Defaults to none.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--roi-size", type=int, default=384)
    parser.add_argument("--roi-stride", type=int, default=192)
    parser.add_argument("--max-rois", type=int, default=8)
    parser.add_argument("--min-foreground-pixels", type=int, default=8)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    parser.add_argument("--fuse-alpha", type=float, default=0.55)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device(args.device or "auto")
    coarse_specs = [_load_model_spec(raw, device=device) for raw in args.coarse_model]
    patch_specs = [_load_model_spec(raw, device=device) for raw in args.patch_model]
    _validate_original_space_specs(coarse_specs)
    _validate_original_space_specs(patch_specs)

    repo_root = coarse_specs[0].repo_root
    if {spec.repo_root for spec in patch_specs} != {repo_root}:
        raise ValueError("Coarse and patch models must share the same repo_root.")
    manifest = (
        _resolve_path(args.manifest, repo_root)
        if args.manifest is not None
        else _resolve_path(coarse_specs[0].config["paths"]["eval_manifest"], repo_root)
    )
    output_dir = _resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or min(
        int(spec.config["training"]["batch_size"]) for spec in coarse_specs
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else min(int(spec.config["training"].get("num_workers", 0)) for spec in coarse_specs)
    )
    coarse_loaders = [
        _build_loader(
            spec,
            eval_manifest=manifest,
            max_samples=args.max_samples,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for spec in coarse_specs
    ]
    reference_records = {
        str(record["sample_id"]): record
        for record in coarse_loaders[0].dataset.records  # type: ignore[attr-defined]
    }
    tta_modes = _normalize_tta_modes(args.tta)
    coarse_weights = _normalized_weights(coarse_specs)
    patch_weights = _normalized_weights(patch_specs)

    rows = _write_refined_predictions(
        coarse_specs,
        patch_specs,
        coarse_loaders,
        reference_records=reference_records,
        coarse_weights=coarse_weights,
        patch_weights=patch_weights,
        tta_modes=tta_modes,
        output_dir=output_dir,
        device=device,
        roi_size=int(args.roi_size),
        roi_stride=int(args.roi_stride),
        max_rois=int(args.max_rois),
        min_foreground_pixels=int(args.min_foreground_pixels),
        nms_iou=float(args.nms_iou),
        fuse_alpha=float(args.fuse_alpha),
    )
    manifest_path = output_dir / "predictions.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "collection",
                "domain",
                "prediction_path",
                "height",
                "width",
                "roi_count",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "coarse_models": _model_summary(coarse_specs, coarse_weights, repo_root),
        "patch_models": _model_summary(patch_specs, patch_weights, repo_root),
        "tta_modes": tta_modes,
        "metric_space": "original_image",
        "manifest": _path_for_json(manifest, repo_root),
        "prediction_dir": _path_for_json(output_dir, repo_root),
        "prediction_manifest": _path_for_json(manifest_path, repo_root),
        "predictions": len(rows),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(device),
        "roi_size": int(args.roi_size),
        "roi_stride": int(args.roi_stride),
        "max_rois": int(args.max_rois),
        "min_foreground_pixels": int(args.min_foreground_pixels),
        "nms_iou": float(args.nms_iou),
        "fuse_alpha": float(args.fuse_alpha),
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


def _model_summary(
    specs: list[ModelSpec],
    weights: list[float],
    repo_root: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "config_path": _path_for_json(spec.config_path, repo_root),
            "checkpoint_path": _path_for_json(spec.checkpoint_path, repo_root),
            "weight": spec.weight,
            "normalized_weight": weights[index],
            "resize_mode": str(spec.config["data"].get("resize_mode", "direct")),
            "image_size": spec.config["data"]["image_size"],
        }
        for index, spec in enumerate(specs)
    ]


def _write_refined_predictions(
    coarse_specs: list[ModelSpec],
    patch_specs: list[ModelSpec],
    coarse_loaders: list[Any],
    *,
    reference_records: dict[str, dict[str, str]],
    coarse_weights: list[float],
    patch_weights: list[float],
    tta_modes: list[str],
    output_dir: Path,
    device: torch.device,
    roi_size: int,
    roi_stride: int,
    max_rois: int,
    min_foreground_pixels: int,
    nms_iou: float,
    fuse_alpha: float,
) -> list[dict[str, str]]:
    repo_root = coarse_specs[0].repo_root
    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batches in tqdm(
            zip(*coarse_loaders),
            total=len(coarse_loaders[0]),
            desc="roi-refine-original",
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
                for spec, batch in zip(coarse_specs, batches)
            ]

            for index, sample_id in enumerate(sample_ids):
                record = reference_records[sample_id]
                image_path = _resolve_path(record["image_path"], repo_root)
                original_shape = _original_image_shape(image_path)
                coarse_probability = _coarse_original_probability(
                    coarse_specs,
                    batch_probabilities,
                    index=index,
                    weights=coarse_weights,
                    original_shape=original_shape,
                )
                coarse_prediction = torch.argmax(coarse_probability, dim=0).numpy().astype(np.uint8)
                boxes = _roi_boxes_from_prediction(
                    coarse_prediction,
                    roi_size=roi_size,
                    roi_stride=roi_stride,
                    max_rois=max_rois,
                    min_foreground_pixels=min_foreground_pixels,
                    nms_iou=nms_iou,
                )
                refined_probability = _fuse_patch_probabilities(
                    coarse_probability,
                    patch_specs,
                    patch_weights=patch_weights,
                    image_path=image_path,
                    boxes=boxes,
                    tta_modes=tta_modes,
                    device=device,
                    fuse_alpha=fuse_alpha,
                )
                prediction = torch.argmax(refined_probability, dim=0).numpy().astype(np.uint8)
                filename = f"{_safe_filename(sample_id)}.png"
                prediction_path = output_dir / filename
                Image.fromarray(prediction).save(prediction_path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "collection": str(record.get("collection", "")),
                        "domain": str(record.get("domain", "")),
                        "prediction_path": _path_for_json(prediction_path, repo_root),
                        "height": str(int(prediction.shape[0])),
                        "width": str(int(prediction.shape[1])),
                        "roi_count": str(len(boxes)),
                    }
                )
    return rows


def _coarse_original_probability(
    coarse_specs: list[ModelSpec],
    batch_probabilities: list[torch.Tensor],
    *,
    index: int,
    weights: list[float],
    original_shape: tuple[int, int],
) -> torch.Tensor:
    probability_sum: torch.Tensor | None = None
    for spec, probability_batch, weight in zip(coarse_specs, batch_probabilities, weights):
        original_probability = _probability_to_original_space(
            probability_batch[index],
            original_shape=original_shape,
            resize_mode=str(spec.config["data"].get("resize_mode", "direct")),
        ).detach().cpu()
        weighted_probability = original_probability * float(weight)
        probability_sum = weighted_probability if probability_sum is None else probability_sum + weighted_probability
    if probability_sum is None:
        raise RuntimeError("No coarse probabilities were produced.")
    return probability_sum


def _fuse_patch_probabilities(
    coarse_probability: torch.Tensor,
    patch_specs: list[ModelSpec],
    *,
    patch_weights: list[float],
    image_path: Path,
    boxes: list[tuple[int, int, int, int]],
    tta_modes: list[str],
    device: torch.device,
    fuse_alpha: float,
) -> torch.Tensor:
    if not boxes:
        return coarse_probability
    channel_count, height, width = coarse_probability.shape
    patch_sum = torch.zeros((channel_count, height, width), dtype=torch.float32)
    patch_count = torch.zeros((1, height, width), dtype=torch.float32)

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        for box in boxes:
            left, top, right, bottom = box
            crop = image.crop((left, top, right, bottom))
            crop_shape = (bottom - top, right - left)
            crop_probability: torch.Tensor | None = None
            for spec, weight in zip(patch_specs, patch_weights):
                patch_tensor = _prepare_patch_tensor(crop, spec).unsqueeze(0).to(device)
                patch_probability = _predict_model_probability(
                    spec,
                    patch_tensor,
                    tta_modes=tta_modes,
                    target_shape=tuple(int(value) for value in patch_tensor.shape[-2:]),
                    device=device,
                )[0]
                original_patch_probability = _probability_to_original_space(
                    patch_probability,
                    original_shape=crop_shape,
                    resize_mode=str(spec.config["data"].get("resize_mode", "direct")),
                ).detach().cpu()
                weighted_probability = original_patch_probability * float(weight)
                crop_probability = (
                    weighted_probability
                    if crop_probability is None
                    else crop_probability + weighted_probability
                )
            if crop_probability is None:
                continue
            patch_sum[:, top:bottom, left:right] += crop_probability
            patch_count[:, top:bottom, left:right] += 1.0

    mask = patch_count > 0
    if not torch.any(mask):
        return coarse_probability
    patch_average = patch_sum / patch_count.clamp_min(1.0)
    alpha = min(1.0, max(0.0, float(fuse_alpha)))
    refined = coarse_probability.clone()
    mask_2d = mask.squeeze(0)
    refined[:, mask_2d] = (
        (1.0 - alpha) * coarse_probability[:, mask_2d]
        + alpha * patch_average[:, mask_2d]
    )
    return refined


def _prepare_patch_tensor(crop: Image.Image, spec: ModelSpec) -> torch.Tensor:
    data_cfg = spec.config["data"]
    image = _image_to_tensor(
        crop,
        _image_size_tuple(data_cfg["image_size"]),
        str(data_cfg.get("resize_mode", "direct")),
    )
    return _normalize_image_tensor(
        image,
        _image_normalization_tensors(data_cfg.get("normalization")),
    )


def _roi_boxes_from_prediction(
    prediction: np.ndarray,
    *,
    roi_size: int,
    roi_stride: int,
    max_rois: int,
    min_foreground_pixels: int,
    nms_iou: float,
) -> list[tuple[int, int, int, int]]:
    height, width = prediction.shape
    roi_height = min(max(1, int(roi_size)), height)
    roi_width = min(max(1, int(roi_size)), width)
    stride = max(1, int(roi_stride))
    min_pixels = max(1, int(min_foreground_pixels))
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    masks = [prediction == 1, prediction == 2, prediction > 0]
    for mask in masks:
        if int(np.count_nonzero(mask)) < min_pixels:
            continue
        for top in _window_starts(height, roi_height, stride):
            for left in _window_starts(width, roi_width, stride):
                right = left + roi_width
                bottom = top + roi_height
                score = int(np.count_nonzero(mask[top:bottom, left:right]))
                if score >= min_pixels:
                    candidates.append((score, (left, top, right, bottom)))

    selected: list[tuple[int, int, int, int]] = []
    for _, box in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(_box_iou(box, kept) > nms_iou for kept in selected):
            continue
        selected.append(box)
        if len(selected) >= max_rois:
            break
    return selected


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if window >= length:
        return [0]
    max_start = length - window
    starts = list(range(0, max_start + 1, stride))
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = float((right - left) * (bottom - top))
    first_area = float((first[2] - first[0]) * (first[3] - first[1]))
    second_area = float((second[2] - second[0]) * (second[3] - second[1]))
    return intersection / max(first_area + second_area - intersection, 1.0)


def _original_image_shape(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return int(height), int(width)


if __name__ == "__main__":
    raise SystemExit(main())
