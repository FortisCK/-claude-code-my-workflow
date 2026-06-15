#!/usr/bin/env python3
"""Gate multiclass Task 1 predictions with a binary ROI foreground refiner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from cathaction.training.task1_baseline import (
    _autocast_context,
    _image_normalization_tensors,
    _image_size_tuple,
    _image_to_tensor,
    _normalize_image_tensor,
    _path_for_json,
    _pil_nearest,
    _resolve_path,
    _safe_filename,
    primary_logits_for_label_mode,
    read_manifest,
    read_prediction_manifest,
    select_device,
    write_json,
)
from scripts.task1.evaluate_tta_ensemble import (
    ModelSpec,
    _apply_tta_image,
    _invert_tta_probability,
    _load_model_spec,
    _normalize_tta_modes,
    _normalized_weights,
)
from scripts.task1.evaluate_tta_ensemble_original_space import _probability_to_original_space
from scripts.task1.predict_roi_patch_refine_original_space import _roi_boxes_from_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a binary patch refiner to remove foreground pixels from an "
            "existing multiclass prediction manifest. Surviving foreground keeps "
            "the original label_1 / label_2 assignment."
        )
    )
    parser.add_argument(
        "--refiner-model",
        action="append",
        nargs=4,
        metavar=("NAME", "CONFIG", "CHECKPOINT", "WEIGHT"),
        required=True,
        help="Binary refiner model spec. Repeat for ensembles.",
    )
    parser.add_argument(
        "--tta",
        choices=("none", "hflip", "vflip", "hvflip"),
        action="append",
        default=None,
        help="TTA transform to include for refiner models. Defaults to none.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coarse-predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--roi-size", type=int, default=512)
    parser.add_argument("--roi-stride", type=int, default=256)
    parser.add_argument("--max-rois", type=int, default=16)
    parser.add_argument("--min-foreground-pixels", type=int, default=8)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    parser.add_argument("--patch-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device(args.device or "auto")
    refiner_specs = [_load_model_spec(raw, device=device) for raw in args.refiner_model]
    _validate_refiner_specs(refiner_specs)

    repo_root = refiner_specs[0].repo_root
    manifest = _resolve_path(args.manifest, repo_root)
    coarse_predictions_csv = _resolve_path(args.coarse_predictions_csv, repo_root)
    output_dir = _resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_records = {row["sample_id"]: row for row in read_manifest(manifest)}
    coarse_rows = _selected_prediction_rows(
        read_prediction_manifest(coarse_predictions_csv),
        reference_records=reference_records,
        domains=set(args.domain or []),
        max_samples=args.max_samples,
    )
    tta_modes = _normalize_tta_modes(args.tta)
    weights = _normalized_weights(refiner_specs)

    rows = _write_gated_predictions(
        refiner_specs,
        coarse_rows,
        reference_records=reference_records,
        weights=weights,
        tta_modes=tta_modes,
        repo_root=repo_root,
        output_dir=output_dir,
        device=device,
        threshold=float(args.threshold),
        roi_size=int(args.roi_size),
        roi_stride=int(args.roi_stride),
        max_rois=int(args.max_rois),
        min_foreground_pixels=int(args.min_foreground_pixels),
        nms_iou=float(args.nms_iou),
        patch_batch_size=int(args.patch_batch_size),
    )

    prediction_manifest = output_dir / "predictions.csv"
    with prediction_manifest.open("w", newline="", encoding="utf-8") as handle:
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
                "gated_pixels",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "refiner_models": [
            {
                "name": spec.name,
                "config_path": _path_for_json(spec.config_path, repo_root),
                "checkpoint_path": _path_for_json(spec.checkpoint_path, repo_root),
                "weight": spec.weight,
                "normalized_weight": weights[index],
                "resize_mode": str(spec.config["data"].get("resize_mode", "direct")),
                "image_size": spec.config["data"]["image_size"],
            }
            for index, spec in enumerate(refiner_specs)
        ],
        "tta_modes": tta_modes,
        "manifest": _path_for_json(manifest, repo_root),
        "coarse_predictions_csv": _path_for_json(coarse_predictions_csv, repo_root),
        "prediction_manifest": _path_for_json(prediction_manifest, repo_root),
        "predictions": len(rows),
        "threshold": float(args.threshold),
        "roi_size": int(args.roi_size),
        "roi_stride": int(args.roi_stride),
        "max_rois": int(args.max_rois),
        "min_foreground_pixels": int(args.min_foreground_pixels),
        "nms_iou": float(args.nms_iou),
        "patch_batch_size": int(args.patch_batch_size),
        "device": str(device),
    }
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


def _validate_refiner_specs(specs: list[ModelSpec]) -> None:
    if not specs:
        raise ValueError("At least one --refiner-model spec is required.")
    label_modes = {str(spec.config["data"]["label_mode"]) for spec in specs}
    if label_modes != {"binary_foreground"}:
        raise ValueError(f"Refiner models must be binary_foreground, got {sorted(label_modes)}")
    repo_roots = {spec.repo_root for spec in specs}
    if len(repo_roots) != 1:
        raise ValueError(f"All refiner models must share repo_root, got {repo_roots}")
    for spec in specs:
        if spec.weight < 0:
            raise ValueError(f"Refiner weight must be non-negative for {spec.name}: {spec.weight}")


def _selected_prediction_rows(
    rows: list[dict[str, str]],
    *,
    reference_records: dict[str, dict[str, str]],
    domains: set[str],
    max_samples: int | None,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        record = reference_records.get(sample_id)
        if record is None:
            continue
        if domains and str(record.get("domain", "")) not in domains:
            continue
        selected.append(row)
        if max_samples is not None and len(selected) >= int(max_samples):
            break
    if not selected:
        raise ValueError("No coarse prediction rows matched the manifest/domain filters.")
    return selected


def _write_gated_predictions(
    refiner_specs: list[ModelSpec],
    coarse_rows: list[dict[str, str]],
    *,
    reference_records: dict[str, dict[str, str]],
    weights: list[float],
    tta_modes: list[str],
    repo_root: Path,
    output_dir: Path,
    device: torch.device,
    threshold: float,
    roi_size: int,
    roi_stride: int,
    max_rois: int,
    min_foreground_pixels: int,
    nms_iou: float,
    patch_batch_size: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for row in tqdm(coarse_rows, desc="binary-roi-gate", leave=False):
            sample_id = str(row["sample_id"])
            record = reference_records[sample_id]
            image_path = _resolve_path(record["image_path"], repo_root)
            coarse_path = _resolve_path(row["prediction_path"], repo_root)
            coarse_prediction = _load_prediction_array(coarse_path)
            image_shape = _original_image_shape(image_path)
            if tuple(coarse_prediction.shape[:2]) != image_shape:
                coarse_prediction = _resize_label_mask(coarse_prediction, image_shape)
            boxes = _roi_boxes_from_prediction(
                coarse_prediction,
                roi_size=roi_size,
                roi_stride=roi_stride,
                max_rois=max_rois,
                min_foreground_pixels=min_foreground_pixels,
                nms_iou=nms_iou,
            )
            gate_probability, patch_count = _refiner_gate_probability(
                refiner_specs,
                image_path=image_path,
                boxes=boxes,
                weights=weights,
                tta_modes=tta_modes,
                image_shape=image_shape,
                device=device,
                patch_batch_size=patch_batch_size,
            )
            prediction = coarse_prediction.copy()
            covered = patch_count > 0
            foreground = prediction > 0
            remove = covered & foreground & (gate_probability < threshold)
            prediction[remove] = 0
            filename = f"{_safe_filename(sample_id)}.png"
            prediction_path = output_dir / filename
            Image.fromarray(prediction.astype(np.uint8)).save(prediction_path)
            rows.append(
                {
                    "sample_id": sample_id,
                    "collection": str(record.get("collection", "")),
                    "domain": str(record.get("domain", "")),
                    "prediction_path": _path_for_json(prediction_path, repo_root),
                    "height": str(int(prediction.shape[0])),
                    "width": str(int(prediction.shape[1])),
                    "roi_count": str(len(boxes)),
                    "gated_pixels": str(int(np.count_nonzero(remove))),
                }
            )
    return rows


def _refiner_gate_probability(
    refiner_specs: list[ModelSpec],
    *,
    image_path: Path,
    boxes: list[tuple[int, int, int, int]],
    weights: list[float],
    tta_modes: list[str],
    image_shape: tuple[int, int],
    device: torch.device,
    patch_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    probability_sum = np.zeros((height, width), dtype=np.float32)
    patch_count = np.zeros((height, width), dtype=np.float32)
    if not boxes:
        return probability_sum, patch_count

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        for spec, weight in zip(refiner_specs, weights):
            for start in range(0, len(boxes), max(1, int(patch_batch_size))):
                batch_boxes = boxes[start : start + max(1, int(patch_batch_size))]
                tensors = [
                    _prepare_patch_tensor(
                        image.crop((left, top, right, bottom)),
                        spec,
                    )
                    for left, top, right, bottom in batch_boxes
                ]
                batch = torch.stack(tensors, dim=0).to(device)
                batch_probability = _predict_binary_probability(
                    spec,
                    batch,
                    tta_modes=tta_modes,
                    target_shape=tuple(int(value) for value in batch.shape[-2:]),
                    device=device,
                )
                for index, box in enumerate(batch_boxes):
                    left, top, right, bottom = box
                    crop_shape = (bottom - top, right - left)
                    crop_probability = _probability_to_original_space(
                        batch_probability[index],
                        original_shape=crop_shape,
                        resize_mode=str(spec.config["data"].get("resize_mode", "direct")),
                    )[0]
                    crop_np = crop_probability.detach().cpu().numpy().astype(np.float32)
                    probability_sum[top:bottom, left:right] += crop_np * float(weight)
                    patch_count[top:bottom, left:right] += float(weight)

    gate_probability = np.zeros((height, width), dtype=np.float32)
    covered = patch_count > 0
    gate_probability[covered] = probability_sum[covered] / patch_count[covered]
    return gate_probability, patch_count


def _predict_binary_probability(
    spec: ModelSpec,
    image: torch.Tensor,
    *,
    tta_modes: list[str],
    target_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    probabilities: list[torch.Tensor] = []
    for mode in tta_modes:
        transformed = _apply_tta_image(image, mode)
        with _autocast_context(device=device, use_amp=spec.use_amp):
            logits = spec.model(transformed)
        primary_logits = primary_logits_for_label_mode(
            logits,
            str(spec.config["data"]["label_mode"]),
        )
        probability = torch.sigmoid(primary_logits.float())
        probabilities.append(_invert_tta_probability(probability, mode))
    probability = torch.stack(probabilities, dim=0).mean(dim=0)
    if tuple(probability.shape[-2:]) != target_shape:
        probability = F.interpolate(
            probability,
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )
    return probability


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


def _load_prediction_array(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        array = np.asarray(handle)
    if array.ndim == 3:
        array = array[..., 0]
    return np.asarray(array, dtype=np.uint8)


def _resize_label_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize((width, height), _pil_nearest()),
        dtype=np.uint8,
    )


def _original_image_shape(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return int(height), int(width)


if __name__ == "__main__":
    raise SystemExit(main())
