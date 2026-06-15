#!/usr/bin/env python3
"""Evaluate Task 1 checkpoints with TTA and softmax ensembling."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cathaction.metrics.segmentation import (
    dice_score,
    iou_score,
    per_class_dice,
    per_class_iou,
    pixel_accuracy,
)
from cathaction.training.task1_baseline import (
    Task1SegmentationDataset,
    _append_sample_metrics,
    _autocast_context,
    _image_size_tuple,
    _mean_or_nan,
    _new_metric_store,
    _path_for_json,
    _per_label_means,
    _prediction_item,
    _resolve_path,
    _summarize_metric_groups,
    _target_item,
    build_model,
    load_checkpoint,
    load_config,
    metric_labels,
    primary_logits_for_label_mode,
    select_device,
    write_json,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_path: Path
    checkpoint_path: Path
    weight: float
    config: dict[str, Any]
    repo_root: Path
    model: nn.Module
    use_amp: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Task 1 checkpoints with flip TTA and probability ensembling."
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
    if not model_specs:
        raise ValueError("At least one --model spec is required.")

    _validate_model_specs(model_specs)
    tta_modes = _normalize_tta_modes(args.tta)
    weights = _normalized_weights(model_specs)
    label_mode = str(model_specs[0].config["data"]["label_mode"])
    if label_mode != "multiclass_012":
        raise ValueError(f"Only multiclass_012 is supported for soft ensembling, got {label_mode!r}.")

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
    metrics = _evaluate_ensemble(
        model_specs,
        loaders,
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
            }
            for index, spec in enumerate(model_specs)
        ],
        "tta_modes": tta_modes,
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


def _load_model_spec(raw: list[str], *, device: torch.device) -> ModelSpec:
    name, config_text, checkpoint_text, weight_text = raw
    config_path = Path(config_text).expanduser()
    checkpoint_path = Path(checkpoint_text).expanduser()
    config = load_config(config_path)
    if device.type != "cuda":
        config["training"]["device"] = str(device)
    repo_root = Path(config["paths"]["repo_root"]).expanduser().resolve()
    checkpoint = _resolve_path(checkpoint_path, repo_root)
    model = build_model(config).to(device)
    payload = load_checkpoint(checkpoint, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    use_amp = bool(config["training"].get("use_amp", False)) and device.type == "cuda"
    return ModelSpec(
        name=name,
        config_path=_resolve_path(config_path, repo_root),
        checkpoint_path=checkpoint,
        weight=float(weight_text),
        config=config,
        repo_root=repo_root,
        model=model,
        use_amp=use_amp,
    )


def _validate_model_specs(model_specs: list[ModelSpec]) -> None:
    label_modes = {str(spec.config["data"]["label_mode"]) for spec in model_specs}
    if len(label_modes) != 1:
        raise ValueError(f"All models must use the same label_mode, got {sorted(label_modes)}")
    resize_modes = {str(spec.config["data"].get("resize_mode", "direct")) for spec in model_specs}
    if len(resize_modes) != 1:
        raise ValueError(f"All models must use the same resize_mode, got {sorted(resize_modes)}")
    for spec in model_specs:
        if spec.weight < 0:
            raise ValueError(f"Model weight must be non-negative for {spec.name}, got {spec.weight}")


def _normalize_tta_modes(values: list[str] | None) -> list[str]:
    modes = values or ["none"]
    ordered: list[str] = []
    for mode in modes:
        if mode not in ordered:
            ordered.append(mode)
    if "none" not in ordered:
        ordered.insert(0, "none")
    return ordered


def _normalized_weights(model_specs: list[ModelSpec]) -> list[float]:
    total = sum(spec.weight for spec in model_specs)
    if total <= 0:
        raise ValueError("At least one model weight must be positive.")
    return [spec.weight / total for spec in model_specs]


def _build_loader(
    spec: ModelSpec,
    *,
    eval_manifest: Path,
    max_samples: int | None,
    batch_size: int,
    num_workers: int,
) -> DataLoader[dict[str, Any]]:
    data_cfg = spec.config["data"]
    dataset = Task1SegmentationDataset(
        manifest_csv=eval_manifest,
        repo_root=spec.repo_root,
        image_size=_image_size_tuple(data_cfg["image_size"]),
        resize_mode=str(data_cfg.get("resize_mode", "direct")),
        label_mode=str(data_cfg["label_mode"]),
        max_samples=max_samples,
        normalization=data_cfg.get("normalization"),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def _evaluate_ensemble(
    model_specs: list[ModelSpec],
    loaders: list[DataLoader[dict[str, Any]]],
    *,
    weights: list[float],
    tta_modes: list[str],
    device: torch.device,
    label_mode: str,
) -> dict[str, Any]:
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
        for batches in tqdm(zip(*loaders), total=len(loaders[0]), desc="evaluate", leave=False):
            reference_batch = batches[0]
            _validate_matching_batches(batches)
            target = reference_batch["mask"].to(device)
            target_shape = tuple(int(value) for value in target.shape[-2:])
            probability_sum: torch.Tensor | None = None

            for spec, batch, weight in zip(model_specs, batches, weights):
                image = batch["image"].to(device)
                model_probability = _predict_model_probability(
                    spec,
                    image,
                    tta_modes=tta_modes,
                    target_shape=target_shape,
                    device=device,
                )
                weighted_probability = model_probability * float(weight)
                probability_sum = (
                    weighted_probability
                    if probability_sum is None
                    else probability_sum + weighted_probability
                )

            if probability_sum is None:
                raise RuntimeError("No model probabilities were produced.")
            prediction = torch.argmax(probability_sum, dim=1).cpu().numpy().astype(np.uint8)
            target_np = target.cpu().numpy().astype(np.uint8)
            num_batches += 1

            for index in range(prediction.shape[0]):
                pred_mask = _prediction_item(prediction, index, label_mode)
                target_mask = _target_item(target_np, index, label_mode)
                dice_scores.append(dice_score(pred_mask, target_mask, labels=labels))
                iou_scores.append(iou_score(pred_mask, target_mask, labels=labels))
                accuracy_scores.append(pixel_accuracy(pred_mask, target_mask))
                for label, score in per_class_dice(pred_mask, target_mask, labels=labels).items():
                    per_label_dice[label].append(score)
                for label, score in per_class_iou(pred_mask, target_mask, labels=labels).items():
                    per_label_iou[label].append(score)
                domain = str(reference_batch["domain"][index])
                collection = str(reference_batch["collection"][index])
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


def _predict_model_probability(
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
        probabilities.append(
            _invert_tta_probability(torch.softmax(primary_logits.float(), dim=1), mode)
        )
    probability = torch.stack(probabilities, dim=0).mean(dim=0)
    if tuple(probability.shape[-2:]) != target_shape:
        probability = F.interpolate(probability, size=target_shape, mode="bilinear", align_corners=False)
    return probability


def _apply_tta_image(image: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return image
    if mode == "hflip":
        return torch.flip(image, dims=(-1,))
    if mode == "vflip":
        return torch.flip(image, dims=(-2,))
    if mode == "hvflip":
        return torch.flip(image, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA mode: {mode}")


def _invert_tta_probability(probability: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return probability
    if mode == "hflip":
        return torch.flip(probability, dims=(-1,))
    if mode == "vflip":
        return torch.flip(probability, dims=(-2,))
    if mode == "hvflip":
        return torch.flip(probability, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA mode: {mode}")


def _validate_matching_batches(batches: tuple[dict[str, Any], ...]) -> None:
    reference = batches[0]
    reference_ids = [str(value) for value in reference["sample_id"]]
    for batch in batches[1:]:
        sample_ids = [str(value) for value in batch["sample_id"]]
        if sample_ids != reference_ids:
            raise ValueError(
                "Ensemble loaders are misaligned: "
                f"reference={reference_ids[:3]}, current={sample_ids[:3]}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
