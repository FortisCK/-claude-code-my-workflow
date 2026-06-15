"""Minimal PyTorch baseline loop for CATHACTION Task 1 segmentation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from tqdm import tqdm

from cathaction.data.task1 import load_task1_mask, mask_to_binary_foreground
from cathaction.metrics.segmentation import (
    dice_score,
    iou_score,
    per_class_dice,
    per_class_iou,
    pixel_accuracy,
)
from cathaction.models import TinyUNet


DEFAULT_CONFIG: dict[str, Any] = {
    "name": "task1_baseline_unet_smoke",
    "seed": 20260520,
    "paths": {
        "repo_root": ".",
        "train_manifest": "configs/task1/splits/released_train.csv",
        "eval_manifest": "configs/task1/splits/released_eval.csv",
        "output_dir": "outputs/task1/baseline_unet_smoke",
    },
    "data": {
        "image_size": [128, 128],
        "resize_mode": "direct",
        "label_mode": "binary_foreground",
        "threshold": 0.5,
        "max_train_samples": 16,
        "max_eval_samples": 8,
        "max_predict_samples": 4,
        "normalization": {
            "enabled": False,
        },
        "augmentation": {
            "enabled": False,
        },
    },
    "model": {
        "name": "tiny_unet",
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": 8,
    },
    "loss": {
        "name": "cross_entropy",
        "class_weights": None,
    },
    "training": {
        "device": "auto",
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "log_interval": 5,
        "monitor_metric": "eval.dice",
        "monitor_mode": "max",
        "use_amp": False,
        "gradient_clip_norm": None,
        "sampling": {
            "domain_balanced": False,
        },
    },
}


class Task1SegmentationDataset(Dataset[dict[str, Any]]):
    """Dataset backed by a Task 1 split manifest CSV."""

    def __init__(
        self,
        *,
        manifest_csv: Path | str,
        repo_root: Path | str = ".",
        image_size: tuple[int, int] = (128, 128),
        resize_mode: str = "direct",
        label_mode: str = "binary_foreground",
        max_samples: int | None = None,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        normalization: dict[str, Any] | None = None,
        patch_sampling: dict[str, Any] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.image_size = tuple(int(value) for value in image_size)
        self.resize_mode = resize_mode
        self.label_mode = label_mode
        self.transform = transform
        self.normalization = _image_normalization_tensors(normalization)
        self.patch_sampling = _patch_sampling_config(patch_sampling)
        self.records = read_manifest(manifest_csv, max_samples=max_samples)
        if not self.records:
            raise ValueError(f"No samples found in manifest: {manifest_csv}")
        self.patch_prediction_paths = _patch_prediction_path_map(
            self.patch_sampling,
            self.repo_root,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = _resolve_path(record["image_path"], self.repo_root)

        # No-GT inference path (hidden test / Docker submission): a manifest may
        # carry an empty mask_path or mask_encoding == "none". The dummy mask is
        # never used for prediction (the inference scripts read only "image"),
        # but a valid target tensor must exist so batches collate.
        raw_mask_path = record.get("mask_path", "")
        mask_encoding = record.get("mask_encoding", "")
        if not raw_mask_path or mask_encoding in ("", "none"):
            mask_path: Path | None = None
            mask_array = np.zeros(self.image_size, dtype=np.uint8)
        else:
            mask_path = _resolve_path(raw_mask_path, self.repo_root)
            mask_array = load_task1_mask(mask_path, mask_encoding)
        if self.patch_sampling is not None:
            patch_prediction_path = (
                self.patch_prediction_paths.get(str(record["sample_id"]))
                if self.patch_prediction_paths is not None
                else None
            )
            image, mask_array = _load_patch_image_and_mask(
                image_path,
                mask_array,
                self.patch_sampling,
                label_mode=self.label_mode,
                prediction_path=patch_prediction_path,
            )
            image = _image_to_tensor(image, self.image_size, self.resize_mode)
        else:
            image = _load_image_tensor(image_path, self.image_size, self.resize_mode)
        target = _prepare_mask_tensor(
            mask_array,
            self.image_size,
            self.label_mode,
            self.resize_mode,
        )
        image, target = _apply_optional_transform(
            image,
            target,
            label_mode=self.label_mode,
            transform=self.transform,
        )
        image = _normalize_image_tensor(image, self.normalization)

        return {
            "image": image,
            "mask": target,
            "sample_id": record["sample_id"],
            "collection": record["collection"],
            "domain": record["domain"],
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None else "",
        }


def load_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return merge_config(DEFAULT_CONFIG, loaded)


def merge_config(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(defaults)
    _deep_update(result, overrides)
    return result


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    paths = config["paths"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    repo_root = Path(paths["repo_root"]).expanduser().resolve()
    output_dir = _resolve_path(paths["output_dir"], repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(training_cfg["device"])
    image_size = _image_size_tuple(data_cfg["image_size"])
    resize_mode = str(data_cfg.get("resize_mode", "direct"))
    train_transform = build_train_transform(data_cfg)
    use_amp = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    gradient_clip_norm = _optional_float(training_cfg.get("gradient_clip_norm"))

    train_dataset = Task1SegmentationDataset(
        manifest_csv=_resolve_path(paths["train_manifest"], repo_root),
        repo_root=repo_root,
        image_size=image_size,
        resize_mode=resize_mode,
        label_mode=data_cfg["label_mode"],
        max_samples=_optional_int(data_cfg.get("max_train_samples")),
        transform=train_transform,
        normalization=data_cfg.get("normalization"),
        patch_sampling=data_cfg.get("patch_sampling"),
    )
    eval_dataset = Task1SegmentationDataset(
        manifest_csv=_resolve_path(paths["eval_manifest"], repo_root),
        repo_root=repo_root,
        image_size=image_size,
        resize_mode=resize_mode,
        label_mode=data_cfg["label_mode"],
        max_samples=_optional_int(data_cfg.get("max_eval_samples")),
        normalization=data_cfg.get("normalization"),
    )
    train_sampler = build_train_sampler(train_dataset, training_cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(training_cfg["num_workers"]),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
    )

    model = build_model(config).to(device)
    initial_checkpoint = training_cfg.get("initial_checkpoint")
    if initial_checkpoint:
        initial_checkpoint_path = _resolve_path(initial_checkpoint, repo_root)
        initial_payload = load_checkpoint(initial_checkpoint_path, device)
        load_result = model.load_state_dict(
            initial_payload["model_state_dict"],
            strict=bool(training_cfg.get("initial_checkpoint_strict", True)),
        )
        print(
            json.dumps(
                {
                    "initial_checkpoint_path": _path_for_json(initial_checkpoint_path, repo_root),
                    "missing_keys": list(getattr(load_result, "missing_keys", [])),
                    "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
                },
                indent=2,
            )
        )
    loss_fn = build_loss(config, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, Any]] = []
    monitor_metric = str(training_cfg.get("monitor_metric", "eval.dice"))
    monitor_mode = str(training_cfg.get("monitor_mode", "max"))
    best_metric: float | None = None
    best_epoch_metrics: dict[str, Any] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs = int(training_cfg["epochs"])
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    checkpoint_path = output_dir / "checkpoint.pt"
    for epoch in range(1, epochs + 1):
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device=device,
            epoch=epoch,
            log_interval=int(training_cfg["log_interval"]),
            use_amp=use_amp,
            scaler=scaler,
            gradient_clip_norm=gradient_clip_norm,
        )
        eval_metrics = evaluate_model(
            model,
            eval_loader,
            loss_fn,
            device=device,
            label_mode=data_cfg["label_mode"],
            threshold=float(data_cfg["threshold"]),
            use_amp=use_amp,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval": eval_metrics,
        }
        history.append(epoch_metrics)
        current_metric = _metric_from_path(epoch_metrics, monitor_metric)
        if _is_better_metric(current_metric, best_metric, monitor_mode):
            best_metric = current_metric
            best_epoch_metrics = copy.deepcopy(epoch_metrics)
            best_state_dict = _clone_state_dict_to_cpu(model)
            _save_training_checkpoint(
                best_checkpoint_path,
                model_state_dict=best_state_dict,
                config=config,
                metrics=_training_metrics_payload(
                    config=config,
                    device=device,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    train_sampler=train_sampler,
                    monitor_metric=monitor_metric,
                    monitor_mode=monitor_mode,
                    best_epoch_metrics=best_epoch_metrics,
                    history=history,
                    checkpoint_path=checkpoint_path,
                    best_checkpoint_path=best_checkpoint_path,
                    repo_root=repo_root,
                ),
            )
        print(json.dumps(epoch_metrics, indent=2))
        latest_metrics = _training_metrics_payload(
            config=config,
            device=device,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_sampler=train_sampler,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            best_epoch_metrics=best_epoch_metrics,
            history=history,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            repo_root=repo_root,
        )
        _save_training_checkpoint(
            checkpoint_path,
            model_state_dict=model.state_dict(),
            config=config,
            metrics=latest_metrics,
        )
        write_json(output_dir / "metrics.json", latest_metrics)

    metrics = _training_metrics_payload(
        config=config,
        device=device,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_sampler=train_sampler,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        best_epoch_metrics=best_epoch_metrics,
        history=history,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        repo_root=repo_root,
    )
    if best_metric is not None and best_state_dict is not None:
        _save_training_checkpoint(
            best_checkpoint_path,
            model_state_dict=best_state_dict,
            config=config,
            metrics=metrics,
        )

    _save_training_checkpoint(
        checkpoint_path,
        model_state_dict=model.state_dict(),
        config=config,
        metrics=metrics,
    )
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "resolved_config.json", config)
    return metrics


def _training_metrics_payload(
    *,
    config: dict[str, Any],
    device: torch.device,
    train_dataset: Dataset[Any],
    eval_dataset: Dataset[Any],
    train_sampler: Sampler[Any] | None,
    monitor_metric: str,
    monitor_mode: str,
    best_epoch_metrics: dict[str, Any] | None,
    history: list[dict[str, Any]],
    checkpoint_path: Path,
    best_checkpoint_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "config_name": config["name"],
        "device": str(device),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "train_sampler": train_sampler.__class__.__name__ if train_sampler is not None else None,
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "best": best_epoch_metrics or {},
        "history": history,
        "final": history[-1] if history else {},
        "best_checkpoint_path": _path_for_json(best_checkpoint_path, repo_root),
        "checkpoint_path": _path_for_json(checkpoint_path, repo_root),
    }


def _save_training_checkpoint(
    path: Path,
    *,
    model_state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def evaluate_from_config(
    config: dict[str, Any],
    *,
    checkpoint_path: Path | str | None = None,
    output_json: Path | str | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    paths = config["paths"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    repo_root = Path(paths["repo_root"]).expanduser().resolve()
    output_dir = _resolve_path(paths["output_dir"], repo_root)
    checkpoint = _checkpoint_path(checkpoint_path, output_dir)
    device = select_device(training_cfg["device"])
    use_amp = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"

    model = build_model(config).to(device)
    checkpoint_payload = load_checkpoint(checkpoint, device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])

    image_size = _image_size_tuple(data_cfg["image_size"])
    resize_mode = str(data_cfg.get("resize_mode", "direct"))
    dataset = Task1SegmentationDataset(
        manifest_csv=_resolve_path(paths["eval_manifest"], repo_root),
        repo_root=repo_root,
        image_size=image_size,
        resize_mode=resize_mode,
        label_mode=data_cfg["label_mode"],
        max_samples=max_samples if max_samples is not None else _optional_int(data_cfg.get("max_eval_samples")),
        normalization=data_cfg.get("normalization"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
    )
    metrics = evaluate_model(
        model,
        loader,
        build_loss(config, device=device),
        device=device,
        label_mode=data_cfg["label_mode"],
        threshold=float(data_cfg["threshold"]),
        use_amp=use_amp,
    )
    result = {
        "checkpoint_path": _path_for_json(checkpoint, repo_root),
        "eval_samples": len(dataset),
        "metrics": metrics,
    }
    if output_json is not None:
        write_json(_resolve_path(output_json, repo_root), result)
    return result


def evaluate_predictions_from_config(
    config: dict[str, Any],
    *,
    predictions_csv: Path | str,
    output_json: Path | str | None = None,
) -> dict[str, Any]:
    paths = config["paths"]
    data_cfg = config["data"]
    repo_root = Path(paths["repo_root"]).expanduser().resolve()
    resize_mode = str(data_cfg.get("resize_mode", "direct"))
    eval_records = {
        record["sample_id"]: record
        for record in read_manifest(_resolve_path(paths["eval_manifest"], repo_root))
    }
    prediction_rows = read_prediction_manifest(_resolve_path(predictions_csv, repo_root))

    dice_scores: list[float] = []
    iou_scores: list[float] = []
    accuracy_scores: list[float] = []
    labels = metric_labels(data_cfg["label_mode"])
    per_label_dice: dict[int, list[float]] = {label: [] for label in labels}
    per_label_iou: dict[int, list[float]] = {label: [] for label in labels}
    domain_metrics: dict[str, dict[str, Any]] = {}
    collection_metrics: dict[str, dict[str, Any]] = {}
    missing_sample_ids: list[str] = []

    for prediction_row in prediction_rows:
        sample_id = prediction_row["sample_id"]
        record = eval_records.get(sample_id)
        if record is None:
            missing_sample_ids.append(sample_id)
            continue

        pred_path = _resolve_path(prediction_row["prediction_path"], repo_root)
        prediction = _load_prediction_mask(pred_path, data_cfg["label_mode"])
        target_raw = load_task1_mask(
            _resolve_path(record["mask_path"], repo_root),
            record["mask_encoding"],
        )
        target = _resize_mask_for_mode(
            target_raw,
            prediction.shape,
            data_cfg["label_mode"],
            resize_mode=resize_mode,
        )
        dice_scores.append(dice_score(prediction, target, labels=labels))
        iou_scores.append(iou_score(prediction, target, labels=labels))
        accuracy_scores.append(pixel_accuracy(prediction, target))
        for label, score in per_class_dice(prediction, target, labels=labels).items():
            per_label_dice[label].append(score)
        for label, score in per_class_iou(prediction, target, labels=labels).items():
            per_label_iou[label].append(score)
        domain = str(record.get("domain", "unknown"))
        collection = str(record.get("collection", "unknown"))
        _append_sample_metrics(
            domain_metrics.setdefault(domain, _new_metric_store(labels)),
            prediction,
            target,
            labels,
        )
        _append_sample_metrics(
            collection_metrics.setdefault(collection, _new_metric_store(labels)),
            prediction,
            target,
            labels,
        )

    result = {
        "prediction_manifest": _path_for_json(_resolve_path(predictions_csv, repo_root), repo_root),
        "predictions": len(prediction_rows),
        "matched_predictions": len(dice_scores),
        "missing_sample_ids": missing_sample_ids,
        "metrics": {
            "dice": _mean_or_nan(dice_scores),
            "iou": _mean_or_nan(iou_scores),
            "miou": _mean_or_nan(iou_scores),
            "per_class_dice": _per_label_means(per_label_dice),
            "per_class_iou": _per_label_means(per_label_iou),
            "pixel_accuracy": _mean_or_nan(accuracy_scores),
            "num_samples": len(dice_scores),
            "by_domain": _summarize_metric_groups(domain_metrics),
            "by_collection": _summarize_metric_groups(collection_metrics),
        },
    }
    if output_json is not None:
        write_json(_resolve_path(output_json, repo_root), result)
    return result


def predict_from_config(
    config: dict[str, Any],
    *,
    checkpoint_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    paths = config["paths"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    repo_root = Path(paths["repo_root"]).expanduser().resolve()
    baseline_output_dir = _resolve_path(paths["output_dir"], repo_root)
    checkpoint = _checkpoint_path(checkpoint_path, baseline_output_dir)
    prediction_dir = _resolve_path(output_dir, repo_root) if output_dir else baseline_output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(training_cfg["device"])
    use_amp = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    model = build_model(config).to(device)
    checkpoint_payload = load_checkpoint(checkpoint, device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()

    image_size = _image_size_tuple(data_cfg["image_size"])
    resize_mode = str(data_cfg.get("resize_mode", "direct"))
    dataset = Task1SegmentationDataset(
        manifest_csv=_resolve_path(paths["eval_manifest"], repo_root),
        repo_root=repo_root,
        image_size=image_size,
        resize_mode=resize_mode,
        label_mode=data_cfg["label_mode"],
        max_samples=max_samples
        if max_samples is not None
        else _optional_int(data_cfg.get("max_predict_samples")),
        normalization=data_cfg.get("normalization"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
    )

    rows: list[dict[str, str]] = []
    threshold = float(data_cfg["threshold"])
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", leave=False):
            with _autocast_context(device=device, use_amp=use_amp):
                logits = model(batch["image"].to(device))
            pred = predict_logits(
                logits,
                label_mode=data_cfg["label_mode"],
                threshold=threshold,
            )
            for i, sample_id in enumerate(batch["sample_id"]):
                filename = f"{_safe_filename(str(sample_id))}.png"
                prediction_path = prediction_dir / filename
                _save_prediction_mask(prediction_path, pred[i], data_cfg["label_mode"])
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "collection": str(batch["collection"][i]),
                        "domain": str(batch["domain"][i]),
                        "prediction_path": _path_for_json(prediction_path, repo_root),
                    }
                )

    manifest_path = prediction_dir / "predictions.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "collection", "domain", "prediction_path"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "checkpoint_path": _path_for_json(checkpoint, repo_root),
        "prediction_dir": _path_for_json(prediction_dir, repo_root),
        "prediction_manifest": _path_for_json(manifest_path, repo_root),
        "predictions": len(rows),
    }
    write_json(prediction_dir / "summary.json", result)
    return result


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    label_mode = str(config["data"]["label_mode"])
    expected_out_channels = expected_model_out_channels(label_mode)
    if _uses_toolness_auxiliary(config):
        if label_mode != "multiclass_012":
            raise ValueError("toolness auxiliary supervision is only supported for multiclass_012.")
        expected_out_channels += 1
    if int(model_cfg["out_channels"]) != expected_out_channels:
        raise ValueError(
            f"label_mode={label_mode} expects "
            f"out_channels={expected_out_channels}, got {model_cfg['out_channels']}"
        )
    if model_cfg["name"] == "tiny_unet":
        return TinyUNet(
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            base_channels=int(model_cfg["base_channels"]),
        )
    if model_cfg["name"] == "monai_unet":
        try:
            from monai.networks.nets import UNet
        except ImportError as error:
            raise ImportError(
                "model.name='monai_unet' requires MONAI. Use an environment with "
                "monai installed, such as cardiac-diffusion on this workstation."
            ) from error

        return UNet(
            spatial_dims=int(model_cfg.get("spatial_dims", 2)),
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            channels=tuple(int(value) for value in model_cfg["channels"]),
            strides=tuple(int(value) for value in model_cfg["strides"]),
            num_res_units=int(model_cfg.get("num_res_units", 0)),
            act=model_cfg.get("act", "PRELU"),
            norm=model_cfg.get("norm", "BATCH"),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if model_cfg["name"] == "smp":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError(
                "model.name='smp' requires segmentation-models-pytorch. Install "
                "it in the active training environment before running Stage 2."
            ) from error

        architecture = str(model_cfg.get("architecture", "Unet"))
        if not hasattr(smp, architecture):
            raise ValueError(f"Unsupported SMP architecture: {architecture}")
        encoder_weights = model_cfg.get("encoder_weights", "imagenet")
        if encoder_weights in {"none", "None", "null", "Null"}:
            encoder_weights = None
        model_kwargs = dict(model_cfg.get("kwargs") or {})
        return getattr(smp, architecture)(
            encoder_name=str(model_cfg["encoder_name"]),
            encoder_weights=encoder_weights,
            in_channels=int(model_cfg["in_channels"]),
            classes=int(model_cfg["out_channels"]),
            activation=None,
            **model_kwargs,
        )
    raise ValueError(f"Unsupported model: {model_cfg['name']}")


def build_loss(config: dict[str, Any], *, device: torch.device | None = None) -> nn.Module:
    label_mode = config["data"]["label_mode"]
    loss_cfg = config.get("loss", {})
    if label_mode == "binary_foreground":
        name = str(loss_cfg.get("name", "bce"))
        pos_weight = _binary_pos_weight_tensor(loss_cfg.get("pos_weight"), device=device)
        if name in {"bce", "binary_cross_entropy", "cross_entropy"}:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if name == "binary_dice_bce":
            return _BinaryDiceBceLoss(
                lambda_bce=float(loss_cfg.get("lambda_bce", 0.5)),
                lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
                pos_weight=pos_weight,
            )
        raise ValueError(f"Unsupported binary_foreground loss: {name}")
    if label_mode == "multiclass_012":
        name = loss_cfg.get("name", "cross_entropy")
        weight = _class_weight_tensor(loss_cfg.get("class_weights"), device=device)
        if name in {"cross_entropy", "weighted_cross_entropy"}:
            return nn.CrossEntropyLoss(weight=weight)
        if name == "monai_dice_ce":
            try:
                from monai.losses import DiceCELoss
            except ImportError as error:
                raise ImportError(
                    "loss.name='monai_dice_ce' requires MONAI. Use an environment "
                    "with monai installed, such as cardiac-diffusion on this workstation."
                ) from error

            loss = DiceCELoss(
                include_background=bool(loss_cfg.get("include_background", True)),
                to_onehot_y=True,
                softmax=True,
                weight=weight,
                lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
                lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
                label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            )
            return _MonaiMulticlassLossWrapper(loss)
        if name == "monai_dice_ce_toolness_aux":
            try:
                from monai.losses import DiceCELoss
            except ImportError as error:
                raise ImportError(
                    "loss.name='monai_dice_ce_toolness_aux' requires MONAI. Use an "
                    "environment with monai installed, such as cardiac-diffusion "
                    "on this workstation."
                ) from error

            dice_ce = DiceCELoss(
                include_background=bool(loss_cfg.get("include_background", True)),
                to_onehot_y=True,
                softmax=True,
                weight=weight,
                lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
                lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
                label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            )
            return _DiceCeToolnessAuxLoss(
                dice_ce=_MonaiMulticlassLossWrapper(dice_ce),
                lambda_toolness=float(loss_cfg.get("lambda_toolness", 0.4)),
                lambda_toolness_bce=float(loss_cfg.get("lambda_toolness_bce", 0.5)),
                lambda_toolness_dice=float(loss_cfg.get("lambda_toolness_dice", 1.0)),
            )
        if name == "monai_dice_ce_toolness_cbdice_hd":
            try:
                from monai.losses import DiceCELoss, HausdorffDTLoss
            except ImportError as error:
                raise ImportError(
                    "loss.name='monai_dice_ce_toolness_cbdice_hd' requires MONAI. "
                    "Use an environment with monai installed, such as "
                    "cardiac-diffusion on this workstation."
                ) from error

            dice_ce = DiceCELoss(
                include_background=bool(loss_cfg.get("include_background", True)),
                to_onehot_y=True,
                softmax=True,
                weight=weight,
                lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
                lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
                label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            )
            hd_loss = HausdorffDTLoss(
                include_background=bool(loss_cfg.get("hd_include_background", False)),
                to_onehot_y=True,
                softmax=True,
                alpha=float(loss_cfg.get("hd_alpha", 2.0)),
                reduction=str(loss_cfg.get("hd_reduction", "mean")),
            )
            return _DiceCeToolnessCbDiceHdLoss(
                dice_ce=_MonaiMulticlassLossWrapper(dice_ce),
                hd_loss=hd_loss,
                lambda_toolness=float(loss_cfg.get("lambda_toolness", 0.4)),
                lambda_toolness_bce=float(loss_cfg.get("lambda_toolness_bce", 0.5)),
                lambda_toolness_dice=float(loss_cfg.get("lambda_toolness_dice", 1.0)),
                lambda_cbdice=float(loss_cfg.get("lambda_cbdice", 0.0)),
                cbdice_iterations=int(loss_cfg.get("cbdice_iterations", 10)),
                cbdice_warmup_epochs=int(loss_cfg.get("cbdice_warmup_epochs", 0)),
                cbdice_threshold=float(loss_cfg.get("cbdice_threshold", 0.5)),
                lambda_hd=float(loss_cfg.get("lambda_hd", 0.0)),
                hd_warmup_epochs=int(loss_cfg.get("hd_warmup_epochs", 0)),
            )
        if name == "monai_dice_ce_cldice":
            try:
                from monai.losses import DiceCELoss
            except ImportError as error:
                raise ImportError(
                    "loss.name='monai_dice_ce_cldice' requires MONAI. Use an "
                    "environment with monai installed, such as cardiac-diffusion "
                    "on this workstation."
                ) from error

            dice_ce = DiceCELoss(
                include_background=bool(loss_cfg.get("include_background", True)),
                to_onehot_y=True,
                softmax=True,
                weight=weight,
                lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
                lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
                label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            )
            return _DiceCeClDiceLoss(
                dice_ce=_MonaiMulticlassLossWrapper(dice_ce),
                lambda_cldice=float(loss_cfg.get("lambda_cldice", 0.3)),
                cldice_labels=[
                    int(label)
                    for label in loss_cfg.get("cldice_labels", metric_labels(label_mode))
                ],
                cldice_class_weights=loss_cfg.get("cldice_class_weights"),
                cldice_iterations=int(loss_cfg.get("cldice_iterations", 10)),
                cldice_warmup_epochs=int(loss_cfg.get("cldice_warmup_epochs", 0)),
            )
        raise ValueError(f"Unsupported multiclass loss: {name}")
    raise ValueError(f"Unsupported label mode for loss: {label_mode}")


class _BinaryDiceBceLoss(nn.Module):
    """Binary foreground BCE plus soft Dice for sparse tool masks."""

    def __init__(
        self,
        *,
        lambda_bce: float,
        lambda_dice: float,
        pos_weight: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.lambda_bce = float(lambda_bce)
        self.lambda_dice = float(lambda_dice)
        if pos_weight is None:
            self.register_buffer("pos_weight", torch.empty(0, dtype=torch.float32))
        else:
            self.register_buffer("pos_weight", pos_weight.detach().float().reshape(1))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(
                "binary Dice+BCE expects logits shaped B,1,H,W, "
                f"got {tuple(logits.shape)}"
            )
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.ndim != logits.ndim:
            raise ValueError(
                "binary target must be shaped B,H,W or B,1,H,W, "
                f"got logits={tuple(logits.shape)}, target={tuple(target.shape)}"
            )
        target = target.float().to(device=logits.device)
        loss = logits.new_tensor(0.0)
        if self.lambda_bce > 0:
            pos_weight = self.pos_weight.to(logits.device) if self.pos_weight.numel() else None
            loss = loss + self.lambda_bce * F.binary_cross_entropy_with_logits(
                logits.float(),
                target,
                pos_weight=pos_weight,
            )
        if self.lambda_dice > 0:
            loss = loss + self.lambda_dice * _binary_soft_dice_loss(
                logits.float(),
                target,
            )
        return loss


class _MonaiMulticlassLossWrapper(nn.Module):
    """Adapt B,H,W class-index masks to MONAI's expected B,1,H,W target shape."""

    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        self.loss = loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        return self.loss(logits, target)


class _DiceCeClDiceLoss(nn.Module):
    """MONAI DiceCE plus class-wise soft-clDice for thin structures."""

    def __init__(
        self,
        *,
        dice_ce: nn.Module,
        lambda_cldice: float,
        cldice_labels: list[int],
        cldice_class_weights: list[float] | tuple[float, ...] | None,
        cldice_iterations: int,
        cldice_warmup_epochs: int,
    ) -> None:
        super().__init__()
        self.dice_ce = dice_ce
        self.lambda_cldice = float(lambda_cldice)
        self.cldice_labels = [int(label) for label in cldice_labels]
        self.cldice_iterations = int(cldice_iterations)
        self.cldice_warmup_epochs = int(cldice_warmup_epochs)
        self.current_epoch = 0
        if cldice_class_weights is None:
            weights = torch.ones(len(self.cldice_labels), dtype=torch.float32)
        else:
            if len(cldice_class_weights) != len(self.cldice_labels):
                raise ValueError(
                    "cldice_class_weights must match cldice_labels length, "
                    f"got weights={cldice_class_weights}, labels={self.cldice_labels}"
                )
            weights = torch.tensor(
                [float(weight) for weight in cldice_class_weights],
                dtype=torch.float32,
            )
        self.register_buffer("cldice_class_weights", weights)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.dice_ce(logits, target)
        if self.lambda_cldice <= 0 or self.current_epoch <= self.cldice_warmup_epochs:
            return loss
        return loss + self.lambda_cldice * _soft_cldice_loss(
            logits,
            target,
            labels=self.cldice_labels,
            class_weights=self.cldice_class_weights.to(logits.device),
            iterations=self.cldice_iterations,
        )


class _DiceCeToolnessAuxLoss(nn.Module):
    """Three-class DiceCE plus binary tool-vs-background auxiliary supervision."""

    def __init__(
        self,
        *,
        dice_ce: nn.Module,
        lambda_toolness: float,
        lambda_toolness_bce: float,
        lambda_toolness_dice: float,
    ) -> None:
        super().__init__()
        self.dice_ce = dice_ce
        self.lambda_toolness = float(lambda_toolness)
        self.lambda_toolness_bce = float(lambda_toolness_bce)
        self.lambda_toolness_dice = float(lambda_toolness_dice)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] < 4:
            raise ValueError(
                "toolness auxiliary loss expects logits shaped B,4,H,W or wider, "
                f"got {tuple(logits.shape)}"
            )
        primary_loss = self.dice_ce(
            primary_logits_for_label_mode(logits, "multiclass_012"),
            target,
        )
        if self.lambda_toolness <= 0:
            return primary_loss

        target_indices = target.squeeze(1) if target.ndim == logits.ndim else target
        if target_indices.ndim != logits.ndim - 1:
            raise ValueError(
                "toolness auxiliary target must be class indices shaped B,H,W or B,1,H,W, "
                f"got logits={tuple(logits.shape)}, target={tuple(target.shape)}"
            )
        toolness_logits = logits[:, 3:4].float()
        toolness_target = (target_indices > 0).float().unsqueeze(1).to(toolness_logits.device)
        toolness_loss = logits.new_tensor(0.0)
        if self.lambda_toolness_bce > 0:
            toolness_loss = toolness_loss + self.lambda_toolness_bce * F.binary_cross_entropy_with_logits(
                toolness_logits,
                toolness_target,
            )
        if self.lambda_toolness_dice > 0:
            toolness_loss = toolness_loss + self.lambda_toolness_dice * _binary_soft_dice_loss(
                toolness_logits,
                toolness_target,
            )
        return primary_loss + self.lambda_toolness * toolness_loss


class _DiceCeToolnessCbDiceHdLoss(nn.Module):
    """Three-class DiceCE plus toolness, foreground-union cbDice, and HD loss."""

    def __init__(
        self,
        *,
        dice_ce: nn.Module,
        hd_loss: nn.Module,
        lambda_toolness: float,
        lambda_toolness_bce: float,
        lambda_toolness_dice: float,
        lambda_cbdice: float,
        cbdice_iterations: int,
        cbdice_warmup_epochs: int,
        cbdice_threshold: float,
        lambda_hd: float,
        hd_warmup_epochs: int,
    ) -> None:
        super().__init__()
        self.dice_ce = dice_ce
        self.hd_loss = hd_loss
        self.lambda_toolness = float(lambda_toolness)
        self.lambda_toolness_bce = float(lambda_toolness_bce)
        self.lambda_toolness_dice = float(lambda_toolness_dice)
        self.lambda_cbdice = float(lambda_cbdice)
        self.cbdice_iterations = int(cbdice_iterations)
        self.cbdice_warmup_epochs = int(cbdice_warmup_epochs)
        self.cbdice_threshold = float(cbdice_threshold)
        self.lambda_hd = float(lambda_hd)
        self.hd_warmup_epochs = int(hd_warmup_epochs)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] < 4:
            raise ValueError(
                "toolness/cbDice/HD loss expects logits shaped B,4,H,W or wider, "
                f"got {tuple(logits.shape)}"
            )
        primary_logits = primary_logits_for_label_mode(logits, "multiclass_012")
        target_indices = _target_indices_for_multiclass(logits, target)

        loss = self.dice_ce(primary_logits, target_indices)
        if self.lambda_toolness > 0:
            toolness_logits = logits[:, 3:4].float()
            toolness_target = (target_indices > 0).float().unsqueeze(1).to(toolness_logits.device)
            toolness_loss = logits.new_tensor(0.0)
            if self.lambda_toolness_bce > 0:
                toolness_loss = toolness_loss + self.lambda_toolness_bce * F.binary_cross_entropy_with_logits(
                    toolness_logits,
                    toolness_target,
                )
            if self.lambda_toolness_dice > 0:
                toolness_loss = toolness_loss + self.lambda_toolness_dice * _binary_soft_dice_loss(
                    toolness_logits,
                    toolness_target,
                )
            loss = loss + self.lambda_toolness * toolness_loss

        if self.lambda_cbdice > 0 and self.current_epoch > self.cbdice_warmup_epochs:
            loss = loss + self.lambda_cbdice * _foreground_union_cbdice_loss(
                primary_logits,
                target_indices,
                iterations=self.cbdice_iterations,
                threshold=self.cbdice_threshold,
            )
        if self.lambda_hd > 0 and self.current_epoch > self.hd_warmup_epochs:
            binary_logits = _foreground_union_binary_logits(primary_logits)
            binary_target = (target_indices > 0).long().unsqueeze(1)
            loss = loss + self.lambda_hd * self.hd_loss(binary_logits, binary_target)
        return loss


def _target_indices_for_multiclass(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_indices = target.squeeze(1) if target.ndim == logits.ndim else target
    if target_indices.ndim != logits.ndim - 1:
        raise ValueError(
            "multiclass target must be class indices shaped B,H,W or B,1,H,W, "
            f"got logits={tuple(logits.shape)}, target={tuple(target.shape)}"
        )
    return target_indices.long()


def _binary_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.to(dtype=probability.dtype, device=probability.device)
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _soft_cldice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    labels: list[int],
    class_weights: torch.Tensor,
    iterations: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not labels:
        return logits.new_tensor(0.0)
    if target.ndim == logits.ndim:
        target = target.squeeze(1)
    if target.ndim != logits.ndim - 1:
        raise ValueError(
            "soft-clDice target must be class indices shaped B,H,W or B,1,H,W, "
            f"got logits={tuple(logits.shape)}, target={tuple(target.shape)}"
        )

    num_classes = int(logits.shape[1])
    for label in labels:
        if label < 0 or label >= num_classes:
            raise ValueError(f"cldice label {label} is outside model classes 0..{num_classes - 1}")

    probabilities = torch.softmax(logits.float(), dim=1)
    one_hot = (
        F.one_hot(target.long(), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .to(dtype=probabilities.dtype, device=probabilities.device)
    )
    weights = class_weights.to(dtype=probabilities.dtype, device=probabilities.device)
    weights = weights / weights.sum().clamp_min(eps)

    label_losses: list[torch.Tensor] = []
    for label in labels:
        prediction = probabilities[:, label : label + 1]
        ground_truth = one_hot[:, label : label + 1]
        prediction_skeleton = _soft_skeleton(prediction, iterations=iterations)
        target_skeleton = _soft_skeleton(ground_truth, iterations=iterations)

        dims = (1, 2, 3)
        target_present = ground_truth.sum(dim=dims) > 0
        if not torch.any(target_present):
            label_losses.append(logits.new_tensor(0.0))
            continue

        topology_precision = (
            (prediction_skeleton * ground_truth).sum(dim=dims)
            / prediction_skeleton.sum(dim=dims).clamp_min(eps)
        )
        topology_sensitivity = (
            (target_skeleton * prediction).sum(dim=dims)
            / target_skeleton.sum(dim=dims).clamp_min(eps)
        )
        cldice = (
            (2.0 * topology_precision * topology_sensitivity + eps)
            / (topology_precision + topology_sensitivity + eps)
        )
        label_losses.append(1.0 - cldice[target_present].mean())

    losses = torch.stack(label_losses)
    return torch.sum(losses * weights)


def _foreground_union_cbdice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int,
    threshold: float,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Foreground-union cbDice with morphological skeletonization.

    This follows the official cbDice adaptation pattern for custom frameworks:
    collapse all foreground logits into one differentiable foreground
    probability, skeletonize hard masks, then weight centerline-boundary overlap
    by distance-transform/radius terms.
    """

    if target.ndim == logits.ndim:
        target = target.squeeze(1)
    if target.ndim != logits.ndim - 1:
        raise ValueError(
            "cbDice target must be class indices shaped B,H,W or B,1,H,W, "
            f"got logits={tuple(logits.shape)}, target={tuple(target.shape)}"
        )
    if logits.ndim != 4:
        raise ValueError(f"cbDice currently expects 2D logits B,C,H,W, got {tuple(logits.shape)}")

    binary_logits = _foreground_union_binary_logits(logits)
    foreground_probability = torch.softmax(binary_logits.float(), dim=1)[:, 1:2]
    with torch.no_grad():
        target_foreground = (target > 0).float().unsqueeze(1).to(foreground_probability.device)
        prediction_hard = (foreground_probability > float(threshold)).float()
        prediction_skeleton_hard = _soft_skeleton(prediction_hard, iterations=iterations)
        target_skeleton = _soft_skeleton(target_foreground, iterations=iterations)
    prediction_skeleton_probability = prediction_skeleton_hard * foreground_probability

    q_vl, q_slvl, q_sl = _cbdice_weights(
        target_foreground,
        target_skeleton,
        prob_flag=False,
        threshold=threshold,
    )
    q_vp, q_spvp, q_sp = _cbdice_weights(
        foreground_probability,
        prediction_skeleton_probability,
        prob_flag=True,
        threshold=threshold,
    )

    w_tprec = (
        torch.sum(q_sp * q_vl) + smooth
    ) / (torch.sum(_combine_cbdice_tensors(q_spvp, q_slvl, q_sp)) + smooth)
    w_tsens = (
        torch.sum(q_sl * q_vp) + smooth
    ) / (torch.sum(_combine_cbdice_tensors(q_slvl, q_spvp, q_sl)) + smooth)
    cbdice = (2.0 * w_tprec * w_tsens + 1e-6) / (w_tprec + w_tsens + 1e-6)
    return 1.0 - cbdice.clamp(0.0, 1.0)


def _foreground_union_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] < 3:
        raise ValueError(f"foreground-union logits expect B,3,H,W or wider, got {tuple(logits.shape)}")
    foreground_logits = torch.max(logits[:, 1:3], dim=1, keepdim=True)[0]
    return torch.cat([logits[:, :1], foreground_logits], dim=1)


def _cbdice_weights(
    mask_input: torch.Tensor,
    skeleton_input: torch.Tensor,
    *,
    prob_flag: bool,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask_input.ndim != 4 or skeleton_input.ndim != 4:
        raise ValueError(
            "cbDice weights expect B,1,H,W tensors, "
            f"got mask={tuple(mask_input.shape)}, skeleton={tuple(skeleton_input.shape)}"
        )
    if prob_flag:
        mask_probability = mask_input.float()
        skeleton_probability = skeleton_input.float()
        mask = (mask_probability > float(threshold)).float()
        skeleton = (skeleton_probability > float(threshold)).float()
    else:
        mask_probability = mask_input.float()
        skeleton_probability = skeleton_input.float()
        mask = mask_probability.float()
        skeleton = skeleton_probability.float()

    distances = _batched_distance_transform(mask)
    distances = distances.to(dtype=mask_probability.dtype, device=mask_probability.device)
    distances = distances * mask
    skeleton_radius = torch.zeros_like(distances, dtype=mask_probability.dtype)
    skeleton_radius = torch.where(skeleton > 0, distances, skeleton_radius)
    dist_map_norm = torch.zeros_like(distances, dtype=mask_probability.dtype)
    skeleton_radius_norm = torch.zeros_like(distances, dtype=mask_probability.dtype)
    inverse_radius_norm = torch.zeros_like(distances, dtype=mask_probability.dtype)

    for index in range(int(mask.shape[0])):
        radius = skeleton_radius[index]
        radius_max = torch.clamp(radius.max(), min=1.0)
        radius_min = torch.clamp(radius[radius > 0].min() if torch.any(radius > 0) else radius.new_tensor(1.0), min=1.0)
        clipped_distances = torch.clamp(distances[index], max=float(radius_max.detach().cpu()))
        dist_map_norm[index] = clipped_distances / radius_max
        skeleton_radius_norm[index] = radius / radius_max
        inverse = (radius_max - radius + radius_min) / radius_max
        inverse_radius_norm[index] = torch.where(skeleton[index] > 0, inverse, torch.zeros_like(inverse))

    if prob_flag:
        return (
            dist_map_norm * mask_probability,
            skeleton_radius_norm * mask_probability,
            inverse_radius_norm * skeleton_probability,
        )
    return (
        dist_map_norm * mask,
        skeleton_radius_norm * mask,
        inverse_radius_norm * skeleton,
    )


def _batched_distance_transform(mask: torch.Tensor) -> torch.Tensor:
    try:
        from monai.transforms import distance_transform_edt
    except ImportError as error:
        raise ImportError(
            "cbDice requires MONAI distance_transform_edt. Use an environment "
            "with monai installed, such as cardiac-diffusion on this workstation."
        ) from error

    outputs: list[torch.Tensor] = []
    for item in mask.detach():
        item_2d = item.to(device="cpu", dtype=torch.uint8)
        distance = distance_transform_edt(item_2d)
        if not isinstance(distance, torch.Tensor):
            distance = torch.as_tensor(distance)
        outputs.append(distance.to(device=mask.device, dtype=torch.float32))
    return torch.stack(outputs, dim=0)


def _combine_cbdice_tensors(
    first: torch.Tensor,
    second: torch.Tensor,
    skeleton: torch.Tensor,
) -> torch.Tensor:
    first_skeleton = first * skeleton
    second_skeleton = second * skeleton
    combined = second_skeleton.clone()
    mask_first_only = (first != 0) & (second == 0)
    combined[mask_first_only] = first_skeleton[mask_first_only]
    return combined


def _soft_skeleton(mask: torch.Tensor, *, iterations: int) -> torch.Tensor:
    mask = mask.clamp(0.0, 1.0)
    skeleton = F.relu(mask - _soft_open(mask))
    for _ in range(max(0, int(iterations))):
        mask = _soft_erode(mask)
        delta = F.relu(mask - _soft_open(mask))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def _soft_erode(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"soft morphology expects B,C,H,W tensors, got {tuple(mask.shape)}")
    vertical = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"soft morphology expects B,C,H,W tensors, got {tuple(mask.shape)}")
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def _soft_open(mask: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(mask))


def expected_model_out_channels(label_mode: str) -> int:
    if label_mode == "binary_foreground":
        return 1
    if label_mode == "multiclass_012":
        return 3
    raise ValueError(f"Unsupported label mode: {label_mode}")


def metric_labels(label_mode: str) -> list[int]:
    if label_mode == "binary_foreground":
        return [1]
    if label_mode == "multiclass_012":
        return [1, 2]
    raise ValueError(f"Unsupported label mode: {label_mode}")


def primary_logits_for_label_mode(logits: torch.Tensor, label_mode: str) -> torch.Tensor:
    """Return logits for the task labels, excluding any auxiliary output channels."""

    expected_channels = expected_model_out_channels(label_mode)
    if logits.ndim < 2:
        raise ValueError(f"Expected logits with channel dimension, got {tuple(logits.shape)}")
    if int(logits.shape[1]) < expected_channels:
        raise ValueError(
            f"label_mode={label_mode} requires at least {expected_channels} logits, "
            f"got {int(logits.shape[1])}"
        )
    return logits[:, :expected_channels]


def _uses_toolness_auxiliary(config: dict[str, Any]) -> bool:
    model_cfg = config.get("model") or {}
    loss_cfg = config.get("loss") or {}
    return (
        bool(model_cfg.get("toolness_auxiliary", False))
        or bool(model_cfg.get("auxiliary_toolness", False))
        or str(loss_cfg.get("name", ""))
        in {
            "monai_dice_ce_toolness_aux",
            "monai_dice_ce_toolness_cbdice_hd",
        }
    )


def _new_metric_store(labels: list[int]) -> dict[str, Any]:
    return {
        "dice": [],
        "iou": [],
        "pixel_accuracy": [],
        "per_class_dice": {label: [] for label in labels},
        "per_class_iou": {label: [] for label in labels},
    }


def _append_sample_metrics(
    store: dict[str, Any],
    prediction: np.ndarray,
    target: np.ndarray,
    labels: list[int],
) -> None:
    store["dice"].append(dice_score(prediction, target, labels=labels))
    store["iou"].append(iou_score(prediction, target, labels=labels))
    store["pixel_accuracy"].append(pixel_accuracy(prediction, target))
    for label, score in per_class_dice(prediction, target, labels=labels).items():
        store["per_class_dice"][label].append(score)
    for label, score in per_class_iou(prediction, target, labels=labels).items():
        store["per_class_iou"][label].append(score)


def _summarize_metric_store(store: dict[str, Any]) -> dict[str, Any]:
    return {
        "dice": _mean_or_nan(store["dice"]),
        "iou": _mean_or_nan(store["iou"]),
        "miou": _mean_or_nan(store["iou"]),
        "per_class_dice": _per_label_means(store["per_class_dice"]),
        "per_class_iou": _per_label_means(store["per_class_iou"]),
        "pixel_accuracy": _mean_or_nan(store["pixel_accuracy"]),
        "num_samples": len(store["dice"]),
    }


def _summarize_metric_groups(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        group_name: _summarize_metric_store(store)
        for group_name, store in sorted(groups.items())
    }


def build_train_transform(
    data_cfg: dict[str, Any],
) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None:
    augmentation_cfg = data_cfg.get("augmentation") or {}
    if not augmentation_cfg.get("enabled", False):
        return None
    try:
        from monai.transforms import Compose, RandAffined, RandFlipd, RandGaussianNoised
    except ImportError as error:
        raise ImportError(
            "data.augmentation.enabled=true requires MONAI. Use an environment "
            "with monai installed, such as cardiac-diffusion on this workstation."
        ) from error

    transforms: list[Callable[..., Any]] = []
    flip_prob = float(augmentation_cfg.get("horizontal_flip_prob", 0.0))
    if flip_prob > 0:
        transforms.append(
            RandFlipd(keys=["image", "mask"], prob=flip_prob, spatial_axis=1)
        )

    affine_prob = float(augmentation_cfg.get("affine_prob", 0.0))
    if affine_prob > 0:
        translate = float(augmentation_cfg.get("translate_pixels", 0.0))
        scale = float(augmentation_cfg.get("scale_range", 0.0))
        transforms.append(
            RandAffined(
                keys=["image", "mask"],
                prob=affine_prob,
                rotate_range=float(augmentation_cfg.get("rotate_range", 0.0)),
                translate_range=(translate, translate),
                scale_range=(scale, scale),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    noise_prob = float(augmentation_cfg.get("gaussian_noise_prob", 0.0))
    if noise_prob > 0:
        transforms.append(
            RandGaussianNoised(
                keys=["image"],
                prob=noise_prob,
                mean=0.0,
                std=float(augmentation_cfg.get("gaussian_noise_std", 0.01)),
            )
        )

    return Compose(transforms) if transforms else None


def build_train_sampler(
    dataset: Task1SegmentationDataset,
    training_cfg: dict[str, Any],
) -> Sampler[int] | None:
    sampling_cfg = training_cfg.get("sampling") or {}
    if not bool(sampling_cfg.get("domain_balanced", False)):
        return None

    domain_counts = Counter(record["domain"] for record in dataset.records)
    if not domain_counts:
        return None
    weights = [
        1.0 / float(domain_counts[record["domain"]])
        for record in dataset.records
    ]
    num_samples = int(sampling_cfg.get("num_samples") or len(dataset.records))
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    log_interval: int,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    gradient_clip_norm: float | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device=device, use_amp=use_amp):
            logits = model(image)
            loss = loss_fn(logits, target)
        if use_amp:
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if log_interval > 0 and step % log_interval == 0:
            progress.set_postfix(loss=f"{loss_value:.4f}")
    return float(np.mean(losses))


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    loss_fn: nn.Module,
    *,
    device: torch.device,
    label_mode: str,
    threshold: float,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    dice_scores: list[float] = []
    iou_scores: list[float] = []
    accuracy_scores: list[float] = []
    labels = metric_labels(label_mode)
    per_label_dice: dict[int, list[float]] = {label: [] for label in labels}
    per_label_iou: dict[int, list[float]] = {label: [] for label in labels}
    domain_metrics: dict[str, dict[str, Any]] = {}
    collection_metrics: dict[str, dict[str, Any]] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            with _autocast_context(device=device, use_amp=use_amp):
                logits = model(image)
                loss = loss_fn(logits, target)
            losses.append(float(loss.detach().cpu()))

            prediction = predict_logits(
                logits,
                label_mode=label_mode,
                threshold=threshold,
            )
            target_np = target.cpu().numpy().astype(np.uint8)

            for i in range(prediction.shape[0]):
                pred_mask = _prediction_item(prediction, i, label_mode)
                target_mask = _target_item(target_np, i, label_mode)
                dice_scores.append(dice_score(pred_mask, target_mask, labels=labels))
                iou_scores.append(iou_score(pred_mask, target_mask, labels=labels))
                accuracy_scores.append(pixel_accuracy(pred_mask, target_mask))
                for label, score in per_class_dice(pred_mask, target_mask, labels=labels).items():
                    per_label_dice[label].append(score)
                for label, score in per_class_iou(pred_mask, target_mask, labels=labels).items():
                    per_label_iou[label].append(score)
                domain = str(batch["domain"][i])
                collection = str(batch["collection"][i])
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
        "loss": _mean_or_nan(losses),
        "dice": _mean_or_nan(dice_scores),
        "iou": _mean_or_nan(iou_scores),
        "miou": _mean_or_nan(iou_scores),
        "per_class_dice": _per_label_means(per_label_dice),
        "per_class_iou": _per_label_means(per_label_iou),
        "pixel_accuracy": _mean_or_nan(accuracy_scores),
        "num_batches": len(losses),
        "num_samples": len(dice_scores),
        "by_domain": _summarize_metric_groups(domain_metrics),
        "by_collection": _summarize_metric_groups(collection_metrics),
    }


def read_manifest(path: Path | str, *, max_samples: int | None = None) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    required = {"image_path", "mask_path", "mask_encoding", "sample_id", "collection", "domain"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing columns: {sorted(missing)}")
    return rows


def read_prediction_manifest(path: Path | str) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "prediction_path"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(
            f"Prediction manifest {manifest_path} is missing columns: {sorted(missing)}"
        )
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: Path | str, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal Task 1 U-Net baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/task1/baseline_unet.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def parse_evaluate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Task 1 baseline checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/task1/baseline_unet.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-manifest", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def parse_predict_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write PNG predictions from a Task 1 baseline checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/task1/baseline_unet.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main_train(argv: list[str] | None = None) -> int:
    args = parse_train_args(argv)
    config = load_config(args.config)
    if args.output_dir is not None:
        config["paths"]["output_dir"] = args.output_dir.as_posix()
    if args.max_train_samples is not None:
        config["data"]["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        config["data"]["max_eval_samples"] = args.max_eval_samples
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device is not None:
        config["training"]["device"] = args.device
    metrics = train_from_config(config)
    print(json.dumps(metrics, indent=2))
    return 0


def main_evaluate(argv: list[str] | None = None) -> int:
    args = parse_evaluate_args(argv)
    config = load_config(args.config)
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.eval_manifest is not None:
        config["paths"]["eval_manifest"] = args.eval_manifest.as_posix()
        config["data"]["max_eval_samples"] = None
    if args.predictions_csv is not None:
        result = evaluate_predictions_from_config(
            config,
            predictions_csv=args.predictions_csv,
            output_json=args.output_json,
        )
    else:
        result = evaluate_from_config(
            config,
            checkpoint_path=args.checkpoint,
            output_json=args.output_json,
            max_samples=args.max_samples,
        )
    print(json.dumps(result, indent=2))
    return 0


def main_predict(argv: list[str] | None = None) -> int:
    args = parse_predict_args(argv)
    config = load_config(args.config)
    if args.device is not None:
        config["training"]["device"] = args.device
    result = predict_from_config(
        config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )
    print(json.dumps(result, indent=2))
    return 0


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _metric_from_path(payload: dict[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Metric path {path!r} not found in payload.")
        value = value[part]
    return float(value)


def _is_better_metric(current: float, best: float | None, mode: str) -> bool:
    if mode not in {"max", "min"}:
        raise ValueError(f"monitor_mode must be 'max' or 'min', got {mode!r}")
    if best is None:
        return True
    if mode == "max":
        return current > best
    return current < best


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _class_weight_tensor(
    weights: list[float] | tuple[float, ...] | None,
    *,
    device: torch.device | None,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if len(weights) != 3:
        raise ValueError(f"multiclass_012 class_weights must contain 3 values, got {weights}")
    return torch.tensor([float(weight) for weight in weights], dtype=torch.float32, device=device)


def _binary_pos_weight_tensor(value: Any, *, device: torch.device | None) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


def _apply_optional_transform(
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    label_mode: str,
    transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if transform is None:
        return image, target

    if label_mode == "binary_foreground":
        transformed = transform({"image": image, "mask": target.float()})
        transformed_image = torch.as_tensor(transformed["image"]).float()
        transformed_target = (torch.as_tensor(transformed["mask"]).float() > 0.5).float()
        return transformed_image.contiguous(), transformed_target.contiguous()

    if label_mode == "multiclass_012":
        transformed = transform({"image": image, "mask": target.unsqueeze(0).float()})
        transformed_image = torch.as_tensor(transformed["image"]).float()
        transformed_target = (
            torch.as_tensor(transformed["mask"])
            .squeeze(0)
            .round()
            .clamp(0, 2)
            .long()
        )
        return transformed_image.contiguous(), transformed_target.contiguous()

    raise ValueError(f"Unsupported label mode for augmentation: {label_mode}")


def _autocast_context(*, device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()




def _patch_sampling_config(patch_sampling: dict[str, Any] | None) -> dict[str, Any] | None:
    if not patch_sampling or not bool(patch_sampling.get("enabled", False)):
        return None

    result = dict(patch_sampling)
    crop_size = result.get("crop_size", result.get("patch_size", [384, 384]))
    result["crop_size"] = _image_size_tuple(crop_size)
    result["foreground_prob"] = float(result.get("foreground_prob", 0.85))
    result["class_balanced_foreground"] = bool(
        result.get("class_balanced_foreground", True)
    )
    result["center_jitter_pixels"] = int(result.get("center_jitter_pixels", 64))
    result["max_attempts"] = max(1, int(result.get("max_attempts", 16)))
    result["min_foreground_pixels"] = max(0, int(result.get("min_foreground_pixels", 8)))
    result["hard_negative_prob"] = float(result.get("hard_negative_prob", 0.0))
    result["hard_negative_dilate_radius"] = max(
        0,
        int(result.get("hard_negative_dilate_radius", 3)),
    )
    result["min_hard_negative_pixels"] = max(
        1,
        int(result.get("min_hard_negative_pixels", 4)),
    )
    labels = result.get("labels")
    result["labels"] = [int(label) for label in labels] if labels is not None else None
    return result


def _patch_prediction_path_map(
    patch_sampling: dict[str, Any] | None,
    repo_root: Path,
) -> dict[str, Path] | None:
    if patch_sampling is None:
        return None
    manifest = patch_sampling.get("hard_negative_prediction_manifest") or patch_sampling.get(
        "prediction_manifest"
    )
    if not manifest:
        return None
    manifest_path = _resolve_path(Path(str(manifest)), repo_root)
    rows = read_prediction_manifest(manifest_path)
    result: dict[str, Path] = {}
    for row in rows:
        result[str(row["sample_id"])] = _resolve_path(row["prediction_path"], repo_root)
    return result


def _load_patch_image_and_mask(
    image_path: Path,
    mask_array: np.ndarray,
    patch_sampling: dict[str, Any],
    *,
    label_mode: str,
    prediction_path: Path | None = None,
) -> tuple[Image.Image, np.ndarray]:
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        image_width, image_height = image.size
        mask_array = _resize_raw_mask_to_shape(
            mask_array,
            (image_height, image_width),
            label_mode=label_mode,
        )
        prediction_array = _load_patch_prediction_mask(
            prediction_path,
            image_shape=(image_height, image_width),
        )
        left, top, right, bottom = _sample_patch_box(
            mask_array,
            image_shape=(image_height, image_width),
            patch_sampling=patch_sampling,
            label_mode=label_mode,
            prediction_array=prediction_array,
        )
        cropped_image = image.crop((left, top, right, bottom))
        cropped_mask = mask_array[top:bottom, left:right]
    return cropped_image, np.asarray(cropped_mask, dtype=np.uint8)


def _resize_raw_mask_to_shape(
    mask_array: np.ndarray,
    image_shape: tuple[int, int],
    *,
    label_mode: str,
) -> np.ndarray:
    target_height, target_width = image_shape
    if tuple(mask_array.shape[:2]) == (target_height, target_width):
        return np.asarray(mask_array, dtype=np.uint8)
    if label_mode == "binary_foreground":
        array = mask_to_binary_foreground(mask_array).astype(np.uint8)
    else:
        array = np.asarray(mask_array, dtype=np.uint8)
    resized = Image.fromarray(array).resize((target_width, target_height), _pil_nearest())
    return np.asarray(resized, dtype=np.uint8)


def _sample_patch_box(
    mask_array: np.ndarray,
    *,
    image_shape: tuple[int, int],
    patch_sampling: dict[str, Any],
    label_mode: str,
    prediction_array: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    image_height, image_width = image_shape
    crop_height, crop_width = patch_sampling["crop_size"]
    crop_height = min(max(1, int(crop_height)), image_height)
    crop_width = min(max(1, int(crop_width)), image_width)
    if crop_height == image_height and crop_width == image_width:
        return 0, 0, image_width, image_height

    hard_negative_prob = float(patch_sampling.get("hard_negative_prob", 0.0))
    if prediction_array is not None and hard_negative_prob > 0 and random.random() < hard_negative_prob:
        for _ in range(int(patch_sampling.get("max_attempts", 16))):
            center = _sample_hard_negative_center(
                mask_array,
                prediction_array,
                patch_sampling=patch_sampling,
                label_mode=label_mode,
            )
            if center is None:
                break
            top, left = _jitter_center(center, patch_sampling)
            box = _box_around_center(
                center_y=top,
                center_x=left,
                crop_height=crop_height,
                crop_width=crop_width,
                image_height=image_height,
                image_width=image_width,
            )
            if _patch_hard_negative_pixels(
                mask_array,
                prediction_array,
                box,
                patch_sampling=patch_sampling,
                label_mode=label_mode,
            ) >= int(patch_sampling.get("min_hard_negative_pixels", 1)):
                return box

    foreground_prob = float(patch_sampling.get("foreground_prob", 0.85))
    if random.random() < foreground_prob:
        for _ in range(int(patch_sampling.get("max_attempts", 16))):
            center = _sample_foreground_center(
                mask_array,
                patch_sampling=patch_sampling,
                label_mode=label_mode,
            )
            if center is None:
                break
            top, left = _jitter_center(center, patch_sampling)
            box = _box_around_center(
                center_y=top,
                center_x=left,
                crop_height=crop_height,
                crop_width=crop_width,
                image_height=image_height,
                image_width=image_width,
            )
            if _patch_foreground_pixels(mask_array, box) >= int(
                patch_sampling.get("min_foreground_pixels", 0)
            ):
                return box

    return _random_patch_box(
        image_height=image_height,
        image_width=image_width,
        crop_height=crop_height,
        crop_width=crop_width,
    )


def _jitter_center(
    center: tuple[int, int],
    patch_sampling: dict[str, Any],
) -> tuple[int, int]:
    top, left = center
    jitter = int(patch_sampling.get("center_jitter_pixels", 0))
    if jitter > 0:
        top += random.randint(-jitter, jitter)
        left += random.randint(-jitter, jitter)
    return top, left


def _sample_foreground_center(
    mask_array: np.ndarray,
    *,
    patch_sampling: dict[str, Any],
    label_mode: str,
) -> tuple[int, int] | None:
    labels = patch_sampling.get("labels") or metric_labels(label_mode)
    if bool(patch_sampling.get("class_balanced_foreground", True)):
        present_labels = [label for label in labels if np.any(mask_array == int(label))]
        if present_labels:
            label = int(random.choice(present_labels))
            coordinates = np.argwhere(mask_array == label)
        else:
            coordinates = np.argwhere(_foreground_mask_for_sampling(mask_array, label_mode))
    else:
        coordinates = np.argwhere(_foreground_mask_for_sampling(mask_array, label_mode))

    if coordinates.size == 0:
        return None
    row, col = coordinates[random.randrange(coordinates.shape[0])]
    return int(row), int(col)


def _sample_hard_negative_center(
    mask_array: np.ndarray,
    prediction_array: np.ndarray,
    *,
    patch_sampling: dict[str, Any],
    label_mode: str,
) -> tuple[int, int] | None:
    hard_negative = _hard_negative_mask(
        mask_array,
        prediction_array,
        patch_sampling=patch_sampling,
        label_mode=label_mode,
    )
    coordinates = np.argwhere(hard_negative)
    if coordinates.size == 0:
        return None
    row, col = coordinates[random.randrange(coordinates.shape[0])]
    return int(row), int(col)


def _foreground_mask_for_sampling(mask_array: np.ndarray, label_mode: str) -> np.ndarray:
    if label_mode == "binary_foreground":
        return mask_to_binary_foreground(mask_array) > 0
    if label_mode == "multiclass_012":
        return np.asarray(mask_array) > 0
    raise ValueError(f"Unsupported label mode for patch sampling: {label_mode}")


def _hard_negative_mask(
    mask_array: np.ndarray,
    prediction_array: np.ndarray,
    *,
    patch_sampling: dict[str, Any],
    label_mode: str,
) -> np.ndarray:
    prediction_foreground = np.asarray(prediction_array) > 0
    if not np.any(prediction_foreground):
        return np.zeros_like(prediction_foreground, dtype=bool)
    foreground = _foreground_mask_for_sampling(mask_array, label_mode)
    radius = int(patch_sampling.get("hard_negative_dilate_radius", 3))
    safe_foreground = _binary_dilate_numpy(foreground, radius=radius)
    return prediction_foreground & ~safe_foreground


def _patch_foreground_pixels(mask_array: np.ndarray, box: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = box
    return int(np.count_nonzero(mask_array[top:bottom, left:right] > 0))


def _patch_hard_negative_pixels(
    mask_array: np.ndarray,
    prediction_array: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    patch_sampling: dict[str, Any],
    label_mode: str,
) -> int:
    left, top, right, bottom = box
    hard_negative = _hard_negative_mask(
        mask_array[top:bottom, left:right],
        prediction_array[top:bottom, left:right],
        patch_sampling=patch_sampling,
        label_mode=label_mode,
    )
    return int(np.count_nonzero(hard_negative))


def _load_patch_prediction_mask(
    prediction_path: Path | None,
    *,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    if prediction_path is None or not prediction_path.is_file():
        return None
    target_height, target_width = image_shape
    with Image.open(prediction_path) as handle:
        prediction = np.asarray(handle)
    if prediction.ndim == 3:
        prediction = prediction[..., 0]
    prediction = np.asarray(prediction, dtype=np.uint8)
    if tuple(prediction.shape[:2]) != (target_height, target_width):
        prediction = np.asarray(
            Image.fromarray(prediction).resize((target_width, target_height), _pil_nearest()),
            dtype=np.uint8,
        )
    return prediction


def _binary_dilate_numpy(mask: np.ndarray, *, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = image.filter(ImageFilter.MaxFilter(size=2 * int(radius) + 1))
    return np.asarray(dilated) > 0


def _box_around_center(
    *,
    center_y: int,
    center_x: int,
    crop_height: int,
    crop_width: int,
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int]:
    top = center_y - crop_height // 2
    left = center_x - crop_width // 2
    top = min(max(0, top), image_height - crop_height)
    left = min(max(0, left), image_width - crop_width)
    return left, top, left + crop_width, top + crop_height


def _random_patch_box(
    *,
    image_height: int,
    image_width: int,
    crop_height: int,
    crop_width: int,
) -> tuple[int, int, int, int]:
    top_max = max(0, image_height - crop_height)
    left_max = max(0, image_width - crop_width)
    top = random.randint(0, top_max) if top_max > 0 else 0
    left = random.randint(0, left_max) if left_max > 0 else 0
    return left, top, left + crop_width, top + crop_height


def _image_to_tensor(
    image: Image.Image,
    image_size: tuple[int, int],
    resize_mode: str = "direct",
) -> torch.Tensor:
    image = _resize_pil_image(
        image.convert("RGB"),
        image_size,
        resize_mode=resize_mode,
        resample=_pil_bilinear(),
        fill=(0, 0, 0),
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _resize_pil_image(
    image: Image.Image,
    image_size: tuple[int, int],
    *,
    resize_mode: str,
    resample: int,
    fill: int | tuple[int, int, int],
) -> Image.Image:
    height, width = image_size
    if resize_mode == "direct":
        return image.resize((width, height), resample)
    if resize_mode in {"aspect_pad", "letterbox"}:
        source_width, source_height = image.size
        scale = min(width / float(source_width), height / float(source_height))
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = image.resize((resized_width, resized_height), resample)
        canvas = Image.new(image.mode, (width, height), fill)
        left = (width - resized_width) // 2
        top = (height - resized_height) // 2
        canvas.paste(resized, (left, top))
        return canvas
    raise ValueError(
        "resize_mode must be one of {'direct', 'aspect_pad', 'letterbox'}, "
        f"got {resize_mode!r}"
    )


def _load_image_tensor(
    path: Path,
    image_size: tuple[int, int],
    resize_mode: str = "direct",
) -> torch.Tensor:
    with Image.open(path) as image:
        return _image_to_tensor(image, image_size, resize_mode)


def _image_normalization_tensors(
    normalization: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not normalization or not bool(normalization.get("enabled", False)):
        return None

    mean = normalization.get("mean")
    std = normalization.get("std")
    if mean is None or std is None:
        preset = str(normalization.get("preset", "imagenet")).lower()
        if preset != "imagenet":
            raise ValueError(
                "normalization preset must be 'imagenet' when explicit mean/std "
                f"are not provided, got {preset!r}"
            )
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if len(mean) != 3 or len(std) != 3:
        raise ValueError(
            "image normalization mean and std must each contain three channel values, "
            f"got mean={mean!r}, std={std!r}"
        )

    mean_tensor = torch.tensor([float(value) for value in mean], dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor([float(value) for value in std], dtype=torch.float32).view(3, 1, 1)
    if torch.any(std_tensor <= 0):
        raise ValueError(f"image normalization std values must be positive, got {std!r}")
    return mean_tensor, std_tensor


def _normalize_image_tensor(
    image: torch.Tensor,
    normalization: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    if normalization is None:
        return image
    mean, std = normalization
    mean = mean.to(device=image.device, dtype=image.dtype)
    std = std.to(device=image.device, dtype=image.dtype)
    return ((image - mean) / std).contiguous()


def _prepare_mask_tensor(
    mask: np.ndarray,
    image_size: tuple[int, int],
    label_mode: str,
    resize_mode: str = "direct",
) -> torch.Tensor:
    if label_mode == "binary_foreground":
        array = _resize_binary_mask(mask, image_size, resize_mode=resize_mode).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0).contiguous()
    if label_mode == "multiclass_012":
        array = _resize_multiclass_012_mask(mask, image_size, resize_mode=resize_mode)
        return torch.from_numpy(array.copy()).long().contiguous()
    raise ValueError(f"Unsupported label mode for baseline: {label_mode}")


def _resize_binary_mask(
    mask: np.ndarray,
    image_size: tuple[int, int],
    *,
    resize_mode: str = "direct",
) -> np.ndarray:
    binary = mask_to_binary_foreground(mask)
    image = _resize_pil_image(
        Image.fromarray(binary.astype(np.uint8)),
        image_size,
        resize_mode=resize_mode,
        resample=_pil_nearest(),
        fill=0,
    )
    return np.asarray(image, dtype=np.uint8)


def _resize_multiclass_012_mask(
    mask: np.ndarray,
    image_size: tuple[int, int],
    *,
    resize_mode: str = "direct",
) -> np.ndarray:
    allowed = {0, 1, 2}
    values = set(int(value) for value in np.unique(mask))
    if not values <= allowed:
        raise ValueError(
            f"multiclass_012 expects mask values within {sorted(allowed)}, "
            f"got {sorted(values)}"
        )
    image = _resize_pil_image(
        Image.fromarray(np.asarray(mask, dtype=np.uint8)),
        image_size,
        resize_mode=resize_mode,
        resample=_pil_nearest(),
        fill=0,
    )
    return np.asarray(image, dtype=np.uint8)


def _resize_mask_for_mode(
    mask: np.ndarray,
    image_size: tuple[int, int],
    label_mode: str,
    resize_mode: str = "direct",
) -> np.ndarray:
    if label_mode == "binary_foreground":
        return _resize_binary_mask(mask, image_size, resize_mode=resize_mode)
    if label_mode == "multiclass_012":
        return _resize_multiclass_012_mask(mask, image_size, resize_mode=resize_mode)
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _load_prediction_mask(path: Path, label_mode: str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    if label_mode == "binary_foreground":
        return (array > 0).astype(np.uint8)
    if label_mode == "multiclass_012":
        values = set(int(value) for value in np.unique(array))
        if not values <= {0, 1, 2}:
            raise ValueError(
                f"multiclass_012 prediction values must be within [0, 1, 2], "
                f"got {sorted(values)} in {path}"
            )
        return array.astype(np.uint8)
    raise ValueError(f"Unsupported label mode: {label_mode}")


def predict_logits(
    logits: torch.Tensor,
    *,
    label_mode: str,
    threshold: float,
) -> np.ndarray:
    logits = primary_logits_for_label_mode(logits, label_mode)
    if label_mode == "binary_foreground":
        return (torch.sigmoid(logits) >= threshold).cpu().numpy().astype(np.uint8)
    if label_mode == "multiclass_012":
        return torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _save_prediction_mask(path: Path, prediction: np.ndarray, label_mode: str) -> None:
    if label_mode == "binary_foreground":
        if prediction.ndim == 3:
            prediction = prediction[0]
        Image.fromarray(prediction.astype(np.uint8) * 255).save(path)
        return
    if label_mode == "multiclass_012":
        Image.fromarray(prediction.astype(np.uint8)).save(path)
        return
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _prediction_item(prediction: np.ndarray, index: int, label_mode: str) -> np.ndarray:
    if label_mode == "binary_foreground":
        return prediction[index, 0]
    if label_mode == "multiclass_012":
        return prediction[index]
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _target_item(target: np.ndarray, index: int, label_mode: str) -> np.ndarray:
    if label_mode == "binary_foreground":
        return target[index, 0]
    if label_mode == "multiclass_012":
        return target[index]
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _resolve_path(path: Path | str, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return base / candidate


def _image_size_tuple(value: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"image_size must contain [height, width], got {value!r}")
    return int(value[0]), int(value[1])


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _checkpoint_path(path: Path | str | None, output_dir: Path) -> Path:
    if path is None:
        return output_dir / "checkpoint.pt"
    return Path(path).expanduser()


def _path_for_json(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _per_label_means(values: dict[int, list[float]]) -> dict[str, float]:
    return {f"label_{label}": _mean_or_nan(scores) for label, scores in sorted(values.items())}


def _pil_bilinear() -> int:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, "BILINEAR", Image.BILINEAR)


def _pil_nearest() -> int:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, "NEAREST", Image.NEAREST)
