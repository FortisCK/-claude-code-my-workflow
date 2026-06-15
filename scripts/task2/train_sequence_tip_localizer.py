#!/usr/bin/env python3
"""Train a sequence-aware class-agnostic tip localizer for CATHACTION Task 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, build_task2_index, repo_relative  # noqa: E402
from cathaction.data.task2_heatmap import draw_gaussian  # noqa: E402
from cathaction.data.task2_roi import class_counts, load_task2_samples_from_split  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402


@dataclass(frozen=True)
class LetterboxMeta:
    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_x: int
    pad_y: int
    resized_width: int
    resized_height: int


@dataclass(frozen=True)
class TipTarget:
    heatmap: torch.Tensor
    size: torch.Tensor
    offset: torch.Tensor
    center_index: tuple[int, int]
    letterbox_box: torch.Tensor
    original_box_xyxy: torch.Tensor
    original_center: torch.Tensor
    meta: LetterboxMeta


@dataclass(frozen=True)
class DecodedTip:
    score: float
    xyxy_original: tuple[float, float, float, float]
    center_original: tuple[float, float]
    xyxy_letterbox: tuple[float, float, float, float]


class SequenceTipDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[Task2Sample],
        *,
        all_samples: list[Task2Sample],
        input_size: int,
        frame_radius: int,
        sigma: float,
        training: bool,
        intensity_jitter: float,
    ) -> None:
        self.samples = samples
        self.input_size = input_size
        self.frame_radius = frame_radius
        self.sigma = sigma
        self.training = training
        self.intensity_jitter = intensity_jitter
        self.frame_lookup = build_frame_lookup(all_samples)
        self.video_frames = build_video_frame_index(all_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        clip_tensors: list[torch.Tensor] = []
        center_meta: LetterboxMeta | None = None
        for offset in range(-self.frame_radius, self.frame_radius + 1):
            frame_sample = self.find_neighbor(sample, offset)
            tensor, meta = load_letterboxed_grayscale(
                frame_sample.image_path,
                input_size=self.input_size,
            )
            if offset == 0:
                center_meta = meta
            clip_tensors.append(tensor)
        if center_meta is None:
            raise RuntimeError("center frame metadata was not created")

        clip = torch.cat(clip_tensors, dim=0)
        if self.training and self.intensity_jitter > 0:
            clip = apply_intensity_jitter(clip, self.intensity_jitter)

        target = make_class_agnostic_tip_target(
            sample,
            meta=center_meta,
            input_size=self.input_size,
            sigma=self.sigma,
        )
        return {
            "image": clip,
            "heatmap": target.heatmap,
            "size": target.size,
            "offset": target.offset,
            "center_index": torch.tensor(target.center_index, dtype=torch.long),
            "letterbox_box": target.letterbox_box,
            "original_box_xyxy": target.original_box_xyxy,
            "original_center": target.original_center,
            "meta": target.meta,
            "gt_class": sample.class_id,
            "sample_id": sample.sample_id,
            "video_id": sample.video_id,
            "frame_index": sample.frame_index,
        }

    def find_neighbor(self, sample: Task2Sample, offset: int) -> Task2Sample:
        target_index = sample.frame_index + offset
        exact = self.frame_lookup.get((sample.video_id, target_index))
        if exact is not None:
            return exact

        frames = self.video_frames[sample.video_id]
        insert_at = bisect_left([item.frame_index for item in frames], target_index)
        candidates: list[Task2Sample] = []
        if insert_at < len(frames):
            candidates.append(frames[insert_at])
        if insert_at > 0:
            candidates.append(frames[insert_at - 1])
        if not candidates:
            return sample
        return min(candidates, key=lambda item: abs(item.frame_index - target_index))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--train-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_train_labels.txt"),
    )
    parser.add_argument(
        "--valid-animal-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_animal_labels.txt"),
    )
    parser.add_argument(
        "--valid-phantom-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/valid_phantom_balanced_small_labels.txt"),
    )
    parser.add_argument(
        "--valid-combined-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_combined_balanced_labels.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/sequence_tip_localizer"))
    parser.add_argument("--name", default="monai_unet5ch_tip512_stage2m")
    parser.add_argument("--model-type", choices=("monai_unet", "smp_fpn"), default="monai_unet")
    parser.add_argument("--encoder-name", default="tu-convnext_tiny")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--frame-radius", type=int, default=2)
    parser.add_argument("--channels", default="16,32,64,128,256")
    parser.add_argument("--strides", default="2,2,2,2")
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--heatmap-pos-weight", type=float, default=80.0)
    parser.add_argument("--heatmap-loss", choices=("bce", "centernet"), default="bce")
    parser.add_argument("--heatmap-loss-weight", type=float, default=1.0)
    parser.add_argument("--coord-loss-weight", type=float, default=5.0)
    parser.add_argument("--size-loss-weight", type=float, default=8.0)
    parser.add_argument("--offset-loss-weight", type=float, default=1.0)
    parser.add_argument("--intensity-jitter", type=float, default=0.15)
    parser.add_argument("--balanced-sampler", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--primary-metric", default="valid_combined/center_recall_20px")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--valid-samples-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    run_dir = resolve_path(args.output_dir) / args.name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_samples = build_task2_index(data_root)
    train_samples = maybe_limit(
        load_task2_samples_from_split(data_root, resolve_path(args.train_split)),
        args.train_limit,
    )
    valid_sets = {
        "valid_animal": prepare_valid_samples(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_animal_split)),
            valid_limit=args.valid_limit,
            samples_per_class=args.valid_samples_per_class,
            seed=args.seed,
        ),
        "valid_phantom": prepare_valid_samples(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_phantom_split)),
            valid_limit=args.valid_limit,
            samples_per_class=args.valid_samples_per_class,
            seed=args.seed,
        ),
        "valid_combined": prepare_valid_samples(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_combined_split)),
            valid_limit=args.valid_limit,
            samples_per_class=args.valid_samples_per_class,
            seed=args.seed,
        ),
    }

    device = choose_device(args.device)
    in_channels = 2 * args.frame_radius + 1
    if args.model_type == "monai_unet":
        model = MonaiSequenceTipLocalizer(
            in_channels=in_channels,
            channels=parse_int_tuple(args.channels),
            strides=parse_int_tuple(args.strides),
        ).to(device)
    elif args.model_type == "smp_fpn":
        model = SmpFpnSequenceTipLocalizer(
            in_channels=in_channels,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights if args.encoder_weights.lower() != "none" else None,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    train_loader = DataLoader(
        SequenceTipDataset(
            train_samples,
            all_samples=all_samples,
            input_size=args.input_size,
            frame_radius=args.frame_radius,
            sigma=args.sigma,
            training=True,
            intensity_jitter=args.intensity_jitter,
        ),
        batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_samples) if args.balanced_sampler else None,
        shuffle=False if args.balanced_sampler else True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=sequence_tip_collate,
    )
    valid_loaders = {
        name: DataLoader(
            SequenceTipDataset(
                samples,
                all_samples=all_samples,
                input_size=args.input_size,
                frame_radius=args.frame_radius,
                sigma=args.sigma,
                training=False,
                intensity_jitter=0.0,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            collate_fn=sequence_tip_collate,
        )
        for name, samples in valid_sets.items()
    }

    args_record = json_ready(vars(args).copy())
    args_record.update(
        {
            "repo_root": REPO_ROOT.as_posix(),
            "data_root": repo_relative(data_root, REPO_ROOT),
            "run_dir": repo_relative(run_dir, REPO_ROOT),
            "device": str(device),
            "model": {
                "type": args.model_type,
                "in_channels": in_channels,
                "channels": parse_int_tuple(args.channels),
                "strides": parse_int_tuple(args.strides),
                "encoder_name": args.encoder_name,
                "encoder_weights": args.encoder_weights,
            },
            "train_samples": len(train_samples),
            "train_class_counts": class_counts(train_samples),
            "valid_class_counts": {name: class_counts(samples) for name, samples in valid_sets.items()},
        }
    )
    (run_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(args_record, indent=2), flush=True)

    best_metric = -float("inf")
    best_epoch = -1
    metrics_path = run_dir / "metrics.csv"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
        )
        row: dict[str, float | int] = {"epoch": epoch, **prefix_metrics("train", train_metrics)}
        prediction_rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for split_name, loader in valid_loaders.items():
            metrics, prediction_rows = evaluate(model=model, loader=loader, device=device, use_amp=args.amp)
            row.update(prefix_metrics(split_name, metrics))
            prediction_rows_by_split[split_name] = prediction_rows

        append_metrics(metrics_path, row)
        primary = float(row[args.primary_metric])
        is_best = primary > best_metric
        if is_best:
            best_metric = primary
            best_epoch = epoch
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row, args_record)
            (run_dir / "best_metrics.json").write_text(
                json.dumps({"best_epoch": best_epoch, "best_metric": best_metric, "metrics": json_ready(row)}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            for split_name, prediction_rows in prediction_rows_by_split.items():
                write_dict_rows(run_dir / f"{split_name}_best_predictions.csv", prediction_rows)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row, args_record)
        print(format_epoch_summary(epoch, row, args.primary_metric, best_epoch, best_metric), flush=True)

    print(f"Best {args.primary_metric}={best_metric:.6f} at epoch {best_epoch}", flush=True)
    print(f"Results saved to {run_dir}", flush=True)
    return 0


class MonaiSequenceTipLocalizer(nn.Module):
    def __init__(self, *, in_channels: int, channels: tuple[int, ...], strides: tuple[int, ...]) -> None:
        super().__init__()
        from monai.networks.nets import UNet

        self.net = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=5,
            channels=channels,
            strides=strides,
            num_res_units=2,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.net(images)
        return {
            "heatmap_logits": output[:, 0:1],
            "size_logits": output[:, 1:3],
            "offset_logits": output[:, 3:5],
        }


class SmpFpnSequenceTipLocalizer(nn.Module):
    def __init__(self, *, in_channels: int, encoder_name: str, encoder_weights: str | None) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=5,
            activation=None,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.net(images)
        return {
            "heatmap_logits": output[:, 0:1],
            "size_logits": output[:, 1:3],
            "offset_logits": output[:, 3:5],
        }


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "heatmap_loss": 0.0,
        "coord_loss": 0.0,
        "size_loss": 0.0,
        "offset_loss": 0.0,
    }
    total_items = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        size_target = batch["size"].to(device, non_blocking=True)
        offset_target = batch["offset"].to(device, non_blocking=True)
        center_indices = batch["center_indices"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            output = model(images)
            loss_parts = compute_losses(
                output=output,
                heatmap_target=heatmap,
                size_target=size_target,
                offset_target=offset_target,
                center_indices=center_indices,
                heatmap_pos_weight=args.heatmap_pos_weight,
                heatmap_loss_name=args.heatmap_loss,
                heatmap_loss_weight=args.heatmap_loss_weight,
                coord_loss_weight=args.coord_loss_weight,
                size_loss_weight=args.size_loss_weight,
                offset_loss_weight=args.offset_loss_weight,
            )
            loss = loss_parts["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {float(loss.detach().cpu())}")
        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(images.shape[0])
        for key in totals:
            totals[key] += float(loss_parts[key].detach().cpu()) * batch_size
        total_items += batch_size
    return {key: value / max(total_items, 1) for key, value in totals.items()}


def compute_losses(
    *,
    output: dict[str, torch.Tensor],
    heatmap_target: torch.Tensor,
    size_target: torch.Tensor,
    offset_target: torch.Tensor,
    center_indices: torch.Tensor,
    heatmap_pos_weight: float,
    heatmap_loss_name: str,
    heatmap_loss_weight: float,
    coord_loss_weight: float,
    size_loss_weight: float,
    offset_loss_weight: float,
) -> dict[str, torch.Tensor]:
    heatmap_logits = output["heatmap_logits"]
    if heatmap_loss_name == "bce":
        heatmap_weight = 1.0 + heatmap_target * heatmap_pos_weight
        heatmap_loss = F.binary_cross_entropy_with_logits(
            heatmap_logits,
            heatmap_target,
            weight=heatmap_weight,
        )
    elif heatmap_loss_name == "centernet":
        heatmap_loss = centernet_focal_heatmap_loss(heatmap_logits, heatmap_target)
    else:
        raise ValueError(f"Unsupported heatmap loss: {heatmap_loss_name}")

    pred_size = torch.sigmoid(output["size_logits"])
    pred_offset = torch.sigmoid(output["offset_logits"])
    batch_indices = torch.arange(center_indices.shape[0], device=center_indices.device)
    y_indices = center_indices[:, 0]
    x_indices = center_indices[:, 1]
    size_at_center = pred_size[batch_indices, :, y_indices, x_indices]
    offset_at_center = pred_offset[batch_indices, :, y_indices, x_indices]
    size_target_at_center = size_target[batch_indices, :, y_indices, x_indices]
    offset_target_at_center = offset_target[batch_indices, :, y_indices, x_indices]
    coord_loss = softargmax_center_loss(
        heatmap_logits=heatmap_logits,
        center_indices=center_indices,
        offset_target_at_center=offset_target_at_center,
    )
    size_loss = F.smooth_l1_loss(size_at_center, size_target_at_center)
    offset_loss = F.smooth_l1_loss(offset_at_center, offset_target_at_center)
    loss = (
        heatmap_loss_weight * heatmap_loss
        + coord_loss_weight * coord_loss
        + size_loss_weight * size_loss
        + offset_loss_weight * offset_loss
    )
    return {
        "loss": loss,
        "heatmap_loss": heatmap_loss,
        "coord_loss": coord_loss,
        "size_loss": size_loss,
        "offset_loss": offset_loss,
    }


def softargmax_center_loss(
    *,
    heatmap_logits: torch.Tensor,
    center_indices: torch.Tensor,
    offset_target_at_center: torch.Tensor,
) -> torch.Tensor:
    logits = heatmap_logits[:, 0].float()
    batch_size, height, width = logits.shape
    probabilities = torch.softmax(logits.reshape(batch_size, -1), dim=1)
    y_grid = torch.arange(height, device=logits.device, dtype=torch.float32).view(1, height, 1)
    x_grid = torch.arange(width, device=logits.device, dtype=torch.float32).view(1, 1, width)
    probabilities_2d = probabilities.reshape(batch_size, height, width)
    pred_x = torch.sum(probabilities_2d * x_grid, dim=(1, 2)) / max(width - 1, 1)
    pred_y = torch.sum(probabilities_2d * y_grid, dim=(1, 2)) / max(height - 1, 1)
    target_x = (center_indices[:, 1].float() + offset_target_at_center[:, 0]) / max(width - 1, 1)
    target_y = (center_indices[:, 0].float() + offset_target_at_center[:, 1]) / max(height - 1, 1)
    pred = torch.stack([pred_x, pred_y], dim=1)
    target = torch.stack([target_x, target_y], dim=1)
    return F.smooth_l1_loss(pred, target)


def centernet_focal_heatmap_loss(
    heatmap_logits: torch.Tensor,
    heatmap_target: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CenterNet-style focal loss for sparse Gaussian heatmaps."""

    heatmap_logits = heatmap_logits.float()
    heatmap_target = heatmap_target.float()
    pred = torch.sigmoid(heatmap_logits).clamp(min=1e-4, max=1.0 - 1e-4)
    positive_mask = heatmap_target.eq(1.0).float()
    negative_mask = heatmap_target.lt(1.0).float()
    negative_weights = torch.pow(1.0 - heatmap_target, beta)

    positive_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * positive_mask
    negative_loss = (
        torch.log(1.0 - pred)
        * torch.pow(pred, alpha)
        * negative_weights
        * negative_mask
    )
    num_positive = torch.clamp(positive_mask.sum(), min=1.0)
    return -(positive_loss.sum() + negative_loss.sum()) / num_positive


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            output = model(images)
            heatmap_logits = output["heatmap_logits"].float().cpu()
            size_map = torch.sigmoid(output["size_logits"].float()).cpu()
            offset_map = torch.sigmoid(output["offset_logits"].float()).cpu()

        for index, sample_id in enumerate(batch["sample_ids"]):
            meta = batch["metas"][index]
            decoded = decode_tip_prediction(
                heatmap_logits[index],
                size_map[index],
                offset_map[index],
                meta=meta,
            )
            gt_xyxy = tuple(float(value) for value in batch["original_box_xyxy"][index].tolist())
            gt_center = tuple(float(value) for value in batch["original_center"][index].tolist())
            pred_center = decoded.center_original
            center_distance = math.hypot(pred_center[0] - gt_center[0], pred_center[1] - gt_center[1])
            overlap = iou_xyxy(decoded.xyxy_original, gt_xyxy)  # type: ignore[arg-type]
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_id": batch["video_ids"][index],
                    "frame_index": int(batch["frame_indices"][index]),
                    "gt_class": int(batch["gt_classes"][index]),
                    "score": decoded.score,
                    "iou": overlap,
                    "center_distance_px": center_distance,
                    "pred_x1": decoded.xyxy_original[0],
                    "pred_y1": decoded.xyxy_original[1],
                    "pred_x2": decoded.xyxy_original[2],
                    "pred_y2": decoded.xyxy_original[3],
                    "gt_x1": gt_xyxy[0],
                    "gt_y1": gt_xyxy[1],
                    "gt_x2": gt_xyxy[2],
                    "gt_y2": gt_xyxy[3],
                }
            )
    return summarize_eval_rows(rows), rows


def summarize_eval_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    ious = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64)
    center_distances = np.asarray([float(row["center_distance_px"]) for row in rows], dtype=np.float64)
    result: dict[str, float] = {
        "samples": float(len(rows)),
        "mean_iou": float(np.mean(ious)) if ious.size else float("nan"),
        "median_iou": float(np.median(ious)) if ious.size else float("nan"),
        "mean_center_distance_px": float(np.mean(center_distances)) if center_distances.size else float("nan"),
        "median_center_distance_px": float(np.median(center_distances)) if center_distances.size else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        result[f"recall_iou_{threshold:.2f}"] = float(np.mean(ious >= threshold)) if ious.size else float("nan")
    for threshold in (5, 10, 20):
        result[f"center_recall_{threshold}px"] = (
            float(np.mean(center_distances <= threshold)) if center_distances.size else float("nan")
        )
    for class_id in (0, 1):
        class_rows = [row for row in rows if int(row["gt_class"]) == class_id]
        class_ious = np.asarray([float(row["iou"]) for row in class_rows], dtype=np.float64)
        class_distances = np.asarray([float(row["center_distance_px"]) for row in class_rows], dtype=np.float64)
        prefix = f"class{class_id}"
        result[f"{prefix}_samples"] = float(len(class_rows))
        result[f"{prefix}_mean_iou"] = float(np.mean(class_ious)) if class_ious.size else float("nan")
        result[f"{prefix}_mean_center_distance_px"] = (
            float(np.mean(class_distances)) if class_distances.size else float("nan")
        )
        for threshold in (0.25, 0.50, 0.75):
            result[f"{prefix}_recall_iou_{threshold:.2f}"] = (
                float(np.mean(class_ious >= threshold)) if class_ious.size else float("nan")
            )
        for threshold in (5, 10, 20):
            result[f"{prefix}_center_recall_{threshold}px"] = (
                float(np.mean(class_distances <= threshold)) if class_distances.size else float("nan")
            )
    return result


def make_class_agnostic_tip_target(
    sample: Task2Sample,
    *,
    meta: LetterboxMeta,
    input_size: int,
    sigma: float,
) -> TipTarget:
    x1, y1, x2, y2 = sample.box.xyxy_pixels(meta.original_width, meta.original_height)
    cx_original = (x1 + x2) / 2.0
    cy_original = (y1 + y2) / 2.0
    width_original = max(1e-6, x2 - x1)
    height_original = max(1e-6, y2 - y1)

    cx = clamp(cx_original * meta.scale + meta.pad_x, 0.0, float(input_size - 1))
    cy = clamp(cy_original * meta.scale + meta.pad_y, 0.0, float(input_size - 1))
    width = clamp(width_original * meta.scale, 1.0, float(input_size))
    height = clamp(height_original * meta.scale, 1.0, float(input_size))
    ix = int(math.floor(cx))
    iy = int(math.floor(cy))

    heatmap = torch.zeros((1, input_size, input_size), dtype=torch.float32)
    draw_gaussian(heatmap[0], cx, cy, sigma=sigma)
    size = torch.zeros((2, input_size, input_size), dtype=torch.float32)
    offset = torch.zeros((2, input_size, input_size), dtype=torch.float32)
    size[0, iy, ix] = float(width / input_size)
    size[1, iy, ix] = float(height / input_size)
    offset[0, iy, ix] = float(cx - ix)
    offset[1, iy, ix] = float(cy - iy)
    return TipTarget(
        heatmap=heatmap,
        size=size,
        offset=offset,
        center_index=(iy, ix),
        letterbox_box=torch.tensor(
            [
                (cx - width / 2.0) / input_size,
                (cy - height / 2.0) / input_size,
                (cx + width / 2.0) / input_size,
                (cy + height / 2.0) / input_size,
            ],
            dtype=torch.float32,
        ),
        original_box_xyxy=torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
        original_center=torch.tensor([cx_original, cy_original], dtype=torch.float32),
        meta=meta,
    )


def decode_tip_prediction(
    heatmap_logits: torch.Tensor,
    size_map: torch.Tensor,
    offset_map: torch.Tensor,
    *,
    meta: LetterboxMeta,
) -> DecodedTip:
    if heatmap_logits.shape[0] != 1:
        raise ValueError(f"expected one heatmap channel, got {tuple(heatmap_logits.shape)}")
    heatmap = torch.sigmoid(heatmap_logits[0])
    flat_index = int(torch.argmax(heatmap).item())
    out_h, out_w = heatmap.shape
    y_index = flat_index // out_w
    x_index = flat_index % out_w
    score = float(heatmap[y_index, x_index].item())
    dx = float(offset_map[0, y_index, x_index].item())
    dy = float(offset_map[1, y_index, x_index].item())
    cx_letterbox = clamp(float(x_index) + dx, 0.0, float(out_w - 1))
    cy_letterbox = clamp(float(y_index) + dy, 0.0, float(out_h - 1))
    width_letterbox = clamp(float(size_map[0, y_index, x_index].item()) * meta.input_size, 1.0, float(meta.input_size))
    height_letterbox = clamp(float(size_map[1, y_index, x_index].item()) * meta.input_size, 1.0, float(meta.input_size))
    xyxy_letterbox = (
        cx_letterbox - width_letterbox / 2.0,
        cy_letterbox - height_letterbox / 2.0,
        cx_letterbox + width_letterbox / 2.0,
        cy_letterbox + height_letterbox / 2.0,
    )
    xyxy_original = letterbox_xyxy_to_original(xyxy_letterbox, meta)
    center_original = letterbox_point_to_original((cx_letterbox, cy_letterbox), meta)
    return DecodedTip(
        score=score,
        xyxy_original=xyxy_original,
        center_original=center_original,
        xyxy_letterbox=xyxy_letterbox,
    )


def load_letterboxed_grayscale(path: Path, *, input_size: int) -> tuple[torch.Tensor, LetterboxMeta]:
    with Image.open(path) as image:
        image = image.convert("L")
        original_width, original_height = image.size
        scale = min(input_size / original_width, input_size / original_height)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
        pad_x = (input_size - resized_width) // 2
        pad_y = (input_size - resized_height) // 2
        canvas = Image.new("L", (input_size, input_size), color=0)
        canvas.paste(resized, (pad_x, pad_y))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.25
    return tensor, LetterboxMeta(
        original_width=original_width,
        original_height=original_height,
        input_size=input_size,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def letterbox_point_to_original(point: tuple[float, float], meta: LetterboxMeta) -> tuple[float, float]:
    x, y = point
    original_x = (x - meta.pad_x) / meta.scale
    original_y = (y - meta.pad_y) / meta.scale
    return (
        clamp(original_x, 0.0, float(meta.original_width)),
        clamp(original_y, 0.0, float(meta.original_height)),
    )


def letterbox_xyxy_to_original(
    box: tuple[float, float, float, float],
    meta: LetterboxMeta,
) -> tuple[float, float, float, float]:
    x1, y1 = letterbox_point_to_original((box[0], box[1]), meta)
    x2, y2 = letterbox_point_to_original((box[2], box[3]), meta)
    return (
        clamp(min(x1, x2), 0.0, float(meta.original_width)),
        clamp(min(y1, y2), 0.0, float(meta.original_height)),
        clamp(max(x1, x2), 0.0, float(meta.original_width)),
        clamp(max(y1, y2), 0.0, float(meta.original_height)),
    )


def sequence_tip_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "heatmap": torch.stack([item["heatmap"] for item in batch], dim=0),
        "size": torch.stack([item["size"] for item in batch], dim=0),
        "offset": torch.stack([item["offset"] for item in batch], dim=0),
        "center_indices": torch.stack([item["center_index"] for item in batch], dim=0),
        "letterbox_box": torch.stack([item["letterbox_box"] for item in batch], dim=0),
        "original_box_xyxy": torch.stack([item["original_box_xyxy"] for item in batch], dim=0),
        "original_center": torch.stack([item["original_center"] for item in batch], dim=0),
        "metas": [item["meta"] for item in batch],
        "gt_classes": torch.tensor([item["gt_class"] for item in batch], dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in batch],
        "video_ids": [item["video_id"] for item in batch],
        "frame_indices": torch.tensor([item["frame_index"] for item in batch], dtype=torch.long),
    }


def apply_intensity_jitter(clip: torch.Tensor, strength: float) -> torch.Tensor:
    contrast = 1.0 + random.uniform(-strength, strength)
    brightness = random.uniform(-strength, strength)
    return torch.clamp(clip * contrast + brightness, -2.0, 2.0)


def build_frame_lookup(samples: list[Task2Sample]) -> dict[tuple[str, int], Task2Sample]:
    return {(sample.video_id, sample.frame_index): sample for sample in samples}


def build_video_frame_index(samples: list[Task2Sample]) -> dict[str, list[Task2Sample]]:
    result: dict[str, list[Task2Sample]] = {}
    for sample in samples:
        result.setdefault(sample.video_id, []).append(sample)
    for video_id in result:
        result[video_id].sort(key=lambda item: item.frame_index)
    return result


def make_balanced_sampler(samples: list[Task2Sample]) -> WeightedRandomSampler:
    counts = class_counts(samples)
    weights = [1.0 / max(counts[sample.class_id], 1) for sample in samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float | int],
    args_record: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": args_record,
        },
        path,
    )


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def format_epoch_summary(
    epoch: int,
    row: dict[str, float | int],
    primary_metric: str,
    best_epoch: int,
    best_metric: float,
) -> str:
    return (
        f"epoch={epoch:03d} "
        f"train_loss={float(row['train/loss']):.4f} "
        f"combined_c20={float(row['valid_combined/center_recall_20px']):.4f} "
        f"combined_r50={float(row['valid_combined/recall_iou_0.50']):.4f} "
        f"animal_c0_c20={float(row['valid_animal/class0_center_recall_20px']):.4f} "
        f"animal_c1_c20={float(row['valid_animal/class1_center_recall_20px']):.4f} "
        f"phantom_c1_c20={float(row['valid_phantom/class1_center_recall_20px']):.4f} "
        f"primary={primary_metric}:{float(row[primary_metric]):.4f} "
        f"best_epoch={best_epoch} best={best_metric:.4f}"
    )


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def maybe_limit(samples: list[Task2Sample], limit: int | None) -> list[Task2Sample]:
    return samples if limit is None else samples[:limit]


def prepare_valid_samples(
    samples: list[Task2Sample],
    *,
    valid_limit: int | None,
    samples_per_class: int | None,
    seed: int,
) -> list[Task2Sample]:
    if samples_per_class is not None:
        samples = stratified_sample(samples, samples_per_class=samples_per_class, seed=seed)
    return maybe_limit(samples, valid_limit)


def stratified_sample(samples: list[Task2Sample], *, samples_per_class: int, seed: int) -> list[Task2Sample]:
    rng = random.Random(seed)
    selected: list[Task2Sample] = []
    for class_id in sorted(class_counts(samples)):
        class_samples = [sample for sample in samples if sample.class_id == class_id]
        picked = class_samples if len(class_samples) <= samples_per_class else rng.sample(class_samples, samples_per_class)
        selected.extend(picked)
    selected.sort(key=lambda item: (item.video_id, item.frame_index, item.sample_id))
    return selected


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
