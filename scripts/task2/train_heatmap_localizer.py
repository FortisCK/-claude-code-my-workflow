#!/usr/bin/env python3
"""Train a single-object heatmap localizer for CATHACTION Task 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_heatmap import (  # noqa: E402
    DecodedBox,
    HeatmapTarget,
    decode_heatmap_prediction,
    make_heatmap_target,
)
from cathaction.data.task2_roi import class_counts, load_task2_samples_from_split  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Task2HeatmapDataset(Dataset[tuple[torch.Tensor, HeatmapTarget, int, torch.Tensor, str]]):
    def __init__(
        self,
        samples: list[Task2Sample],
        *,
        input_size: int,
        sigma: float,
        training: bool,
    ) -> None:
        self.samples = samples
        self.input_size = input_size
        self.sigma = sigma
        self.transform = build_transform(input_size=input_size, training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, HeatmapTarget, int, torch.Tensor, str]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        target = make_heatmap_target(
            sample.box,
            output_size=(self.input_size, self.input_size),
            num_classes=2,
            sigma=self.sigma,
        )
        box_tensor = torch.tensor(
            [
                sample.box.x_center,
                sample.box.y_center,
                sample.box.width,
                sample.box.height,
            ],
            dtype=torch.float32,
        )
        return image_tensor, target, sample.class_id, box_tensor, sample.sample_id


def heatmap_collate(batch: list[tuple[torch.Tensor, HeatmapTarget, int, torch.Tensor, str]]) -> dict[str, Any]:
    images, targets, labels, boxes, sample_ids = zip(*batch)
    center_indices = torch.tensor([target.center_index for target in targets], dtype=torch.long)
    return {
        "images": torch.stack(list(images), dim=0),
        "heatmap": torch.stack([target.heatmap for target in targets], dim=0),
        "size": torch.stack([target.size for target in targets], dim=0),
        "offset": torch.stack([target.offset for target in targets], dim=0),
        "center_indices": center_indices,
        "labels": torch.tensor(labels, dtype=torch.long),
        "boxes": torch.stack(list(boxes), dim=0),
        "sample_ids": list(sample_ids),
    }


def build_transform(*, input_size: int, training: bool) -> transforms.Compose:
    steps: list[Any] = [transforms.Resize((input_size, input_size))]
    if training:
        steps.append(
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.25)],
                p=0.5,
            )
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


class MonaiHeatmapLocalizer(nn.Module):
    def __init__(self, *, channels: tuple[int, ...], strides: tuple[int, ...]) -> None:
        super().__init__()
        from monai.networks.nets import UNet

        self.net = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=6,
            channels=channels,
            strides=strides,
            num_res_units=2,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.net(images)
        return {
            "heatmap_logits": output[:, 0:2],
            "size_logits": output[:, 2:4],
            "offset_logits": output[:, 4:6],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument("--train-split", type=Path, default=Path("configs/task2/splits/train_clean_labels.txt"))
    parser.add_argument(
        "--valid-combined-split",
        type=Path,
        default=Path("configs/task2/splits/valid_combined_labels.txt"),
    )
    parser.add_argument(
        "--valid-phantom-split",
        type=Path,
        default=Path("configs/task2/splits/valid_phantom_labels.txt"),
    )
    parser.add_argument(
        "--valid-animal-split",
        type=Path,
        default=Path("configs/task2/splits/valid_animal_labels.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/heatmap_localizer"))
    parser.add_argument("--name", default="monai_unet_heatmap256")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--channels", default="16,32,64,128")
    parser.add_argument("--strides", default="2,2,2")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heatmap-pos-weight", type=float, default=30.0)
    parser.add_argument("--heatmap-loss-weight", type=float, default=1.0)
    parser.add_argument("--size-loss-weight", type=float, default=5.0)
    parser.add_argument("--offset-loss-weight", type=float, default=1.0)
    parser.add_argument("--balanced-sampler", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--primary-metric", default="valid_combined/mean_iou")
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

    train_samples = maybe_limit(
        load_task2_samples_from_split(data_root, resolve_path(args.train_split)),
        args.train_limit,
    )
    valid_sets = {
        "valid_combined": prepare_valid_samples(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_combined_split)),
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
        "valid_animal": prepare_valid_samples(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_animal_split)),
            valid_limit=args.valid_limit,
            samples_per_class=args.valid_samples_per_class,
            seed=args.seed,
        ),
    }

    device = choose_device(args.device)
    model = MonaiHeatmapLocalizer(
        channels=parse_int_tuple(args.channels),
        strides=parse_int_tuple(args.strides),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    train_loader = DataLoader(
        Task2HeatmapDataset(
            train_samples,
            input_size=args.input_size,
            sigma=args.sigma,
            training=True,
        ),
        batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_samples) if args.balanced_sampler else None,
        shuffle=False if args.balanced_sampler else True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=heatmap_collate,
    )
    valid_loaders = {
        name: DataLoader(
            Task2HeatmapDataset(
                samples,
                input_size=args.input_size,
                sigma=args.sigma,
                training=False,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            collate_fn=heatmap_collate,
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
                "type": "MonaiHeatmapLocalizer",
                "channels": parse_int_tuple(args.channels),
                "strides": parse_int_tuple(args.strides),
            },
            "train_samples": len(train_samples),
            "train_class_counts": class_counts(train_samples),
            "valid_class_counts": {name: class_counts(samples) for name, samples in valid_sets.items()},
        }
    )
    (run_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(args_record, indent=2), flush=True)

    best_metric = float("-inf")
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
        for split_name, loader in valid_loaders.items():
            metrics = evaluate(model=model, loader=loader, device=device, use_amp=args.amp)
            row.update(prefix_metrics(split_name, metrics))

        append_metrics(metrics_path, row)
        primary = float(row[args.primary_metric])
        is_best = primary > best_metric
        if is_best:
            best_metric = primary
            best_epoch = epoch
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row, args_record)
            (run_dir / "best_metrics.json").write_text(
                json.dumps({"best_epoch": best_epoch, "best_metric": best_metric, "metrics": row}, indent=2)
                + "\n",
                encoding="utf-8",
            )
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row, args_record)
        print(format_epoch_summary(epoch, row, args.primary_metric, best_epoch, best_metric), flush=True)

    print(f"Best {args.primary_metric}={best_metric:.6f} at epoch {best_epoch}", flush=True)
    print(f"Results saved to {run_dir}", flush=True)
    return 0


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
    totals = {"loss": 0.0, "heatmap_loss": 0.0, "size_loss": 0.0, "offset_loss": 0.0}
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
                heatmap_loss_weight=args.heatmap_loss_weight,
                size_loss_weight=args.size_loss_weight,
                offset_loss_weight=args.offset_loss_weight,
            )
            loss = loss_parts["loss"]
        scaler.scale(loss).backward()
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
    heatmap_loss_weight: float,
    size_loss_weight: float,
    offset_loss_weight: float,
) -> dict[str, torch.Tensor]:
    heatmap_logits = output["heatmap_logits"]
    heatmap_weight = 1.0 + heatmap_target * heatmap_pos_weight
    heatmap_loss = F.binary_cross_entropy_with_logits(
        heatmap_logits,
        heatmap_target,
        weight=heatmap_weight,
    )

    pred_size = torch.sigmoid(output["size_logits"])
    pred_offset = torch.sigmoid(output["offset_logits"])
    batch_indices = torch.arange(center_indices.shape[0], device=center_indices.device)
    y_indices = center_indices[:, 0]
    x_indices = center_indices[:, 1]
    size_at_center = pred_size[batch_indices, :, y_indices, x_indices]
    offset_at_center = pred_offset[batch_indices, :, y_indices, x_indices]
    size_target_at_center = size_target[batch_indices, :, y_indices, x_indices]
    offset_target_at_center = offset_target[batch_indices, :, y_indices, x_indices]
    size_loss = F.smooth_l1_loss(size_at_center, size_target_at_center)
    offset_loss = F.smooth_l1_loss(offset_at_center, offset_target_at_center)
    loss = (
        heatmap_loss_weight * heatmap_loss
        + size_loss_weight * size_loss
        + offset_loss_weight * offset_loss
    )
    return {
        "loss": loss,
        "heatmap_loss": heatmap_loss,
        "size_loss": size_loss,
        "offset_loss": offset_loss,
    }


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].cpu().numpy().tolist()
        gt_boxes = batch["boxes"].cpu().numpy()
        sample_ids = batch["sample_ids"]
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            output = model(images)
            heatmap_logits = output["heatmap_logits"].float().cpu()
            size_map = torch.sigmoid(output["size_logits"].float()).cpu()
            offset_map = torch.sigmoid(output["offset_logits"].float()).cpu()
        for index, sample_id in enumerate(sample_ids):
            decoded = decode_heatmap_prediction(
                heatmap_logits[index],
                size_map[index],
                offset_map[index],
            )
            gt_class = int(labels[index])
            gt_xyxy = normalized_xywh_to_xyxy(gt_boxes[index])
            pred_xyxy = decoded_to_xyxy(decoded)
            rows.append(
                {
                    "sample_id": sample_id,
                    "gt_class": gt_class,
                    "pred_class": decoded.class_id,
                    "score": decoded.score,
                    "iou": iou_xyxy(pred_xyxy, gt_xyxy),  # type: ignore[arg-type]
                }
            )
    return summarize_eval_rows(rows)


def summarize_eval_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    ious = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64)
    class_correct = np.asarray([int(row["pred_class"]) == int(row["gt_class"]) for row in rows], dtype=np.float64)
    result: dict[str, float] = {
        "samples": float(len(rows)),
        "mean_iou": float(np.mean(ious)) if ious.size else float("nan"),
        "median_iou": float(np.median(ious)) if ious.size else float("nan"),
        "class_accuracy": float(np.mean(class_correct)) if class_correct.size else float("nan"),
    }
    for threshold in (0.10, 0.25, 0.50):
        result[f"recall_iou_{threshold:.2f}"] = float(np.mean(ious >= threshold)) if ious.size else float("nan")
    for class_id in (0, 1):
        class_ious = np.asarray(
            [float(row["iou"]) for row in rows if int(row["gt_class"]) == class_id],
            dtype=np.float64,
        )
        prefix = f"class{class_id}"
        result[f"{prefix}_samples"] = float(class_ious.size)
        result[f"{prefix}_mean_iou"] = float(np.mean(class_ious)) if class_ious.size else float("nan")
        for threshold in (0.10, 0.25, 0.50):
            result[f"{prefix}_recall_iou_{threshold:.2f}"] = (
                float(np.mean(class_ious >= threshold)) if class_ious.size else float("nan")
            )
    return result


def normalized_xywh_to_xyxy(box: np.ndarray) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = [float(value) for value in box.tolist()]
    return (
        x_center - width / 2.0,
        y_center - height / 2.0,
        x_center + width / 2.0,
        y_center + height / 2.0,
    )


def decoded_to_xyxy(decoded: DecodedBox) -> tuple[float, float, float, float]:
    return (
        decoded.x_center - decoded.width / 2.0,
        decoded.y_center - decoded.height / 2.0,
        decoded.x_center + decoded.width / 2.0,
        decoded.y_center + decoded.height / 2.0,
    )


def create_prediction_rows(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> list[dict[str, Any]]:
    _ = model, loader, device, use_amp
    return []


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
        f"combined_iou={float(row['valid_combined/mean_iou']):.4f} "
        f"combined_r25={float(row['valid_combined/recall_iou_0.25']):.4f} "
        f"animal_c1_iou={float(row['valid_animal/class1_mean_iou']):.4f} "
        f"animal_c1_r25={float(row['valid_animal/class1_recall_iou_0.25']):.4f} "
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
    if limit is None:
        return samples
    return samples[:limit]


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
        if len(class_samples) <= samples_per_class:
            picked = class_samples
        else:
            picked = rng.sample(class_samples, samples_per_class)
        selected.extend(picked)
    selected.sort(key=lambda item: (item.video_id, item.frame_index, item.sample_id))
    return selected


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


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
