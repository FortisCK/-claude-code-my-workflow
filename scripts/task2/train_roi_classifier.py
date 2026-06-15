#!/usr/bin/env python3
"""Train a GT-ROI classifier for CATHACTION Task 2 collision detection."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import (  # noqa: E402
    RoiCropConfig,
    class_counts,
    crop_task2_roi,
    load_task2_samples_from_split,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Task2RoiDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: list[Task2Sample],
        *,
        crop_config: RoiCropConfig,
        input_size: int,
        training: bool,
    ) -> None:
        self.samples = samples
        self.crop_config = crop_config
        self.transform = build_transform(input_size=input_size, training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            roi = crop_task2_roi(image, sample, self.crop_config)
        return self.transform(roi), sample.class_id


def build_transform(*, input_size: int, training: bool) -> transforms.Compose:
    steps: list[Any] = [transforms.Resize((input_size, input_size))]
    if training:
        steps.extend(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.15, contrast=0.25)],
                    p=0.5,
                ),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.04, 0.04),
                    scale=(0.92, 1.08),
                    fill=0,
                ),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/roi_classifier"))
    parser.add_argument("--name", default="convnext_tiny_gtroi224_scale8")

    parser.add_argument("--backend", choices=("auto", "timm", "torchvision"), default="auto")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=8.0)
    parser.add_argument("--min-crop-size", type=int, default=224)
    parser.add_argument("--max-crop-size", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--balanced-sampler", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--primary-metric", default="valid_combined/average_precision")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    run_dir = resolve_path(args.output_dir) / args.name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    crop_config = RoiCropConfig(
        crop_scale=args.crop_scale,
        min_crop_size=args.min_crop_size,
        max_crop_size=args.max_crop_size,
    )

    train_samples = maybe_limit(
        load_task2_samples_from_split(data_root, resolve_path(args.train_split)),
        args.train_limit,
    )
    valid_sets = {
        "valid_combined": maybe_limit(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_combined_split)),
            args.valid_limit,
        ),
        "valid_phantom": maybe_limit(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_phantom_split)),
            args.valid_limit,
        ),
        "valid_animal": maybe_limit(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_animal_split)),
            args.valid_limit,
        ),
    }

    device = choose_device(args.device)
    model = create_model(args.backend, args.model, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    class_weight = make_class_weight(train_samples, device) if args.class_weight else None

    train_loader = DataLoader(
        Task2RoiDataset(
            train_samples,
            crop_config=crop_config,
            input_size=args.input_size,
            training=True,
        ),
        batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_samples) if args.balanced_sampler else None,
        shuffle=False if args.balanced_sampler else True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    valid_loaders = {
        name: DataLoader(
            Task2RoiDataset(
                samples,
                crop_config=crop_config,
                input_size=args.input_size,
                training=False,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        for name, samples in valid_sets.items()
    }

    args_record = json_ready(vars(args).copy())
    args_record.update(
        {
            "repo_root": REPO_ROOT.as_posix(),
            "data_root": repo_relative(data_root, REPO_ROOT),
            "run_dir": repo_relative(run_dir, REPO_ROOT),
            "crop_config": asdict(crop_config),
            "device": str(device),
            "train_samples": len(train_samples),
            "train_class_counts": class_counts(train_samples),
            "valid_class_counts": {name: class_counts(samples) for name, samples in valid_sets.items()},
        }
    )
    (run_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(args_record, indent=2))

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
            class_weight=class_weight,
            use_amp=args.amp,
        )
        row: dict[str, float | int] = {"epoch": epoch, **prefix_metrics("train", train_metrics)}
        for split_name, loader in valid_loaders.items():
            metrics = evaluate(
                model=model,
                loader=loader,
                device=device,
                threshold=args.threshold,
                use_amp=args.amp,
            )
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

    print(f"Best {args.primary_metric}={best_metric:.6f} at epoch {best_epoch}")
    print(f"Results saved to {run_dir}")
    return 0


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    class_weight: torch.Tensor | None,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, weight=class_weight)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return {"loss": total_loss / max(total_items, 1)}


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    probs: list[float] = []
    labels_out: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="none")
            prob_collision = torch.softmax(logits, dim=1)[:, 1]
        losses.extend(float(value) for value in loss.detach().cpu())
        probs.extend(float(value) for value in prob_collision.detach().cpu())
        labels_out.extend(int(value) for value in labels.detach().cpu())
    return binary_classification_metrics(labels_out, probs, losses, threshold=threshold)


def binary_classification_metrics(
    labels: list[int],
    probs: list[float],
    losses: list[float],
    *,
    threshold: float,
) -> dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int64)
    probs_np = np.asarray(probs, dtype=np.float64)
    preds_np = (probs_np >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np,
        preds_np,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    result = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds_np)),
        "collision_precision": float(precision),
        "collision_recall": float(recall),
        "collision_f1": float(f1),
        "predicted_collision_rate": float(np.mean(preds_np)) if preds_np.size else float("nan"),
        "mean_collision_prob": float(np.mean(probs_np)) if probs_np.size else float("nan"),
    }
    try:
        result["average_precision"] = float(average_precision_score(labels_np, probs_np))
    except ValueError:
        result["average_precision"] = float("nan")
    try:
        result["auroc"] = float(roc_auc_score(labels_np, probs_np))
    except ValueError:
        result["auroc"] = float("nan")
    return result


def create_model(backend: str, model_name: str, *, pretrained: bool) -> nn.Module:
    if backend in {"auto", "timm"}:
        try:
            import timm

            return timm.create_model(model_name, pretrained=pretrained, num_classes=2, in_chans=3)
        except Exception:
            if backend == "timm":
                raise
    import torchvision.models as models

    if not hasattr(models, model_name):
        raise ValueError(f"Unknown torchvision model: {model_name}")
    factory = getattr(models, model_name)
    weights = "DEFAULT" if pretrained else None
    model = factory(weights=weights)
    return replace_torchvision_classifier(model, num_classes=2)


def replace_torchvision_classifier(model: nn.Module, *, num_classes: int) -> nn.Module:
    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if isinstance(classifier, nn.Sequential):
            in_features = classifier[-1].in_features
            classifier[-1] = nn.Linear(in_features, num_classes)
            return model
        if isinstance(classifier, nn.Linear):
            setattr(model, "classifier", nn.Linear(classifier.in_features, num_classes))
            return model
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Cannot replace classifier for model type {type(model).__name__}")


def make_class_weight(samples: list[Task2Sample], device: torch.device) -> torch.Tensor:
    counts = class_counts(samples)
    total = sum(counts.values())
    weights = [total / max(counts.get(class_id, 1), 1) for class_id in (0, 1)]
    mean_weight = sum(weights) / len(weights)
    normalized = [weight / mean_weight for weight in weights]
    return torch.tensor(normalized, dtype=torch.float32, device=device)


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
        f"combined_ap={float(row['valid_combined/average_precision']):.4f} "
        f"combined_bacc={float(row['valid_combined/balanced_accuracy']):.4f} "
        f"collision_recall={float(row['valid_combined/collision_recall']):.4f} "
        f"animal_ap={float(row['valid_animal/average_precision']):.4f} "
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
    return value


if __name__ == "__main__":
    raise SystemExit(main())
