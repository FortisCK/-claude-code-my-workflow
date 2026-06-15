#!/usr/bin/env python3
"""Train a local box refiner for Task 2 Stage2P candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import RoiCropConfig, crop_roi_with_padding, load_task2_samples_from_split  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402
from scripts.task2.train_stage2o_candidate_ranker import (  # noqa: E402
    CandidateExample,
    apply_candidate_budget,
    create_model,
    load_candidate_csv,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BoxRefinerDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        candidates: list[CandidateExample],
        *,
        crop_config: RoiCropConfig,
        input_size: int,
        target_clip: float,
        training: bool,
    ) -> None:
        self.candidates = candidates
        self.crop_config = crop_config
        self.target_clip = float(target_clip)
        self.transform = build_transform(input_size=input_size, training=training)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        candidate = self.candidates[index]
        with Image.open(candidate.image_path) as image:
            bounds = compute_roi_bounds_from_xyxy(candidate.xyxy, self.crop_config)
            roi = crop_roi_with_padding(image, bounds, fill=self.crop_config.fill)
        target = encode_box_delta(candidate.xyxy, candidate.gt_xyxy, clip=self.target_clip)
        return self.transform(roi), torch.tensor(target, dtype=torch.float32), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("outputs/task2/stage2o_candidate_pool/stage2o_train_subset_yolo_geometry_top50/valid_combined_candidates.csv"),
    )
    parser.add_argument(
        "--valid-csv",
        action="append",
        default=None,
        help="Validation CSV as split_name=path. Defaults to YOLO+geometry Stage2O valid candidates.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--valid-combined-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_combined_balanced_labels.txt"),
    )
    parser.add_argument(
        "--valid-phantom-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/valid_phantom_balanced_small_labels.txt"),
    )
    parser.add_argument(
        "--valid-animal-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_animal_labels.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/stage2p_box_refiner"))
    parser.add_argument("--name", default="convnext_tiny_yolo_geometry_iou20_e6")

    parser.add_argument("--backend", choices=("auto", "timm", "torchvision"), default="auto")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=8.0)
    parser.add_argument("--min-crop-size", type=int, default=224)
    parser.add_argument("--max-crop-size", type=int, default=512)

    parser.add_argument("--train-min-iou", type=float, default=0.20)
    parser.add_argument("--eval-max-candidates-per-sample", type=int, default=100)
    parser.add_argument("--candidate-budget-sort", choices=("source_order", "raw_conf"), default="source_order")
    parser.add_argument("--target-clip", type=float, default=4.0)
    parser.add_argument("--decode-clip", type=float, default=2.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balanced-sampler", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--primary-valid-split", default="valid_combined")
    parser.add_argument(
        "--primary-metric",
        choices=("after_r75", "after_mean_iou", "r75_gain", "augmented_r75_gain", "augmented_mean_iou_gain"),
        default="augmented_r75_gain",
    )
    parser.add_argument(
        "--skip-overlap-check",
        default=False,
        action="store_true",
        help="Allow eval-only use on the training split when generating refined train candidates.",
    )
    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    train_csv = resolve_path(args.train_csv)
    valid_specs = parse_valid_csv_specs(args.valid_csv)
    run_dir = resolve_path(args.output_dir) / args.name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    crop_config = RoiCropConfig(
        crop_scale=float(args.crop_scale),
        min_crop_size=int(args.min_crop_size),
        max_crop_size=int(args.max_crop_size),
    )

    train_candidates_all = load_candidate_csv(train_csv, split_override="train", repo_root=REPO_ROOT)
    train_candidates = [
        candidate for candidate in train_candidates_all if candidate.gt_iou >= float(args.train_min_iou)
    ]
    if args.train_limit is not None:
        train_candidates = stratified_candidate_limit(train_candidates, int(args.train_limit), seed=int(args.seed))
    if not train_candidates:
        raise ValueError(f"No train candidates with IoU >= {args.train_min_iou} in {train_csv}")

    valid_candidates_all = {
        name: apply_candidate_budget(
            load_candidate_csv(resolve_path(path), split_override=name, repo_root=REPO_ROOT),
            max_candidates_per_sample=int(args.eval_max_candidates_per_sample),
            sort_mode=str(args.candidate_budget_sort),
        )
        for name, path in valid_specs.items()
    }
    valid_splits = {
        "valid_combined": load_task2_samples_from_split(data_root, resolve_path(args.valid_combined_split)),
        "valid_phantom": load_task2_samples_from_split(data_root, resolve_path(args.valid_phantom_split)),
        "valid_animal": load_task2_samples_from_split(data_root, resolve_path(args.valid_animal_split)),
    }
    if args.valid_limit is not None:
        valid_splits = {name: samples[: int(args.valid_limit)] for name, samples in valid_splits.items()}
        allowed_ids = {name: {sample.sample_id for sample in samples} for name, samples in valid_splits.items()}
        valid_candidates = {
            name: [candidate for candidate in candidates if candidate.sample_id in allowed_ids[name]]
            for name, candidates in valid_candidates_all.items()
        }
    else:
        valid_candidates = valid_candidates_all
    if not args.skip_overlap_check:
        assert_no_train_valid_overlap(train_candidates, valid_splits)

    args_record = json_ready(
        {
            **vars(args),
            "repo_root": REPO_ROOT.as_posix(),
            "train_csv": repo_relative(train_csv, REPO_ROOT),
            "valid_csvs": {name: repo_relative(resolve_path(path), REPO_ROOT) for name, path in valid_specs.items()},
            "data_root": repo_relative(data_root, REPO_ROOT),
            "run_dir": repo_relative(run_dir, REPO_ROOT),
            "device": str(device),
            "crop_config": asdict(crop_config),
            "train_candidate_summary": summarize_candidates(train_candidates),
            "valid_candidate_summary": {name: summarize_candidates(candidates) for name, candidates in valid_candidates.items()},
            "valid_sample_counts": {name: len(samples) for name, samples in valid_splits.items()},
        }
    )
    (run_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(args_record, indent=2), flush=True)

    model = create_model(args.backend, args.model, pretrained=args.pretrained, num_classes=4).to(device)
    if args.eval_checkpoint is not None:
        load_model_state(model, resolve_path(args.eval_checkpoint))
        eval_metrics = evaluate_all_splits(
            model=model,
            valid_candidates=valid_candidates,
            valid_splits=valid_splits,
            crop_config=crop_config,
            args=args,
            device=device,
            run_dir=run_dir,
            write_prefix="eval",
        )
        (run_dir / "eval_metrics.json").write_text(
            json.dumps(
                json_ready(
                    {
                        "checkpoint": repo_relative(resolve_path(args.eval_checkpoint), REPO_ROOT),
                        "metrics": eval_metrics,
                    }
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")
    train_loader = DataLoader(
        BoxRefinerDataset(
            train_candidates,
            crop_config=crop_config,
            input_size=int(args.input_size),
            target_clip=float(args.target_clip),
            training=True,
        ),
        batch_size=int(args.batch_size),
        sampler=make_balanced_sampler(train_candidates) if args.balanced_sampler else None,
        shuffle=False if args.balanced_sampler else True,
        num_workers=int(args.workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
    )

    best_metric = float("-inf")
    best_epoch = -1
    best_metrics: dict[str, Any] | None = None
    metrics_path = run_dir / "metrics.csv"
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            smooth_l1_beta=float(args.smooth_l1_beta),
            use_amp=bool(args.amp),
        )
        eval_metrics = evaluate_all_splits(
            model=model,
            valid_candidates=valid_candidates,
            valid_splits=valid_splits,
            crop_config=crop_config,
            args=args,
            device=device,
            run_dir=run_dir,
            write_prefix=None,
        )
        row = {"epoch": epoch, **prefix_metrics("train", train_metrics)}
        row.update(flatten_eval_metrics(eval_metrics))
        append_metrics(metrics_path, row)
        primary = primary_metric(eval_metrics, str(args.primary_valid_split), str(args.primary_metric))
        if primary > best_metric:
            best_metric = primary
            best_epoch = epoch
            best_metrics = eval_metrics
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row, args_record)
            (run_dir / "best_metrics.json").write_text(
                json.dumps(
                    json_ready(
                        {
                            "best_epoch": best_epoch,
                            "best_metric": best_metric,
                            "primary_valid_split": args.primary_valid_split,
                            "primary_metric": args.primary_metric,
                            "row": row,
                            "metrics": eval_metrics,
                        }
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row, args_record)
        print(format_epoch_summary(epoch, row, str(args.primary_valid_split), best_epoch, best_metric), flush=True)

    if best_metrics is not None:
        load_model_state(model, checkpoint_dir / "best.pt")
        evaluate_all_splits(
            model=model,
            valid_candidates=valid_candidates,
            valid_splits=valid_splits,
            crop_config=crop_config,
            args=args,
            device=device,
            run_dir=run_dir,
            write_prefix="best",
        )
    print(f"Best {args.primary_valid_split}/{args.primary_metric}={best_metric:.6f} at epoch {best_epoch}", flush=True)
    print(f"Results saved to {run_dir}", flush=True)
    return 0


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    smooth_l1_beta: float,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, targets, _indices in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            predictions = model(images)
            loss = F.smooth_l1_loss(predictions, targets, beta=smooth_l1_beta, reduction="mean")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return {"loss": total_loss / max(total_items, 1)}


@torch.no_grad()
def evaluate_all_splits(
    *,
    model: nn.Module,
    valid_candidates: dict[str, list[CandidateExample]],
    valid_splits: dict[str, list[Task2Sample]],
    crop_config: RoiCropConfig,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    write_prefix: str | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split_name, candidates in valid_candidates.items():
        loader = DataLoader(
            BoxRefinerDataset(
                candidates,
                crop_config=crop_config,
                input_size=int(args.input_size),
                target_clip=float(args.target_clip),
                training=False,
            ),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.workers) > 0,
        )
        split_metrics, rows = evaluate_refiner(
            model=model,
            loader=loader,
            candidates=candidates,
            samples=valid_splits[split_name],
            device=device,
            decode_clip=float(args.decode_clip),
            use_amp=bool(args.amp),
        )
        metrics[split_name] = split_metrics
        if write_prefix is not None:
            write_refined_csv(run_dir / f"{split_name}_{write_prefix}_refined_candidates.csv", rows)
        print(format_split_summary(split_name, split_metrics), flush=True)
    return metrics


@torch.no_grad()
def evaluate_refiner(
    *,
    model: nn.Module,
    loader: DataLoader,
    candidates: list[CandidateExample],
    samples: list[Task2Sample],
    device: torch.device,
    decode_clip: float,
    use_amp: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for images, _targets, indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            deltas = model(images).detach().cpu().numpy()
        for delta, index in zip(deltas, indices.numpy()):
            candidate = candidates[int(index)]
            refined = decode_box_delta(
                candidate.xyxy,
                tuple(float(value) for value in delta.tolist()),
                image_width=candidate.image_width,
                image_height=candidate.image_height,
                clip=decode_clip,
            )
            rows.append(
                {
                    "sample_id": candidate.sample_id,
                    "domain": candidate.domain,
                    "gt_class": candidate.gt_class,
                    "source": candidate.source,
                    "source_rank": candidate.source_rank,
                    "before_iou": candidate.gt_iou,
                    "after_iou": iou_xyxy(refined, candidate.gt_xyxy),
                    "x1": candidate.xyxy[0],
                    "y1": candidate.xyxy[1],
                    "x2": candidate.xyxy[2],
                    "y2": candidate.xyxy[3],
                    "refined_x1": refined[0],
                    "refined_y1": refined[1],
                    "refined_x2": refined[2],
                    "refined_y2": refined[3],
                    "gt_x1": candidate.gt_xyxy[0],
                    "gt_y1": candidate.gt_xyxy[1],
                    "gt_x2": candidate.gt_xyxy[2],
                    "gt_y2": candidate.gt_xyxy[3],
                }
            )
    metrics = summarize_rows(samples, rows)
    return metrics, rows


def summarize_rows(samples: list[Task2Sample], rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    result = {
        "samples": len(samples),
        "rows": len(rows),
        "all": summarize_sample_group(samples, rows_by_sample),
        "by_gt_class": {},
        "by_domain": {},
        "by_domain_gt_class": {},
    }
    for class_id in sorted({sample.class_id for sample in samples}):
        group = [sample for sample in samples if sample.class_id == class_id]
        result["by_gt_class"][str(class_id)] = summarize_sample_group(group, rows_by_sample)
    for domain in sorted({domain_of_sample(sample) for sample in samples}):
        group = [sample for sample in samples if domain_of_sample(sample) == domain]
        result["by_domain"][domain] = summarize_sample_group(group, rows_by_sample)
    for domain in sorted({domain_of_sample(sample) for sample in samples}):
        for class_id in sorted({sample.class_id for sample in samples}):
            group = [
                sample for sample in samples if domain_of_sample(sample) == domain and sample.class_id == class_id
            ]
            if group:
                result["by_domain_gt_class"][f"{domain}:{class_id}"] = summarize_sample_group(group, rows_by_sample)
    return result


def summarize_sample_group(
    samples: list[Task2Sample],
    rows_by_sample: dict[str, list[dict[str, Any]]],
) -> dict[str, float | int]:
    before_best: list[float] = []
    after_best: list[float] = []
    augmented_best: list[float] = []
    for sample in samples:
        rows = rows_by_sample.get(sample.sample_id, [])
        sample_before_best = max((float(row["before_iou"]) for row in rows), default=0.0)
        sample_after_best = max((float(row["after_iou"]) for row in rows), default=0.0)
        before_best.append(sample_before_best)
        after_best.append(sample_after_best)
        augmented_best.append(max(sample_before_best, sample_after_best))
    result: dict[str, float | int] = {
        "samples": len(samples),
        "before_mean_best_iou": float(np.mean(before_best)) if before_best else float("nan"),
        "after_mean_best_iou": float(np.mean(after_best)) if after_best else float("nan"),
        "augmented_mean_best_iou": float(np.mean(augmented_best)) if augmented_best else float("nan"),
        "mean_best_iou_gain": float(np.mean(np.asarray(after_best) - np.asarray(before_best))) if before_best else float("nan"),
        "augmented_mean_best_iou_gain": (
            float(np.mean(np.asarray(augmented_best) - np.asarray(before_best))) if before_best else float("nan")
        ),
    }
    for threshold in (0.25, 0.50, 0.75):
        before_recall = float(np.mean([value >= threshold for value in before_best])) if before_best else float("nan")
        after_recall = float(np.mean([value >= threshold for value in after_best])) if after_best else float("nan")
        augmented_recall = float(np.mean([value >= threshold for value in augmented_best])) if before_best else float("nan")
        result[f"before_recall_iou_{threshold:.2f}"] = before_recall
        result[f"after_recall_iou_{threshold:.2f}"] = after_recall
        result[f"augmented_recall_iou_{threshold:.2f}"] = augmented_recall
        result[f"recall_iou_{threshold:.2f}_gain"] = after_recall - before_recall
        result[f"augmented_recall_iou_{threshold:.2f}_gain"] = augmented_recall - before_recall
    return result


def encode_box_delta(
    source_xyxy: tuple[float, float, float, float],
    target_xyxy: tuple[float, float, float, float],
    *,
    clip: float,
) -> tuple[float, float, float, float]:
    sx1, sy1, sx2, sy2 = source_xyxy
    tx1, ty1, tx2, ty2 = target_xyxy
    source_width = max(1.0, sx2 - sx1)
    source_height = max(1.0, sy2 - sy1)
    source_cx = (sx1 + sx2) / 2.0
    source_cy = (sy1 + sy2) / 2.0
    target_width = max(1.0, tx2 - tx1)
    target_height = max(1.0, ty2 - ty1)
    target_cx = (tx1 + tx2) / 2.0
    target_cy = (ty1 + ty2) / 2.0
    delta = (
        (target_cx - source_cx) / source_width,
        (target_cy - source_cy) / source_height,
        math.log(target_width / source_width),
        math.log(target_height / source_height),
    )
    return tuple(float(np.clip(value, -clip, clip)) for value in delta)  # type: ignore[return-value]


def decode_box_delta(
    source_xyxy: tuple[float, float, float, float],
    delta: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    clip: float,
) -> tuple[float, float, float, float]:
    sx1, sy1, sx2, sy2 = source_xyxy
    source_width = max(1.0, sx2 - sx1)
    source_height = max(1.0, sy2 - sy1)
    source_cx = (sx1 + sx2) / 2.0
    source_cy = (sy1 + sy2) / 2.0
    dx, dy, dw, dh = (float(np.clip(value, -clip, clip)) for value in delta)
    refined_cx = source_cx + dx * source_width
    refined_cy = source_cy + dy * source_height
    refined_width = source_width * math.exp(dw)
    refined_height = source_height * math.exp(dh)
    x1 = refined_cx - refined_width / 2.0
    y1 = refined_cy - refined_height / 2.0
    x2 = refined_cx + refined_width / 2.0
    y2 = refined_cy + refined_height / 2.0
    return clip_xyxy((x1, y1, x2, y2), image_width=image_width, image_height=image_height)


def clip_xyxy(
    xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), float(image_width)))
    y1 = max(0.0, min(float(y1), float(image_height)))
    x2 = max(0.0, min(float(x2), float(image_width)))
    y2 = max(0.0, min(float(y2), float(image_height)))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def compute_roi_bounds_from_xyxy(
    box_xyxy: tuple[float, float, float, float],
    config: RoiCropConfig,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    side = max(box_width, box_height) * config.crop_scale
    side = max(side, float(config.min_crop_size))
    side = min(side, float(config.max_crop_size))
    side_int = max(1, int(math.ceil(side)))
    left = int(round(center_x - side_int / 2.0))
    top = int(round(center_y - side_int / 2.0))
    return left, top, left + side_int, top + side_int


def build_transform(*, input_size: int, training: bool) -> transforms.Compose:
    steps: list[Any] = [transforms.Resize((input_size, input_size))]
    if training:
        steps.extend(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.10, contrast=0.20)],
                    p=0.5,
                ),
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
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


def parse_valid_csv_specs(values: list[str] | None) -> dict[str, Path]:
    if values is None:
        base = Path("outputs/task2/stage2o_candidate_pool/stage2o_valid_yolo_geometry_top50")
        return {
            "valid_combined": base / "valid_combined_candidates.csv",
            "valid_phantom": base / "valid_phantom_candidates.csv",
            "valid_animal": base / "valid_animal_candidates.csv",
        }
    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--valid-csv must be split_name=path, got {item!r}")
        name, path_text = item.split("=", 1)
        result[name.strip()] = Path(path_text.strip())
    return result


def make_balanced_sampler(candidates: list[CandidateExample]) -> WeightedRandomSampler:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = f"{candidate.domain}:{candidate.gt_class}:{candidate.source}"
        counts[key] = counts.get(key, 0) + 1
    weights = [
        1.0 / max(counts[f"{candidate.domain}:{candidate.gt_class}:{candidate.source}"], 1)
        for candidate in candidates
    ]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def stratified_candidate_limit(
    candidates: list[CandidateExample],
    limit: int,
    *,
    seed: int,
) -> list[CandidateExample]:
    if limit <= 0 or len(candidates) <= limit:
        return candidates
    rng = random.Random(seed)
    grouped: dict[str, list[CandidateExample]] = {}
    for candidate in candidates:
        key = f"{candidate.domain}:{candidate.gt_class}:{candidate.source}"
        grouped.setdefault(key, []).append(candidate)
    selected: list[CandidateExample] = []
    per_group = max(1, math.ceil(limit / max(len(grouped), 1)))
    for key in sorted(grouped):
        group = grouped[key]
        selected.extend(group if len(group) <= per_group else rng.sample(group, per_group))
    if len(selected) > limit:
        selected = rng.sample(selected, limit)
    selected.sort(key=lambda item: item.row_index)
    return selected


def assert_no_train_valid_overlap(
    train_candidates: list[CandidateExample],
    valid_splits: dict[str, list[Task2Sample]],
) -> None:
    train_ids = {candidate.sample_id for candidate in train_candidates}
    overlaps = {
        split_name: sorted(train_ids & {sample.sample_id for sample in samples})[:10]
        for split_name, samples in valid_splits.items()
    }
    overlaps = {name: items for name, items in overlaps.items() if items}
    if overlaps:
        raise ValueError(f"Train candidates overlap validation samples: {overlaps}")


def summarize_candidates(candidates: list[CandidateExample]) -> dict[str, Any]:
    return {
        "candidates": len(candidates),
        "source_counts": count_by(candidates, lambda item: item.source),
        "domain_class_counts": count_by(candidates, lambda item: f"{item.domain}:{item.gt_class}"),
        "mean_iou": float(np.mean([item.gt_iou for item in candidates])) if candidates else float("nan"),
        "min_iou": float(np.min([item.gt_iou for item in candidates])) if candidates else float("nan"),
    }


def count_by(candidates: Iterable[CandidateExample], key_fn: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = str(key_fn(candidate))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def flatten_eval_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for split_name, split_metrics in metrics.items():
        all_metrics = split_metrics["all"]
        for key, value in all_metrics.items():
            result[f"{split_name}/{key}"] = float(value)
    return result


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def primary_metric(metrics: dict[str, Any], split_name: str, metric_name: str) -> float:
    item = metrics[split_name]["all"]
    if metric_name == "after_r75":
        return float(item["after_recall_iou_0.75"])
    if metric_name == "after_mean_iou":
        return float(item["after_mean_best_iou"])
    if metric_name == "r75_gain":
        return float(item["recall_iou_0.75_gain"])
    if metric_name == "augmented_r75_gain":
        return float(item["augmented_recall_iou_0.75_gain"])
    if metric_name == "augmented_mean_iou_gain":
        return float(item["augmented_mean_best_iou_gain"])
    raise ValueError(f"Unknown primary metric: {metric_name}")


def format_epoch_summary(
    epoch: int,
    row: dict[str, float | int],
    primary_split: str,
    best_epoch: int,
    best_metric: float,
) -> str:
    return (
        f"epoch={epoch:03d} "
        f"train_loss={float(row['train/loss']):.4f} "
        f"{primary_split}_before_r75={float(row[f'{primary_split}/before_recall_iou_0.75']):.4f} "
        f"{primary_split}_after_r75={float(row[f'{primary_split}/after_recall_iou_0.75']):.4f} "
        f"{primary_split}_aug_r75={float(row[f'{primary_split}/augmented_recall_iou_0.75']):.4f} "
        f"{primary_split}_r75_gain={float(row[f'{primary_split}/recall_iou_0.75_gain']):.4f} "
        f"{primary_split}_aug_r75_gain={float(row[f'{primary_split}/augmented_recall_iou_0.75_gain']):.4f} "
        f"{primary_split}_aug_mean_iou_gain={float(row[f'{primary_split}/augmented_mean_best_iou_gain']):.4f} "
        f"best_epoch={best_epoch} best={best_metric:.4f}"
    )


def format_split_summary(split_name: str, metrics: dict[str, Any]) -> str:
    item = metrics["all"]
    return (
        f"{split_name} "
        f"before_r50={float(item['before_recall_iou_0.50']):.4f} "
        f"after_r50={float(item['after_recall_iou_0.50']):.4f} "
        f"aug_r50={float(item['augmented_recall_iou_0.50']):.4f} "
        f"before_r75={float(item['before_recall_iou_0.75']):.4f} "
        f"after_r75={float(item['after_recall_iou_0.75']):.4f} "
        f"aug_r75={float(item['augmented_recall_iou_0.75']):.4f} "
        f"aug_mean_iou_gain={float(item['augmented_mean_best_iou_gain']):.4f}"
    )


def write_refined_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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


def load_model_state(model: nn.Module, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=True)


def domain_of_sample(sample: Task2Sample) -> str:
    return "animal" if "animal" in sample.video_id.lower() else "phantom"


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value, REPO_ROOT)
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
