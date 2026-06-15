#!/usr/bin/env python3
"""Train a background-aware ROI verifier for CATHACTION Task 2 proposals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
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
from cathaction.data.task2_roi import (  # noqa: E402
    RoiCropConfig,
    class_counts,
    crop_roi_with_padding,
    load_task2_samples_from_split,
)
from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth,
    DetectionPrediction,
    compute_detection_map,
    iou_xyxy,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VERIFIER_CLASS_NAMES = ("background", "normal", "collision")


@dataclass(frozen=True)
class CandidateExample:
    split: str
    sample_id: str
    image_path: Path
    gt_class: int
    verifier_label: int
    xyxy: tuple[float, float, float, float]
    gt_xyxy: tuple[float, float, float, float]
    gt_iou: float
    det_conf: float
    source: str
    rank: int
    image_width: int
    image_height: int


class CandidateRoiDataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(
        self,
        candidates: list[CandidateExample],
        *,
        crop_config: RoiCropConfig,
        input_size: int,
        training: bool,
    ) -> None:
        self.candidates = candidates
        self.crop_config = crop_config
        self.transform = build_transform(input_size=input_size, training=training)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        candidate = self.candidates[index]
        with Image.open(candidate.image_path) as image:
            bounds = compute_roi_bounds_from_xyxy(candidate.xyxy, self.crop_config)
            roi = crop_roi_with_padding(image, bounds, fill=self.crop_config.fill)
        return self.transform(roi), candidate.verifier_label, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--train-split",
        type=Path,
        default=Path("configs/task2/splits_adapt_animal/train_v1_val_v2_train_labels.txt"),
    )
    parser.add_argument(
        "--valid-animal-split",
        type=Path,
        default=Path("configs/task2/splits_stage2j_balanced/pilot_train_v1_val_v2_valid_animal_labels.txt"),
    )
    parser.add_argument(
        "--valid-phantom-split",
        type=Path,
        default=Path("configs/task2/splits_stage2j_balanced/valid_phantom_balanced_small_labels.txt"),
    )
    parser.add_argument(
        "--proposal-weights",
        type=Path,
        default=Path("outputs/task2/yolo_adapt_animal/yolo11s_1024_train_v1_val_v2_e40/weights/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/roi_verifier"))
    parser.add_argument("--name", default="convnext_tiny_stage2h_proposals_v1_val_v2_small")

    parser.add_argument("--backend", choices=("auto", "timm", "torchvision"), default="auto")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=8.0)
    parser.add_argument("--min-crop-size", type=int, default=224)
    parser.add_argument("--max-crop-size", type=int, default=512)

    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--det-conf", type=float, default=0.001)
    parser.add_argument("--det-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--det-batch-size", type=int, default=32)
    parser.add_argument("--source-chunk-size", type=int, default=64)
    parser.add_argument("--positive-iou", type=float, default=0.50)
    parser.add_argument("--background-iou", type=float, default=0.20)
    parser.add_argument("--score-mode", choices=("roi", "det_roi"), default="det_roi")

    parser.add_argument("--train-samples-per-domain-class", type=int, default=1000)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--balanced-sampler", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--primary-metric", default="valid_animal/detection_argmax_mAP50")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    proposal_weights = resolve_path(args.proposal_weights)
    run_dir = resolve_path(args.output_dir) / args.name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    device_arg = "0" if device.type == "cuda" and device.index in (None, 0) else str(device)
    crop_config = RoiCropConfig(
        crop_scale=args.crop_scale,
        min_crop_size=args.min_crop_size,
        max_crop_size=args.max_crop_size,
    )

    train_samples_all = load_task2_samples_from_split(data_root, resolve_path(args.train_split))
    train_samples = stratified_sample_by_domain_class(
        train_samples_all,
        samples_per_domain_class=args.train_samples_per_domain_class,
        seed=args.seed,
    )
    valid_sets = {
        "valid_animal": maybe_limit(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_animal_split)),
            args.valid_limit,
        ),
        "valid_phantom": maybe_limit(
            load_task2_samples_from_split(data_root, resolve_path(args.valid_phantom_split)),
            args.valid_limit,
        ),
    }

    args_record = json_ready(
        {
            **vars(args),
            "repo_root": REPO_ROOT.as_posix(),
            "data_root": repo_relative(data_root, REPO_ROOT),
            "proposal_weights": repo_relative(proposal_weights, REPO_ROOT),
            "run_dir": repo_relative(run_dir, REPO_ROOT),
            "crop_config": asdict(crop_config),
            "device": str(device),
            "train_samples_all": len(train_samples_all),
            "train_samples": len(train_samples),
            "train_sample_class_counts": class_counts(train_samples),
            "train_sample_domain_class_counts": domain_class_counts(train_samples),
            "valid_sample_class_counts": {
                name: class_counts(samples) for name, samples in valid_sets.items()
            },
            "verifier_class_names": VERIFIER_CLASS_NAMES,
        }
    )
    (run_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(args_record, indent=2), flush=True)

    from ultralytics import YOLO

    proposal_model = YOLO(proposal_weights.as_posix())
    train_candidates = build_candidates(
        samples=train_samples,
        split_name="train",
        proposal_model=proposal_model,
        device_arg=device_arg,
        include_gt=True,
        training=True,
        args=args,
    )
    valid_candidates = {
        name: build_candidates(
            samples=samples,
            split_name=name,
            proposal_model=proposal_model,
            device_arg=device_arg,
            include_gt=False,
            training=False,
            args=args,
        )
        for name, samples in valid_sets.items()
    }

    write_candidate_csv(run_dir / "train_candidates.csv", train_candidates)
    for name, candidates in valid_candidates.items():
        write_candidate_csv(run_dir / f"{name}_candidates.csv", candidates)

    candidate_summary = {
        "train": summarize_candidate_labels(train_candidates),
        **{name: summarize_candidate_labels(candidates) for name, candidates in valid_candidates.items()},
    }
    (run_dir / "candidate_summary.json").write_text(
        json.dumps(json_ready(candidate_summary), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(candidate_summary), indent=2), flush=True)

    model = create_model(args.backend, args.model, pretrained=args.pretrained, num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    class_weight = make_class_weight(train_candidates, device) if args.class_weight else None

    train_loader = DataLoader(
        CandidateRoiDataset(
            train_candidates,
            crop_config=crop_config,
            input_size=args.input_size,
            training=True,
        ),
        batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_candidates) if args.balanced_sampler else None,
        shuffle=False if args.balanced_sampler else True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    valid_loaders = {
        name: DataLoader(
            CandidateRoiDataset(
                candidates,
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
        for name, candidates in valid_candidates.items()
    }

    best_metric = float("-inf")
    best_epoch = -1
    metrics_path = run_dir / "metrics.csv"
    best_metrics: dict[str, Any] | None = None
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
        detailed_metrics: dict[str, Any] = {}
        for split_name, loader in valid_loaders.items():
            metrics = evaluate_verifier(
                model=model,
                loader=loader,
                candidates=valid_candidates[split_name],
                samples=valid_sets[split_name],
                device=device,
                use_amp=args.amp,
                score_mode=args.score_mode,
            )
            detailed_metrics[split_name] = metrics
            row.update(flatten_metrics(split_name, metrics))

        append_metrics(metrics_path, row)
        primary = float(row[args.primary_metric])
        is_best = primary > best_metric
        if is_best:
            best_metric = primary
            best_epoch = epoch
            best_metrics = detailed_metrics
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row, args_record)
            (run_dir / "best_metrics.json").write_text(
                json.dumps(
                    json_ready(
                        {
                            "best_epoch": best_epoch,
                            "best_metric": best_metric,
                            "row": row,
                            "detailed_metrics": detailed_metrics,
                        }
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row, args_record)
        print(format_epoch_summary(epoch, row, args.primary_metric, best_epoch, best_metric), flush=True)

    print(f"Best {args.primary_metric}={best_metric:.6f} at epoch {best_epoch}", flush=True)
    print(f"Results saved to {run_dir}", flush=True)
    _ = best_metrics
    return 0


def build_candidates(
    *,
    samples: list[Task2Sample],
    split_name: str,
    proposal_model: Any,
    device_arg: str,
    include_gt: bool,
    training: bool,
    args: argparse.Namespace,
) -> list[CandidateExample]:
    candidates: list[CandidateExample] = []
    for chunk_start in range(0, len(samples), args.source_chunk_size):
        chunk_samples = samples[chunk_start : chunk_start + args.source_chunk_size]
        image_paths = [sample.image_path.as_posix() for sample in chunk_samples]
        results = proposal_model.predict(
            source=image_paths,
            imgsz=args.imgsz,
            conf=args.det_conf,
            iou=args.det_iou,
            max_det=args.max_det,
            batch=args.det_batch_size,
            device=device_arg,
            stream=True,
            verbose=False,
            agnostic_nms=True,
        )
        for chunk_offset, (sample, result) in enumerate(zip(chunk_samples, results)):
            sample_index = chunk_start + chunk_offset + 1
            image_height, image_width = int(result.orig_shape[0]), int(result.orig_shape[1])
            gt_xyxy = tuple(float(v) for v in sample.box.xyxy_pixels(image_width, image_height))
            if include_gt:
                candidates.append(
                    CandidateExample(
                        split=split_name,
                        sample_id=sample.sample_id,
                        image_path=sample.image_path,
                        gt_class=sample.class_id,
                        verifier_label=sample.class_id + 1,
                        xyxy=gt_xyxy,  # type: ignore[arg-type]
                        gt_xyxy=gt_xyxy,  # type: ignore[arg-type]
                        gt_iou=1.0,
                        det_conf=1.0,
                        source="gt",
                        rank=0,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                confidences = boxes.conf.detach().cpu().numpy()
                order = np.argsort(-confidences)[: args.max_det]
                for rank, box_index in enumerate(order.tolist(), start=1):
                    xyxy = tuple(
                        float(value)
                        for value in boxes.xyxy[box_index].detach().cpu().numpy().tolist()
                    )
                    gt_iou = iou_xyxy(xyxy, gt_xyxy)  # type: ignore[arg-type]
                    verifier_label = verifier_label_from_iou(
                        gt_class=sample.class_id,
                        gt_iou=gt_iou,
                        positive_iou=args.positive_iou,
                        background_iou=args.background_iou,
                        training=training,
                    )
                    if verifier_label is None:
                        continue
                    candidates.append(
                        CandidateExample(
                            split=split_name,
                            sample_id=sample.sample_id,
                            image_path=sample.image_path,
                            gt_class=sample.class_id,
                            verifier_label=verifier_label,
                            xyxy=xyxy,  # type: ignore[arg-type]
                            gt_xyxy=gt_xyxy,  # type: ignore[arg-type]
                            gt_iou=gt_iou,
                            det_conf=float(confidences[box_index]),
                            source="proposal",
                            rank=rank,
                            image_width=image_width,
                            image_height=image_height,
                        )
                    )

            if args.progress_every > 0 and sample_index % args.progress_every == 0:
                print(
                    f"{split_name}: generated candidates for {sample_index}/{len(samples)} samples",
                    flush=True,
                )
    return candidates


def verifier_label_from_iou(
    *,
    gt_class: int,
    gt_iou: float,
    positive_iou: float,
    background_iou: float,
    training: bool,
) -> int | None:
    if gt_iou >= positive_iou:
        return gt_class + 1
    if gt_iou <= background_iou:
        return 0
    return None if training else -1


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
    for images, labels, _indices in loader:
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
def evaluate_verifier(
    *,
    model: nn.Module,
    loader: DataLoader,
    candidates: list[CandidateExample],
    samples: list[Task2Sample],
    device: torch.device,
    use_amp: bool,
    score_mode: str,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    valid_labels: list[int] = []
    valid_preds: list[int] = []
    two_class_predictions: list[DetectionPrediction] = []
    argmax_predictions: list[DetectionPrediction] = []

    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            loss_mask = labels_device >= 0
            if torch.any(loss_mask):
                loss_values = F.cross_entropy(
                    logits[loss_mask],
                    labels_device[loss_mask],
                    reduction="none",
                )
                losses.extend(float(value) for value in loss_values.detach().cpu())
        probabilities_np = probabilities.detach().cpu().numpy()
        labels_np = labels.numpy()
        indices_np = indices.numpy()
        for probs, label, candidate_index in zip(probabilities_np, labels_np, indices_np):
            candidate = candidates[int(candidate_index)]
            pred_label = int(np.argmax(probs))
            if int(label) >= 0:
                valid_labels.append(int(label))
                valid_preds.append(pred_label)
            p_bg, p_normal, p_collision = (float(value) for value in probs.tolist())
            score_scale = candidate.det_conf if score_mode == "det_roi" else 1.0
            score_normal = score_scale * p_normal
            score_collision = score_scale * p_collision
            row = {
                "split": candidate.split,
                "sample_id": candidate.sample_id,
                "gt_class": candidate.gt_class,
                "verifier_label": int(label),
                "pred_label": pred_label,
                "prob_background": p_bg,
                "prob_normal": p_normal,
                "prob_collision": p_collision,
                "score_normal": score_normal,
                "score_collision": score_collision,
                "det_conf": candidate.det_conf,
                "gt_iou": candidate.gt_iou,
                "source": candidate.source,
                "rank": candidate.rank,
            }
            rows.append(row)
            box = candidate.xyxy
            two_class_predictions.append(
                DetectionPrediction(candidate.sample_id, 0, score_normal, box)
            )
            two_class_predictions.append(
                DetectionPrediction(candidate.sample_id, 1, score_collision, box)
            )
            if pred_label in (1, 2):
                pred_class = pred_label - 1
                pred_score = score_normal if pred_class == 0 else score_collision
                argmax_predictions.append(
                    DetectionPrediction(candidate.sample_id, pred_class, pred_score, box)
                )

    ground_truths = [
        DetectionGroundTruth(
            sample.sample_id,
            sample.class_id,
            sample.box.xyxy_pixels(
                infer_image_width(sample, candidates),
                infer_image_height(sample, candidates),
            ),
        )
        for sample in samples
    ]
    two_class_scores = compute_detection_map(
        ground_truths,
        two_class_predictions,
        class_ids=(0, 1),
    )
    argmax_scores = compute_detection_map(
        ground_truths,
        argmax_predictions,
        class_ids=(0, 1),
    )
    return {
        "candidate_classification": multiclass_metrics(valid_labels, valid_preds),
        "localization": summarize_best_localization(samples, candidates),
        "detection_two_class_scores": two_class_scores,
        "detection_argmax": argmax_scores,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "rows": len(rows),
    }


def infer_image_width(sample: Task2Sample, candidates: list[CandidateExample]) -> int:
    for candidate in candidates:
        if candidate.sample_id == sample.sample_id:
            return candidate.image_width
    with Image.open(sample.image_path) as image:
        return image.width


def infer_image_height(sample: Task2Sample, candidates: list[CandidateExample]) -> int:
    for candidate in candidates:
        if candidate.sample_id == sample.sample_id:
            return candidate.image_height
    with Image.open(sample.image_path) as image:
        return image.height


def summarize_best_localization(
    samples: list[Task2Sample],
    candidates: list[CandidateExample],
) -> dict[str, float | int]:
    by_sample: dict[str, list[CandidateExample]] = {}
    for candidate in candidates:
        by_sample.setdefault(candidate.sample_id, []).append(candidate)
    best_ious = [
        max((candidate.gt_iou for candidate in by_sample.get(sample.sample_id, [])), default=0.0)
        for sample in samples
    ]
    result: dict[str, float | int] = {
        "samples": len(samples),
        "candidates": len(candidates),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        result[f"recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in best_ious])) if best_ious else float("nan")
        )
    return result


def multiclass_metrics(labels: list[int], preds: list[int]) -> dict[str, Any]:
    if not labels:
        return {"accuracy": float("nan"), "per_class_recall": {}}
    labels_np = np.asarray(labels, dtype=np.int64)
    preds_np = np.asarray(preds, dtype=np.int64)
    metrics: dict[str, Any] = {"accuracy": float(np.mean(labels_np == preds_np))}
    per_class_recall: dict[str, float] = {}
    for class_id, class_name in enumerate(VERIFIER_CLASS_NAMES):
        mask = labels_np == class_id
        per_class_recall[class_name] = (
            float(np.mean(preds_np[mask] == class_id)) if np.any(mask) else float("nan")
        )
    metrics["per_class_recall"] = per_class_recall
    metrics["pred_counts"] = {
        VERIFIER_CLASS_NAMES[class_id]: int(np.sum(preds_np == class_id))
        for class_id in range(len(VERIFIER_CLASS_NAMES))
    }
    metrics["label_counts"] = {
        VERIFIER_CLASS_NAMES[class_id]: int(np.sum(labels_np == class_id))
        for class_id in range(len(VERIFIER_CLASS_NAMES))
    }
    return metrics


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float | int]:
    return {
        f"{prefix}/loss": float(metrics["loss"]),
        f"{prefix}/candidate_accuracy": float(metrics["candidate_classification"]["accuracy"]),
        f"{prefix}/loc_recall50": float(metrics["localization"]["recall_iou_0.50"]),
        f"{prefix}/detection_two_class_mAP50": float(metrics["detection_two_class_scores"]["mAP50"]),
        f"{prefix}/detection_two_class_mAP50_95": float(metrics["detection_two_class_scores"]["mAP50-95"]),
        f"{prefix}/detection_argmax_mAP50": float(metrics["detection_argmax"]["mAP50"]),
        f"{prefix}/detection_argmax_mAP50_95": float(metrics["detection_argmax"]["mAP50-95"]),
    }


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


def create_model(backend: str, model_name: str, *, pretrained: bool, num_classes: int) -> nn.Module:
    if backend in {"auto", "timm"}:
        try:
            import timm

            return timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=3,
            )
        except Exception:
            if backend == "timm":
                raise
    import torchvision.models as models

    if not hasattr(models, model_name):
        raise ValueError(f"Unknown torchvision model: {model_name}")
    factory = getattr(models, model_name)
    weights = "DEFAULT" if pretrained else None
    model = factory(weights=weights)
    return replace_torchvision_classifier(model, num_classes=num_classes)


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


def make_class_weight(candidates: list[CandidateExample], device: torch.device) -> torch.Tensor:
    counts = verifier_label_counts(candidates)
    total = sum(counts.values())
    weights = [total / max(counts.get(class_id, 1), 1) for class_id in range(3)]
    mean_weight = sum(weights) / len(weights)
    normalized = [weight / mean_weight for weight in weights]
    return torch.tensor(normalized, dtype=torch.float32, device=device)


def make_balanced_sampler(candidates: list[CandidateExample]) -> WeightedRandomSampler:
    counts = verifier_label_counts(candidates)
    weights = [1.0 / max(counts[candidate.verifier_label], 1) for candidate in candidates]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def verifier_label_counts(candidates: Iterable[CandidateExample]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for candidate in candidates:
        if candidate.verifier_label < 0:
            continue
        counts[candidate.verifier_label] = counts.get(candidate.verifier_label, 0) + 1
    return dict(sorted(counts.items()))


def summarize_candidate_labels(candidates: list[CandidateExample]) -> dict[str, Any]:
    return {
        "candidates": len(candidates),
        "label_counts": {
            VERIFIER_CLASS_NAMES[key] if key >= 0 else "ignore": value
            for key, value in verifier_label_counts_including_ignore(candidates).items()
        },
        "source_counts": counts_by(candidates, lambda item: item.source),
        "gt_class_counts": counts_by(candidates, lambda item: str(item.gt_class)),
        "mean_gt_iou": float(np.mean([item.gt_iou for item in candidates])) if candidates else float("nan"),
    }


def verifier_label_counts_including_ignore(candidates: Iterable[CandidateExample]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for candidate in candidates:
        counts[candidate.verifier_label] = counts.get(candidate.verifier_label, 0) + 1
    return dict(sorted(counts.items()))


def counts_by(candidates: Iterable[CandidateExample], key_fn: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = str(key_fn(candidate))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def stratified_sample_by_domain_class(
    samples: list[Task2Sample],
    *,
    samples_per_domain_class: int,
    seed: int,
) -> list[Task2Sample]:
    if samples_per_domain_class <= 0:
        return samples
    rng = random.Random(seed)
    grouped: dict[tuple[str, int], list[Task2Sample]] = {}
    for sample in samples:
        grouped.setdefault((domain_of(sample), sample.class_id), []).append(sample)
    selected: list[Task2Sample] = []
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) <= samples_per_domain_class:
            picked = group
        else:
            picked = rng.sample(group, samples_per_domain_class)
        selected.extend(picked)
    selected.sort(key=lambda item: (item.video_id, item.frame_index, item.sample_id))
    return selected


def domain_of(sample: Task2Sample) -> str:
    return "animal" if "animal" in sample.video_id else "phantom"


def domain_class_counts(samples: Iterable[Task2Sample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = f"{domain_of(sample)}:{sample.class_id}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def maybe_limit(samples: list[Task2Sample], limit: int | None) -> list[Task2Sample]:
    if limit is None:
        return samples
    return samples[:limit]


def write_candidate_csv(path: Path, candidates: list[CandidateExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "sample_id",
        "image_path",
        "gt_class",
        "verifier_label",
        "x1",
        "y1",
        "x2",
        "y2",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "gt_iou",
        "det_conf",
        "source",
        "rank",
        "image_width",
        "image_height",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "split": candidate.split,
                    "sample_id": candidate.sample_id,
                    "image_path": repo_relative(candidate.image_path, REPO_ROOT),
                    "gt_class": candidate.gt_class,
                    "verifier_label": candidate.verifier_label,
                    "x1": candidate.xyxy[0],
                    "y1": candidate.xyxy[1],
                    "x2": candidate.xyxy[2],
                    "y2": candidate.xyxy[3],
                    "gt_x1": candidate.gt_xyxy[0],
                    "gt_y1": candidate.gt_xyxy[1],
                    "gt_x2": candidate.gt_xyxy[2],
                    "gt_y2": candidate.gt_xyxy[3],
                    "gt_iou": candidate.gt_iou,
                    "det_conf": candidate.det_conf,
                    "source": candidate.source,
                    "rank": candidate.rank,
                    "image_width": candidate.image_width,
                    "image_height": candidate.image_height,
                }
            )


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
        f"animal_argmax_mAP50={float(row['valid_animal/detection_argmax_mAP50']):.4f} "
        f"animal_two_cls_mAP50={float(row['valid_animal/detection_two_class_mAP50']):.4f} "
        f"phantom_argmax_mAP50={float(row['valid_phantom/detection_argmax_mAP50']):.4f} "
        f"phantom_two_cls_mAP50={float(row['valid_phantom/detection_two_class_mAP50']):.4f} "
        f"primary={primary_metric}:{float(row[primary_metric]):.4f} "
        f"best_epoch={best_epoch} best={best_metric:.4f}"
    )


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
