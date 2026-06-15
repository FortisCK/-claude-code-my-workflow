#!/usr/bin/env python3
"""Train an image-aware ROI quality ranker for Task 2 Stage2U."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import (  # noqa: E402
    RoiCropConfig,
    crop_roi_with_padding,
    load_task2_samples_from_split,
)
from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth,
    DetectionPrediction,
    compute_detection_map,
)
from scripts.task2.train_stage2o_candidate_ranker import (  # noqa: E402
    CandidateExample,
    RANKER_CLASS_NAMES,
    SCORE_MODES,
    append_metrics,
    apply_candidate_budget,
    build_transform,
    candidate_score,
    compute_roi_bounds_from_xyxy,
    create_model,
    first_non_empty,
    infer_height,
    infer_width,
    json_ready,
    load_candidate_csv,
    make_class_weight,
    multiclass_metrics,
    parse_valid_csv_specs,
    resolve_path,
    save_checkpoint,
    set_seed,
    stratified_candidate_limit,
    summarize_best_localization,
    summarize_candidate_table,
    write_candidate_csv,
    write_prediction_csv,
)


QUALITY_MODES = ("pred_iou", "prob_iou50", "prob_iou75", "blend")
QUALITY_SCORE_MODES = tuple(f"{quality}_{base}" for quality in QUALITY_MODES for base in SCORE_MODES)
ALL_DETECTION_MODES = SCORE_MODES + QUALITY_SCORE_MODES
SOURCE_NAMES = (
    "yolo_stage2l",
    "task1_geometry_rect",
    "sequence_tip",
    "stage2w_dense",
    "stage2x_class1",
    "stage2p_refined",
    "unknown",
)
SOURCE_TO_INDEX = {name: index for index, name in enumerate(SOURCE_NAMES)}
SOURCE_METADATA_DIM = len(SOURCE_NAMES) + 9


class SourceAwareResidualRanker(nn.Module):
    def __init__(self, image_model: nn.Module, *, metadata_dim: int, output_dim: int = 6) -> None:
        super().__init__()
        self.image_model = image_model
        hidden_dim = max(16, metadata_dim * 2)
        self.metadata_head = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        final = self.metadata_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, images: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        image_outputs = self.image_model(images)
        if metadata is None:
            return image_outputs
        return image_outputs + self.metadata_head(metadata)


class QualityRankerDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]]):
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]:
        candidate = self.candidates[index]
        with Image.open(candidate.image_path) as image:
            bounds = compute_roi_bounds_from_xyxy(candidate.xyxy, self.crop_config)
            roi = crop_roi_with_padding(image, bounds, fill=self.crop_config.fill)
        targets = encode_quality_targets(candidate.gt_iou)
        return self.transform(roi), encode_source_metadata(candidate), candidate.verifier_label, targets, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path(
            "outputs/task2/stage2o_candidate_pool/"
            "stage2o_train_subset_yolo_geometry_top50/valid_combined_candidates.csv"
        ),
        help="Non-leakage training candidate CSV. Rows with verifier_label < 0 are skipped by default.",
    )
    parser.add_argument(
        "--valid-csv",
        action="append",
        default=None,
        help="Validation CSV as split_name=path. Defaults to Stage2O valid candidate CSVs.",
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/stage2u_quality_ranker"))
    parser.add_argument("--name", default="convnext_tiny_stage2u_quality_e6")

    parser.add_argument("--backend", choices=("auto", "timm", "torchvision"), default="auto")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--source-aware",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Add a zero-initialized source/geometry metadata residual head on top of the ROI image model.",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=8.0)
    parser.add_argument("--min-crop-size", type=int, default=224)
    parser.add_argument("--max-crop-size", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--iou-loss-weight", type=float, default=1.0)
    parser.add_argument("--iou50-loss-weight", type=float, default=0.5)
    parser.add_argument("--iou75-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--balanced-sampler", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--keep-unlabeled-train", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--primary-score-mode", choices=ALL_DETECTION_MODES, default="prob_iou75_rank_decay_roi")
    parser.add_argument("--primary-valid-split", default="valid_combined")
    parser.add_argument("--primary-map-key", choices=("mAP50", "mAP50-95"), default="mAP50-95")
    parser.add_argument("--max-candidates-per-sample", type=int, default=100)
    parser.add_argument("--candidate-budget-sort", choices=("source_order", "raw_conf"), default="source_order")
    parser.add_argument(
        "--valid-source-keep",
        default=None,
        help=(
            "Comma-separated validation candidate sources to keep, e.g. yolo_stage2l. "
            "Applied only to validation/eval candidates, not training candidates."
        ),
    )
    parser.add_argument(
        "--train-source-keep",
        default=None,
        help=(
            "Comma-separated training candidate sources to keep, e.g. yolo_stage2l. "
            "Useful when the final export uses a restricted source set."
        ),
    )
    parser.add_argument(
        "--primary-metric-key",
        default=None,
        help=(
            "Optional exact flattened metric key to select the best checkpoint, "
            "for example valid_phantom/prob_iou75_rank_decay_roi/class1_ap50. "
            "Overrides --primary-score-mode/--primary-valid-split/--primary-map-key."
        ),
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print train/eval progress every N batches. Default 0 preserves compact epoch-only logging.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint for partial warm-start. Matching-shape tensors are loaded; "
            "the final classifier head can differ, e.g. Stage2O 3-class -> Stage2U 6-output."
        ),
    )
    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--skip-overlap-check",
        default=False,
        action="store_true",
        help="Skip train/valid sample-id overlap checks. Use only for intentional train-side exports.",
    )
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

    crop_config = RoiCropConfig(
        crop_scale=float(args.crop_scale),
        min_crop_size=int(args.min_crop_size),
        max_crop_size=int(args.max_crop_size),
    )
    device = choose_device(args.device)

    train_source_keep = parse_source_keep(args.train_source_keep)
    train_candidates = filter_candidates_by_source(
        load_candidate_csv(train_csv, split_override="train", repo_root=REPO_ROOT),
        train_source_keep,
    )
    if not args.keep_unlabeled_train:
        train_candidates = [candidate for candidate in train_candidates if candidate.verifier_label >= 0]
    train_candidates = apply_candidate_budget(
        train_candidates,
        max_candidates_per_sample=int(args.max_candidates_per_sample),
        sort_mode=str(args.candidate_budget_sort),
    )
    if args.train_limit is not None:
        train_candidates = stratified_candidate_limit(train_candidates, int(args.train_limit), seed=int(args.seed))
    if not train_candidates:
        raise ValueError(f"No usable training candidates loaded from {train_csv}")

    valid_source_keep = parse_source_keep(args.valid_source_keep)
    valid_candidates_all = {
        name: filter_candidates_by_source(
            apply_candidate_budget(
                load_candidate_csv(resolve_path(path), split_override=name, repo_root=REPO_ROOT),
                max_candidates_per_sample=int(args.max_candidates_per_sample),
                sort_mode=str(args.candidate_budget_sort),
            ),
            valid_source_keep,
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
            "valid_source_keep": sorted(valid_source_keep) if valid_source_keep is not None else None,
            "train_source_keep": sorted(train_source_keep) if train_source_keep is not None else None,
            "data_root": repo_relative(data_root, REPO_ROOT),
            "run_dir": repo_relative(run_dir, REPO_ROOT),
            "device": str(device),
            "crop_config": asdict(crop_config),
            "ranker_class_names": RANKER_CLASS_NAMES,
            "base_score_modes": SCORE_MODES,
            "quality_modes": QUALITY_MODES,
            "detection_modes": ALL_DETECTION_MODES,
            "source_metadata_dim": SOURCE_METADATA_DIM if args.source_aware else 0,
            "source_names": SOURCE_NAMES if args.source_aware else (),
            "train_candidate_summary": summarize_candidate_table(train_candidates),
            "valid_candidate_summary": {
                name: summarize_candidate_table(candidates)
                for name, candidates in valid_candidates.items()
            },
            "valid_sample_counts": {name: len(samples) for name, samples in valid_splits.items()},
        }
    )
    (run_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(args_record, indent=2), flush=True)

    write_candidate_csv(run_dir / "train_candidates_used.csv", train_candidates)
    for split_name, candidates in valid_candidates.items():
        write_candidate_csv(run_dir / f"{split_name}_candidates_used.csv", candidates)

    image_model = create_model(args.backend, args.model, pretrained=args.pretrained, num_classes=6)
    model: nn.Module
    if args.source_aware:
        model = SourceAwareResidualRanker(image_model, metadata_dim=SOURCE_METADATA_DIM, output_dim=6)
    else:
        model = image_model
    model = model.to(device)
    if args.init_checkpoint is not None and args.eval_checkpoint is None:
        init_summary = load_partial_checkpoint(model, resolve_path(args.init_checkpoint))
        (run_dir / "init_checkpoint_summary.json").write_text(
            json.dumps(
                json_ready(
                    {
                        "init_checkpoint": repo_relative(resolve_path(args.init_checkpoint), REPO_ROOT),
                        **init_summary,
                    }
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Warm-started from {args.init_checkpoint}: {init_summary}", flush=True)
    if args.eval_checkpoint is not None:
        checkpoint_path = resolve_path(args.eval_checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict, strict=True)
        eval_metrics, eval_rows = evaluate_all_splits(
            model=model,
            valid_candidates=valid_candidates,
            valid_splits=valid_splits,
            crop_config=crop_config,
            args=args,
            device=device,
        )
        for split_name, rows in eval_rows.items():
            write_prediction_csv(run_dir / f"{split_name}_eval_prediction_rows.csv", rows)
        (run_dir / "eval_metrics.json").write_text(
            json.dumps(
                json_ready(
                    {
                        "checkpoint": repo_relative(checkpoint_path, REPO_ROOT),
                        "metrics": eval_metrics,
                    }
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Eval-only results saved to {run_dir}", flush=True)
        return 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")
    class_weight = make_class_weight(train_candidates, device) if args.class_weight else None

    train_loader = DataLoader(
        QualityRankerDataset(
            train_candidates,
            crop_config=crop_config,
            input_size=int(args.input_size),
            training=True,
        ),
        batch_size=int(args.batch_size),
        sampler=make_quality_sampler(train_candidates) if args.balanced_sampler else None,
        shuffle=False if args.balanced_sampler else True,
        num_workers=int(args.workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
    )

    best_metric = float("-inf")
    best_epoch = -1
    metrics_path = run_dir / "metrics.csv"
    primary_metric_key = str(args.primary_metric_key) if args.primary_metric_key else (
        f"{args.primary_valid_split}/{args.primary_score_mode}/"
        f"{'mAP50_95' if args.primary_map_key == 'mAP50-95' else 'mAP50'}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            class_weight=class_weight,
            args=args,
        )
        detailed_metrics, prediction_rows = evaluate_all_splits(
            model=model,
            valid_candidates=valid_candidates,
            valid_splits=valid_splits,
            crop_config=crop_config,
            args=args,
            device=device,
        )
        row: dict[str, float | int] = {"epoch": epoch, **prefix_metrics("train", train_metrics)}
        for split_name, metrics in detailed_metrics.items():
            row.update(flatten_metrics(split_name, metrics))
        append_metrics(metrics_path, row)
        primary = float(row[primary_metric_key])
        if primary > best_metric:
            best_metric = primary
            best_epoch = epoch
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row, args_record)
            for split_name, rows in prediction_rows.items():
                write_prediction_csv(run_dir / f"{split_name}_best_prediction_rows.csv", rows)
            (run_dir / "best_metrics.json").write_text(
                json.dumps(
                    json_ready(
                        {
                            "best_epoch": best_epoch,
                            "best_metric": best_metric,
                            "primary_metric_key": primary_metric_key,
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
        print(format_epoch_summary(epoch, row, primary_metric_key, best_epoch, best_metric), flush=True)

    print(f"Best {primary_metric_key}={best_metric:.6f} at epoch {best_epoch}", flush=True)
    print(f"Results saved to {run_dir}", flush=True)
    return 0


def parse_source_keep(value: str | None) -> set[str] | None:
    if value is None:
        return None
    sources = {item.strip() for item in value.split(",") if item.strip()}
    if not sources:
        raise ValueError("--valid-source-keep was provided but no non-empty sources were parsed")
    return sources


def filter_candidates_by_source(
    candidates: list[CandidateExample],
    source_keep: set[str] | None,
) -> list[CandidateExample]:
    if source_keep is None:
        return candidates
    return [candidate for candidate in candidates if candidate.source in source_keep]


def encode_quality_targets(gt_iou: float) -> torch.Tensor:
    clipped = float(np.clip(gt_iou, 0.0, 1.0))
    return torch.tensor(
        (
            clipped,
            1.0 if clipped >= 0.50 else 0.0,
            1.0 if clipped >= 0.75 else 0.0,
        ),
        dtype=torch.float32,
    )


def split_quality_outputs(outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if outputs.ndim != 2 or outputs.shape[1] != 6:
        raise ValueError(f"Stage2U output must have shape (N, 6), got {tuple(outputs.shape)}")
    class_logits = outputs[:, :3]
    pred_iou = torch.sigmoid(outputs[:, 3])
    iou50_logits = outputs[:, 4]
    iou75_logits = outputs[:, 5]
    return class_logits, pred_iou, iou50_logits, iou75_logits


def encode_source_metadata(candidate: CandidateExample) -> torch.Tensor:
    source_index = SOURCE_TO_INDEX.get(candidate.source, SOURCE_TO_INDEX["unknown"])
    source_one_hot = [0.0] * len(SOURCE_NAMES)
    source_one_hot[source_index] = 1.0

    x1, y1, x2, y2 = candidate.xyxy
    image_width = max(float(candidate.image_width), 1.0)
    image_height = max(float(candidate.image_height), 1.0)
    box_width = max(float(x2 - x1), 1.0)
    box_height = max(float(y2 - y1), 1.0)
    center_x = ((float(x1) + float(x2)) * 0.5) / image_width
    center_y = ((float(y1) + float(y2)) * 0.5) / image_height
    width_norm = box_width / image_width
    height_norm = box_height / image_height
    area_norm = (box_width * box_height) / max(image_width * image_height, 1.0)
    aspect_log = math.log(box_width / box_height)
    rank_decay = 1.0 / math.sqrt(max(float(candidate.source_rank), 1.0))
    priority_norm = 1.0 / (1.0 + max(float(candidate.source_priority), 0.0))
    source_conf = float(np.clip(candidate.source_conf, 0.0, 1.0))
    features = [
        *source_one_hot,
        source_conf,
        rank_decay,
        priority_norm,
        float(np.clip(center_x, 0.0, 1.0)),
        float(np.clip(center_y, 0.0, 1.0)),
        float(np.clip(width_norm, 0.0, 1.0)),
        float(np.clip(height_norm, 0.0, 1.0)),
        float(np.clip(area_norm, 0.0, 1.0)),
        float(np.clip(aspect_log, -5.0, 5.0) / 5.0),
    ]
    return torch.tensor(features, dtype=torch.float32)


def forward_ranker(model: nn.Module, images: torch.Tensor, metadata: torch.Tensor | None) -> torch.Tensor:
    if isinstance(model, SourceAwareResidualRanker):
        return model(images, metadata)
    return model(images)


def quality_blend(pred_iou: float, prob_iou50: float, prob_iou75: float) -> float:
    return float(np.clip(0.50 * pred_iou + 0.25 * prob_iou50 + 0.25 * prob_iou75, 0.0, 1.0))


def quality_values(pred_iou: float, prob_iou50: float, prob_iou75: float) -> dict[str, float]:
    return {
        "pred_iou": float(np.clip(pred_iou, 0.0, 1.0)),
        "prob_iou50": float(np.clip(prob_iou50, 0.0, 1.0)),
        "prob_iou75": float(np.clip(prob_iou75, 0.0, 1.0)),
        "blend": quality_blend(pred_iou, prob_iou50, prob_iou75),
    }


def quality_adjusted_score(
    candidate: CandidateExample,
    p_bg: float,
    p_class: float,
    *,
    base_mode: str,
    quality: float,
) -> float:
    return float(np.clip(quality, 0.0, 1.0)) * candidate_score(candidate, p_bg, p_class, mode=base_mode)


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    class_weight: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "class_loss": 0.0,
        "iou_loss": 0.0,
        "iou50_loss": 0.0,
        "iou75_loss": 0.0,
    }
    total_items = 0
    log_interval = max(int(getattr(args, "log_interval", 0)), 0)
    start_time = time.monotonic()
    total_batches = len(loader)
    for batch_index, (images, metadata, labels, quality_targets, _indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        quality_targets = quality_targets.to(device, non_blocking=True)
        gt_iou = quality_targets[:, 0]
        iou50_targets = quality_targets[:, 1]
        iou75_targets = quality_targets[:, 2]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=bool(args.amp) and device.type == "cuda"):
            outputs = forward_ranker(model, images, metadata)
            class_logits, pred_iou, iou50_logits, iou75_logits = split_quality_outputs(outputs)
            label_mask = labels >= 0
            if torch.any(label_mask):
                class_loss = F.cross_entropy(class_logits[label_mask], labels[label_mask], weight=class_weight)
            else:
                class_loss = class_logits.sum() * 0.0
            iou_loss = F.smooth_l1_loss(pred_iou, gt_iou, beta=float(args.smooth_l1_beta), reduction="mean")
            iou50_loss = F.binary_cross_entropy_with_logits(iou50_logits, iou50_targets)
            iou75_loss = F.binary_cross_entropy_with_logits(iou75_logits, iou75_targets)
            loss = (
                float(args.class_loss_weight) * class_loss
                + float(args.iou_loss_weight) * iou_loss
                + float(args.iou50_loss_weight) * iou50_loss
                + float(args.iou75_loss_weight) * iou75_loss
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.numel())
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["class_loss"] += float(class_loss.detach().cpu()) * batch_size
        totals["iou_loss"] += float(iou_loss.detach().cpu()) * batch_size
        totals["iou50_loss"] += float(iou50_loss.detach().cpu()) * batch_size
        totals["iou75_loss"] += float(iou75_loss.detach().cpu()) * batch_size
        total_items += batch_size
        if log_interval and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches):
            elapsed = time.monotonic() - start_time
            print(
                "train progress: "
                f"batch={batch_index}/{total_batches} "
                f"items={total_items} "
                f"loss={totals['loss'] / max(total_items, 1):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return {key: value / max(total_items, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_all_splits(
    *,
    model: nn.Module,
    valid_candidates: dict[str, list[CandidateExample]],
    valid_splits: dict[str, list[Task2Sample]],
    crop_config: RoiCropConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    metrics: dict[str, Any] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name, candidates in valid_candidates.items():
        loader = DataLoader(
            QualityRankerDataset(
                candidates,
                crop_config=crop_config,
                input_size=int(args.input_size),
                training=False,
            ),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.workers) > 0,
        )
        split_metrics, rows = evaluate_quality_ranker(
            model=model,
            loader=loader,
            candidates=candidates,
            samples=valid_splits[split_name],
            device=device,
            use_amp=bool(args.amp),
            smooth_l1_beta=float(args.smooth_l1_beta),
            log_interval=max(int(getattr(args, "log_interval", 0)), 0),
            split_name=split_name,
        )
        metrics[split_name] = split_metrics
        rows_by_split[split_name] = rows
        print(format_eval_summary(split_name, split_metrics), flush=True)
    return metrics, rows_by_split


@torch.no_grad()
def evaluate_quality_ranker(
    *,
    model: nn.Module,
    loader: DataLoader,
    candidates: list[CandidateExample],
    samples: list[Task2Sample],
    device: torch.device,
    use_amp: bool,
    smooth_l1_beta: float,
    log_interval: int = 0,
    split_name: str = "valid",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    valid_labels: list[int] = []
    valid_preds: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    predictions_by_mode: dict[str, list[DetectionPrediction]] = {mode: [] for mode in ALL_DETECTION_MODES}
    quality_abs_errors: list[float] = []
    losses = {"class_loss": [], "iou_loss": [], "iou50_loss": [], "iou75_loss": []}

    log_interval = max(int(log_interval), 0)
    start_time = time.monotonic()
    total_batches = len(loader)
    total_items = 0
    for batch_index, (images, metadata, labels, quality_targets, indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        quality_targets_device = quality_targets.to(device, non_blocking=True)
        gt_iou = quality_targets_device[:, 0]
        iou50_targets = quality_targets_device[:, 1]
        iou75_targets = quality_targets_device[:, 2]
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            outputs = forward_ranker(model, images, metadata)
            class_logits, pred_iou, iou50_logits, iou75_logits = split_quality_outputs(outputs)
            probabilities = torch.softmax(class_logits, dim=1)
            label_mask = labels_device >= 0
            if torch.any(label_mask):
                class_loss_values = F.cross_entropy(
                    class_logits[label_mask],
                    labels_device[label_mask],
                    reduction="none",
                )
                losses["class_loss"].extend(float(value) for value in class_loss_values.detach().cpu())
            iou_loss_values = F.smooth_l1_loss(
                pred_iou,
                gt_iou,
                beta=smooth_l1_beta,
                reduction="none",
            )
            iou50_loss_values = F.binary_cross_entropy_with_logits(iou50_logits, iou50_targets, reduction="none")
            iou75_loss_values = F.binary_cross_entropy_with_logits(iou75_logits, iou75_targets, reduction="none")
            losses["iou_loss"].extend(float(value) for value in iou_loss_values.detach().cpu())
            losses["iou50_loss"].extend(float(value) for value in iou50_loss_values.detach().cpu())
            losses["iou75_loss"].extend(float(value) for value in iou75_loss_values.detach().cpu())

        probabilities_np = probabilities.detach().cpu().numpy()
        pred_iou_np = pred_iou.detach().cpu().numpy()
        prob_iou50_np = torch.sigmoid(iou50_logits).detach().cpu().numpy()
        prob_iou75_np = torch.sigmoid(iou75_logits).detach().cpu().numpy()
        labels_np = labels.numpy()
        gt_iou_np = quality_targets[:, 0].numpy()
        indices_np = indices.numpy()

        for probs, pred_iou_value, prob_iou50, prob_iou75, label, gt_iou_value, candidate_index in zip(
            probabilities_np,
            pred_iou_np,
            prob_iou50_np,
            prob_iou75_np,
            labels_np,
            gt_iou_np,
            indices_np,
        ):
            candidate = candidates[int(candidate_index)]
            pred_label = int(np.argmax(probs))
            if int(label) >= 0:
                valid_labels.append(int(label))
                valid_preds.append(pred_label)
            p_bg, p_normal, p_collision = (float(value) for value in probs.tolist())
            q_values = quality_values(float(pred_iou_value), float(prob_iou50), float(prob_iou75))
            quality_abs_errors.append(abs(float(pred_iou_value) - float(gt_iou_value)))
            row = {
                "split": candidate.split,
                "sample_id": candidate.sample_id,
                "gt_class": candidate.gt_class,
                "verifier_label": int(label),
                "pred_label": pred_label,
                "prob_background": p_bg,
                "prob_normal": p_normal,
                "prob_collision": p_collision,
                "source_conf": candidate.source_conf,
                "source": candidate.source,
                "source_rank": candidate.source_rank,
                "gt_iou": candidate.gt_iou,
                "pred_iou": q_values["pred_iou"],
                "prob_iou50": q_values["prob_iou50"],
                "prob_iou75": q_values["prob_iou75"],
                "quality_blend": q_values["blend"],
                "iou_abs_error": abs(q_values["pred_iou"] - float(gt_iou_value)),
            }
            for base_mode in SCORE_MODES:
                score_normal = candidate_score(candidate, p_bg, p_normal, mode=base_mode)
                score_collision = candidate_score(candidate, p_bg, p_collision, mode=base_mode)
                row[f"{base_mode}_score_normal"] = score_normal
                row[f"{base_mode}_score_collision"] = score_collision
                predictions_by_mode[base_mode].append(
                    DetectionPrediction(candidate.sample_id, 0, score_normal, candidate.xyxy)
                )
                predictions_by_mode[base_mode].append(
                    DetectionPrediction(candidate.sample_id, 1, score_collision, candidate.xyxy)
                )
                for quality_mode, quality in q_values.items():
                    mode = f"{quality_mode}_{base_mode}" if quality_mode != "blend" else f"blend_{base_mode}"
                    q_score_normal = quality_adjusted_score(
                        candidate,
                        p_bg,
                        p_normal,
                        base_mode=base_mode,
                        quality=quality,
                    )
                    q_score_collision = quality_adjusted_score(
                        candidate,
                        p_bg,
                        p_collision,
                        base_mode=base_mode,
                        quality=quality,
                    )
                    row[f"{mode}_score_normal"] = q_score_normal
                    row[f"{mode}_score_collision"] = q_score_collision
                    predictions_by_mode[mode].append(
                        DetectionPrediction(candidate.sample_id, 0, q_score_normal, candidate.xyxy)
                    )
                    predictions_by_mode[mode].append(
                        DetectionPrediction(candidate.sample_id, 1, q_score_collision, candidate.xyxy)
                    )
            prediction_rows.append(row)

        total_items += int(labels.numel())
        if log_interval and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches):
            elapsed = time.monotonic() - start_time
            print(
                "eval progress: "
                f"split={split_name} "
                f"batch={batch_index}/{total_batches} "
                f"items={total_items} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    ground_truths = [
        DetectionGroundTruth(
            sample.sample_id,
            sample.class_id,
            sample.box.xyxy_pixels(infer_width(sample, candidates), infer_height(sample, candidates)),
        )
        for sample in samples
    ]
    detection_scores = {
        mode: compute_detection_map(ground_truths, predictions, class_ids=(0, 1))
        for mode, predictions in predictions_by_mode.items()
    }
    loss_means = {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in losses.items()
    }
    return (
        {
            "candidate_classification": multiclass_metrics(valid_labels, valid_preds),
            "localization": summarize_best_localization(samples, candidates),
            "quality": {
                "pred_iou_mae": float(np.mean(quality_abs_errors)) if quality_abs_errors else float("nan"),
                **loss_means,
            },
            "detection": detection_scores,
            "loss": float(np.nansum(list(loss_means.values()))),
            "rows": len(prediction_rows),
        },
        prediction_rows,
    )


def make_quality_sampler(candidates: list[CandidateExample]) -> WeightedRandomSampler:
    counts: dict[str, int] = {}
    keys: list[str] = []
    for candidate in candidates:
        if candidate.gt_iou >= 0.75:
            bucket = "iou75"
        elif candidate.gt_iou >= 0.50:
            bucket = "iou50"
        elif candidate.gt_iou >= 0.25:
            bucket = "iou25"
        else:
            bucket = "low"
        key = f"{candidate.domain}:{candidate.gt_class}:{candidate.verifier_label}:{bucket}"
        keys.append(key)
        counts[key] = counts.get(key, 0) + 1
    weights = torch.tensor([1.0 / counts[key] for key in keys], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(candidates), replacement=True)


def assert_no_train_valid_overlap(
    train_candidates: list[CandidateExample],
    valid_splits: dict[str, list[Task2Sample]],
) -> None:
    train_ids = {candidate.sample_id for candidate in train_candidates}
    for split_name, samples in valid_splits.items():
        overlap = train_ids & {sample.sample_id for sample in samples}
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Train/valid sample overlap for {split_name}: {preview}")


def load_partial_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = payload.get("model_state_dict", payload)
    target_state = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in source_state.items():
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            matched[key] = value
        elif f"image_model.{key}" in target_state and tuple(target_state[f"image_model.{key}"].shape) == tuple(value.shape):
            matched[f"image_model.{key}"] = value
        else:
            skipped.append(key)
    target_state.update(matched)
    model.load_state_dict(target_state, strict=True)
    return {
        "loaded_tensors": len(matched),
        "skipped_tensors": len(skipped),
        "skipped_preview": skipped[:20],
    }


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        f"{prefix}/loss": float(metrics["loss"]),
        f"{prefix}/candidate_accuracy": float(metrics["candidate_classification"]["accuracy"]),
        f"{prefix}/quality_pred_iou_mae": float(metrics["quality"]["pred_iou_mae"]),
        f"{prefix}/quality_class_loss": float(metrics["quality"]["class_loss"]),
        f"{prefix}/quality_iou_loss": float(metrics["quality"]["iou_loss"]),
        f"{prefix}/quality_iou50_loss": float(metrics["quality"]["iou50_loss"]),
        f"{prefix}/quality_iou75_loss": float(metrics["quality"]["iou75_loss"]),
        f"{prefix}/loc_recall50": float(metrics["localization"]["recall_iou_0.50"]),
        f"{prefix}/loc_recall75": float(metrics["localization"]["recall_iou_0.75"]),
    }
    for mode, scores in metrics["detection"].items():
        result[f"{prefix}/{mode}/mAP50"] = float(scores["mAP50"])
        result[f"{prefix}/{mode}/mAP50_95"] = float(scores["mAP50-95"])
        classes = scores["classes"]
        for class_id in ("0", "1"):
            result[f"{prefix}/{mode}/class{class_id}_ap50"] = float(classes[class_id]["ap50"])
            result[f"{prefix}/{mode}/class{class_id}_ap50_95"] = float(classes[class_id]["ap50_95"])
    return result


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def format_epoch_summary(
    epoch: int,
    row: dict[str, float | int],
    primary_metric_key: str,
    best_epoch: int,
    best_metric: float,
) -> str:
    return (
        f"epoch={epoch} "
        f"train_loss={float(row['train/loss']):.4f} "
        f"{primary_metric_key}={float(row[primary_metric_key]):.4f} "
        f"best_epoch={best_epoch} best={best_metric:.4f}"
    )


def format_eval_summary(split_name: str, metrics: dict[str, Any]) -> str:
    best_modes = sorted(
        (
            (mode, float(scores["mAP50"]), float(scores["mAP50-95"]))
            for mode, scores in metrics["detection"].items()
        ),
        key=lambda item: (item[2], item[1]),
        reverse=True,
    )[:3]
    best_text = ", ".join(
        f"{mode}:mAP50={map50:.4f}/mAP50-95={map5095:.4f}"
        for mode, map50, map5095 in best_modes
    )
    return (
        f"{split_name}: loss={float(metrics['loss']):.4f} "
        f"acc={float(metrics['candidate_classification']['accuracy']):.4f} "
        f"iou_mae={float(metrics['quality']['pred_iou_mae']):.4f} "
        f"locR50={float(metrics['localization']['recall_iou_0.50']):.4f} "
        f"locR75={float(metrics['localization']['recall_iou_0.75']):.4f} "
        f"best=[{best_text}]"
    )


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    raise SystemExit(main())
