#!/usr/bin/env python3
"""Train a CSV-driven background-aware ROI ranker for Task 2 Stage2O."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    crop_roi_with_padding,
    load_task2_samples_from_split,
)
from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth,
    DetectionPrediction,
    compute_detection_map,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RANKER_CLASS_NAMES = ("background", "normal", "collision")
SCORE_MODES = (
    "roi",
    "bg_suppressed_roi",
    "source_roi",
    "sqrt_source_roi",
    "rank_decay_roi",
    "source_rank_decay_roi",
)


@dataclass(frozen=True)
class CandidateExample:
    split: str
    sample_id: str
    video_id: str
    frame_index: int
    domain: str
    image_path: Path
    gt_class: int
    verifier_label: int
    xyxy: tuple[float, float, float, float]
    gt_xyxy: tuple[float, float, float, float]
    gt_iou: float
    source_conf: float
    source: str
    source_priority: int
    source_rank: int
    row_index: int
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
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("outputs/task2/roi_verifier/convnext_tiny_stage2h_proposals_v1_val_v2_spdc1000_e8/train_candidates.csv"),
        help="Non-leakage training candidate CSV. Rows with verifier_label < 0 are skipped.",
    )
    parser.add_argument(
        "--valid-csv",
        action="append",
        default=None,
        help="Validation CSV as split_name=path. Defaults to Stage2O valid_combined/phantom/animal candidates.",
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/stage2o_ranker"))
    parser.add_argument("--name", default="convnext_tiny_stage2o_from_stage2h_traincsv_e6")

    parser.add_argument("--backend", choices=("auto", "timm", "torchvision"), default="auto")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--pretrained", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=8.0)
    parser.add_argument("--min-crop-size", type=int, default=224)
    parser.add_argument("--max-crop-size", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--balanced-sampler", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--primary-score-mode", choices=SCORE_MODES, default="source_roi")
    parser.add_argument("--primary-valid-split", default="valid_combined")
    parser.add_argument("--primary-map-key", choices=("mAP50", "mAP50-95"), default="mAP50-95")
    parser.add_argument("--max-candidates-per-sample", type=int, default=100)
    parser.add_argument(
        "--candidate-budget-sort",
        choices=("source_order", "raw_conf"),
        default="source_order",
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="If set, load this checkpoint and run validation only.",
    )
    parser.add_argument(
        "--skip-overlap-check",
        default=False,
        action="store_true",
        help=(
            "Skip train/valid sample-id overlap checks. Use only for intentional "
            "eval-only export on train candidates; normal validation should keep "
            "the default leakage guard enabled."
        ),
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

    train_candidates = load_candidate_csv(train_csv, split_override="train", repo_root=REPO_ROOT)
    train_candidates = [candidate for candidate in train_candidates if candidate.verifier_label >= 0]
    train_candidates = apply_candidate_budget(
        train_candidates,
        max_candidates_per_sample=int(args.max_candidates_per_sample),
        sort_mode=str(args.candidate_budget_sort),
    )
    if args.train_limit is not None:
        train_candidates = stratified_candidate_limit(train_candidates, args.train_limit, seed=args.seed)
    if not train_candidates:
        raise ValueError(f"No usable training candidates loaded from {train_csv}")

    valid_candidates_all = {
        name: apply_candidate_budget(
            load_candidate_csv(resolve_path(path), split_override=name, repo_root=REPO_ROOT),
            max_candidates_per_sample=int(args.max_candidates_per_sample),
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
        valid_splits = {name: samples[: args.valid_limit] for name, samples in valid_splits.items()}
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
            "ranker_class_names": RANKER_CLASS_NAMES,
            "score_modes": SCORE_MODES,
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

    model = create_model(args.backend, args.model, pretrained=args.pretrained, num_classes=3).to(device)
    if args.eval_checkpoint is not None:
        checkpoint_path = resolve_path(args.eval_checkpoint)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict, strict=True)
        eval_only_metrics: dict[str, Any] = {}
        for split_name, candidates in valid_candidates.items():
            loader = DataLoader(
                CandidateRoiDataset(
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
            metrics, rows = evaluate_ranker(
                model=model,
                loader=loader,
                candidates=candidates,
                samples=valid_splits[split_name],
                device=device,
                use_amp=bool(args.amp),
            )
            eval_only_metrics[split_name] = metrics
            write_prediction_csv(run_dir / f"{split_name}_eval_prediction_rows.csv", rows)
            print(format_eval_summary(split_name, metrics), flush=True)
        (run_dir / "eval_metrics.json").write_text(
            json.dumps(
                json_ready(
                    {
                        "checkpoint": repo_relative(checkpoint_path, REPO_ROOT),
                        "metrics": eval_only_metrics,
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
        CandidateRoiDataset(
            train_candidates,
            crop_config=crop_config,
            input_size=int(args.input_size),
            training=True,
        ),
        batch_size=int(args.batch_size),
        sampler=make_balanced_sampler(train_candidates) if args.balanced_sampler else None,
        shuffle=False if args.balanced_sampler else True,
        num_workers=int(args.workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.workers) > 0,
    )
    valid_loaders = {
        name: DataLoader(
            CandidateRoiDataset(
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
        for name, candidates in valid_candidates.items()
    }

    best_metric = float("-inf")
    best_epoch = -1
    metrics_path = run_dir / "metrics.csv"
    primary_metric_key = (
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
            use_amp=bool(args.amp),
        )
        row: dict[str, float | int] = {"epoch": epoch, **prefix_metrics("train", train_metrics)}
        detailed_metrics: dict[str, Any] = {}
        prediction_rows: dict[str, list[dict[str, Any]]] = {}
        for split_name, loader in valid_loaders.items():
            metrics, rows = evaluate_ranker(
                model=model,
                loader=loader,
                candidates=valid_candidates[split_name],
                samples=valid_splits[split_name],
                device=device,
                use_amp=bool(args.amp),
            )
            detailed_metrics[split_name] = metrics
            prediction_rows[split_name] = rows
            row.update(flatten_metrics(split_name, metrics))

        append_metrics(metrics_path, row)
        primary = float(row[primary_metric_key])
        is_best = primary > best_metric
        if is_best:
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


def load_candidate_csv(path: Path, *, split_override: str | None, repo_root: Path) -> list[CandidateExample]:
    candidates: list[CandidateExample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, raw in enumerate(reader):
            image_path = resolve_path(Path(first_non_empty(raw, "image_path")))
            gt_xyxy = parse_gt_xyxy(raw)
            xyxy = parse_candidate_xyxy(raw)
            verifier_label = int(float(first_non_empty(raw, "verifier_label", default="-1")))
            sample_id = first_non_empty(raw, "sample_id")
            video_id = first_non_empty(raw, "video_id", default=sample_id.rsplit("_", 1)[0])
            frame_index = int(float(first_non_empty(raw, "frame_index", default=sample_id.rsplit("_", 1)[-1])))
            domain = first_non_empty(raw, "domain", default=domain_from_video_id(video_id))
            candidates.append(
                CandidateExample(
                    split=split_override or first_non_empty(raw, "split", default=path.stem),
                    sample_id=sample_id,
                    video_id=video_id,
                    frame_index=frame_index,
                    domain=domain,
                    image_path=image_path,
                    gt_class=int(float(first_non_empty(raw, "gt_class"))),
                    verifier_label=verifier_label,
                    xyxy=xyxy,
                    gt_xyxy=gt_xyxy,
                    gt_iou=float(first_non_empty(raw, "candidate_iou", "gt_iou", default="0")),
                    source_conf=float(first_non_empty(raw, "source_conf", "det_conf", default="0")),
                    source=first_non_empty(raw, "source", default="proposal"),
                    source_priority=int(float(first_non_empty(raw, "source_priority", default="0"))),
                    source_rank=int(float(first_non_empty(raw, "source_rank", "rank", default="9999"))),
                    row_index=row_index,
                    image_width=int(float(first_non_empty(raw, "image_width"))),
                    image_height=int(float(first_non_empty(raw, "image_height"))),
                )
            )
    return candidates


def apply_candidate_budget(
    candidates: list[CandidateExample],
    *,
    max_candidates_per_sample: int,
    sort_mode: str,
) -> list[CandidateExample]:
    if max_candidates_per_sample <= 0:
        return candidates
    grouped: dict[str, list[CandidateExample]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.sample_id, []).append(candidate)
    selected: list[CandidateExample] = []
    for sample_id in sorted(grouped):
        ordered = sorted(grouped[sample_id], key=lambda item: candidate_budget_key(item, sort_mode))
        selected.extend(ordered[:max_candidates_per_sample])
    selected.sort(key=lambda item: item.row_index)
    return selected


def candidate_budget_key(candidate: CandidateExample, sort_mode: str) -> tuple[Any, ...]:
    if sort_mode == "source_order":
        return (
            candidate.source_priority,
            candidate.source_rank,
            -candidate.source_conf,
            candidate.source,
            candidate.row_index,
        )
    if sort_mode == "raw_conf":
        return (
            -candidate.source_conf,
            candidate.source_priority,
            candidate.source_rank,
            candidate.source,
            candidate.row_index,
        )
    raise ValueError(f"Unknown candidate budget sort mode: {sort_mode}")


def parse_valid_csv_specs(values: list[str] | None) -> dict[str, Path]:
    if values is None:
        base = Path("outputs/task2/stage2o_candidate_pool/stage2o_aprime_default_sources_top50")
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


def parse_gt_xyxy(raw: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(first_non_empty(raw, "gt_x1")),
        float(first_non_empty(raw, "gt_y1")),
        float(first_non_empty(raw, "gt_x2")),
        float(first_non_empty(raw, "gt_y2")),
    )


def parse_candidate_xyxy(raw: dict[str, str]) -> tuple[float, float, float, float]:
    if first_non_empty(raw, "candidate_x1", default="") != "":
        return (
            float(first_non_empty(raw, "candidate_x1")),
            float(first_non_empty(raw, "candidate_y1")),
            float(first_non_empty(raw, "candidate_x2")),
            float(first_non_empty(raw, "candidate_y2")),
        )
    return (
        float(first_non_empty(raw, "x1")),
        float(first_non_empty(raw, "y1")),
        float(first_non_empty(raw, "x2")),
        float(first_non_empty(raw, "y2")),
    )


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
def evaluate_ranker(
    *,
    model: nn.Module,
    loader: DataLoader,
    candidates: list[CandidateExample],
    samples: list[Task2Sample],
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    losses: list[float] = []
    valid_labels: list[int] = []
    valid_preds: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    predictions_by_mode: dict[str, list[DetectionPrediction]] = {mode: [] for mode in SCORE_MODES}

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
            }
            for mode in SCORE_MODES:
                score_normal = candidate_score(candidate, p_bg, p_normal, mode=mode)
                score_collision = candidate_score(candidate, p_bg, p_collision, mode=mode)
                row[f"{mode}_score_normal"] = score_normal
                row[f"{mode}_score_collision"] = score_collision
                predictions_by_mode[mode].append(
                    DetectionPrediction(candidate.sample_id, 0, score_normal, candidate.xyxy)
                )
                predictions_by_mode[mode].append(
                    DetectionPrediction(candidate.sample_id, 1, score_collision, candidate.xyxy)
                )
            prediction_rows.append(row)

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
    return (
        {
            "candidate_classification": multiclass_metrics(valid_labels, valid_preds),
            "localization": summarize_best_localization(samples, candidates),
            "detection": detection_scores,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "rows": len(prediction_rows),
        },
        prediction_rows,
    )


def candidate_score(candidate: CandidateExample, p_bg: float, p_class: float, *, mode: str) -> float:
    source_conf = max(0.0, min(float(candidate.source_conf), 1.0))
    rank_decay = 1.0 / math.sqrt(max(float(candidate.source_rank), 1.0))
    if mode == "roi":
        return p_class
    if mode == "bg_suppressed_roi":
        return p_class * max(0.0, 1.0 - p_bg)
    if mode == "source_roi":
        return source_conf * p_class
    if mode == "sqrt_source_roi":
        return math.sqrt(source_conf) * p_class
    if mode == "rank_decay_roi":
        return rank_decay * p_class
    if mode == "source_rank_decay_roi":
        return source_conf * rank_decay * p_class
    raise ValueError(f"Unknown score mode: {mode}")


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
    per_class_recall: dict[str, float] = {}
    for class_id, class_name in enumerate(RANKER_CLASS_NAMES):
        mask = labels_np == class_id
        per_class_recall[class_name] = (
            float(np.mean(preds_np[mask] == class_id)) if np.any(mask) else float("nan")
        )
    return {
        "accuracy": float(np.mean(labels_np == preds_np)),
        "per_class_recall": per_class_recall,
        "pred_counts": {
            RANKER_CLASS_NAMES[class_id]: int(np.sum(preds_np == class_id))
            for class_id in range(len(RANKER_CLASS_NAMES))
        },
        "label_counts": {
            RANKER_CLASS_NAMES[class_id]: int(np.sum(labels_np == class_id))
            for class_id in range(len(RANKER_CLASS_NAMES))
        },
    }


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        f"{prefix}/loss": float(metrics["loss"]),
        f"{prefix}/candidate_accuracy": float(metrics["candidate_classification"]["accuracy"]),
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


def summarize_candidate_table(candidates: list[CandidateExample]) -> dict[str, Any]:
    return {
        "candidates": len(candidates),
        "label_counts": {
            RANKER_CLASS_NAMES[key] if key >= 0 else "ignore": value
            for key, value in count_by(candidates, lambda item: item.verifier_label).items()
        },
        "source_counts": count_by(candidates, lambda item: item.source),
        "domain_class_counts": count_by(candidates, lambda item: f"{item.domain}:{item.gt_class}"),
        "mean_gt_iou": float(np.mean([item.gt_iou for item in candidates])) if candidates else float("nan"),
    }


def count_by(candidates: Iterable[CandidateExample], key_fn: Any) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for candidate in candidates:
        key = key_fn(candidate)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def stratified_candidate_limit(
    candidates: list[CandidateExample],
    limit: int,
    *,
    seed: int,
) -> list[CandidateExample]:
    if limit <= 0 or len(candidates) <= limit:
        return candidates
    rng = random.Random(seed)
    grouped: dict[tuple[str, int, int], list[CandidateExample]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate.domain, candidate.gt_class, candidate.verifier_label), []).append(candidate)
    selected: list[CandidateExample] = []
    per_group = max(1, math.ceil(limit / max(len(grouped), 1)))
    for key in sorted(grouped):
        group = grouped[key]
        selected.extend(group if len(group) <= per_group else rng.sample(group, per_group))
    if len(selected) > limit:
        selected = rng.sample(selected, limit)
    selected.sort(key=lambda item: (item.video_id, item.frame_index, item.sample_id, item.source, item.source_rank))
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


def infer_width(sample: Task2Sample, candidates: Sequence[CandidateExample]) -> int:
    for candidate in candidates:
        if candidate.sample_id == sample.sample_id:
            return candidate.image_width
    with Image.open(sample.image_path) as image:
        return image.width


def infer_height(sample: Task2Sample, candidates: Sequence[CandidateExample]) -> int:
    for candidate in candidates:
        if candidate.sample_id == sample.sample_id:
            return candidate.image_height
    with Image.open(sample.image_path) as image:
        return image.height


def write_candidate_csv(path: Path, candidates: list[CandidateExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "sample_id",
        "video_id",
        "frame_index",
        "domain",
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
        "source_conf",
        "source",
        "source_priority",
        "source_rank",
        "row_index",
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
                    "video_id": candidate.video_id,
                    "frame_index": candidate.frame_index,
                    "domain": candidate.domain,
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
                    "source_conf": candidate.source_conf,
                    "source": candidate.source,
                    "source_priority": candidate.source_priority,
                    "source_rank": candidate.source_rank,
                    "row_index": candidate.row_index,
                    "image_width": candidate.image_width,
                    "image_height": candidate.image_height,
                }
            )


def write_prediction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def format_epoch_summary(
    epoch: int,
    row: dict[str, float | int],
    primary_metric_key: str,
    best_epoch: int,
    best_metric: float,
) -> str:
    return (
        f"epoch={epoch:03d} "
        f"train_loss={float(row['train/loss']):.4f} "
        f"combined_source_mAP50={float(row['valid_combined/source_roi/mAP50']):.4f} "
        f"combined_source_mAP50_95={float(row['valid_combined/source_roi/mAP50_95']):.4f} "
        f"combined_roi_mAP50={float(row['valid_combined/roi/mAP50']):.4f} "
        f"animal_source_mAP50={float(row['valid_animal/source_roi/mAP50']):.4f} "
        f"phantom_source_mAP50={float(row['valid_phantom/source_roi/mAP50']):.4f} "
        f"primary={primary_metric_key}:{float(row[primary_metric_key]):.4f} "
        f"best_epoch={best_epoch} best={best_metric:.4f}"
    )


def format_eval_summary(split_name: str, metrics: dict[str, Any]) -> str:
    rank_decay = metrics["detection"]["rank_decay_roi"]
    source_rank_decay = metrics["detection"]["source_rank_decay_roi"]
    return (
        f"{split_name} "
        f"rank_decay_mAP50={float(rank_decay['mAP50']):.4f} "
        f"rank_decay_mAP50_95={float(rank_decay['mAP50-95']):.4f} "
        f"source_rank_decay_mAP50={float(source_rank_decay['mAP50']):.4f} "
        f"source_rank_decay_mAP50_95={float(source_rank_decay['mAP50-95']):.4f} "
        f"loc_r50={float(metrics['localization']['recall_iou_0.50']):.4f} "
        f"loc_r75={float(metrics['localization']['recall_iou_0.75']):.4f}"
    )


def first_non_empty(raw: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def domain_from_video_id(video_id: str) -> str:
    lowered = video_id.lower()
    if "animal" in lowered:
        return "animal"
    if "human" in lowered:
        return "human"
    return "phantom"


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
