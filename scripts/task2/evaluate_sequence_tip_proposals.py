#!/usr/bin/env python3
"""Export top-K proposals from the sequence-aware Task 2 tip localizer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, build_task2_index, repo_relative  # noqa: E402
from cathaction.data.task2_roi import class_counts, load_task2_samples_from_split  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402
from scripts.task2.train_sequence_tip_localizer import (  # noqa: E402
    LetterboxMeta,
    MonaiSequenceTipLocalizer,
    SequenceTipDataset,
    SmpFpnSequenceTipLocalizer,
    letterbox_point_to_original,
    sequence_tip_collate,
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class TemplateSize:
    width: float
    height: float


@dataclass(frozen=True)
class Peak:
    x_index: int
    y_index: int
    score: float


@dataclass(frozen=True)
class Candidate:
    xyxy: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    source: str
    source_peak_rank: int
    source_template: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a train_sequence_tip_localizer.py checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/sequence_tip_proposals"))
    parser.add_argument("--name", default="convnext_tip384_topk_templates")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-peaks", type=int, default=20)
    parser.add_argument("--max-proposals", type=int, default=50)
    parser.add_argument("--peak-nms-kernel", type=int, default=17)
    parser.add_argument("--min-peak-score", type=float, default=0.0)
    parser.add_argument("--top-k", default="1,5,10,20,50")
    parser.add_argument(
        "--templates",
        default="20x20,30x30,44x44,52x52,20x58,28x58,30x70,58x20,70x30",
        help="Comma-separated WxH original-pixel template sizes around each predicted center.",
    )
    parser.add_argument("--include-predicted-size", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    checkpoint_path = resolve_path(args.checkpoint)
    run_dir = resolve_path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = parse_top_k_values(args.top_k)
    templates = parse_templates(args.templates)
    device = choose_device(args.device)

    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_args = checkpoint["args"]
    model, model_config = build_model_from_checkpoint(checkpoint_args)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    input_size = int(model_config["input_size"])
    frame_radius = int(model_config["frame_radius"])
    sigma = float(model_config.get("sigma", 3.0))
    all_samples = build_task2_index(data_root)

    split_specs = [
        SplitSpec("valid_combined", resolve_path(args.valid_combined_split)),
        SplitSpec("valid_phantom", resolve_path(args.valid_phantom_split)),
        SplitSpec("valid_animal", resolve_path(args.valid_animal_split)),
    ]
    args_record = {
        **vars(args),
        "repo_root": REPO_ROOT.as_posix(),
        "data_root": repo_relative(data_root, REPO_ROOT),
        "checkpoint": repo_relative(checkpoint_path, REPO_ROOT),
        "run_dir": repo_relative(run_dir, REPO_ROOT),
        "device": str(device),
        "model_config": model_config,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "templates": [f"{item.width:g}x{item.height:g}" for item in templates],
        "top_k_values": top_k_values,
    }
    (run_dir / "args.json").write_text(json.dumps(json_ready(args_record), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(args_record), indent=2), flush=True)

    all_metrics: dict[str, Any] = {}
    for split_spec in split_specs:
        samples = load_task2_samples_from_split(data_root, split_spec.path)
        if args.samples_per_class is not None:
            samples = stratified_sample(samples, samples_per_class=args.samples_per_class, seed=args.seed)
        if args.limit is not None:
            samples = samples[: args.limit]
        print(f"Evaluating {split_spec.name}: {len(samples)} samples", flush=True)

        loader = DataLoader(
            SequenceTipDataset(
                samples,
                all_samples=all_samples,
                input_size=input_size,
                frame_radius=frame_radius,
                sigma=sigma,
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
        rows = evaluate_split(
            split_name=split_spec.name,
            loader=loader,
            model=model,
            device=device,
            args=args,
            templates=templates,
        )
        metrics = summarize_proposal_rows(samples, rows, top_k_values=top_k_values)
        all_metrics[split_spec.name] = metrics
        write_rows_csv(run_dir / f"{split_spec.name}_proposals.csv", rows)
        (run_dir / f"{split_spec.name}_metrics.json").write_text(
            json.dumps(json_ready(metrics), indent=2) + "\n",
            encoding="utf-8",
        )
        print(format_summary(split_spec.name, metrics, top_k_values), flush=True)

    (run_dir / "summary_metrics.json").write_text(
        json.dumps(json_ready(all_metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved sequence-tip proposal evaluation to {run_dir}", flush=True)
    return 0


@torch.no_grad()
def evaluate_split(
    *,
    split_name: str,
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    templates: list[TemplateSize],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    processed = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
            output = model(images)
        heatmap_logits = output["heatmap_logits"].float().cpu()
        size_map = torch.sigmoid(output["size_logits"].float()).cpu()
        offset_map = torch.sigmoid(output["offset_logits"].float()).cpu()

        for batch_index, sample_id in enumerate(batch["sample_ids"]):
            meta: LetterboxMeta = batch["metas"][batch_index]
            gt_box = tuple(float(value) for value in batch["original_box_xyxy"][batch_index].tolist())
            gt_center = tuple(float(value) for value in batch["original_center"][batch_index].tolist())
            candidates = decode_candidates(
                heatmap_logits[batch_index],
                size_map[batch_index],
                offset_map[batch_index],
                meta=meta,
                templates=templates,
                top_peaks=args.top_peaks,
                max_proposals=args.max_proposals,
                peak_nms_kernel=args.peak_nms_kernel,
                min_peak_score=args.min_peak_score,
                include_predicted_size=args.include_predicted_size,
            )
            if candidates:
                for proposal_rank, candidate in enumerate(candidates, start=1):
                    rows.append(
                        {
                            "split": split_name,
                            "sample_id": sample_id,
                            "gt_class": int(batch["gt_classes"][batch_index]),
                            "image_path": f"datasets/collision_detection/images/{sample_id}.jpg",
                            "image_width": meta.original_width,
                            "image_height": meta.original_height,
                            "gt_x1": gt_box[0],
                            "gt_y1": gt_box[1],
                            "gt_x2": gt_box[2],
                            "gt_y2": gt_box[3],
                            "has_proposal": True,
                            "proposal_rank": proposal_rank,
                            "proposal_class": 0,
                            "proposal_conf": candidate.confidence,
                            "proposal_x1": candidate.xyxy[0],
                            "proposal_y1": candidate.xyxy[1],
                            "proposal_x2": candidate.xyxy[2],
                            "proposal_y2": candidate.xyxy[3],
                            "proposal_gt_iou": iou_xyxy(candidate.xyxy, gt_box),  # type: ignore[arg-type]
                            "proposal_center_distance_px": math.hypot(
                                candidate.center[0] - gt_center[0],
                                candidate.center[1] - gt_center[1],
                            ),
                            "proposal_source": candidate.source,
                            "proposal_source_peak_rank": candidate.source_peak_rank,
                            "proposal_template": candidate.source_template,
                        }
                    )
            else:
                rows.append(
                    {
                        "split": split_name,
                        "sample_id": sample_id,
                        "gt_class": int(batch["gt_classes"][batch_index]),
                        "image_path": f"datasets/collision_detection/images/{sample_id}.jpg",
                        "image_width": meta.original_width,
                        "image_height": meta.original_height,
                        "gt_x1": gt_box[0],
                        "gt_y1": gt_box[1],
                        "gt_x2": gt_box[2],
                        "gt_y2": gt_box[3],
                        "has_proposal": False,
                        "proposal_rank": "",
                        "proposal_class": "",
                        "proposal_conf": 0.0,
                        "proposal_x1": "",
                        "proposal_y1": "",
                        "proposal_x2": "",
                        "proposal_y2": "",
                        "proposal_gt_iou": 0.0,
                        "proposal_center_distance_px": float("inf"),
                        "proposal_source": "",
                        "proposal_source_peak_rank": "",
                        "proposal_template": "",
                    }
                )
            processed += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(f"{split_name}: processed {processed}", flush=True)
    return rows


def decode_candidates(
    heatmap_logits: torch.Tensor,
    size_map: torch.Tensor,
    offset_map: torch.Tensor,
    *,
    meta: LetterboxMeta,
    templates: list[TemplateSize],
    top_peaks: int,
    max_proposals: int,
    peak_nms_kernel: int,
    min_peak_score: float,
    include_predicted_size: bool,
) -> list[Candidate]:
    peaks = extract_peaks(
        heatmap_logits,
        top_peaks=top_peaks,
        peak_nms_kernel=peak_nms_kernel,
        min_peak_score=min_peak_score,
    )
    candidates: list[Candidate] = []
    for peak_rank, peak in enumerate(peaks, start=1):
        center = peak_to_original_center(peak, offset_map, meta=meta)
        if include_predicted_size:
            width, height = predicted_size_original(peak, size_map, meta=meta)
            candidates.append(
                make_candidate(
                    center=center,
                    width=width,
                    height=height,
                    confidence=peak.score,
                    source="predicted_size",
                    source_peak_rank=peak_rank,
                    source_template="predicted",
                    meta=meta,
                )
            )
        for template in templates:
            candidates.append(
                make_candidate(
                    center=center,
                    width=template.width,
                    height=template.height,
                    confidence=peak.score,
                    source="template",
                    source_peak_rank=peak_rank,
                    source_template=f"{template.width:g}x{template.height:g}",
                    meta=meta,
                )
            )
        if len(candidates) >= max_proposals:
            break
    return candidates[:max_proposals]


def extract_peaks(
    heatmap_logits: torch.Tensor,
    *,
    top_peaks: int,
    peak_nms_kernel: int,
    min_peak_score: float,
) -> list[Peak]:
    if heatmap_logits.shape[0] != 1:
        raise ValueError(f"expected a one-channel heatmap, got {tuple(heatmap_logits.shape)}")
    if peak_nms_kernel <= 0 or peak_nms_kernel % 2 == 0:
        raise ValueError("--peak-nms-kernel must be a positive odd integer")
    heatmap = torch.sigmoid(heatmap_logits[0])
    pooled = F.max_pool2d(
        heatmap.unsqueeze(0).unsqueeze(0),
        kernel_size=peak_nms_kernel,
        stride=1,
        padding=peak_nms_kernel // 2,
    )[0, 0]
    peak_scores = torch.where(heatmap == pooled, heatmap, torch.zeros_like(heatmap))
    count = min(top_peaks, peak_scores.numel())
    scores, flat_indices = torch.topk(peak_scores.reshape(-1), k=count)
    width = int(peak_scores.shape[1])
    peaks: list[Peak] = []
    for score_tensor, flat_index_tensor in zip(scores, flat_indices):
        score = float(score_tensor.item())
        if score < min_peak_score:
            continue
        flat_index = int(flat_index_tensor.item())
        y_index = flat_index // width
        x_index = flat_index % width
        peaks.append(Peak(x_index=x_index, y_index=y_index, score=score))
    return peaks


def peak_to_original_center(peak: Peak, offset_map: torch.Tensor, *, meta: LetterboxMeta) -> tuple[float, float]:
    out_h, out_w = int(offset_map.shape[1]), int(offset_map.shape[2])
    dx = float(offset_map[0, peak.y_index, peak.x_index].item())
    dy = float(offset_map[1, peak.y_index, peak.x_index].item())
    scale_x = meta.input_size / max(out_w, 1)
    scale_y = meta.input_size / max(out_h, 1)
    cx_letterbox = (float(peak.x_index) + dx) * scale_x
    cy_letterbox = (float(peak.y_index) + dy) * scale_y
    return letterbox_point_to_original((cx_letterbox, cy_letterbox), meta)


def predicted_size_original(peak: Peak, size_map: torch.Tensor, *, meta: LetterboxMeta) -> tuple[float, float]:
    width_letterbox = float(size_map[0, peak.y_index, peak.x_index].item()) * meta.input_size
    height_letterbox = float(size_map[1, peak.y_index, peak.x_index].item()) * meta.input_size
    width_original = clamp(width_letterbox / meta.scale, 1.0, float(meta.original_width))
    height_original = clamp(height_letterbox / meta.scale, 1.0, float(meta.original_height))
    return width_original, height_original


def make_candidate(
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    confidence: float,
    source: str,
    source_peak_rank: int,
    source_template: str,
    meta: LetterboxMeta,
) -> Candidate:
    cx, cy = center
    x1 = clamp(cx - width / 2.0, 0.0, float(meta.original_width))
    y1 = clamp(cy - height / 2.0, 0.0, float(meta.original_height))
    x2 = clamp(cx + width / 2.0, 0.0, float(meta.original_width))
    y2 = clamp(cy + height / 2.0, 0.0, float(meta.original_height))
    return Candidate(
        xyxy=(x1, y1, x2, y2),
        center=center,
        confidence=confidence,
        source=source,
        source_peak_rank=source_peak_rank,
        source_template=source_template,
    )


def summarize_proposal_rows(
    samples: list[Task2Sample],
    rows: list[dict[str, Any]],
    *,
    top_k_values: list[int],
) -> dict[str, Any]:
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    metrics: dict[str, Any] = {
        "samples": len(samples),
        "class_counts": class_counts(samples),
        "rows": len(rows),
        "proposal_count": sum(1 for row in rows if row["has_proposal"]),
        "samples_with_proposal": sum(
            1
            for sample in samples
            if any(row["has_proposal"] for row in rows_by_sample.get(sample.sample_id, []))
        ),
        "topk": {},
        "topk_by_gt_class": {},
    }
    for top_k in top_k_values:
        metrics["topk"][str(top_k)] = summarize_topk(samples, rows_by_sample, top_k=top_k)
    for class_id in sorted(class_counts(samples)):
        class_samples = [sample for sample in samples if sample.class_id == class_id]
        metrics["topk_by_gt_class"][str(class_id)] = {
            str(top_k): summarize_topk(class_samples, rows_by_sample, top_k=top_k)
            for top_k in top_k_values
        }
    return metrics


def summarize_topk(
    samples: list[Task2Sample],
    rows_by_sample: dict[str, list[dict[str, Any]]],
    *,
    top_k: int,
) -> dict[str, float | int]:
    best_ious: list[float] = []
    best_center_distances: list[float] = []
    proposal_counts: list[int] = []
    for sample in samples:
        proposal_rows = [
            row
            for row in rows_by_sample.get(sample.sample_id, [])
            if row["has_proposal"] and int(row["proposal_rank"]) <= top_k
        ]
        proposal_counts.append(len(proposal_rows))
        best_ious.append(max([float(row["proposal_gt_iou"]) for row in proposal_rows], default=0.0))
        best_center_distances.append(
            min([float(row["proposal_center_distance_px"]) for row in proposal_rows], default=float("inf"))
        )

    summary: dict[str, float | int] = {
        "samples": len(samples),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else float("nan"),
        "median_best_iou": float(np.median(best_ious)) if best_ious else float("nan"),
        "mean_best_center_distance_px": (
            float(np.mean(best_center_distances)) if best_center_distances else float("nan")
        ),
        "median_best_center_distance_px": (
            float(np.median(best_center_distances)) if best_center_distances else float("nan")
        ),
        "mean_proposals_per_sample": float(np.mean(proposal_counts)) if proposal_counts else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        summary[f"recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in best_ious])) if best_ious else float("nan")
        )
    for threshold in (5, 10, 20):
        summary[f"center_recall_{threshold}px"] = (
            float(np.mean([distance <= threshold for distance in best_center_distances]))
            if best_center_distances
            else float("nan")
        )
    return summary


def build_model_from_checkpoint(args_record: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_record = dict(args_record.get("model", {}))
    model_type = str(model_record.get("type", args_record.get("model_type", "monai_unet")))
    frame_radius = int(args_record.get("frame_radius", (int(model_record.get("in_channels", 5)) - 1) // 2))
    in_channels = int(model_record.get("in_channels", 2 * frame_radius + 1))
    input_size = int(args_record.get("input_size", 512))
    sigma = float(args_record.get("sigma", 3.0))
    if model_type == "monai_unet":
        channels = tuple(int(item) for item in model_record.get("channels", [16, 32, 64, 128, 256]))
        strides = tuple(int(item) for item in model_record.get("strides", [2, 2, 2, 2]))
        model = MonaiSequenceTipLocalizer(in_channels=in_channels, channels=channels, strides=strides)
        model_config: dict[str, Any] = {
            "type": model_type,
            "in_channels": in_channels,
            "channels": channels,
            "strides": strides,
            "input_size": input_size,
            "frame_radius": frame_radius,
            "sigma": sigma,
        }
    elif model_type == "smp_fpn":
        encoder_name = str(model_record.get("encoder_name", args_record.get("encoder_name", "tu-convnext_tiny")))
        model = SmpFpnSequenceTipLocalizer(
            in_channels=in_channels,
            encoder_name=encoder_name,
            encoder_weights=None,
        )
        model_config = {
            "type": model_type,
            "in_channels": in_channels,
            "encoder_name": encoder_name,
            "encoder_weights": None,
            "input_size": input_size,
            "frame_radius": frame_radius,
            "sigma": sigma,
        }
    else:
        raise ValueError(f"Unsupported checkpoint model type: {model_type}")
    return model, model_config


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_templates(value: str) -> list[TemplateSize]:
    result: list[TemplateSize] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid template size {item!r}; expected WxH")
        width_text, height_text = item.split("x", 1)
        width = float(width_text)
        height = float(height_text)
        if width <= 0 or height <= 0:
            raise ValueError(f"Template sizes must be positive: {item!r}")
        result.append(TemplateSize(width=width, height=height))
    if not result:
        raise ValueError("--templates must contain at least one size")
    return result


def parse_top_k_values(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or result[0] <= 0:
        raise ValueError(f"--top-k must contain positive integers: {value!r}")
    return result


def stratified_sample(samples: list[Task2Sample], *, samples_per_class: int, seed: int) -> list[Task2Sample]:
    rng = random.Random(seed)
    selected: list[Task2Sample] = []
    for class_id in sorted(class_counts(samples)):
        class_samples = [sample for sample in samples if sample.class_id == class_id]
        picked = class_samples if len(class_samples) <= samples_per_class else rng.sample(class_samples, samples_per_class)
        selected.extend(picked)
    selected.sort(key=lambda item: (item.video_id, item.frame_index, item.sample_id))
    return selected


def format_summary(split_name: str, metrics: dict[str, Any], top_k_values: list[int]) -> str:
    parts = [split_name]
    for top_k in top_k_values:
        item = metrics["topk"][str(top_k)]
        parts.append(
            f"top{top_k}:r50={item['recall_iou_0.50']:.4f},"
            f"c20={item['center_recall_20px']:.4f},"
            f"mean_iou={item['mean_best_iou']:.4f}"
        )
    return " ".join(parts)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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
