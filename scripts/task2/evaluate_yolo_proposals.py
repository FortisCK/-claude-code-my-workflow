#!/usr/bin/env python3
"""Evaluate class-agnostic YOLO proposal recall for CATHACTION Task 2."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import class_counts, load_task2_samples_from_split  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class Proposal:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
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
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/task2/yolo_proposal/yolo11s_1024_agnostic_clean_combined_e100/weights/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/yolo_proposal_eval"))
    parser.add_argument("--name", default="yolo11s_1024_agnostic_clean_combined_e100")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-chunk-size", type=int, default=64)
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    weights = resolve_path(args.weights)
    run_dir = resolve_path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    device_arg = "0" if device.type == "cuda" and device.index in (None, 0) else str(device)
    top_k_values = parse_top_k_values(args.top_k)

    args_record = json_ready(
        {
            **vars(args),
            "repo_root": REPO_ROOT.as_posix(),
            "data_root": repo_relative(data_root, REPO_ROOT),
            "weights": repo_relative(weights, REPO_ROOT),
            "run_dir": repo_relative(run_dir, REPO_ROOT),
            "device": str(device),
            "top_k_values": top_k_values,
        }
    )
    (run_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(args_record, indent=2), flush=True)

    from ultralytics import YOLO

    yolo_model = YOLO(weights.as_posix())
    split_specs = [
        SplitSpec("valid_combined", resolve_path(args.valid_combined_split)),
        SplitSpec("valid_phantom", resolve_path(args.valid_phantom_split)),
        SplitSpec("valid_animal", resolve_path(args.valid_animal_split)),
    ]

    all_metrics: dict[str, Any] = {}
    for split_spec in split_specs:
        samples = load_task2_samples_from_split(data_root, split_spec.path)
        if args.limit is not None:
            samples = samples[: args.limit]
        print(f"Evaluating {split_spec.name}: {len(samples)} samples", flush=True)
        rows = evaluate_split(
            samples=samples,
            split_name=split_spec.name,
            yolo_model=yolo_model,
            device_arg=device_arg,
            args=args,
        )
        metrics = summarize_proposals(samples, rows, top_k_values=top_k_values)
        all_metrics[split_spec.name] = metrics
        (run_dir / f"{split_spec.name}_metrics.json").write_text(
            json.dumps(json_ready(metrics), indent=2) + "\n",
            encoding="utf-8",
        )
        write_rows_csv(run_dir / f"{split_spec.name}_proposals.csv", rows)
        print(format_summary(split_spec.name, metrics, top_k_values), flush=True)

    (run_dir / "summary_metrics.json").write_text(
        json.dumps(json_ready(all_metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved proposal evaluation to {run_dir}", flush=True)
    return 0


def evaluate_split(
    *,
    samples: list[Task2Sample],
    split_name: str,
    yolo_model: Any,
    device_arg: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_start in range(0, len(samples), args.source_chunk_size):
        chunk_samples = samples[chunk_start : chunk_start + args.source_chunk_size]
        image_paths = [sample.image_path.as_posix() for sample in chunk_samples]
        results = yolo_model.predict(
            source=image_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            batch=args.batch_size,
            device=device_arg,
            stream=True,
            verbose=False,
        )
        for chunk_offset, (sample, result) in enumerate(zip(chunk_samples, results)):
            index = chunk_start + chunk_offset + 1
            image_height, image_width = int(result.orig_shape[0]), int(result.orig_shape[1])
            gt_box = tuple(float(value) for value in sample.box.xyxy_pixels(image_width, image_height))
            proposals = select_proposals(result)
            if proposals:
                for proposal in proposals:
                    rows.append(
                        {
                            "split": split_name,
                            "sample_id": sample.sample_id,
                            "gt_class": sample.class_id,
                            "image_path": repo_relative(sample.image_path, REPO_ROOT),
                            "image_width": image_width,
                            "image_height": image_height,
                            "gt_x1": gt_box[0],
                            "gt_y1": gt_box[1],
                            "gt_x2": gt_box[2],
                            "gt_y2": gt_box[3],
                            "has_proposal": True,
                            "proposal_rank": proposal.rank,
                            "proposal_class": proposal.class_id,
                            "proposal_conf": proposal.confidence,
                            "proposal_x1": proposal.xyxy[0],
                            "proposal_y1": proposal.xyxy[1],
                            "proposal_x2": proposal.xyxy[2],
                            "proposal_y2": proposal.xyxy[3],
                            "proposal_gt_iou": iou_xyxy(proposal.xyxy, gt_box),  # type: ignore[arg-type]
                        }
                    )
            else:
                rows.append(
                    {
                        "split": split_name,
                        "sample_id": sample.sample_id,
                        "gt_class": sample.class_id,
                        "image_path": repo_relative(sample.image_path, REPO_ROOT),
                        "image_width": image_width,
                        "image_height": image_height,
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
                    }
                )
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"{split_name}: processed {index}/{len(samples)}", flush=True)
    return rows


def select_proposals(result: Any) -> list[Proposal]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    confidences = boxes.conf.detach().cpu().numpy()
    order = np.argsort(-confidences)
    proposals: list[Proposal] = []
    for rank, box_index in enumerate(order.tolist(), start=1):
        xyxy = tuple(float(value) for value in boxes.xyxy[box_index].detach().cpu().numpy().tolist())
        proposals.append(
            Proposal(
                xyxy=xyxy,  # type: ignore[arg-type]
                confidence=float(confidences[box_index]),
                class_id=int(boxes.cls[box_index].detach().cpu().item()),
                rank=rank,
            )
        )
    return proposals


def summarize_proposals(
    samples: list[Task2Sample],
    rows: list[dict[str, Any]],
    *,
    top_k_values: list[int],
) -> dict[str, Any]:
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_sample.setdefault(str(row["sample_id"]), []).append(row)

    result: dict[str, Any] = {
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
        result["topk"][str(top_k)] = summarize_topk(samples, rows_by_sample, top_k=top_k)
    for class_id in sorted(class_counts(samples)):
        class_samples = [sample for sample in samples if sample.class_id == class_id]
        result["topk_by_gt_class"][str(class_id)] = {
            str(top_k): summarize_topk(class_samples, rows_by_sample, top_k=top_k)
            for top_k in top_k_values
        }
    return result


def summarize_topk(
    samples: list[Task2Sample],
    rows_by_sample: dict[str, list[dict[str, Any]]],
    *,
    top_k: int,
) -> dict[str, float | int]:
    best_ious: list[float] = []
    proposal_counts: list[int] = []
    for sample in samples:
        proposal_rows = [
            row
            for row in rows_by_sample.get(sample.sample_id, [])
            if row["has_proposal"] and int(row["proposal_rank"]) <= top_k
        ]
        proposal_counts.append(len(proposal_rows))
        if proposal_rows:
            best_ious.append(max(float(row["proposal_gt_iou"]) for row in proposal_rows))
        else:
            best_ious.append(0.0)

    summary: dict[str, float | int] = {
        "samples": len(samples),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else float("nan"),
        "median_best_iou": float(np.median(best_ious)) if best_ious else float("nan"),
        "mean_proposals_per_sample": float(np.mean(proposal_counts)) if proposal_counts else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        summary[f"recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in best_ious])) if best_ious else float("nan")
        )
    return summary


def parse_top_k_values(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or result[0] <= 0:
        raise ValueError(f"--top-k must contain positive integers: {value!r}")
    return result


def format_summary(split_name: str, metrics: dict[str, Any], top_k_values: list[int]) -> str:
    parts = [split_name]
    for top_k in top_k_values:
        item = metrics["topk"][str(top_k)]
        parts.append(
            f"top{top_k}:r50={item['recall_iou_0.50']:.4f},"
            f"r75={item['recall_iou_0.75']:.4f},mean_iou={item['mean_best_iou']:.4f}"
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
