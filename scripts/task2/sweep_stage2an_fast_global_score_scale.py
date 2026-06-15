#!/usr/bin/env python3
"""Fast global class/domain scale sweep for clean Task2 predictions."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import ap_101_point, iou_xyxy
from scripts.task2.export_clean_predictions import write_rows
from scripts.task2.export_stage2ab_domain_policy_predictions import DEFAULT_SPLITS


DOMAINS = ("phantom", "animal")
CLASSES = (0, 1)
IOU_THRESHOLDS = tuple(float(round(value, 2)) for value in np.arange(0.5, 0.96, 0.05))


@dataclass(frozen=True)
class CleanSplitCache:
    rows: list[dict[str, str]]
    sample_ids: list[str]
    domains: np.ndarray
    class_ids: np.ndarray
    gt_classes: dict[str, int]
    gt_boxes: dict[str, tuple[float, float, float, float]]
    base_scores: np.ndarray
    ious_to_gt: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--optimize-split", default="valid_combined")
    parser.add_argument("--scale", action="append", type=float, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_gt(run_dir: Path, split: str) -> tuple[dict[str, int], dict[str, tuple[float, float, float, float]]]:
    gt_classes: dict[str, int] = {}
    gt_boxes: dict[str, tuple[float, float, float, float]] = {}
    for row in read_csv(run_dir / f"{split}_candidates_used.csv"):
        sample_id = row["sample_id"]
        if sample_id in gt_classes:
            continue
        gt_classes[sample_id] = int(row["gt_class"])
        gt_boxes[sample_id] = (
            float(row["gt_x1"]),
            float(row["gt_y1"]),
            float(row["gt_x2"]),
            float(row["gt_y2"]),
        )
    return gt_classes, gt_boxes


def load_cache(run_dir: Path, prediction_dir: Path, split: str) -> CleanSplitCache:
    rows = read_csv(prediction_dir / f"{split}_domain_policy_predictions.csv")
    gt_classes, gt_boxes = load_gt(run_dir, split)
    sample_ids: list[str] = []
    domains: list[str] = []
    class_ids: list[int] = []
    base_scores: list[float] = []
    ious: list[float] = []
    for row in rows:
        sample_id = row["sample_id"]
        class_id = int(row["class_id"])
        domain = row.get("domain", "phantom")
        if domain not in DOMAINS:
            domain = "animal"
        pred_box = (
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
        )
        sample_ids.append(sample_id)
        domains.append(domain)
        class_ids.append(class_id)
        base_scores.append(float(row["score"]))
        ious.append(iou_xyxy(pred_box, gt_boxes[sample_id]) if gt_classes[sample_id] == class_id else 0.0)
    return CleanSplitCache(
        rows=rows,
        sample_ids=sample_ids,
        domains=np.asarray(domains, dtype=object),
        class_ids=np.asarray(class_ids, dtype=np.int64),
        gt_classes=gt_classes,
        gt_boxes=gt_boxes,
        base_scores=np.asarray(base_scores, dtype=np.float64),
        ious_to_gt=np.asarray(ious, dtype=np.float64),
    )


def scale_key(class_id: int, domain: str) -> str:
    return f"class{class_id}_{domain}"


def scaled_scores(cache: CleanSplitCache, policy: dict[str, float]) -> np.ndarray:
    scores = cache.base_scores.copy()
    for class_id in CLASSES:
        for domain in DOMAINS:
            mask = (cache.class_ids == class_id) & (cache.domains == domain)
            scores[mask] *= float(policy.get(scale_key(class_id, domain), 1.0))
    return scores


def ap_for_class(cache: CleanSplitCache, scores: np.ndarray, *, class_id: int, threshold: float) -> float:
    gt_sample_ids = {sid for sid, cls in cache.gt_classes.items() if cls == class_id}
    num_gt = len(gt_sample_ids)
    if num_gt == 0:
        return float("nan")
    indices = np.where(cache.class_ids == class_id)[0]
    indices = indices[np.argsort(-scores[indices], kind="mergesort")]
    matched: set[str] = set()
    tp: list[float] = []
    fp: list[float] = []
    for index in indices:
        sample_id = cache.sample_ids[int(index)]
        if cache.gt_classes[sample_id] == class_id and cache.ious_to_gt[int(index)] >= threshold and sample_id not in matched:
            tp.append(1.0)
            fp.append(0.0)
            matched.add(sample_id)
        else:
            tp.append(0.0)
            fp.append(1.0)
    tp_arr = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp_arr = np.cumsum(np.asarray(fp, dtype=np.float64))
    recalls = tp_arr / max(num_gt, 1)
    precisions = tp_arr / np.maximum(tp_arr + fp_arr, 1e-12)
    return float(ap_101_point(recalls, precisions))


def metrics_for_policy(cache: CleanSplitCache, policy: dict[str, float]) -> dict[str, Any]:
    scores = scaled_scores(cache, policy)
    class_metrics: dict[str, Any] = {}
    ap50_values: list[float] = []
    ap5095_values: list[float] = []
    for class_id in CLASSES:
        ap_values = [
            ap_for_class(cache, scores, class_id=class_id, threshold=threshold)
            for threshold in IOU_THRESHOLDS
        ]
        class_metrics[str(class_id)] = {
            "ap50": float(ap_values[0]),
            "ap50_95": float(np.nanmean(ap_values)),
            "ap_by_iou": {f"{threshold:.2f}": float(ap) for threshold, ap in zip(IOU_THRESHOLDS, ap_values)},
            "num_gt": int(sum(cls == class_id for cls in cache.gt_classes.values())),
            "num_predictions": int(np.sum(cache.class_ids == class_id)),
        }
        ap50_values.append(float(ap_values[0]))
        ap5095_values.append(float(np.nanmean(ap_values)))
    return {
        "mAP50": float(np.nanmean(ap50_values)),
        "mAP50-95": float(np.nanmean(ap5095_values)),
        "classes": class_metrics,
    }


def policy_grid(scales: list[float]) -> list[dict[str, float]]:
    keys = [scale_key(class_id, domain) for class_id in CLASSES for domain in DOMAINS]
    return [dict(zip(keys, values)) for values in itertools.product(scales, repeat=len(keys))]


def output_rows(cache: CleanSplitCache, policy: dict[str, float]) -> list[dict[str, Any]]:
    scores = scaled_scores(cache, policy)
    result: list[dict[str, Any]] = []
    for row, score in zip(cache.rows, scores):
        class_id = int(row["class_id"])
        domain = row.get("domain", "phantom")
        if domain not in DOMAINS:
            domain = "animal"
        factor = float(policy.get(scale_key(class_id, domain), 1.0))
        result.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row.get("video_id", ""),
                "frame_index": row.get("frame_index", ""),
                "domain": domain,
                "class_id": class_id,
                "score": float(score),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "source": row.get("source", ""),
                "source_rank": row.get("source_rank", ""),
                "score_mode": row.get("score_mode", ""),
                "policy_name": f"{row.get('policy_name', '')}_scale_{factor:.3f}",
            }
        )
    return result


def main() -> int:
    args = parse_args()
    scales = [float(value) for value in (args.scale or [0.5, 0.75, 1.0, 1.25, 1.5])]
    if 1.0 not in scales:
        scales.append(1.0)
    scales = sorted(scales)
    identity = {scale_key(class_id, domain): 1.0 for class_id in CLASSES for domain in DOMAINS}
    optimize_cache = load_cache(args.run_dir, args.prediction_dir, args.optimize_split)

    best_policy: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for policy in policy_grid(scales):
        metrics = metrics_for_policy(optimize_cache, policy)
        rows.append(
            {
                **policy,
                "mAP50": float(metrics["mAP50"]),
                "mAP50-95": float(metrics["mAP50-95"]),
                "class0_ap50": float(metrics["classes"]["0"]["ap50"]),
                "class0_ap50_95": float(metrics["classes"]["0"]["ap50_95"]),
                "class1_ap50": float(metrics["classes"]["1"]["ap50"]),
                "class1_ap50_95": float(metrics["classes"]["1"]["ap50_95"]),
            }
        )
        if best_metrics is None or (metrics["mAP50-95"], metrics["mAP50"]) > (
            best_metrics["mAP50-95"],
            best_metrics["mAP50"],
        ):
            best_metrics = metrics
            best_policy = policy
    if best_policy is None or best_metrics is None:
        raise ValueError("No policy evaluated")

    metrics_by_split: dict[str, Any] = {}
    for split in args.splits:
        cache = optimize_cache if split == args.optimize_split else load_cache(args.run_dir, args.prediction_dir, split)
        metrics_by_split[split] = metrics_for_policy(cache, best_policy)
        if args.output_dir is not None:
            write_rows(args.output_dir / f"{split}_domain_policy_predictions.csv", output_rows(cache, best_policy))

    output = {
        "artifact_type": "task2_stage2an_fast_global_score_scale",
        "run_dir": args.run_dir.as_posix(),
        "prediction_dir": args.prediction_dir.as_posix(),
        "optimize_split": args.optimize_split,
        "scale_values": scales,
        "identity_policy": identity,
        "identity_metrics": metrics_for_policy(optimize_cache, identity),
        "best_policy": best_policy,
        "best_metrics": best_metrics,
        "metrics": metrics_by_split,
        "top_policies": sorted(rows, key=lambda row: (row["mAP50-95"], row["mAP50"]), reverse=True)[:25],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
