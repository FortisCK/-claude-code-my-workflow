#!/usr/bin/env python3
"""Sweep global class/domain score scales on clean Task2 predictions."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import DetectionGroundTruth, DetectionPrediction, compute_detection_map
from scripts.task2.export_clean_predictions import CLEAN_FIELDS, write_rows
from scripts.task2.export_stage2ab_domain_policy_predictions import DEFAULT_SPLITS


DOMAINS = ("phantom", "animal")
CLASSES = (0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--optimize-split", default="valid_combined")
    parser.add_argument(
        "--scale",
        action="append",
        type=float,
        default=None,
        help="Candidate multiplicative scale. Defaults to a conservative grid.",
    )
    parser.add_argument(
        "--optimize-key",
        choices=("mAP50", "mAP50-95", "balanced"),
        default="mAP50-95",
        help="balanced maximizes mAP50-95 with mAP50 as tie breaker.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ground_truths_from_candidates(path: Path) -> list[DetectionGroundTruth]:
    rows = read_csv(path)
    seen: set[str] = set()
    result: list[DetectionGroundTruth] = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        result.append(
            DetectionGroundTruth(
                sample_id=sample_id,
                class_id=int(row["gt_class"]),
                box_xyxy=(
                    float(row["gt_x1"]),
                    float(row["gt_y1"]),
                    float(row["gt_x2"]),
                    float(row["gt_y2"]),
                ),
            )
        )
    return result


def scale_key(class_id: int, domain: str) -> str:
    return f"class{class_id}_{domain}"


def scaled_rows(rows: list[dict[str, str]], scales: dict[str, float]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        class_id = int(row["class_id"])
        domain = row.get("domain", "phantom")
        if domain not in DOMAINS:
            domain = "animal"
        factor = float(scales.get(scale_key(class_id, domain), 1.0))
        result.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row.get("video_id", ""),
                "frame_index": row.get("frame_index", ""),
                "domain": domain,
                "class_id": class_id,
                "score": max(0.0, float(row["score"]) * factor),
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


def predictions_from_rows(rows: list[dict[str, Any]]) -> list[DetectionPrediction]:
    return [
        DetectionPrediction(
            sample_id=str(row["sample_id"]),
            class_id=int(row["class_id"]),
            score=float(row["score"]),
            box_xyxy=(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ),
        )
        for row in rows
    ]


def evaluate_split(
    *,
    run_dir: Path,
    prediction_dir: Path,
    split: str,
    scales: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ground_truths = ground_truths_from_candidates(run_dir / f"{split}_candidates_used.csv")
    rows = scaled_rows(read_csv(prediction_dir / f"{split}_domain_policy_predictions.csv"), scales)
    metrics = compute_detection_map(ground_truths, predictions_from_rows(rows), class_ids=CLASSES)
    return metrics, rows


def metric_objective(metrics: dict[str, Any], optimize_key: str) -> tuple[float, float]:
    if optimize_key == "mAP50":
        return (float(metrics["mAP50"]), float(metrics["mAP50-95"]))
    if optimize_key == "mAP50-95":
        return (float(metrics["mAP50-95"]), float(metrics["mAP50"]))
    return (float(metrics["mAP50-95"]), float(metrics["mAP50"]))


def policy_grid(scales: list[float]) -> list[dict[str, float]]:
    keys = [scale_key(class_id, domain) for class_id in CLASSES for domain in DOMAINS]
    policies: list[dict[str, float]] = []
    # Full grid is still small with the default 5-value scale list.
    for values in itertools.product(scales, repeat=len(keys)):
        policies.append(dict(zip(keys, values)))
    return policies


def main() -> int:
    args = parse_args()
    scale_values = [float(value) for value in (args.scale or [0.5, 0.75, 1.0, 1.25, 1.5])]
    if 1.0 not in scale_values:
        scale_values.append(1.0)
        scale_values = sorted(scale_values)
    identity = {scale_key(class_id, domain): 1.0 for class_id in CLASSES for domain in DOMAINS}

    candidates = policy_grid(scale_values)
    best_policy: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    summary_rows: list[dict[str, Any]] = []
    for policy in candidates:
        metrics, _rows = evaluate_split(
            run_dir=args.run_dir,
            prediction_dir=args.prediction_dir,
            split=args.optimize_split,
            scales=policy,
        )
        row = {
            **policy,
            "mAP50": float(metrics["mAP50"]),
            "mAP50-95": float(metrics["mAP50-95"]),
            "class0_ap50": float(metrics["classes"]["0"]["ap50"]),
            "class0_ap50_95": float(metrics["classes"]["0"]["ap50_95"]),
            "class1_ap50": float(metrics["classes"]["1"]["ap50"]),
            "class1_ap50_95": float(metrics["classes"]["1"]["ap50_95"]),
        }
        summary_rows.append(row)
        if best_metrics is None or metric_objective(metrics, args.optimize_key) > metric_objective(best_metrics, args.optimize_key):
            best_policy = policy
            best_metrics = metrics

    if best_policy is None or best_metrics is None:
        raise ValueError("No policy evaluated")

    identity_metrics, _identity_rows = evaluate_split(
        run_dir=args.run_dir,
        prediction_dir=args.prediction_dir,
        split=args.optimize_split,
        scales=identity,
    )
    metrics_by_split: dict[str, Any] = {}
    output_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in args.splits:
        metrics, rows = evaluate_split(
            run_dir=args.run_dir,
            prediction_dir=args.prediction_dir,
            split=split,
            scales=best_policy,
        )
        metrics_by_split[split] = metrics
        output_rows_by_split[split] = rows
        if args.output_dir is not None:
            write_rows(args.output_dir / f"{split}_domain_policy_predictions.csv", rows)

    output = {
        "artifact_type": "task2_stage2an_global_score_scale",
        "run_dir": args.run_dir.as_posix(),
        "prediction_dir": args.prediction_dir.as_posix(),
        "optimize_split": args.optimize_split,
        "optimize_key": args.optimize_key,
        "scale_values": scale_values,
        "identity_policy": identity,
        "identity_metrics": identity_metrics,
        "best_policy": best_policy,
        "best_metrics": best_metrics,
        "metrics": metrics_by_split,
        "top_policies": sorted(
            summary_rows,
            key=lambda row: (
                row["mAP50-95"] if args.optimize_key != "mAP50" else row["mAP50"],
                row["mAP50"],
            ),
            reverse=True,
        )[:25],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
