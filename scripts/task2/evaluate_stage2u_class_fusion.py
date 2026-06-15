#!/usr/bin/env python3
"""Evaluate class-aware fusion of saved Stage2U eval runs.

This script combines prediction rows from two Stage2U evaluation directories.
It is useful when one candidate source/score mode is better for class 0 and a
different source/score mode is better for class 1.
"""

from __future__ import annotations

import argparse
import csv
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

from cathaction.metrics.detection import (
    DetectionGroundTruth,
    DetectionPrediction,
    compute_detection_map,
)


CLASS_SUFFIX = {0: "normal", 1: "collision"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class0-run-dir", type=Path, required=True)
    parser.add_argument("--class1-run-dir", type=Path, required=True)
    parser.add_argument("--class0-score-mode", required=True)
    parser.add_argument("--class1-score-mode", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("valid_combined", "valid_phantom", "valid_animal"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Optional directory for {split}_fused_predictions.csv files.",
    )
    return parser.parse_args()


def load_run_rows(run_dir: Path, split: str) -> tuple[list[DetectionGroundTruth], list[tuple[dict[str, str], dict[str, str]]]]:
    candidates_path = run_dir / f"{split}_candidates_used.csv"
    predictions_path = run_dir / f"{split}_eval_prediction_rows.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    with candidates_path.open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    with predictions_path.open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(candidates) != len(predictions):
        raise ValueError(
            f"{split}: row count mismatch for {run_dir}: "
            f"{len(candidates)} candidates vs {len(predictions)} predictions"
        )

    ground_truths: list[DetectionGroundTruth] = []
    seen: set[str] = set()
    for candidate in candidates:
        sample_id = candidate["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        ground_truths.append(
            DetectionGroundTruth(
                sample_id,
                int(candidate["gt_class"]),
                (
                    float(candidate["gt_x1"]),
                    float(candidate["gt_y1"]),
                    float(candidate["gt_x2"]),
                    float(candidate["gt_y2"]),
                ),
            )
        )
    return ground_truths, list(zip(candidates, predictions))


def predictions_for_class(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    score_mode: str,
) -> list[DetectionPrediction]:
    suffix = CLASS_SUFFIX[class_id]
    score_key = f"{score_mode}_score_{suffix}"
    result: list[DetectionPrediction] = []
    for candidate, prediction in rows:
        if score_key not in prediction:
            raise KeyError(f"Missing score column {score_key}")
        result.append(
            DetectionPrediction(
                candidate["sample_id"],
                class_id,
                float(prediction[score_key]),
                (
                    float(candidate["x1"]),
                    float(candidate["y1"]),
                    float(candidate["x2"]),
                    float(candidate["y2"]),
                ),
            )
        )
    return result


def prediction_rows_for_class(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    score_mode: str,
    policy_name: str,
) -> list[dict[str, Any]]:
    suffix = CLASS_SUFFIX[class_id]
    score_key = f"{score_mode}_score_{suffix}"
    result: list[dict[str, Any]] = []
    for candidate, prediction in rows:
        if score_key not in prediction:
            raise KeyError(f"Missing score column {score_key}")
        result.append(
            {
                "split": candidate["split"],
                "sample_id": candidate["sample_id"],
                "video_id": candidate.get("video_id", ""),
                "frame_index": candidate.get("frame_index", ""),
                "domain": candidate.get("domain", ""),
                "class_id": class_id,
                "score": float(prediction[score_key]),
                "x1": float(candidate["x1"]),
                "y1": float(candidate["y1"]),
                "x2": float(candidate["x2"]),
                "y2": float(candidate["y2"]),
                "source": candidate.get("source", ""),
                "source_rank": candidate.get("source_rank", ""),
                "score_mode": score_mode,
                "policy_name": policy_name,
                "gt_class": int(candidate["gt_class"]),
                "gt_iou": float(candidate["gt_iou"]),
            }
        )
    return result


def candidate_summary(rows: list[tuple[dict[str, str], dict[str, str]]]) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    sample_ids: set[str] = set()
    for candidate, _ in rows:
        sample_ids.add(candidate["sample_id"])
        source_counts[candidate["source"]] = source_counts.get(candidate["source"], 0) + 1
    return {
        "candidates": len(rows),
        "samples": len(sample_ids),
        "source_counts": source_counts,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output: dict[str, Any] = {
        "class0_run_dir": args.class0_run_dir.as_posix(),
        "class1_run_dir": args.class1_run_dir.as_posix(),
        "class0_score_mode": args.class0_score_mode,
        "class1_score_mode": args.class1_score_mode,
        "metrics": {},
    }

    for split in args.splits:
        gt0, rows0 = load_run_rows(args.class0_run_dir, split)
        gt1, rows1 = load_run_rows(args.class1_run_dir, split)
        gt0_key = [(gt.sample_id, gt.class_id, gt.box_xyxy) for gt in gt0]
        gt1_key = [(gt.sample_id, gt.class_id, gt.box_xyxy) for gt in gt1]
        if gt0_key != gt1_key:
            raise ValueError(f"{split}: ground-truth rows do not match between run dirs")

        predictions = [
            *predictions_for_class(rows0, class_id=0, score_mode=args.class0_score_mode),
            *predictions_for_class(rows1, class_id=1, score_mode=args.class1_score_mode),
        ]
        fused_rows = [
            *prediction_rows_for_class(
                rows0,
                class_id=0,
                score_mode=args.class0_score_mode,
                policy_name="class0",
            ),
            *prediction_rows_for_class(
                rows1,
                class_id=1,
                score_mode=args.class1_score_mode,
                policy_name="class1",
            ),
        ]
        if args.prediction_dir is not None:
            write_csv(args.prediction_dir / f"{split}_fused_predictions.csv", fused_rows)
        output["metrics"][split] = {
            "detection": compute_detection_map(gt0, predictions, class_ids=(0, 1)),
            "class0_source": candidate_summary(rows0),
            "class1_source": candidate_summary(rows1),
            "prediction_rows": len(fused_rows),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
