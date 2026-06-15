#!/usr/bin/env python3
"""Export Stage2V class-aware clean predictions.

This is the single-entry wrapper for the current best Task2 validation policy:

- class 0 predictions are taken from one Stage2U eval run/score mode;
- class 1 predictions are taken from another Stage2U eval run/score mode;
- validation-only fields are omitted from the exported CSVs;
- if ground-truth columns are present in the source runs, mAP is also computed.

The output CSV is an internal clean prediction artifact. A final CATHACTION
official formatter should map this file to the challenge result schema once the
official hidden-test instructions are fixed.
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

from cathaction.metrics.detection import compute_detection_map
from scripts.task2.evaluate_stage2u_class_fusion import (
    load_run_rows,
    prediction_rows_for_class,
    predictions_for_class,
)
from scripts.task2.export_clean_predictions import CLEAN_FIELDS, apply_topk, clean_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class0-run-dir", type=Path, required=True)
    parser.add_argument("--class0-score-mode", required=True)
    parser.add_argument("--class1-run-dir", type=Path, required=True)
    parser.add_argument("--class1-score-mode", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("valid_combined", "valid_phantom", "valid_animal"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk-per-sample-class", type=int, default=0)
    return parser.parse_args()


def write_clean_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLEAN_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "class0_run_dir": args.class0_run_dir.as_posix(),
        "class0_score_mode": args.class0_score_mode,
        "class1_run_dir": args.class1_run_dir.as_posix(),
        "class1_score_mode": args.class1_score_mode,
        "topk_per_sample_class": int(args.topk_per_sample_class),
        "splits": {},
    }

    for split in args.splits:
        ground_truths0, rows0 = load_run_rows(args.class0_run_dir, split)
        ground_truths1, rows1 = load_run_rows(args.class1_run_dir, split)
        gt_key0 = [(gt.sample_id, gt.class_id, gt.box_xyxy) for gt in ground_truths0]
        gt_key1 = [(gt.sample_id, gt.class_id, gt.box_xyxy) for gt in ground_truths1]
        if gt_key0 != gt_key1:
            raise ValueError(f"{split}: ground-truth rows do not match between run dirs")

        internal_rows = [
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
        clean_rows = [clean_row({key: str(value) for key, value in row.items()}) for row in internal_rows]
        clean_rows = apply_topk(clean_rows, int(args.topk_per_sample_class))
        write_clean_csv(args.output_dir / f"{split}_predictions.csv", clean_rows)

        predictions = [
            *predictions_for_class(rows0, class_id=0, score_mode=args.class0_score_mode),
            *predictions_for_class(rows1, class_id=1, score_mode=args.class1_score_mode),
        ]
        if int(args.topk_per_sample_class) > 0:
            keep_keys = {
                (
                    str(row["sample_id"]),
                    int(row["class_id"]),
                    float(row["score"]),
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                )
                for row in clean_rows
            }
            predictions = [
                prediction
                for prediction in predictions
                if (
                    prediction.sample_id,
                    prediction.class_id,
                    float(prediction.score),
                    float(prediction.box_xyxy[0]),
                    float(prediction.box_xyxy[1]),
                    float(prediction.box_xyxy[2]),
                    float(prediction.box_xyxy[3]),
                )
                in keep_keys
            ]

        metrics["splits"][split] = {
            "prediction_rows": len(clean_rows),
            "detection": compute_detection_map(ground_truths0, predictions, class_ids=(0, 1)),
        }

    metrics_path = args.output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
