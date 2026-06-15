#!/usr/bin/env python3
"""Summarize Task2 candidate-pool localization ceiling from saved candidate CSVs.

The Stage2 ranker can only choose among candidate boxes. This diagnostic checks
whether those candidates contain boxes that overlap the ground truth well enough
before any learned scoring is considered.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        required=True,
        help="Path to a *_candidates_used.csv file.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.3, 0.5, 0.75, 0.9, 0.95),
    )
    return parser.parse_args()


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_group(rows: list[dict[str, str]], thresholds: list[float]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)

    best_ious: list[float] = []
    best_sources: Counter[str] = Counter()
    candidates_per_sample: list[int] = []
    missing_candidates = 0
    for sample_id, sample_rows in by_sample.items():
        candidates_per_sample.append(len(sample_rows))
        if not sample_rows:
            missing_candidates += 1
            best_ious.append(0.0)
            continue
        best_row = max(sample_rows, key=lambda row: safe_float(row.get("gt_iou", "0")))
        best_ious.append(safe_float(best_row.get("gt_iou", "0")))
        best_sources[best_row.get("source", "")] += 1

    if not best_ious:
        return {
            "samples": 0,
            "candidate_rows": len(rows),
            "missing_candidates": missing_candidates,
            "mean_best_iou": None,
            "median_best_iou": None,
            "mean_candidates_per_sample": None,
            "recall_at_iou": {str(threshold): None for threshold in thresholds},
            "best_source_counts": {},
        }

    return {
        "samples": len(by_sample),
        "candidate_rows": len(rows),
        "missing_candidates": missing_candidates,
        "mean_best_iou": mean(best_ious),
        "median_best_iou": median(best_ious),
        "mean_candidates_per_sample": mean(candidates_per_sample),
        "recall_at_iou": {
            f"{threshold:.2f}": sum(iou >= threshold for iou in best_ious) / len(best_ious)
            for threshold in thresholds
        },
        "best_source_counts": dict(best_sources.most_common()),
    }


def main() -> int:
    args = parse_args()
    with args.candidate_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    thresholds = list(args.thresholds)
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    groups["all"].extend(rows)
    for row in rows:
        groups[f"domain={row.get('domain', '')}"].append(row)
        groups[f"class={row.get('gt_class', '')}"].append(row)
        groups[f"domain={row.get('domain', '')}|class={row.get('gt_class', '')}"].append(row)

    output = {
        "candidate_csv": args.candidate_csv.as_posix(),
        "thresholds": thresholds,
        "groups": {
            group_name: summarize_group(group_rows, thresholds)
            for group_name, group_rows in sorted(groups.items())
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
