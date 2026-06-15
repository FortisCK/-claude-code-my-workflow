#!/usr/bin/env python3
"""Export clean Task2 prediction CSVs from internal fused prediction rows.

The class-aware fusion artifacts include validation-only diagnostic fields such
as ``gt_class`` and ``gt_iou``. This script removes those fields and optionally
applies per-sample/per-class top-k pruning. The output is still an internal CSV,
not the final official CATHACTION hidden-test format.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


CLEAN_FIELDS = (
    "sample_id",
    "video_id",
    "frame_index",
    "domain",
    "class_id",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "source",
    "source_rank",
    "score_mode",
    "policy_name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--topk-per-sample-class",
        type=int,
        default=0,
        help="Keep only top-k predictions per sample_id/class_id. 0 keeps all rows.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "video_id": row.get("video_id", ""),
        "frame_index": row.get("frame_index", ""),
        "domain": row.get("domain", ""),
        "class_id": int(row["class_id"]),
        "score": float(row["score"]),
        "x1": float(row["x1"]),
        "y1": float(row["y1"]),
        "x2": float(row["x2"]),
        "y2": float(row["y2"]),
        "source": row.get("source", ""),
        "source_rank": row.get("source_rank", ""),
        "score_mode": row.get("score_mode", ""),
        "policy_name": row.get("policy_name", ""),
    }


def apply_topk(rows: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
    if topk <= 0:
        return rows
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sample_id"]), int(row["class_id"]))].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        result.extend(
            sorted(grouped[key], key=lambda row: float(row["score"]), reverse=True)[:topk]
        )
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLEAN_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = [clean_row(row) for row in read_rows(args.input_csv)]
    rows = apply_topk(rows, int(args.topk_per_sample_class))
    write_rows(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
