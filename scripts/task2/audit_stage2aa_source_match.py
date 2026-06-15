#!/usr/bin/env python3
"""Audit whether Task2 selector train/valid artifacts are source-matched."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CsvSummary:
    path: str
    rows: int
    samples: int
    sources: dict[str, int]
    fields: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--valid-candidates", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, default=None)
    parser.add_argument("--valid-predictions", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="Report source mismatch without returning a failing exit code.",
    )
    return parser.parse_args()


def read_summary(path: Path) -> CsvSummary:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = 0
    samples: set[str] = set()
    sources: dict[str, int] = {}
    fields: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            sample_id = row.get("sample_id", "")
            if sample_id:
                samples.add(sample_id)
            source = row.get("source", "")
            sources[source] = sources.get(source, 0) + 1
    return CsvSummary(
        path=path.as_posix(),
        rows=rows,
        samples=len(samples),
        sources=dict(sorted(sources.items())),
        fields=fields,
    )


def compare_prediction_alignment(
    candidate_path: Path,
    prediction_path: Path | None,
) -> dict[str, Any] | None:
    if prediction_path is None:
        return None
    candidate_summary = read_summary(candidate_path)
    prediction_summary = read_summary(prediction_path)
    source_match = set(candidate_summary.sources) == set(prediction_summary.sources)
    return {
        "candidate_rows": candidate_summary.rows,
        "prediction_rows": prediction_summary.rows,
        "row_count_match": candidate_summary.rows == prediction_summary.rows,
        "candidate_sources": candidate_summary.sources,
        "prediction_sources": prediction_summary.sources,
        "source_set_match": source_match,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    train = read_summary(args.train_candidates)
    valid = read_summary(args.valid_candidates)
    train_sources = set(train.sources)
    valid_sources = set(valid.sources)
    return {
        "train_candidates": asdict(train),
        "valid_candidates": asdict(valid),
        "source_set_match": train_sources == valid_sources,
        "train_only_sources": sorted(train_sources - valid_sources),
        "valid_only_sources": sorted(valid_sources - train_sources),
        "common_sources": sorted(train_sources & valid_sources),
        "train_prediction_alignment": compare_prediction_alignment(
            args.train_candidates,
            args.train_predictions,
        ),
        "valid_prediction_alignment": compare_prediction_alignment(
            args.valid_candidates,
            args.valid_predictions,
        ),
    }


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report["source_set_match"]) and not args.allow_source_mismatch:
        return 2
    for key in ("train_prediction_alignment", "valid_prediction_alignment"):
        alignment = report.get(key)
        if alignment is None:
            continue
        if not bool(alignment["row_count_match"]) or not bool(alignment["source_set_match"]):
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

