#!/usr/bin/env python3
"""Filter aligned Task2 candidate/prediction CSV rows by proposal source."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-candidates", type=Path, required=True)
    parser.add_argument("--input-predictions", type=Path, default=None)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument(
        "--keep-source",
        action="append",
        required=True,
        help="Proposal source to keep. Repeat for multiple sources.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = row.get("source", "")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def assert_candidate_prediction_aligned(
    candidates: list[dict[str, str]],
    predictions: list[dict[str, str]],
) -> None:
    if len(candidates) != len(predictions):
        raise ValueError(
            f"candidate/prediction row mismatch: {len(candidates)} vs {len(predictions)}"
        )
    for index, (candidate, prediction) in enumerate(zip(candidates, predictions)):
        for key in ("sample_id", "source", "source_rank"):
            left = candidate.get(key, "")
            right = prediction.get(key, "")
            if key == "source_rank" and left and right:
                left = str(int(float(left)))
                right = str(int(float(right)))
            if left != right:
                raise ValueError(f"row {index}: {key} mismatch: {left!r} vs {right!r}")


def filter_rows(rows: list[dict[str, str]], keep_sources: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("source", "") in keep_sources]


def run_filter(args: argparse.Namespace) -> dict[str, Any]:
    keep_sources = set(args.keep_source)
    candidate_fields, candidate_rows = read_rows(args.input_candidates)
    filtered_candidates = filter_rows(candidate_rows, keep_sources)
    result: dict[str, Any] = {
        "keep_sources": sorted(keep_sources),
        "input_candidates": args.input_candidates.as_posix(),
        "output_candidates": args.output_candidates.as_posix(),
        "candidate_rows_before": len(candidate_rows),
        "candidate_rows_after": len(filtered_candidates),
        "candidate_source_counts_before": source_counts(candidate_rows),
        "candidate_source_counts_after": source_counts(filtered_candidates),
    }
    write_rows(args.output_candidates, candidate_fields, filtered_candidates)

    if args.input_predictions is not None:
        if args.output_predictions is None:
            raise ValueError("--output-predictions is required with --input-predictions")
        prediction_fields, prediction_rows = read_rows(args.input_predictions)
        assert_candidate_prediction_aligned(candidate_rows, prediction_rows)
        filtered_predictions = filter_rows(prediction_rows, keep_sources)
        if len(filtered_predictions) != len(filtered_candidates):
            raise ValueError(
                f"filtered row mismatch: {len(filtered_candidates)} candidates vs "
                f"{len(filtered_predictions)} predictions"
            )
        write_rows(args.output_predictions, prediction_fields, filtered_predictions)
        result.update(
            {
                "input_predictions": args.input_predictions.as_posix(),
                "output_predictions": args.output_predictions.as_posix(),
                "prediction_rows_before": len(prediction_rows),
                "prediction_rows_after": len(filtered_predictions),
                "prediction_source_counts_before": source_counts(prediction_rows),
                "prediction_source_counts_after": source_counts(filtered_predictions),
            }
        )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run_filter(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

