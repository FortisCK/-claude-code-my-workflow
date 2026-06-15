#!/usr/bin/env python3
"""Apply fixed hard-mask morphology post-processing for CATHACTION Task 1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from scripts.task1.search_morphology_postprocess import (
    _apply_candidate,
    _load_prediction,
    _repo_relative,
    _resolve,
    _safe_filename,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-id", default="remove_small_min32")
    parser.add_argument("--label-1-op", default="none")
    parser.add_argument("--label-1-radius", type=int, default=0)
    parser.add_argument("--label-2-op", default="none")
    parser.add_argument("--label-2-radius", type=int, default=0)
    parser.add_argument("--overlap-priority", default="original")
    parser.add_argument("--min-component-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    predictions_csv = _resolve(args.predictions_csv, repo_root)
    output_dir = _resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(predictions_csv)
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]

    candidate = {
        "candidate_id": str(args.candidate_id),
        "label_1_op": str(args.label_1_op),
        "label_1_radius": int(args.label_1_radius),
        "label_2_op": str(args.label_2_op),
        "label_2_radius": int(args.label_2_radius),
        "overlap_priority": str(args.overlap_priority),
        "min_component_size": int(args.min_component_size),
    }
    output_manifest = _write_processed_predictions(
        rows,
        candidate=candidate,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    summary = {
        "predictions_csv": _repo_relative(predictions_csv, repo_root),
        "output_dir": _repo_relative(output_dir, repo_root),
        "prediction_manifest": _repo_relative(output_manifest, repo_root),
        "predictions": len(rows),
        "candidate": candidate,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _write_processed_predictions(
    rows: list[dict[str, str]],
    *,
    candidate: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
) -> Path:
    processed_rows: list[dict[str, str]] = []
    for row in tqdm(rows, desc="apply-morphology", leave=False):
        prediction = _load_prediction(_resolve(row["prediction_path"], repo_root))
        processed = _apply_candidate(prediction, candidate)
        prediction_path = output_dir / f"{_safe_filename(str(row['sample_id']))}.png"
        Image.fromarray(processed).save(prediction_path)
        processed_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "collection": str(row.get("collection", "")),
                "domain": str(row.get("domain", "")),
                "prediction_path": _repo_relative(prediction_path, repo_root),
                "height": str(processed.shape[0]),
                "width": str(processed.shape[1]),
            }
        )
    manifest_path = output_dir / "predictions.csv"
    _write_csv(manifest_path, processed_rows)
    return manifest_path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
