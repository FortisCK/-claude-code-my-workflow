#!/usr/bin/env python3
"""Evaluate saved Task 1 predictions with MSLNet-style binary metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cathaction.data.task1 import load_task1_mask
from cathaction.metrics.mslnet_style import (
    MslnetStyleMetrics,
    aggregate_mslnet_style_metrics,
    mslnet_style_metrics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--per-sample-csv", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--tolerance-radius", type=int, action="append", default=None)
    parser.add_argument(
        "--resize-target",
        action="store_true",
        help="Nearest-resize target masks to prediction shape when shapes differ.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    eval_manifest = _resolve(args.eval_manifest, repo_root)
    predictions_csv = _resolve(args.predictions_csv, repo_root)
    output_json = _resolve(args.output_json, repo_root)
    per_sample_csv = _resolve(args.per_sample_csv, repo_root) if args.per_sample_csv else None
    tolerance_radii = sorted(set(args.tolerance_radius or [0, 1, 2, 3, 4]))

    eval_rows = {row["sample_id"]: row for row in _read_rows(eval_manifest)}
    prediction_rows = _read_rows(predictions_csv)
    per_sample_rows: list[dict[str, Any]] = []
    metric_rows: list[MslnetStyleMetrics] = []
    metrics_by_domain: dict[str, list[MslnetStyleMetrics]] = defaultdict(list)
    metrics_by_collection: dict[str, list[MslnetStyleMetrics]] = defaultdict(list)
    missing_sample_ids: list[str] = []

    for prediction_row in prediction_rows:
        sample_id = prediction_row["sample_id"]
        eval_row = eval_rows.get(sample_id)
        if eval_row is None:
            missing_sample_ids.append(sample_id)
            continue

        prediction = _load_prediction(_resolve(prediction_row["prediction_path"], repo_root))
        target = load_task1_mask(
            _resolve(eval_row["mask_path"], repo_root),
            eval_row["mask_encoding"],
        )
        if prediction.shape != target.shape:
            if not args.resize_target:
                raise ValueError(
                    f"Shape mismatch for {sample_id}: prediction={prediction.shape}, "
                    f"target={target.shape}. Use --resize-target only for non-original-space outputs."
                )
            target = _resize_mask_nearest(target, prediction.shape)

        metrics = mslnet_style_metrics(
            prediction,
            target,
            tolerance_radii=tolerance_radii,
        )
        metric_rows.append(metrics)
        domain = eval_row.get("domain", "")
        collection = eval_row.get("collection", "")
        metrics_by_domain[domain].append(metrics)
        metrics_by_collection[collection].append(metrics)
        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "collection": collection,
                "domain": domain,
                "image_path": eval_row.get("image_path", ""),
                "mask_path": eval_row.get("mask_path", ""),
                "prediction_path": prediction_row.get("prediction_path", ""),
                **metrics.to_flat_dict(),
            }
        )

    if not metric_rows:
        raise ValueError("No prediction rows matched the evaluation manifest.")

    if per_sample_csv is not None:
        per_sample_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(per_sample_csv, per_sample_rows)

    output = {
        "metric_style": "MSLNet-style binary foreground segmentation",
        "metric_definition": {
            "foreground": "all non-background labels collapsed to foreground",
            "precision": "detected foreground pixels within radius of an annotated foreground pixel",
            "recall": "annotated foreground pixels within radius of a detected foreground pixel",
            "f1": "harmonic mean of tolerance precision and recall",
            "ahd": "average of prediction-to-target and target-to-prediction nearest-pixel distances",
        },
        "eval_manifest": _repo_relative(eval_manifest, repo_root),
        "predictions_csv": _repo_relative(predictions_csv, repo_root),
        "output_json": _repo_relative(output_json, repo_root),
        "per_sample_csv": _repo_relative(per_sample_csv, repo_root) if per_sample_csv else None,
        "samples": len(metric_rows),
        "missing_sample_ids": missing_sample_ids,
        "tolerance_radii": tolerance_radii,
        "sample_mean": aggregate_mslnet_style_metrics(
            metric_rows,
            tolerance_radii=tolerance_radii,
        ),
        "sample_mean_by_domain": {
            domain: aggregate_mslnet_style_metrics(rows, tolerance_radii=tolerance_radii)
            for domain, rows in sorted(metrics_by_domain.items())
        },
        "sample_mean_by_collection": {
            collection: aggregate_mslnet_style_metrics(rows, tolerance_radii=tolerance_radii)
            for collection, rows in sorted(metrics_by_collection.items())
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_console_summary(output), indent=2))
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_prediction(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8)


def _resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    return np.asarray(image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST), dtype=np.uint8)


def _resolve(path: str | Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _repo_relative(path: Path | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _console_summary(output: dict[str, Any]) -> dict[str, Any]:
    sample_mean = output["sample_mean"]
    return {
        "samples": output["samples"],
        "dice": sample_mean["dice"],
        "iou": sample_mean["iou"],
        "ahd": sample_mean["ahd"],
        "f1_r0": sample_mean["f1_r0"],
        "f1_r1": sample_mean["f1_r1"],
        "f1_r2": sample_mean["f1_r2"],
        "f1_r3": sample_mean["f1_r3"],
        "f1_r4": sample_mean["f1_r4"],
        "by_domain": output["sample_mean_by_domain"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
