#!/usr/bin/env python3
"""Diagnose why Task 1 multiclass Dice is low for an existing prediction set."""

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


LABELS = (1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--tolerance-radius", type=int, action="append", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--resize-target",
        action="store_true",
        help="Nearest-resize target masks to prediction shape when shapes differ.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    eval_manifest = _resolve(args.eval_manifest, repo_root)
    predictions_csv = _resolve(args.predictions_csv, repo_root)
    output_dir = _resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    radii = sorted(set(args.tolerance_radius or [1, 2, 3]))
    eval_rows = {row["sample_id"]: row for row in _read_rows(eval_manifest)}
    prediction_rows = _read_rows(predictions_csv)
    per_sample: list[dict[str, Any]] = []
    global_counts = _new_global_counts()
    missing_sample_ids: list[str] = []

    for prediction_row in prediction_rows:
        sample_id = prediction_row["sample_id"]
        eval_row = eval_rows.get(sample_id)
        if eval_row is None:
            missing_sample_ids.append(sample_id)
            continue

        target = load_task1_mask(
            _resolve(eval_row["mask_path"], repo_root),
            eval_row["mask_encoding"],
        ).astype(np.uint8)
        prediction = _load_prediction(_resolve(prediction_row["prediction_path"], repo_root))
        if prediction.shape != target.shape:
            if not args.resize_target:
                raise ValueError(
                    f"Shape mismatch for {sample_id}: prediction={prediction.shape}, "
                    f"target={target.shape}. Use --resize-target only for non-original-space outputs."
                )
            target = _resize_mask_nearest(target, prediction.shape)

        row = _diagnose_sample(
            prediction=prediction,
            target=target,
            radii=radii,
        )
        row.update(
            {
                "sample_id": sample_id,
                "collection": eval_row.get("collection", ""),
                "domain": eval_row.get("domain", ""),
                "image_path": eval_row.get("image_path", ""),
                "mask_path": eval_row.get("mask_path", ""),
                "prediction_path": prediction_row.get("prediction_path", ""),
                "target_label_1_px": int(np.count_nonzero(target == 1)),
                "target_label_2_px": int(np.count_nonzero(target == 2)),
                "pred_label_1_px": int(np.count_nonzero(prediction == 1)),
                "pred_label_2_px": int(np.count_nonzero(prediction == 2)),
            }
        )
        per_sample.append(row)
        _update_global_counts(global_counts, prediction, target)

    if not per_sample:
        raise ValueError("No prediction rows matched the evaluation manifest.")

    per_sample_path = output_dir / "per_sample_diagnostics.csv"
    _write_csv(per_sample_path, per_sample)
    top_cases = _write_top_case_tables(
        output_dir / "top_cases",
        per_sample,
        top_k=int(args.top_k),
        radii=radii,
    )
    summary = {
        "eval_manifest": _repo_relative(eval_manifest, repo_root),
        "predictions_csv": _repo_relative(predictions_csv, repo_root),
        "output_dir": _repo_relative(output_dir, repo_root),
        "samples": len(per_sample),
        "missing_sample_ids": missing_sample_ids,
        "tolerance_radii": radii,
        "per_sample_csv": _repo_relative(per_sample_path, repo_root),
        "aggregate": _aggregate(per_sample, radii=radii),
        "global_pixel_aggregate": _global_aggregate(global_counts),
        "top_cases": top_cases,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_compact_console_summary(summary), indent=2))
    return 0


def _diagnose_sample(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    radii: list[int],
) -> dict[str, float]:
    exact_per_label = {label: _dice_for_label(prediction, target, label) for label in LABELS}
    exact_dice = _mean(exact_per_label.values())

    swapped = _swap_labels(prediction)
    swap_per_label = {label: _dice_for_label(swapped, target, label) for label in LABELS}
    swap_dice = _mean(swap_per_label.values())
    oracle_swap_dice = max(exact_dice, swap_dice)

    binary_dice = _dice_masks(prediction > 0, target > 0)
    result: dict[str, float] = {
        "exact_multiclass_dice": exact_dice,
        "exact_label_1_dice": exact_per_label[1],
        "exact_label_2_dice": exact_per_label[2],
        "binary_foreground_dice": binary_dice,
        "binary_minus_multiclass": binary_dice - exact_dice,
        "class_swap_dice": swap_dice,
        "class_swap_gain": swap_dice - exact_dice,
        "oracle_swap_dice": oracle_swap_dice,
        "oracle_swap_gain": oracle_swap_dice - exact_dice,
    }

    for radius in radii:
        tolerance_per_label = {
            label: _tolerant_dice_masks(prediction == label, target == label, radius=radius)
            for label in LABELS
        }
        tolerance_dice = _mean(tolerance_per_label.values())
        binary_tolerance = _tolerant_dice_masks(prediction > 0, target > 0, radius=radius)
        result[f"tolerance_r{radius}_multiclass_dice"] = tolerance_dice
        result[f"tolerance_r{radius}_label_1_dice"] = tolerance_per_label[1]
        result[f"tolerance_r{radius}_label_2_dice"] = tolerance_per_label[2]
        result[f"tolerance_r{radius}_gain"] = tolerance_dice - exact_dice
        result[f"tolerance_r{radius}_binary_dice"] = binary_tolerance
        result[f"tolerance_r{radius}_binary_gain"] = binary_tolerance - binary_dice
    return result


def _dice_for_label(prediction: np.ndarray, target: np.ndarray, label: int) -> float:
    return _dice_masks(prediction == label, target == label)


def _dice_masks(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=bool)
    tgt = np.asarray(target, dtype=bool)
    denominator = int(pred.sum() + tgt.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(pred, tgt).sum())
    return float(2.0 * intersection / denominator)


def _tolerant_dice_masks(prediction: np.ndarray, target: np.ndarray, *, radius: int) -> float:
    pred = np.asarray(prediction, dtype=bool)
    tgt = np.asarray(target, dtype=bool)
    pred_count = int(pred.sum())
    tgt_count = int(tgt.sum())
    if pred_count == 0 and tgt_count == 0:
        return 1.0
    if pred_count == 0 or tgt_count == 0:
        return 0.0

    dilated_target = _dilate_binary(tgt, radius)
    dilated_prediction = _dilate_binary(pred, radius)
    precision = float(np.logical_and(pred, dilated_target).sum() / pred_count)
    recall = float(np.logical_and(tgt, dilated_prediction).sum() / tgt_count)
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return result


def _swap_labels(prediction: np.ndarray) -> np.ndarray:
    swapped = np.asarray(prediction, dtype=np.uint8).copy()
    one = swapped == 1
    two = swapped == 2
    swapped[one] = 2
    swapped[two] = 1
    return swapped


def _new_global_counts() -> dict[str, dict[int, int]]:
    return {
        "intersection": {label: 0 for label in LABELS},
        "prediction": {label: 0 for label in LABELS},
        "target": {label: 0 for label in LABELS},
        "binary_intersection": {1: 0},
        "binary_prediction": {1: 0},
        "binary_target": {1: 0},
    }


def _update_global_counts(
    counts: dict[str, dict[int, int]],
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    for label in LABELS:
        pred = prediction == label
        tgt = target == label
        counts["intersection"][label] += int(np.logical_and(pred, tgt).sum())
        counts["prediction"][label] += int(pred.sum())
        counts["target"][label] += int(tgt.sum())
    pred_fg = prediction > 0
    tgt_fg = target > 0
    counts["binary_intersection"][1] += int(np.logical_and(pred_fg, tgt_fg).sum())
    counts["binary_prediction"][1] += int(pred_fg.sum())
    counts["binary_target"][1] += int(tgt_fg.sum())


def _global_aggregate(counts: dict[str, dict[int, int]]) -> dict[str, Any]:
    per_label = {
        f"label_{label}": _dice_from_counts(
            counts["intersection"][label],
            counts["prediction"][label],
            counts["target"][label],
        )
        for label in LABELS
    }
    binary = _dice_from_counts(
        counts["binary_intersection"][1],
        counts["binary_prediction"][1],
        counts["binary_target"][1],
    )
    return {
        "multiclass_macro_dice": _mean(per_label.values()),
        "per_class_dice": per_label,
        "binary_foreground_dice": binary,
        "counts": counts,
    }


def _dice_from_counts(intersection: int, prediction: int, target: int) -> float:
    denominator = int(prediction + target)
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def _aggregate(rows: list[dict[str, Any]], *, radii: list[int]) -> dict[str, Any]:
    metric_keys = [
        "exact_multiclass_dice",
        "exact_label_1_dice",
        "exact_label_2_dice",
        "binary_foreground_dice",
        "binary_minus_multiclass",
        "class_swap_dice",
        "class_swap_gain",
        "oracle_swap_dice",
        "oracle_swap_gain",
    ]
    for radius in radii:
        metric_keys.extend(
            [
                f"tolerance_r{radius}_multiclass_dice",
                f"tolerance_r{radius}_label_1_dice",
                f"tolerance_r{radius}_label_2_dice",
                f"tolerance_r{radius}_gain",
                f"tolerance_r{radius}_binary_dice",
                f"tolerance_r{radius}_binary_gain",
            ]
        )
    return {
        "overall": _mean_metrics(rows, metric_keys),
        "by_domain": _grouped_metrics(rows, "domain", metric_keys),
        "by_collection": _grouped_metrics(rows, "collection", metric_keys),
    }


def _grouped_metrics(
    rows: list[dict[str, Any]],
    group_key: str,
    metric_keys: list[str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    return {
        group: {"num_samples": len(group_rows), **_mean_metrics(group_rows, metric_keys)}
        for group, group_rows in sorted(grouped.items())
    }


def _mean_metrics(rows: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, float]:
    return {key: _mean(float(row[key]) for row in rows) for key in metric_keys}


def _write_top_case_tables(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    radii: list[int],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict[str, Any]]] = {
        "worst_exact": sorted(rows, key=lambda row: float(row["exact_multiclass_dice"]))[:top_k],
        "worst_binary": sorted(rows, key=lambda row: float(row["binary_foreground_dice"]))[:top_k],
        "largest_binary_gap": sorted(
            rows,
            key=lambda row: float(row["binary_minus_multiclass"]),
            reverse=True,
        )[:top_k],
        "largest_swap_gain": sorted(
            rows,
            key=lambda row: float(row["oracle_swap_gain"]),
            reverse=True,
        )[:top_k],
    }
    if radii:
        radius = 2 if 2 in radii else radii[-1]
        buckets[f"largest_tolerance_r{radius}_gain"] = sorted(
            rows,
            key=lambda row: float(row[f"tolerance_r{radius}_gain"]),
            reverse=True,
        )[:top_k]
    for domain in sorted(set(str(row["domain"]) for row in rows)):
        buckets[f"worst_{domain}"] = sorted(
            [row for row in rows if str(row["domain"]) == domain],
            key=lambda row: float(row["exact_multiclass_dice"]),
        )[:top_k]

    paths: dict[str, str] = {}
    for name, bucket_rows in buckets.items():
        path = output_dir / f"{name}.csv"
        _write_csv(path, bucket_rows)
        paths[name] = path.as_posix()
    return paths


def _compact_console_summary(summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary["aggregate"]["overall"]
    by_domain = summary["aggregate"]["by_domain"]
    keys = [
        "exact_multiclass_dice",
        "binary_foreground_dice",
        "binary_minus_multiclass",
        "oracle_swap_dice",
        "oracle_swap_gain",
        "tolerance_r1_multiclass_dice",
        "tolerance_r2_multiclass_dice",
        "tolerance_r3_multiclass_dice",
    ]
    return {
        "samples": summary["samples"],
        "overall": {key: overall.get(key) for key in keys if key in overall},
        "by_domain": {
            domain: {key: metrics.get(key) for key in keys if key in metrics}
            for domain, metrics in by_domain.items()
        },
        "global_pixel_aggregate": summary["global_pixel_aggregate"],
        "summary_json": str(Path(summary["output_dir"]) / "summary.json"),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_prediction(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def _resolve(path: Path | str, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _mean(values: Any) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(np.mean(values))


if __name__ == "__main__":
    raise SystemExit(main())

