#!/usr/bin/env python3
"""Search hard-mask morphology post-processing for CATHACTION Task 1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from cathaction.data.task1 import load_task1_mask


LABELS = (1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--preset",
        choices=("compact", "expanded"),
        default="compact",
        help="Candidate grid size. Use expanded only after compact shows promise.",
    )
    parser.add_argument("--write-best-predictions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    eval_manifest = _resolve(args.eval_manifest, repo_root)
    predictions_csv = _resolve(args.predictions_csv, repo_root)
    output_dir = _resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = {row["sample_id"]: row for row in _read_rows(eval_manifest)}
    prediction_rows = _read_rows(predictions_csv)
    if args.max_samples is not None:
        prediction_rows = prediction_rows[: int(args.max_samples)]

    samples = _load_samples(
        prediction_rows,
        eval_rows=eval_rows,
        repo_root=repo_root,
    )
    candidates = _candidate_grid(str(args.preset))
    results: list[dict[str, Any]] = []
    for candidate in tqdm(candidates, desc="morphology-grid", leave=False):
        results.append(_evaluate_candidate(candidate, samples))

    baseline = next(row for row in results if row["candidate_id"] == "baseline")
    for row in results:
        row["delta_dice"] = float(row["dice"] - baseline["dice"])
        row["delta_label_1_dice"] = float(row["label_1_dice"] - baseline["label_1_dice"])
        row["delta_label_2_dice"] = float(row["label_2_dice"] - baseline["label_2_dice"])
        row["delta_animal_dice"] = float(row.get("animal_dice", np.nan) - baseline.get("animal_dice", np.nan))
        row["delta_phantom_dice"] = float(row.get("phantom_dice", np.nan) - baseline.get("phantom_dice", np.nan))

    results = sorted(results, key=lambda row: float(row["dice"]), reverse=True)
    results_csv = output_dir / "candidate_metrics.csv"
    _write_csv(results_csv, results)
    summary = {
        "eval_manifest": _repo_relative(eval_manifest, repo_root),
        "predictions_csv": _repo_relative(predictions_csv, repo_root),
        "output_dir": _repo_relative(output_dir, repo_root),
        "samples": len(samples),
        "candidates": len(candidates),
        "preset": str(args.preset),
        "baseline": _public_candidate_row(baseline),
        "best": _public_candidate_row(results[0]),
        "top": [_public_candidate_row(row) for row in results[: int(args.top_k)]],
        "candidate_metrics_csv": _repo_relative(results_csv, repo_root),
    }
    if args.write_best_predictions:
        best_dir = output_dir / "best_predictions"
        best_manifest = _write_candidate_predictions(
            best_dir,
            samples,
            results[0],
            repo_root=repo_root,
        )
        summary["best_prediction_manifest"] = _repo_relative(best_manifest, repo_root)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _load_samples(
    prediction_rows: list[dict[str, str]],
    *,
    eval_rows: dict[str, dict[str, str]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in tqdm(prediction_rows, desc="load-samples", leave=False):
        sample_id = row["sample_id"]
        eval_row = eval_rows.get(sample_id)
        if eval_row is None:
            continue
        prediction = _load_prediction(_resolve(row["prediction_path"], repo_root))
        target = load_task1_mask(
            _resolve(eval_row["mask_path"], repo_root),
            eval_row["mask_encoding"],
        ).astype(np.uint8)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {sample_id}: prediction={prediction.shape}, target={target.shape}"
            )
        samples.append(
            {
                "sample_id": sample_id,
                "collection": eval_row.get("collection", ""),
                "domain": eval_row.get("domain", ""),
                "prediction": prediction,
                "target": target,
            }
        )
    if not samples:
        raise ValueError("No prediction rows matched the evaluation manifest.")
    return samples


def _candidate_grid(preset: str = "compact") -> list[dict[str, Any]]:
    if preset == "compact":
        return _compact_candidate_grid()
    if preset != "expanded":
        raise ValueError(f"Unsupported candidate preset: {preset}")
    return _expanded_candidate_grid()


def _compact_candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        _candidate("baseline", "none", "none", 0, 0, "original"),
    ]
    single_label_variants = [
        ("dilate", 1),
        ("erode", 1),
        ("open", 1),
        ("close", 1),
        ("close", 2),
    ]
    for label in [1, 2]:
        for op, radius in single_label_variants:
            op1 = op if label == 1 else "none"
            op2 = op if label == 2 else "none"
            radius1 = radius if label == 1 else 0
            radius2 = radius if label == 2 else 0
            candidates.append(
                _candidate(
                    f"label{label}_{op}{radius}",
                    op1,
                    op2,
                    radius1,
                    radius2,
                    "original",
                )
            )
    for op, radius in single_label_variants:
        candidates.append(
            _candidate(
                f"both_{op}{radius}",
                op,
                op,
                radius,
                radius,
                "original",
            )
        )
    for min_size in [4, 8, 16, 32]:
        candidates.append(
            _candidate(
                f"remove_small_min{min_size}",
                "none",
                "none",
                0,
                0,
                "original",
                min_component_size=min_size,
            )
        )
    for op, radius in [("close", 1), ("close", 2), ("open", 1)]:
        for min_size in [4, 8, 16]:
            candidates.append(
                _candidate(
                    f"both_{op}{radius}_min{min_size}",
                    op,
                    op,
                    radius,
                    radius,
                    "original",
                    min_component_size=min_size,
                )
            )
    return _deduplicate_candidates(candidates)


def _expanded_candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        _candidate("baseline", "none", "none", 0, 0, "original"),
    ]
    label_ops = ["none", "dilate", "erode", "open", "close"]
    radii_for_op = {
        "none": [0],
        "dilate": [1],
        "erode": [1],
        "open": [1],
        "close": [1, 2],
    }
    min_component_sizes = [0, 4, 8, 16]
    priorities = ["original", "label1", "label2"]

    for op1 in label_ops:
        for op2 in label_ops:
            for radius1 in radii_for_op[op1]:
                for radius2 in radii_for_op[op2]:
                    if op1 == "none" and op2 == "none":
                        continue
                    for min_size in min_component_sizes:
                        for priority in priorities:
                            candidate_id = (
                                f"l1_{op1}{radius1}_l2_{op2}{radius2}"
                                f"_min{min_size}_p{priority}"
                            )
                            candidates.append(
                                _candidate(
                                    candidate_id,
                                    op1,
                                    op2,
                                    radius1,
                                    radius2,
                                    priority,
                                    min_component_size=min_size,
                                )
                            )

    # Small, symmetric variants are useful and cheaper to interpret.
    for op in ["dilate", "erode", "open", "close"]:
        for radius in ([1, 2] if op == "close" else [1]):
            for min_size in min_component_sizes:
                candidate_id = f"both_{op}{radius}_min{min_size}_poriginal"
                candidates.append(
                    _candidate(
                        candidate_id,
                        op,
                        op,
                        radius,
                        radius,
                        "original",
                        min_component_size=min_size,
                    )
                )
    return _deduplicate_candidates(candidates)


def _candidate(
    candidate_id: str,
    label_1_op: str,
    label_2_op: str,
    label_1_radius: int,
    label_2_radius: int,
    overlap_priority: str,
    *,
    min_component_size: int = 0,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "label_1_op": label_1_op,
        "label_2_op": label_2_op,
        "label_1_radius": int(label_1_radius),
        "label_2_radius": int(label_2_radius),
        "overlap_priority": overlap_priority,
        "min_component_size": int(min_component_size),
    }


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (
            candidate["label_1_op"],
            candidate["label_2_op"],
            candidate["label_1_radius"],
            candidate["label_2_radius"],
            candidate["overlap_priority"],
            candidate["min_component_size"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _evaluate_candidate(
    candidate: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    scores: list[float] = []
    label_scores: dict[int, list[float]] = {label: [] for label in LABELS}
    domain_scores: dict[str, list[float]] = defaultdict(list)
    collection_scores: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        prediction = _apply_candidate(sample["prediction"], candidate)
        target = sample["target"]
        per_label = {label: _dice_for_label(prediction, target, label) for label in LABELS}
        score = float(np.mean(list(per_label.values())))
        scores.append(score)
        for label, value in per_label.items():
            label_scores[label].append(value)
        domain_scores[str(sample["domain"])].append(score)
        collection_scores[str(sample["collection"])].append(score)

    result = dict(candidate)
    result.update(
        {
            "dice": _mean(scores),
            "label_1_dice": _mean(label_scores[1]),
            "label_2_dice": _mean(label_scores[2]),
            "num_samples": len(samples),
        }
    )
    for domain, values in sorted(domain_scores.items()):
        result[f"{domain}_dice"] = _mean(values)
        result[f"{domain}_samples"] = len(values)
    for collection, values in sorted(collection_scores.items()):
        result[f"{collection}_dice"] = _mean(values)
    return result


def _apply_candidate(prediction: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    label_1 = _apply_label_op(
        prediction == 1,
        str(candidate["label_1_op"]),
        int(candidate["label_1_radius"]),
    )
    label_2 = _apply_label_op(
        prediction == 2,
        str(candidate["label_2_op"]),
        int(candidate["label_2_radius"]),
    )
    min_component_size = int(candidate.get("min_component_size", 0))
    if min_component_size > 0:
        label_1 = _remove_small_components(label_1, min_component_size)
        label_2 = _remove_small_components(label_2, min_component_size)
    return _resolve_label_masks(
        label_1,
        label_2,
        prediction,
        priority=str(candidate.get("overlap_priority", "original")),
    )


def _apply_label_op(mask: np.ndarray, op: str, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if op == "none" or radius <= 0:
        return mask.copy()
    if op == "dilate":
        return _dilate(mask, radius)
    if op == "erode":
        return _erode(mask, radius)
    if op == "open":
        return _dilate(_erode(mask, radius), radius)
    if op == "close":
        return _erode(_dilate(mask, radius), radius)
    raise ValueError(f"Unsupported morphology op: {op}")


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
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


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(radius))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[:-2, :-2]
            & padded[:-2, 1:-1]
            & padded[:-2, 2:]
            & padded[1:-1, :-2]
            & padded[1:-1, 1:-1]
            & padded[1:-1, 2:]
            & padded[2:, :-2]
            & padded[2:, 1:-1]
            & padded[2:, 2:]
        )
    return result


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    labeled, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labeled]


def _resolve_label_masks(
    label_1: np.ndarray,
    label_2: np.ndarray,
    original: np.ndarray,
    *,
    priority: str,
) -> np.ndarray:
    label_1 = np.asarray(label_1, dtype=bool)
    label_2 = np.asarray(label_2, dtype=bool)
    output = np.zeros(original.shape, dtype=np.uint8)
    output[label_1 & ~label_2] = 1
    output[label_2 & ~label_1] = 2
    overlap = label_1 & label_2
    if not np.any(overlap):
        return output

    if priority == "label1":
        output[overlap] = 1
    elif priority == "label2":
        output[overlap] = 2
    elif priority == "original":
        original_labels = original[overlap]
        overlap_values = np.zeros(int(np.count_nonzero(overlap)), dtype=np.uint8)
        overlap_values[original_labels == 1] = 1
        overlap_values[original_labels == 2] = 2
        overlap_values[overlap_values == 0] = 2
        output[overlap] = overlap_values
    else:
        raise ValueError(f"Unsupported overlap priority: {priority}")
    return output


def _dice_for_label(prediction: np.ndarray, target: np.ndarray, label: int) -> float:
    pred = prediction == label
    tgt = target == label
    denom = int(pred.sum() + tgt.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, tgt).sum() / denom)


def _write_candidate_predictions(
    output_dir: Path,
    samples: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for sample in tqdm(samples, desc="write-best", leave=False):
        processed = _apply_candidate(sample["prediction"], candidate)
        prediction_path = output_dir / f"{_safe_filename(str(sample['sample_id']))}.png"
        Image.fromarray(processed).save(prediction_path)
        rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                "collection": str(sample["collection"]),
                "domain": str(sample["domain"]),
                "prediction_path": _repo_relative(prediction_path, repo_root),
                "height": str(processed.shape[0]),
                "width": str(processed.shape[1]),
            }
        )
    manifest_path = output_dir / "predictions.csv"
    _write_csv(manifest_path, rows)
    return manifest_path


def _public_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate_id",
        "label_1_op",
        "label_1_radius",
        "label_2_op",
        "label_2_radius",
        "overlap_priority",
        "min_component_size",
        "dice",
        "delta_dice",
        "label_1_dice",
        "delta_label_1_dice",
        "label_2_dice",
        "delta_label_2_dice",
        "animal_dice",
        "delta_animal_dice",
        "phantom_dice",
        "delta_phantom_dice",
    ]
    return {key: row.get(key) for key in keys if key in row}


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


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
