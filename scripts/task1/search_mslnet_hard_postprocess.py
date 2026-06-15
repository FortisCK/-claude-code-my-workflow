#!/usr/bin/env python3
"""Search hard-mask foreground gates against MSLNet-style F1/Dice."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from cathaction.data.task1 import load_task1_mask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--domain", action="append", default=None)
    parser.add_argument("--preset", choices=("quick", "expanded"), default="quick")
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=None,
        help="Restrict the search to one or more candidate IDs from the selected preset. Baseline is added automatically.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--write-best-predictions", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    eval_manifest = _resolve(args.eval_manifest, repo_root)
    predictions_csv = _resolve(args.predictions_csv, repo_root)
    output_dir = _resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = {row["sample_id"]: row for row in _read_rows(eval_manifest)}
    prediction_rows = _read_rows(predictions_csv)

    domain_filter = set(str(value) for value in args.domain) if args.domain else None
    samples = _load_samples(
        prediction_rows,
        eval_rows=eval_rows,
        repo_root=repo_root,
        domain_filter=domain_filter,
    )
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    candidates = _candidate_grid(str(args.preset), domain_filter=domain_filter)
    candidates = _select_candidates(candidates, args.candidate_id)
    metric_states = {candidate["candidate_id"]: _new_state() for candidate in candidates}
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    disk3 = _disk_structure(3)

    for sample in tqdm(samples, desc="mslnet-hard-grid", leave=False):
        target_fg = sample["target"] > 0
        target_count = int(target_fg.sum())
        distance_to_target = ndimage.distance_transform_edt(~target_fg)
        target_by_dilation_cache: dict[str, np.ndarray] = {}
        for candidate in candidates:
            processed = _apply_candidate(sample["prediction"], candidate, domain=sample["domain"])
            pred_fg = processed > 0
            pred_count = int(pred_fg.sum())
            intersection = int(np.logical_and(pred_fg, target_fg).sum())
            union = int(np.logical_or(pred_fg, target_fg).sum())
            dice = 1.0 if pred_count + target_count == 0 else float(2.0 * intersection / (pred_count + target_count))
            iou = 1.0 if union == 0 else float(intersection / union)
            if pred_count == 0 and target_count == 0:
                precision_r3 = 1.0
                recall_r3 = 1.0
            elif pred_count == 0 or target_count == 0:
                precision_r3 = 0.0
                recall_r3 = 0.0
            else:
                precision_r3 = float(np.count_nonzero(distance_to_target[pred_fg] <= 3) / pred_count)
                # Radius-3 Euclidean tolerance for GT pixels close to predicted foreground.
                cache_key = candidate["candidate_id"]
                dilated_prediction = target_by_dilation_cache.get(cache_key)
                if dilated_prediction is None:
                    dilated_prediction = ndimage.binary_dilation(pred_fg, structure=disk3)
                    target_by_dilation_cache[cache_key] = dilated_prediction
                recall_r3 = float(np.logical_and(target_fg, dilated_prediction).sum() / target_count)
            f1_r3 = _f1(precision_r3, recall_r3)

            state = metric_states[candidate["candidate_id"]]
            _append_state(
                state,
                sample=sample,
                dice=dice,
                iou=iou,
                precision_r3=precision_r3,
                recall_r3=recall_r3,
                f1_r3=f1_r3,
            )

    rows = [_summarize_state(candidate_by_id[candidate_id], state) for candidate_id, state in metric_states.items()]
    baseline = next(row for row in rows if row["candidate_id"] == "baseline")
    for row in rows:
        row["delta_dice"] = float(row["dice"] - baseline["dice"])
        row["delta_iou"] = float(row["iou"] - baseline["iou"])
        row["delta_f1_r3"] = float(row["f1_r3" ] - baseline["f1_r3"])
        row["delta_precision_r3"] = float(row["precision_r3"] - baseline["precision_r3"])
        row["delta_recall_r3"] = float(row["recall_r3"] - baseline["recall_r3"])
    rows.sort(key=lambda row: (float(row["f1_r3"]), float(row["dice"])), reverse=True)

    metrics_csv = output_dir / "candidate_metrics.csv"
    _write_csv(metrics_csv, rows)
    summary = {
        "metric_style": "fast MSLNet-style hard-mask search",
        "search_note": "Uses exact binary Dice/IoU and distance-to-target precision r3; recall r3 is computed via radius-3 binary dilation. Full AHD/F1 r0-r4 must be recomputed for promoted candidates.",
        "eval_manifest": _repo_relative(eval_manifest, repo_root),
        "predictions_csv": _repo_relative(predictions_csv, repo_root),
        "output_dir": _repo_relative(output_dir, repo_root),
        "samples": len(samples),
        "domain_filter": sorted(domain_filter) if domain_filter else None,
        "candidates": len(candidates),
        "preset": str(args.preset),
        "baseline": _public_row(baseline),
        "best": _public_row(rows[0]),
        "top": [_public_row(row) for row in rows[: int(args.top_k)]],
        "candidate_metrics_csv": _repo_relative(metrics_csv, repo_root),
    }
    if args.write_best_predictions:
        best_dir = output_dir / "best_predictions"
        best_manifest = _write_candidate_predictions(
            best_dir,
            samples,
            candidate_by_id[str(rows[0]["candidate_id"])],
            repo_root=repo_root,
        )
        summary["best_prediction_manifest"] = _repo_relative(best_manifest, repo_root)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _candidate_grid(preset: str, *, domain_filter: set[str] | None) -> list[dict[str, Any]]:
    if preset == "quick":
        return _quick_candidate_grid(domain_filter=domain_filter)
    if preset != "expanded":
        raise ValueError(f"Unsupported preset: {preset}")
    return _expanded_candidate_grid(domain_filter=domain_filter)


def _select_candidates(
    candidates: list[dict[str, Any]],
    candidate_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not candidate_ids:
        return candidates
    requested = {"baseline", *(str(candidate_id) for candidate_id in candidate_ids)}
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    missing = sorted(requested.difference(by_id))
    if missing:
        available = ", ".join(sorted(by_id))
        raise ValueError(f"Unknown candidate ID(s): {missing}. Available candidates: {available}")
    return [candidate for candidate in candidates if str(candidate["candidate_id"]) in requested]


def _quick_candidate_grid(*, domain_filter: set[str] | None) -> list[dict[str, Any]]:
    domains = ["all"] if domain_filter is not None else ["all", "phantom"]
    candidates: list[dict[str, Any]] = []
    for domain in domains:
        candidates.append(_candidate("baseline" if domain == "all" else f"{domain}_baseline", domain))
        for min_size in [64, 256, 1024]:
            candidates.append(_candidate(f"{domain}_min{min_size}", domain, min_component_size=min_size))
        for keep_top_k in [1, 2, 3, 5, 8]:
            candidates.append(_candidate(f"{domain}_top{keep_top_k}", domain, keep_top_k=keep_top_k))
        candidates.append(_candidate(f"{domain}_erode1", domain, op="erode", radius=1))
        candidates.append(_candidate(f"{domain}_open1", domain, op="open", radius=1))
        for keep_top_k in [2, 3, 5]:
            candidates.append(
                _candidate(
                    f"{domain}_erode1_top{keep_top_k}",
                    domain,
                    op="erode",
                    radius=1,
                    keep_top_k=keep_top_k,
                )
            )
            candidates.append(
                _candidate(
                    f"{domain}_open1_top{keep_top_k}",
                    domain,
                    op="open",
                    radius=1,
                    keep_top_k=keep_top_k,
                )
            )
        for min_size in [128, 512]:
            candidates.append(
                _candidate(
                    f"{domain}_erode1_min{min_size}",
                    domain,
                    op="erode",
                    radius=1,
                    min_component_size=min_size,
                )
            )
            candidates.append(
                _candidate(
                    f"{domain}_open1_min{min_size}",
                    domain,
                    op="open",
                    radius=1,
                    min_component_size=min_size,
                )
            )
    return _deduplicate(candidates)


def _expanded_candidate_grid(*, domain_filter: set[str] | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    domains = ["all"] if domain_filter is not None else ["all", "phantom"]
    for domain in domains:
        candidates.append(_candidate("baseline" if domain == "all" else f"{domain}_baseline", domain))
        for min_size in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            candidates.append(_candidate(f"{domain}_min{min_size}", domain, min_component_size=min_size))
        for keep_top_k in [1, 2, 3, 4, 5, 8, 12]:
            candidates.append(_candidate(f"{domain}_top{keep_top_k}", domain, keep_top_k=keep_top_k))
        for op in ["erode", "open"]:
            for radius in [1, 2]:
                candidates.append(_candidate(f"{domain}_{op}{radius}", domain, op=op, radius=radius))
                for min_size in [64, 128, 256, 512]:
                    candidates.append(
                        _candidate(
                            f"{domain}_{op}{radius}_min{min_size}",
                            domain,
                            op=op,
                            radius=radius,
                            min_component_size=min_size,
                        )
                    )
                for keep_top_k in [2, 3, 4, 5, 8]:
                    candidates.append(
                        _candidate(
                            f"{domain}_{op}{radius}_top{keep_top_k}",
                            domain,
                            op=op,
                            radius=radius,
                            keep_top_k=keep_top_k,
                        )
                    )
    return _deduplicate(candidates)


def _candidate(
    candidate_id: str,
    domain: str,
    *,
    op: str = "none",
    radius: int = 0,
    min_component_size: int = 0,
    keep_top_k: int = 0,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "domain": domain,
        "op": op,
        "radius": int(radius),
        "min_component_size": int(min_component_size),
        "keep_top_k": int(keep_top_k),
    }


def _deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (
            candidate["domain"],
            candidate["op"],
            candidate["radius"],
            candidate["min_component_size"],
            candidate["keep_top_k"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _load_samples(
    prediction_rows: list[dict[str, str]],
    *,
    eval_rows: dict[str, dict[str, str]],
    repo_root: Path,
    domain_filter: set[str] | None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in tqdm(prediction_rows, desc="load-samples", leave=False):
        sample_id = row["sample_id"]
        eval_row = eval_rows.get(sample_id)
        if eval_row is None:
            continue
        if domain_filter is not None and str(eval_row.get("domain", "")) not in domain_filter:
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


def _apply_candidate(prediction: np.ndarray, candidate: dict[str, Any], *, domain: str) -> np.ndarray:
    if str(candidate["domain"]) == "phantom" and str(domain) != "phantom":
        return np.asarray(prediction, dtype=np.uint8)

    prediction = np.asarray(prediction, dtype=np.uint8)
    foreground = prediction > 0
    op = str(candidate["op"])
    radius = int(candidate["radius"])
    if op == "erode" and radius > 0:
        foreground = ndimage.binary_erosion(foreground, structure=_disk_structure(radius), border_value=0)
    elif op == "open" and radius > 0:
        foreground = ndimage.binary_opening(foreground, structure=_disk_structure(radius), border_value=0)
    elif op != "none":
        raise ValueError(f"Unsupported op: {op}")

    min_component_size = int(candidate["min_component_size"])
    keep_top_k = int(candidate["keep_top_k"])
    if min_component_size > 0 or keep_top_k > 0:
        foreground = _filter_components(
            foreground,
            min_component_size=min_component_size,
            keep_top_k=keep_top_k,
        )

    output = prediction.copy()
    output[~foreground] = 0
    return output


def _filter_components(mask: np.ndarray, *, min_component_size: int, keep_top_k: int) -> np.ndarray:
    labeled, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    keep = np.ones(len(sizes), dtype=bool)
    keep[0] = False
    if min_component_size > 0:
        keep &= sizes >= int(min_component_size)
    if keep_top_k > 0:
        component_ids = np.arange(len(sizes))
        eligible = component_ids[keep]
        eligible = eligible[np.argsort(sizes[eligible])[::-1]]
        top = set(int(value) for value in eligible[: int(keep_top_k)])
        keep &= np.array([idx in top for idx in component_ids], dtype=bool)
    return keep[labeled]


def _new_state() -> dict[str, Any]:
    return {
        "dice": [],
        "iou": [],
        "precision_r3": [],
        "recall_r3": [],
        "f1_r3": [],
        "by_domain": defaultdict(lambda: defaultdict(list)),
    }


def _append_state(
    state: dict[str, Any],
    *,
    sample: dict[str, Any],
    dice: float,
    iou: float,
    precision_r3: float,
    recall_r3: float,
    f1_r3: float,
) -> None:
    values = {
        "dice": dice,
        "iou": iou,
        "precision_r3": precision_r3,
        "recall_r3": recall_r3,
        "f1_r3": f1_r3,
    }
    for key, value in values.items():
        state[key].append(value)
        state["by_domain"][str(sample["domain"])][key].append(value)


def _summarize_state(candidate: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    for key in ["dice", "iou", "precision_r3", "recall_r3", "f1_r3"]:
        result[key] = _mean(state[key])
    for domain, metrics in sorted(state["by_domain"].items()):
        for key in ["dice", "iou", "precision_r3", "recall_r3", "f1_r3"]:
            result[f"{domain}_{key}"] = _mean(metrics[key])
        result[f"{domain}_samples"] = len(metrics["dice"])
    result["num_samples"] = len(state["dice"])
    return result


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
        processed = _apply_candidate(sample["prediction"], candidate, domain=str(sample["domain"]))
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


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate_id",
        "domain",
        "op",
        "radius",
        "min_component_size",
        "keep_top_k",
        "dice",
        "delta_dice",
        "iou",
        "delta_iou",
        "precision_r3",
        "delta_precision_r3",
        "recall_r3",
        "delta_recall_r3",
        "f1_r3",
        "delta_f1_r3",
        "animal_f1_r3",
        "phantom_f1_r3",
    ]
    return {key: row.get(key) for key in keys if key in row}


def _disk_structure(radius: int) -> np.ndarray:
    radius = int(radius)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
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


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
