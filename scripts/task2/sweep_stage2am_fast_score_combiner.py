#!/usr/bin/env python3
"""Fast Stage2AM score-combiner sweep for saved Task2 candidate rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import ap_101_point, iou_xyxy
from scripts.task2.export_stage2ab_domain_policy_predictions import CLASS_SUFFIX, DEFAULT_SPLITS
from scripts.task2.export_stage2ab_domain_policy_predictions import load_box_transforms
from scripts.task2.sweep_stage2ae_calibrated_policy import transformed_box
from scripts.task2.sweep_stage2v_domain_policy import DEFAULT_SCORE_MODES


DOMAINS = ("phantom", "animal")
IOU_THRESHOLDS = tuple(float(round(value, 2)) for value in np.arange(0.5, 0.96, 0.05))
BASE_POLICY = {
    0: {"phantom": "prob_iou75_source_rank_decay_roi", "animal": "prob_iou75_source_rank_decay_roi"},
    1: {"phantom": "rank_decay_roi", "animal": "prob_iou75_roi"},
}


@dataclass(frozen=True)
class SplitCache:
    sample_ids: list[str]
    video_ids: list[str]
    frame_indices: list[str]
    domains: np.ndarray
    gt_classes: np.ndarray
    boxes: dict[int, list[tuple[float, float, float, float]]]
    gt_ious: dict[int, np.ndarray]
    scores: dict[int, dict[str, np.ndarray]]
    candidate_rows: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--box-transform-json", type=Path, default=None)
    parser.add_argument("--optimize-split", default="valid_combined")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--score-mode", action="append", default=None)
    parser.add_argument("--alpha", action="append", type=float, default=None)
    parser.add_argument("--blend-type", action="append", default=None, choices=("linear", "geometric", "max"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_split_cache(
    run_dir: Path,
    split: str,
    *,
    score_modes: list[str],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> SplitCache:
    candidates = read_csv(run_dir / f"{split}_candidates_used.csv")
    predictions = read_csv(run_dir / f"{split}_eval_prediction_rows.csv")
    if len(candidates) != len(predictions):
        raise ValueError(f"{split}: row count mismatch")
    sample_ids: list[str] = []
    video_ids: list[str] = []
    frame_indices: list[str] = []
    domains: list[str] = []
    gt_classes: list[int] = []
    boxes: dict[int, list[tuple[float, float, float, float]]] = {0: [], 1: []}
    gt_ious: dict[int, list[float]] = {0: [], 1: []}
    scores: dict[int, dict[str, list[float]]] = {
        class_id: {mode: [] for mode in score_modes} for class_id in (0, 1)
    }
    for row_index, (candidate, prediction) in enumerate(zip(candidates, predictions), start=2):
        for field in ("sample_id", "source", "source_rank"):
            if str(candidate.get(field, "")) != str(prediction.get(field, "")):
                raise ValueError(f"{split}: row {row_index} alignment mismatch on {field}")
        sample_ids.append(candidate["sample_id"])
        video_ids.append(candidate.get("video_id", ""))
        frame_indices.append(candidate.get("frame_index", ""))
        domain = candidate.get("domain", "phantom")
        if domain not in DOMAINS:
            domain = "animal"
        domains.append(domain)
        gt_class = int(candidate["gt_class"])
        gt_classes.append(gt_class)
        gt_box = (
            float(candidate["gt_x1"]),
            float(candidate["gt_y1"]),
            float(candidate["gt_x2"]),
            float(candidate["gt_y2"]),
        )
        for class_id in (0, 1):
            box = transformed_box(candidate, class_id=class_id, box_transforms=box_transforms)
            boxes[class_id].append(box)
            gt_ious[class_id].append(iou_xyxy(box, gt_box) if gt_class == class_id else 0.0)
            suffix = CLASS_SUFFIX[class_id]
            for mode in score_modes:
                scores[class_id][mode].append(float(prediction[f"{mode}_score_{suffix}"]))
    return SplitCache(
        sample_ids=sample_ids,
        video_ids=video_ids,
        frame_indices=frame_indices,
        domains=np.asarray(domains, dtype=object),
        gt_classes=np.asarray(gt_classes, dtype=np.int64),
        boxes=boxes,
        gt_ious={class_id: np.asarray(values, dtype=np.float64) for class_id, values in gt_ious.items()},
        scores={
            class_id: {mode: np.asarray(values, dtype=np.float64) for mode, values in per_mode.items()}
            for class_id, per_mode in scores.items()
        },
        candidate_rows=candidates,
    )


def available_score_modes(run_dir: Path, split: str, requested: list[str] | None) -> list[str]:
    predictions = read_csv(run_dir / f"{split}_eval_prediction_rows.csv")
    fields = set(predictions[0]) if predictions else set()
    modes = requested or list(DEFAULT_SCORE_MODES)
    return [
        mode
        for mode in modes
        if all(f"{mode}_score_{suffix}" in fields for suffix in CLASS_SUFFIX.values())
    ]


def blend_scores(base: np.ndarray, alt: np.ndarray, *, alpha: float, blend_type: str) -> np.ndarray:
    base = np.clip(base.astype(np.float64), 0.0, 1.0)
    alt = np.clip(alt.astype(np.float64), 0.0, 1.0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if blend_type == "linear":
        return (1.0 - alpha) * base + alpha * alt
    if blend_type == "geometric":
        eps = 1e-12
        return np.exp((1.0 - alpha) * np.log(base + eps) + alpha * np.log(alt + eps))
    if blend_type == "max":
        return np.maximum((1.0 - alpha) * base, alpha * alt)
    raise ValueError(blend_type)


def ap_for_scores(
    cache: SplitCache,
    *,
    class_id: int,
    scores: np.ndarray,
    iou_threshold: float,
    row_mask: np.ndarray | None = None,
) -> float:
    if row_mask is None:
        row_mask = np.ones(len(cache.sample_ids), dtype=bool)
    gt_samples = {
        sample_id
        for sample_id, gt_class, keep in zip(cache.sample_ids, cache.gt_classes, row_mask)
        if keep and int(gt_class) == class_id
    }
    num_gt = len(gt_samples)
    if num_gt == 0:
        return float("nan")
    indices = np.where(row_mask)[0]
    indices = indices[np.argsort(-scores[indices], kind="mergesort")]
    matched: set[str] = set()
    tp: list[float] = []
    fp: list[float] = []
    gt_ious = cache.gt_ious[class_id]
    for index in indices:
        sample_id = cache.sample_ids[int(index)]
        if (
            int(cache.gt_classes[int(index)]) == class_id
            and float(gt_ious[int(index)]) >= iou_threshold
            and sample_id not in matched
        ):
            tp.append(1.0)
            fp.append(0.0)
            matched.add(sample_id)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return 0.0
    tp_arr = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp_arr = np.cumsum(np.asarray(fp, dtype=np.float64))
    recalls = tp_arr / max(num_gt, 1)
    precisions = tp_arr / np.maximum(tp_arr + fp_arr, 1e-12)
    return float(ap_101_point(recalls, precisions))


def class_metrics_for_scores(cache: SplitCache, *, class_id: int, scores: np.ndarray) -> dict[str, Any]:
    ap_values = [
        ap_for_scores(cache, class_id=class_id, scores=scores, iou_threshold=threshold)
        for threshold in IOU_THRESHOLDS
    ]
    return {
        "ap50": float(ap_values[0]),
        "ap50_95": float(np.nanmean(ap_values)),
        "ap_by_iou": {f"{threshold:.2f}": float(value) for threshold, value in zip(IOU_THRESHOLDS, ap_values)},
        "num_gt": int(len({sid for sid, cls in zip(cache.sample_ids, cache.gt_classes) if int(cls) == class_id})),
        "num_predictions": int(len(scores)),
    }


def detection_metrics(cache: SplitCache, *, policy: dict[str, Any]) -> dict[str, Any]:
    class_metrics: dict[str, Any] = {}
    ap50_values: list[float] = []
    ap5095_values: list[float] = []
    for class_id in (0, 1):
        scores = scores_for_policy(cache, class_id=class_id, policy=policy)
        metrics = class_metrics_for_scores(cache, class_id=class_id, scores=scores)
        class_metrics[str(class_id)] = metrics
        ap50_values.append(float(metrics["ap50"]))
        ap5095_values.append(float(metrics["ap50_95"]))
    return {
        "mAP50": float(np.nanmean(ap50_values)),
        "mAP50-95": float(np.nanmean(ap5095_values)),
        "classes": class_metrics,
    }


def scores_for_policy(cache: SplitCache, *, class_id: int, policy: dict[str, Any]) -> np.ndarray:
    out = np.zeros(len(cache.sample_ids), dtype=np.float64)
    for domain in DOMAINS:
        mask = cache.domains == domain
        domain_policy = policy[str(class_id)][domain]
        base = cache.scores[class_id][domain_policy["base_mode"]]
        alt = cache.scores[class_id][domain_policy["alt_mode"]]
        out[mask] = blend_scores(
            base[mask],
            alt[mask],
            alpha=float(domain_policy["alpha"]),
            blend_type=str(domain_policy["blend_type"]),
        )
    return out


def neutral_policy(class_id: int) -> dict[str, dict[str, Any]]:
    return {
        domain: {
            "base_mode": BASE_POLICY[class_id][domain],
            "alt_mode": BASE_POLICY[class_id][domain],
            "alpha": 0.0,
            "blend_type": "linear",
        }
        for domain in DOMAINS
    }


def search_domain_policy(
    cache: SplitCache,
    *,
    class_id: int,
    domain: str,
    score_modes: list[str],
    alphas: list[float],
    blend_types: list[str],
) -> dict[str, Any]:
    mask = cache.domains == domain
    base_mode = BASE_POLICY[class_id][domain]
    best: dict[str, Any] | None = None
    for alt_mode in score_modes:
        for alpha in alphas:
            for blend_type in blend_types:
                scores = blend_scores(
                    cache.scores[class_id][base_mode],
                    cache.scores[class_id][alt_mode],
                    alpha=alpha,
                    blend_type=blend_type,
                )
                ap_values = [
                    ap_for_scores(
                        cache,
                        class_id=class_id,
                        scores=scores,
                        iou_threshold=threshold,
                        row_mask=mask,
                    )
                    for threshold in IOU_THRESHOLDS
                ]
                row = {
                    "base_mode": base_mode,
                    "alt_mode": alt_mode,
                    "alpha": float(alpha),
                    "blend_type": blend_type,
                    "domain_ap50": float(ap_values[0]),
                    "domain_ap50_95": float(np.nanmean(ap_values)),
                }
                if best is None or (row["domain_ap50_95"], row["domain_ap50"]) > (
                    best["domain_ap50_95"],
                    best["domain_ap50"],
                ):
                    best = row
    if best is None:
        raise ValueError(f"No policy for class={class_id} domain={domain}")
    return best


def prediction_rows(cache: SplitCache, *, policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id in (0, 1):
        scores = scores_for_policy(cache, class_id=class_id, policy=policy)
        for index, (candidate, score) in enumerate(zip(cache.candidate_rows, scores)):
            domain = cache.domains[index]
            domain_policy = policy[str(class_id)][str(domain)]
            x1, y1, x2, y2 = cache.boxes[class_id][index]
            rows.append(
                {
                    "sample_id": cache.sample_ids[index],
                    "video_id": cache.video_ids[index],
                    "frame_index": cache.frame_indices[index],
                    "domain": str(domain),
                    "class_id": class_id,
                    "score": float(score),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "source": candidate.get("source", ""),
                    "source_rank": candidate.get("source_rank", ""),
                    "score_mode": (
                        f"{domain_policy['blend_type']}:{domain_policy['base_mode']}:"
                        f"{domain_policy['alt_mode']}:{float(domain_policy['alpha']):.3f}"
                    ),
                    "policy_name": f"class{class_id}_{domain}",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    box_transforms = load_box_transforms(args.box_transform_json)
    score_modes = available_score_modes(args.run_dir, args.optimize_split, args.score_mode)
    if not score_modes:
        raise ValueError("No score modes available")
    alphas = [float(value) for value in (args.alpha or [0.0, 0.1, 0.25, 0.5, 1.0])]
    blend_types = list(args.blend_type or ["linear", "geometric", "max"])
    optimize_cache = load_split_cache(
        args.run_dir,
        args.optimize_split,
        score_modes=score_modes,
        box_transforms=box_transforms,
    )
    policy = {"0": neutral_policy(0), "1": neutral_policy(1)}
    best_by_class_domain: dict[str, dict[str, Any]] = {}
    for class_id in (0, 1):
        best_by_class_domain[str(class_id)] = {}
        for domain in DOMAINS:
            best = search_domain_policy(
                optimize_cache,
                class_id=class_id,
                domain=domain,
                score_modes=score_modes,
                alphas=alphas,
                blend_types=blend_types,
            )
            best_by_class_domain[str(class_id)][domain] = best
            policy[str(class_id)][domain] = {
                "base_mode": best["base_mode"],
                "alt_mode": best["alt_mode"],
                "alpha": best["alpha"],
                "blend_type": best["blend_type"],
            }

    metrics: dict[str, Any] = {}
    for split in args.splits:
        cache = optimize_cache if split == args.optimize_split else load_split_cache(
            args.run_dir,
            split,
            score_modes=score_modes,
            box_transforms=box_transforms,
        )
        metrics[split] = detection_metrics(cache, policy=policy)
        if args.prediction_dir is not None:
            write_csv(args.prediction_dir / f"{split}_stage2am_fast_predictions.csv", prediction_rows(cache, policy=policy))

    output = {
        "artifact_type": "task2_stage2am_fast_score_combiner",
        "run_dir": args.run_dir.as_posix(),
        "box_transform_json": args.box_transform_json.as_posix() if args.box_transform_json else None,
        "optimize_split": args.optimize_split,
        "score_modes": score_modes,
        "alphas": alphas,
        "blend_types": blend_types,
        "base_policy": BASE_POLICY,
        "best_by_class_domain": best_by_class_domain,
        "final_policy": policy,
        "metrics": metrics,
    }
    write_json(args.output_json, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
