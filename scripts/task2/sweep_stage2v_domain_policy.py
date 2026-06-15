#!/usr/bin/env python3
"""Sweep domain-aware score-mode policies for a Stage2U eval run.

This is a diagnostic on top of the current Stage2V idea. It keeps the candidate
source fixed and only asks whether different domains/classes prefer different
score columns. The search is class-separable: for each class, choose the
phantom/animal score-mode pair that maximizes class AP50-95 on the optimization
split, then evaluate the resulting policy on requested splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth,
    DetectionPrediction,
    compute_class_ap,
    compute_detection_map,
)
from scripts.task2.evaluate_stage2u_class_fusion import CLASS_SUFFIX, load_run_rows  # noqa: E402


DEFAULT_SCORE_MODES = (
    "roi",
    "bg_suppressed_roi",
    "source_roi",
    "sqrt_source_roi",
    "rank_decay_roi",
    "source_rank_decay_roi",
    "pred_iou_roi",
    "pred_iou_rank_decay_roi",
    "pred_iou_source_rank_decay_roi",
    "prob_iou50_roi",
    "prob_iou50_rank_decay_roi",
    "prob_iou50_source_rank_decay_roi",
    "prob_iou75_roi",
    "prob_iou75_rank_decay_roi",
    "prob_iou75_source_rank_decay_roi",
    "blend_roi",
    "blend_rank_decay_roi",
    "blend_source_rank_decay_roi",
)
DOMAINS = ("phantom", "animal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--optimize-split", default="valid_combined")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("valid_combined", "valid_phantom", "valid_animal"),
    )
    parser.add_argument("--score-mode", action="append", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    return parser.parse_args()


def available_score_modes(rows: list[tuple[dict[str, str], dict[str, str]]]) -> list[str]:
    if not rows:
        return []
    fields = set(rows[0][1])
    modes: list[str] = []
    for mode in DEFAULT_SCORE_MODES:
        if all(f"{mode}_score_{suffix}" in fields for suffix in CLASS_SUFFIX.values()):
            modes.append(mode)
    return modes


def class_predictions_for_domain_policy(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    policy: dict[str, str],
) -> list[DetectionPrediction]:
    suffix = CLASS_SUFFIX[class_id]
    predictions: list[DetectionPrediction] = []
    for candidate, prediction in rows:
        domain = candidate.get("domain", "phantom")
        score_mode = policy.get(domain, policy.get("default", "roi"))
        score_key = f"{score_mode}_score_{suffix}"
        predictions.append(
            DetectionPrediction(
                candidate["sample_id"],
                class_id,
                float(prediction[score_key]),
                (
                    float(candidate["x1"]),
                    float(candidate["y1"]),
                    float(candidate["x2"]),
                    float(candidate["y2"]),
                ),
            )
        )
    return predictions


def prediction_rows_for_domain_policy(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    policy: dict[str, str],
) -> list[dict[str, Any]]:
    suffix = CLASS_SUFFIX[class_id]
    result: list[dict[str, Any]] = []
    for candidate, prediction in rows:
        domain = candidate.get("domain", "phantom")
        score_mode = policy.get(domain, policy.get("default", "roi"))
        score_key = f"{score_mode}_score_{suffix}"
        result.append(
            {
                "sample_id": candidate["sample_id"],
                "video_id": candidate.get("video_id", ""),
                "frame_index": candidate.get("frame_index", ""),
                "domain": domain,
                "class_id": class_id,
                "score": float(prediction[score_key]),
                "x1": float(candidate["x1"]),
                "y1": float(candidate["y1"]),
                "x2": float(candidate["x2"]),
                "y2": float(candidate["y2"]),
                "source": candidate.get("source", ""),
                "source_rank": candidate.get("source_rank", ""),
                "score_mode": score_mode,
                "policy_name": f"class{class_id}_{domain}",
            }
        )
    return result


def search_class_policy(
    ground_truths: list[DetectionGroundTruth],
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    score_modes: list[str],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for phantom_mode in score_modes:
        for animal_mode in score_modes:
            policy = {"phantom": phantom_mode, "animal": animal_mode, "default": phantom_mode}
            predictions = class_predictions_for_domain_policy(rows, class_id=class_id, policy=policy)
            ap_values = [
                compute_class_ap(
                    ground_truths,
                    predictions,
                    class_id=class_id,
                    iou_threshold=threshold,
                )["ap"]
                for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
            ]
            ap50 = float(ap_values[0])
            ap50_95 = float(sum(ap_values) / len(ap_values))
            row = {
                "class_id": class_id,
                "policy": policy,
                "ap50": ap50,
                "ap50_95": ap50_95,
            }
            if best is None or ap50_95 > float(best["ap50_95"]):
                best = row
    if best is None:
        raise ValueError("No class policy was evaluated")
    return best


def evaluate_policy(
    run_dir: Path,
    split: str,
    *,
    class_policies: dict[int, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ground_truths, rows = load_run_rows(run_dir, split)
    predictions = [
        *class_predictions_for_domain_policy(rows, class_id=0, policy=class_policies[0]),
        *class_predictions_for_domain_policy(rows, class_id=1, policy=class_policies[1]),
    ]
    prediction_rows = [
        *prediction_rows_for_domain_policy(rows, class_id=0, policy=class_policies[0]),
        *prediction_rows_for_domain_policy(rows, class_id=1, policy=class_policies[1]),
    ]
    return compute_detection_map(ground_truths, predictions, class_ids=(0, 1)), prediction_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    optimize_ground_truths, optimize_rows = load_run_rows(run_dir, args.optimize_split)
    score_modes = args.score_mode or available_score_modes(optimize_rows)
    if not score_modes:
        raise ValueError("No score modes available for search")

    best_class = {
        class_id: search_class_policy(
            optimize_ground_truths,
            optimize_rows,
            class_id=class_id,
            score_modes=list(score_modes),
        )
        for class_id in (0, 1)
    }
    class_policies = {
        class_id: dict(best_class[class_id]["policy"])
        for class_id in (0, 1)
    }

    split_metrics: dict[str, Any] = {}
    for split in args.splits:
        metrics, rows = evaluate_policy(run_dir, split, class_policies=class_policies)
        split_metrics[split] = {
            "detection": metrics,
            "prediction_rows": len(rows),
        }
        if args.prediction_dir is not None:
            write_csv(args.prediction_dir / f"{split}_domain_policy_predictions.csv", rows)

    result = {
        "run_dir": run_dir.as_posix(),
        "optimize_split": args.optimize_split,
        "score_modes": list(score_modes),
        "best_class_policies": best_class,
        "metrics": split_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

