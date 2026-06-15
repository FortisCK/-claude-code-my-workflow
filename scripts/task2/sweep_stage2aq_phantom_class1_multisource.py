#!/usr/bin/env python3
"""Sweep controlled phantom-class1 multi-source fusion policies.

Stage2AQ keeps the Stage2AE champion predictions everywhere except phantom
class1. For phantom class1, it swaps in candidates from a multi-source Stage2U
eval run and sweeps source subsets, score modes, top-k, and score scale.

This is a public-validation experiment. It uses GT only for metric computation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
    compute_detection_map,
)
from scripts.task2.export_clean_predictions import CLEAN_FIELDS, apply_topk, write_rows  # noqa: E402


DEFAULT_BASELINE_DIR = Path(
    "outputs/task2/stage2ao_champion_pipeline_with_an/public_valid_ab_ad_ae_ai_an/stage2ae"
)
DEFAULT_MULTISOURCE_RUN_DIR = Path("outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval")
DEFAULT_OUTPUT_DIR = Path("outputs/task2/stage2aq_phantom_class1_multisource/stage2ae_stage2x_sweep")

SOURCE_POLICIES: dict[str, set[str]] = {
    "yolo_stage2x": {"yolo_stage2l", "stage2x_class1"},
    "yolo_geom_stage2x": {"yolo_stage2l", "task1_geometry_rect", "stage2x_class1"},
    "original_sources": {"yolo_stage2l", "task1_geometry_rect"},
}
SCORE_MODES = (
    "rank_decay_roi",
    "prob_iou75_rank_decay_roi",
    "blend_rank_decay_roi",
)
TOPK_VALUES = (1, 3, 5, 10)
SCALE_VALUES = (0.75, 1.0, 1.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--multisource-run-dir", type=Path, default=DEFAULT_MULTISOURCE_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("valid_combined", "valid_phantom", "valid_animal"),
    )
    parser.add_argument("--optimize-split", default="valid_combined")
    parser.add_argument("--max-export-rank", type=int, default=25)
    parser.add_argument(
        "--skip-full-metrics",
        action="store_true",
        help="Only run the fast phantom-class1 AP sweep and skip full detector metric export.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str | float | int | None, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def box_from_row(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (fnum(row["x1"]), fnum(row["y1"]), fnum(row["x2"]), fnum(row["y2"]))


def ground_truths_from_candidates(candidates: list[dict[str, str]]) -> list[DetectionGroundTruth]:
    seen: set[str] = set()
    ground_truths: list[DetectionGroundTruth] = []
    for row in candidates:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        ground_truths.append(
            DetectionGroundTruth(
                sample_id=sample_id,
                class_id=int(row["gt_class"]),
                box_xyxy=(
                    fnum(row["gt_x1"]),
                    fnum(row["gt_y1"]),
                    fnum(row["gt_x2"]),
                    fnum(row["gt_y2"]),
                ),
            )
        )
    return ground_truths


def baseline_predictions(rows: list[dict[str, str]], *, keep_phantom_class1: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        class_id = int(row["class_id"])
        domain = row["domain"].strip().lower()
        if not keep_phantom_class1 and domain == "phantom" and class_id == 1:
            continue
        output.append(
            {
                "sample_id": row["sample_id"],
                "video_id": row.get("video_id", ""),
                "frame_index": row.get("frame_index", ""),
                "domain": domain,
                "class_id": class_id,
                "score": fnum(row["score"]),
                "x1": fnum(row["x1"]),
                "y1": fnum(row["y1"]),
                "x2": fnum(row["x2"]),
                "y2": fnum(row["y2"]),
                "source": row.get("source", ""),
                "source_rank": row.get("source_rank", ""),
                "score_mode": row.get("score_mode", ""),
                "policy_name": row.get("policy_name", ""),
            }
        )
    return output


def multisource_phantom_class1_rows(
    candidates: list[dict[str, str]],
    predictions: list[dict[str, str]],
    *,
    source_keep: set[str],
    score_mode: str,
    topk: int,
    scale: float,
) -> list[dict[str, Any]]:
    if len(candidates) != len(predictions):
        raise ValueError(f"row mismatch: {len(candidates)} candidates vs {len(predictions)} predictions")
    score_key = f"{score_mode}_score_collision"
    if predictions and score_key not in predictions[0]:
        raise KeyError(score_key)
    rows: list[dict[str, Any]] = []
    for row_index, (candidate, prediction) in enumerate(zip(candidates, predictions), start=2):
        if candidate.get("domain", "").strip().lower() != "phantom":
            continue
        if candidate.get("source", "") not in source_keep:
            continue
        for key in ("sample_id", "source", "source_rank"):
            if str(candidate.get(key, "")) != str(prediction.get(key, "")):
                raise ValueError(
                    f"alignment mismatch at row {row_index} for {key}: "
                    f"{candidate.get(key)!r} vs {prediction.get(key)!r}"
                )
        rows.append(
            {
                "sample_id": candidate["sample_id"],
                "video_id": candidate.get("video_id", ""),
                "frame_index": candidate.get("frame_index", ""),
                "domain": "phantom",
                "class_id": 1,
                "score": fnum(prediction[score_key]) * scale,
                "x1": fnum(candidate["x1"]),
                "y1": fnum(candidate["y1"]),
                "x2": fnum(candidate["x2"]),
                "y2": fnum(candidate["y2"]),
                "source": candidate.get("source", ""),
                "source_rank": candidate.get("source_rank", ""),
                "score_mode": score_mode,
                "policy_name": f"stage2aq_{score_mode}_top{topk}_x{scale:g}",
            }
        )
    return apply_topk(rows, topk)


def ap_101_point(tp_sequence: list[bool], num_gt: int) -> float:
    if num_gt <= 0:
        return float("nan")
    true_positive = 0
    false_positive = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for is_true_positive in tp_sequence:
        if is_true_positive:
            true_positive += 1
        else:
            false_positive += 1
        recalls.append(true_positive / num_gt)
        precisions.append(true_positive / max(true_positive + false_positive, 1))
    if not recalls:
        return 0.0

    padded_recalls = [0.0, *recalls, 1.0]
    padded_precisions = [0.0, *precisions, 0.0]
    for index in range(len(padded_precisions) - 2, -1, -1):
        padded_precisions[index] = max(padded_precisions[index], padded_precisions[index + 1])

    total = 0.0
    for index in range(101):
        threshold = index / 100
        eligible = [
            precision
            for recall, precision in zip(padded_recalls, padded_precisions)
            if recall >= threshold
        ]
        total += max(eligible) if eligible else 0.0
    return total / 101


def fast_phantom_class1_ap(
    candidates: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    *,
    source_policy_name: str,
    score_mode: str,
    topk: int,
    scale: float,
    iou_threshold: float,
) -> dict[str, Any]:
    rows = multisource_phantom_class1_rows(
        candidates,
        prediction_rows,
        source_keep=SOURCE_POLICIES[source_policy_name],
        score_mode=score_mode,
        topk=topk,
        scale=scale,
    )
    positive_samples = {
        row["sample_id"]
        for row in candidates
        if row.get("domain", "").strip().lower() == "phantom" and int(row["gt_class"]) == 1
    }
    gt_iou_by_key: dict[tuple[str, str, str], tuple[int, float]] = {}
    for candidate in candidates:
        if candidate.get("domain", "").strip().lower() != "phantom":
            continue
        gt_iou_by_key[
            (
                candidate["sample_id"],
                candidate.get("source", ""),
                str(candidate.get("source_rank", "")),
            )
        ] = (int(candidate["gt_class"]), fnum(candidate.get("gt_iou")))

    matched: set[str] = set()
    tp_sequence: list[bool] = []
    true_positives = 0
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        key = (str(row["sample_id"]), str(row["source"]), str(row["source_rank"]))
        gt_class, gt_iou = gt_iou_by_key.get(key, (0, 0.0))
        is_tp = gt_class == 1 and gt_iou >= iou_threshold and str(row["sample_id"]) not in matched
        if is_tp:
            matched.add(str(row["sample_id"]))
            true_positives += 1
        tp_sequence.append(is_tp)
    return {
        "ap": ap_101_point(tp_sequence, len(positive_samples)),
        "num_gt": len(positive_samples),
        "num_predictions": len(rows),
        "true_positives": true_positives,
        "false_positives": len(rows) - true_positives,
    }


def detection_predictions(rows: list[dict[str, Any]]) -> list[DetectionPrediction]:
    return [
        DetectionPrediction(
            sample_id=str(row["sample_id"]),
            class_id=int(row["class_id"]),
            score=float(row["score"]),
            box_xyxy=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
        )
        for row in rows
    ]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str, int] = defaultdict(int)
    domain_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        source_counts[str(row["source"])] += 1
        class_counts[str(row["class_id"])] += 1
        domain_counts[str(row["domain"])] += 1
    return {
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


def build_prediction_rows(
    *,
    baseline_rows: list[dict[str, str]],
    candidates: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    source_policy_name: str,
    score_mode: str,
    topk: int,
    scale: float,
) -> list[dict[str, Any]]:
    return [
        *baseline_predictions(baseline_rows, keep_phantom_class1=False),
        *multisource_phantom_class1_rows(
            candidates,
            prediction_rows,
            source_keep=SOURCE_POLICIES[source_policy_name],
            score_mode=score_mode,
            topk=topk,
            scale=scale,
        ),
    ]


def evaluate_policy_for_split(
    *,
    split: str,
    baseline_dir: Path,
    multisource_run_dir: Path,
    source_policy_name: str,
    score_mode: str,
    topk: int,
    scale: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_rows = read_csv(baseline_dir / f"{split}_domain_policy_predictions.csv")
    candidates = read_csv(multisource_run_dir / f"{split}_candidates_used.csv")
    prediction_rows = read_csv(multisource_run_dir / f"{split}_eval_prediction_rows.csv")
    ground_truths = ground_truths_from_candidates(candidates)
    rows = build_prediction_rows(
        baseline_rows=baseline_rows,
        candidates=candidates,
        prediction_rows=prediction_rows,
        source_policy_name=source_policy_name,
        score_mode=score_mode,
        topk=topk,
        scale=scale,
    )
    metrics = compute_detection_map(ground_truths, detection_predictions(rows), class_ids=(0, 1))
    return metrics, rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimize_candidates = read_csv(args.multisource_run_dir / f"{args.optimize_split}_candidates_used.csv")
    optimize_predictions = read_csv(args.multisource_run_dir / f"{args.optimize_split}_eval_prediction_rows.csv")
    policies: list[dict[str, Any]] = []
    for source_policy_name in SOURCE_POLICIES:
        for score_mode in SCORE_MODES:
            for topk in TOPK_VALUES:
                for scale in SCALE_VALUES:
                    ap50 = fast_phantom_class1_ap(
                        optimize_candidates,
                        optimize_predictions,
                        source_policy_name=source_policy_name,
                        score_mode=score_mode,
                        topk=topk,
                        scale=scale,
                        iou_threshold=0.50,
                    )
                    ap75 = fast_phantom_class1_ap(
                        optimize_candidates,
                        optimize_predictions,
                        source_policy_name=source_policy_name,
                        score_mode=score_mode,
                        topk=topk,
                        scale=scale,
                        iou_threshold=0.75,
                    )
                    policies.append(
                        {
                            "source_policy": source_policy_name,
                            "score_mode": score_mode,
                            "topk": topk,
                            "scale": scale,
                            "optimize_phantom_class1_ap50": ap50,
                            "optimize_phantom_class1_ap75": ap75,
                        }
                    )

    policies.sort(
        key=lambda item: (
            float(item["optimize_phantom_class1_ap50"]["ap"]),
            float(item["optimize_phantom_class1_ap75"]["ap"]),
            -float(item["optimize_phantom_class1_ap50"]["num_predictions"]),
        ),
        reverse=True,
    )
    best = policies[0]
    best_split_metrics: dict[str, Any] = {}
    best_prediction_dir = args.output_dir / "best_predictions"
    if not args.skip_full_metrics:
        for split in args.splits:
            metrics, rows = evaluate_policy_for_split(
                split=split,
                baseline_dir=args.baseline_dir,
                multisource_run_dir=args.multisource_run_dir,
                source_policy_name=str(best["source_policy"]),
                score_mode=str(best["score_mode"]),
                topk=int(best["topk"]),
                scale=float(best["scale"]),
            )
            rows = sorted(rows, key=lambda row: (str(row["sample_id"]), int(row["class_id"]), -float(row["score"])))
            write_rows(best_prediction_dir / f"{split}_domain_policy_predictions.csv", rows)
            best_split_metrics[split] = {
                "metrics": metrics,
                "row_summary": summarize_rows(rows),
            }

    summary = {
        "artifact_type": "task2_stage2aq_phantom_class1_multisource_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_dir": args.baseline_dir.as_posix(),
        "multisource_run_dir": args.multisource_run_dir.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "optimize_split": args.optimize_split,
        "source_policies": {name: sorted(values) for name, values in SOURCE_POLICIES.items()},
        "score_modes": list(SCORE_MODES),
        "topk_values": list(TOPK_VALUES),
        "scale_values": list(SCALE_VALUES),
        "best_policy": {
            key: best[key]
            for key in ("source_policy", "score_mode", "topk", "scale")
        },
        "best_optimize_metrics": {
            "phantom_class1_ap50": best["optimize_phantom_class1_ap50"],
            "phantom_class1_ap75": best["optimize_phantom_class1_ap75"],
        },
        "best_split_metrics": best_split_metrics,
        "top_policies": policies[: int(args.max_export_rank)],
        "prediction_dir": best_prediction_dir.as_posix() if not args.skip_full_metrics else None,
    }
    output_json = args.output_dir / "stage2aq_sweep_summary.json"
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
