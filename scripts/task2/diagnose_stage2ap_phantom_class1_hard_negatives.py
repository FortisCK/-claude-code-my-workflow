#!/usr/bin/env python3
"""Diagnose phantom class1 hard negatives in the Task2 champion pipeline.

This is a public-validation diagnostic. It requires candidate rows with ground
truth fields and must not be used as a hidden-test inference component.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import iou_xyxy  # noqa: E402


DEFAULT_RUN_DIR = Path("outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval")
DEFAULT_PREDICTION_DIR = Path(
    "outputs/task2/stage2ao_champion_pipeline_with_an/public_valid_ab_ad_ae_ai_an/stage2ae"
)
DEFAULT_OUTPUT_DIR = Path("outputs/task2/stage2ap_phantom_class1_hard_negatives/stage2ae_public_valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--split", default="valid_phantom")
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=200)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def fnum(value: str | float | int | None, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def box_from_row(row: dict[str, str], prefix: str = "") -> tuple[float, float, float, float]:
    return (
        fnum(row[f"{prefix}x1"]),
        fnum(row[f"{prefix}y1"]),
        fnum(row[f"{prefix}x2"]),
        fnum(row[f"{prefix}y2"]),
    )


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def score_bucket(score: float) -> str:
    if score >= 0.75:
        return ">=0.75"
    if score >= 0.50:
        return "0.50-0.75"
    if score >= 0.25:
        return "0.25-0.50"
    if score >= 0.10:
        return "0.10-0.25"
    return "<0.10"


def iou_bucket(iou: float) -> str:
    if iou >= 0.75:
        return ">=0.75"
    if iou >= 0.50:
        return "0.50-0.75"
    if iou >= 0.25:
        return "0.25-0.50"
    if iou > 0.0:
        return "0.00-0.25"
    return "0.00"


def align_candidate_prediction_rows(
    candidates: list[dict[str, str]],
    predictions: list[dict[str, str]],
    *,
    class_id: int,
) -> list[dict[str, Any]]:
    class_predictions = [row for row in predictions if int(row["class_id"]) == class_id]
    if len(class_predictions) != len(candidates):
        raise ValueError(
            f"class {class_id}: expected {len(candidates)} prediction rows, got {len(class_predictions)}"
        )

    joined: list[dict[str, Any]] = []
    for row_index, (candidate, prediction) in enumerate(zip(candidates, class_predictions), start=2):
        for key in ("sample_id", "source", "source_rank"):
            if str(candidate.get(key, "")) != str(prediction.get(key, "")):
                raise ValueError(
                    f"candidate/prediction alignment mismatch at row {row_index} for {key}: "
                    f"{candidate.get(key)!r} vs {prediction.get(key)!r}"
                )
        gt_class = int(candidate["gt_class"])
        pred_box = box_from_row(prediction)
        gt_box = box_from_row(candidate, "gt_")
        overlap = iou_xyxy(pred_box, gt_box) if gt_class == class_id else 0.0
        joined.append(
            {
                "sample_id": candidate["sample_id"],
                "video_id": candidate.get("video_id", ""),
                "frame_index": candidate.get("frame_index", ""),
                "image_path": candidate.get("image_path", ""),
                "domain": candidate.get("domain", ""),
                "gt_class": gt_class,
                "target_class": class_id,
                "score": fnum(prediction["score"]),
                "score_mode": prediction.get("score_mode", ""),
                "policy_name": prediction.get("policy_name", ""),
                "source": candidate.get("source", ""),
                "source_rank": int(fnum(candidate.get("source_rank", "0"))),
                "source_conf": fnum(candidate.get("source_conf")),
                "raw_gt_iou": fnum(candidate.get("gt_iou")),
                "final_iou_to_gt": overlap,
                "candidate_x1": fnum(candidate["x1"]),
                "candidate_y1": fnum(candidate["y1"]),
                "candidate_x2": fnum(candidate["x2"]),
                "candidate_y2": fnum(candidate["y2"]),
                "pred_x1": fnum(prediction["x1"]),
                "pred_y1": fnum(prediction["y1"]),
                "pred_x2": fnum(prediction["x2"]),
                "pred_y2": fnum(prediction["y2"]),
                "gt_x1": fnum(candidate["gt_x1"]),
                "gt_y1": fnum(candidate["gt_y1"]),
                "gt_x2": fnum(candidate["gt_x2"]),
                "gt_y2": fnum(candidate["gt_y2"]),
            }
        )
    return joined


def per_sample_diagnostics(rows: list[dict[str, Any]], *, class_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_sample[str(row["sample_id"])].append(row)

    sample_rows: list[dict[str, Any]] = []
    positive_samples = 0
    oracle_iou50 = 0
    oracle_iou75 = 0
    top1_iou50 = 0
    top1_iou75 = 0
    rank_gaps: list[int] = []
    score_margins: list[float] = []

    for sample_id, sample_candidates in rows_by_sample.items():
        gt_class = int(sample_candidates[0]["gt_class"])
        ranked = sorted(sample_candidates, key=lambda item: float(item["score"]), reverse=True)
        top = ranked[0]
        target_candidates = [row for row in ranked if gt_class == class_id]
        best = max(target_candidates, key=lambda item: float(item["final_iou_to_gt"])) if target_candidates else None
        best_iou = float(best["final_iou_to_gt"]) if best is not None else 0.0
        top_iou = float(top["final_iou_to_gt"]) if gt_class == class_id else 0.0
        best_tp_rank = None
        best_tp_score = None
        for rank, row in enumerate(ranked, start=1):
            if gt_class == class_id and float(row["final_iou_to_gt"]) >= 0.50:
                best_tp_rank = rank
                best_tp_score = float(row["score"])
                break
        if gt_class == class_id:
            positive_samples += 1
            oracle_iou50 += int(best_iou >= 0.50)
            oracle_iou75 += int(best_iou >= 0.75)
            top1_iou50 += int(top_iou >= 0.50)
            top1_iou75 += int(top_iou >= 0.75)
            if best_tp_rank is not None:
                rank_gaps.append(int(best_tp_rank) - 1)
                score_margins.append(float(top["score"]) - float(best_tp_score))
        sample_rows.append(
            {
                "sample_id": sample_id,
                "video_id": top["video_id"],
                "frame_index": top["frame_index"],
                "image_path": top["image_path"],
                "gt_class": gt_class,
                "num_candidates": len(sample_candidates),
                "top_score": float(top["score"]),
                "top_source": top["source"],
                "top_source_rank": top["source_rank"],
                "top_iou": top_iou,
                "best_iou": best_iou,
                "best_source": best["source"] if best is not None else "",
                "best_source_rank": best["source_rank"] if best is not None else "",
                "best_score": float(best["score"]) if best is not None else 0.0,
                "best_tp_rank": best_tp_rank if best_tp_rank is not None else "",
                "best_tp_score": best_tp_score if best_tp_score is not None else "",
                "top_minus_best_tp_score": (
                    float(top["score"]) - float(best_tp_score) if best_tp_score is not None else ""
                ),
            }
        )

    summary = {
        "num_samples": len(rows_by_sample),
        "num_positive_samples": positive_samples,
        "oracle_iou50_recall": oracle_iou50 / positive_samples if positive_samples else None,
        "oracle_iou75_recall": oracle_iou75 / positive_samples if positive_samples else None,
        "top1_iou50_recall": top1_iou50 / positive_samples if positive_samples else None,
        "top1_iou75_recall": top1_iou75 / positive_samples if positive_samples else None,
        "rank_gap_for_first_iou50_candidate": numeric_summary(rank_gaps),
        "top_score_minus_first_iou50_score": numeric_summary(score_margins),
    }
    return summary, sample_rows


def row_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "source": counter_dict(Counter(str(row["source"]) for row in rows)),
        "score_mode": counter_dict(Counter(str(row["score_mode"]) for row in rows)),
        "gt_class": counter_dict(Counter(str(row["gt_class"]) for row in rows)),
        "score_bucket": counter_dict(Counter(score_bucket(float(row["score"])) for row in rows)),
        "iou_bucket": counter_dict(Counter(iou_bucket(float(row["final_iou_to_gt"])) for row in rows)),
    }


def main() -> int:
    args = parse_args()
    candidates_path = args.run_dir / f"{args.split}_candidates_used.csv"
    predictions_path = args.prediction_dir / f"{args.split}_domain_policy_predictions.csv"
    candidates = read_csv(candidates_path)
    predictions = read_csv(predictions_path)
    joined = align_candidate_prediction_rows(candidates, predictions, class_id=int(args.class_id))

    positive_rows = [row for row in joined if int(row["gt_class"]) == int(args.class_id)]
    true_candidate_rows = [row for row in positive_rows if float(row["final_iou_to_gt"]) >= 0.50]
    near_miss_rows = [
        row for row in positive_rows if 0.25 <= float(row["final_iou_to_gt"]) < 0.50
    ]
    false_positive_rows = [
        row for row in joined if not (int(row["gt_class"]) == int(args.class_id) and float(row["final_iou_to_gt"]) >= 0.50)
    ]
    high_score_false_positive_rows = sorted(
        false_positive_rows,
        key=lambda item: float(item["score"]),
        reverse=True,
    )[: int(args.max_examples)]

    sample_summary, sample_rows = per_sample_diagnostics(joined, class_id=int(args.class_id))
    positive_sample_rows = [row for row in sample_rows if int(row["gt_class"]) == int(args.class_id)]
    low_rank_true_positive_samples = sorted(
        [
            row
            for row in positive_sample_rows
            if row["best_tp_rank"] != "" and int(row["best_tp_rank"]) > 1
        ],
        key=lambda item: (int(item["best_tp_rank"]), -float(item["top_score"])),
        reverse=True,
    )[: int(args.max_examples)]
    missed_positive_samples = sorted(
        [
            row
            for row in positive_sample_rows
            if float(row["best_iou"]) < 0.50
        ],
        key=lambda item: float(item["best_iou"]),
    )[: int(args.max_examples)]

    top_rows_by_sample = []
    for row in sorted(joined, key=lambda item: (str(item["sample_id"]), -float(item["score"]))):
        if top_rows_by_sample and top_rows_by_sample[-1]["sample_id"] == row["sample_id"]:
            continue
        top_rows_by_sample.append(row)

    summary = {
        "artifact_type": "task2_stage2ap_phantom_class1_hard_negative_diagnostic",
        "run_dir": args.run_dir.as_posix(),
        "prediction_dir": args.prediction_dir.as_posix(),
        "split": args.split,
        "class_id": int(args.class_id),
        "candidate_rows": len(candidates),
        "prediction_rows_for_class": len(joined),
        "sample_summary": sample_summary,
        "row_distributions": {
            "all_class1_predictions": row_distribution(joined),
            "true_candidates_iou50": row_distribution(true_candidate_rows),
            "near_misses_iou25_50": row_distribution(near_miss_rows),
            "false_positive_candidates": row_distribution(false_positive_rows),
            "high_score_false_positive_top_examples": row_distribution(high_score_false_positive_rows),
            "top1_by_sample": row_distribution(top_rows_by_sample),
        },
        "output_files": {
            "summary_json": (args.output_dir / "stage2ap_summary.json").as_posix(),
            "high_score_false_positives_csv": (
                args.output_dir / "high_score_false_positives.csv"
            ).as_posix(),
            "low_rank_true_positive_samples_csv": (
                args.output_dir / "low_rank_true_positive_samples.csv"
            ).as_posix(),
            "missed_positive_samples_csv": (
                args.output_dir / "missed_positive_samples.csv"
            ).as_posix(),
        },
    }

    candidate_fields = [
        "sample_id",
        "video_id",
        "frame_index",
        "image_path",
        "gt_class",
        "score",
        "score_mode",
        "source",
        "source_rank",
        "source_conf",
        "raw_gt_iou",
        "final_iou_to_gt",
        "candidate_x1",
        "candidate_y1",
        "candidate_x2",
        "candidate_y2",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
    ]
    sample_fields = [
        "sample_id",
        "video_id",
        "frame_index",
        "image_path",
        "gt_class",
        "num_candidates",
        "top_score",
        "top_source",
        "top_source_rank",
        "top_iou",
        "best_iou",
        "best_source",
        "best_source_rank",
        "best_score",
        "best_tp_rank",
        "best_tp_score",
        "top_minus_best_tp_score",
    ]
    write_json(args.output_dir / "stage2ap_summary.json", summary)
    write_csv(
        args.output_dir / "high_score_false_positives.csv",
        high_score_false_positive_rows,
        candidate_fields,
    )
    write_csv(
        args.output_dir / "low_rank_true_positive_samples.csv",
        low_rank_true_positive_samples,
        sample_fields,
    )
    write_csv(
        args.output_dir / "missed_positive_samples.csv",
        missed_positive_samples,
        sample_fields,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
