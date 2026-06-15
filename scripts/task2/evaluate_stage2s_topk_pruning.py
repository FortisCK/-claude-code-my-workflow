#!/usr/bin/env python3
"""Evaluate per-frame top-k output-control policies for Task 2 ranker outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import repo_relative  # noqa: E402
from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth,
    DetectionPrediction,
    compute_detection_map,
)


SCORE_MODES = (
    "roi",
    "bg_suppressed_roi",
    "source_roi",
    "sqrt_source_roi",
    "rank_decay_roi",
    "source_rank_decay_roi",
)
POLICIES = (
    "all",
    "candidate_top",
    "prediction_top",
    "class_top",
    "source_candidate_top",
    "source_class_top",
    "oracle_candidate_top",
)


@dataclass(frozen=True)
class CandidateRecord:
    row_index: int
    split: str
    sample_id: str
    source: str
    source_rank: int
    gt_class: int
    gt_xyxy: tuple[float, float, float, float]
    xyxy: tuple[float, float, float, float]
    gt_iou: float


@dataclass(frozen=True)
class ScoredPrediction:
    row_index: int
    sample_id: str
    source: str
    class_id: int
    score: float
    xyxy: tuple[float, float, float, float]
    gt_iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/stage2s_topk_diagnostic"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--split", action="append", default=None, help="Split name. Defaults to valid_combined/phantom/animal.")
    parser.add_argument("--score-mode", action="append", default=None, choices=SCORE_MODES)
    parser.add_argument("--policy", action="append", default=None, choices=POLICIES)
    parser.add_argument("--k", action="append", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    name = args.name or run_dir.name
    output_dir = resolve_path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = args.split or ["valid_combined", "valid_phantom", "valid_animal"]
    score_modes = tuple(args.score_mode or SCORE_MODES)
    policies = tuple(args.policy or POLICIES)
    k_values = tuple(args.k or [1, 2, 3, 5, 10, 20, 50])

    all_rows: list[dict[str, Any]] = []
    for split in splits:
        candidates, prediction_rows = load_split_rows(run_dir, split)
        ground_truths = build_ground_truths(candidates)
        for score_mode in score_modes:
            candidate_scores, predictions = build_scored_predictions(
                candidates,
                prediction_rows,
                score_mode=score_mode,
            )
            for policy in policies:
                policy_k_values = [0] if policy == "all" else list(k_values)
                for k in policy_k_values:
                    selected = select_predictions(
                        candidates=candidates,
                        predictions=predictions,
                        candidate_scores=candidate_scores,
                        policy=policy,
                        k=k,
                    )
                    metrics = compute_detection_map(ground_truths, to_detection_predictions(selected), class_ids=(0, 1))
                    loc = summarize_localization(candidates, selected)
                    row = {
                        "split": split,
                        "score_mode": score_mode,
                        "policy": policy,
                        "k": "" if policy == "all" else k,
                        "mAP50": metrics["mAP50"],
                        "mAP50_95": metrics["mAP50-95"],
                        "class0_ap50": metrics["classes"]["0"]["ap50"],
                        "class0_ap50_95": metrics["classes"]["0"]["ap50_95"],
                        "class1_ap50": metrics["classes"]["1"]["ap50"],
                        "class1_ap50_95": metrics["classes"]["1"]["ap50_95"],
                        "num_predictions": len(selected),
                        **loc,
                    }
                    all_rows.append(row)
                    print(
                        f"{split} {score_mode} {policy}"
                        f"{'' if policy == 'all' else f'@{k}'} "
                        f"mAP50={float(row['mAP50']):.4f} "
                        f"mAP50-95={float(row['mAP50_95']):.4f} "
                        f"locR75={float(row['loc_recall_iou_0.75']):.4f}",
                        flush=True,
                    )

    write_csv(output_dir / "stage2s_topk_summary.csv", all_rows)
    write_json(output_dir / "stage2s_topk_summary.json", {"rows": all_rows})
    write_markdown(output_dir / "stage2s_topk_report.md", all_rows, run_dir=run_dir)
    print(f"Saved Stage2S top-k diagnostic to {output_dir}", flush=True)
    return 0


def load_split_rows(run_dir: Path, split: str) -> tuple[list[CandidateRecord], list[dict[str, str]]]:
    candidate_path = run_dir / f"{split}_candidates_used.csv"
    prediction_path = run_dir / f"{split}_eval_prediction_rows.csv"
    candidates_raw = read_csv(candidate_path)
    prediction_rows = read_csv(prediction_path)
    if len(candidates_raw) != len(prediction_rows):
        raise ValueError(
            f"{split}: candidate/prediction row count mismatch: "
            f"{len(candidates_raw)} vs {len(prediction_rows)}"
        )
    candidates: list[CandidateRecord] = []
    for row_index, (candidate_raw, prediction_raw) in enumerate(zip(candidates_raw, prediction_rows)):
        assert_aligned(split, row_index, candidate_raw, prediction_raw)
        candidates.append(
            CandidateRecord(
                row_index=row_index,
                split=first_non_empty(candidate_raw, "split", default=split),
                sample_id=first_non_empty(candidate_raw, "sample_id"),
                source=first_non_empty(candidate_raw, "source"),
                source_rank=int(float(first_non_empty(candidate_raw, "source_rank", default="9999"))),
                gt_class=int(float(first_non_empty(candidate_raw, "gt_class"))),
                gt_xyxy=parse_xyxy(candidate_raw, "gt_x1", "gt_y1", "gt_x2", "gt_y2"),
                xyxy=parse_candidate_xyxy(candidate_raw),
                gt_iou=float(first_non_empty(candidate_raw, "gt_iou", "candidate_iou", default="0")),
            )
        )
    return candidates, prediction_rows


def assert_aligned(split: str, row_index: int, candidate: dict[str, str], prediction: dict[str, str]) -> None:
    checks = [
        ("sample_id", first_non_empty(candidate, "sample_id"), first_non_empty(prediction, "sample_id")),
        ("source", first_non_empty(candidate, "source"), first_non_empty(prediction, "source")),
        (
            "source_rank",
            str(int(float(first_non_empty(candidate, "source_rank", default="9999")))),
            str(int(float(first_non_empty(prediction, "source_rank", default="9999")))),
        ),
    ]
    for name, left, right in checks:
        if left != right:
            raise ValueError(f"{split} row {row_index}: {name} mismatch: {left!r} vs {right!r}")
    candidate_iou = float(first_non_empty(candidate, "gt_iou", "candidate_iou", default="0"))
    prediction_iou = float(first_non_empty(prediction, "gt_iou", default="0"))
    if abs(candidate_iou - prediction_iou) > 1e-6:
        raise ValueError(f"{split} row {row_index}: gt_iou mismatch: {candidate_iou} vs {prediction_iou}")


def build_ground_truths(candidates: list[CandidateRecord]) -> list[DetectionGroundTruth]:
    by_sample: dict[str, CandidateRecord] = {}
    for candidate in candidates:
        previous = by_sample.get(candidate.sample_id)
        if previous is None:
            by_sample[candidate.sample_id] = candidate
        elif previous.gt_class != candidate.gt_class or previous.gt_xyxy != candidate.gt_xyxy:
            raise ValueError(f"Inconsistent GT metadata for sample {candidate.sample_id}")
    return [
        DetectionGroundTruth(item.sample_id, item.gt_class, item.gt_xyxy)
        for item in sorted(by_sample.values(), key=lambda candidate: candidate.sample_id)
    ]


def build_scored_predictions(
    candidates: list[CandidateRecord],
    prediction_rows: list[dict[str, str]],
    *,
    score_mode: str,
) -> tuple[dict[int, float], list[ScoredPrediction]]:
    predictions: list[ScoredPrediction] = []
    candidate_scores: dict[int, float] = {}
    for candidate, row in zip(candidates, prediction_rows):
        normal_score = float(first_non_empty(row, f"{score_mode}_score_normal"))
        collision_score = float(first_non_empty(row, f"{score_mode}_score_collision"))
        candidate_scores[candidate.row_index] = max(normal_score, collision_score)
        predictions.append(
            ScoredPrediction(
                row_index=candidate.row_index,
                sample_id=candidate.sample_id,
                source=candidate.source,
                class_id=0,
                score=normal_score,
                xyxy=candidate.xyxy,
                gt_iou=candidate.gt_iou,
            )
        )
        predictions.append(
            ScoredPrediction(
                row_index=candidate.row_index,
                sample_id=candidate.sample_id,
                source=candidate.source,
                class_id=1,
                score=collision_score,
                xyxy=candidate.xyxy,
                gt_iou=candidate.gt_iou,
            )
        )
    return candidate_scores, predictions


def select_predictions(
    *,
    candidates: list[CandidateRecord],
    predictions: list[ScoredPrediction],
    candidate_scores: dict[int, float],
    policy: str,
    k: int,
) -> list[ScoredPrediction]:
    if policy == "all":
        return list(predictions)
    if k <= 0:
        raise ValueError(f"k must be positive for {policy}, got {k}")

    if policy == "candidate_top":
        selected_candidates = select_candidate_ids(candidates, candidate_scores, k=k, group_key=lambda item: item.sample_id)
        return [prediction for prediction in predictions if prediction.row_index in selected_candidates]
    if policy == "source_candidate_top":
        selected_candidates = select_candidate_ids(
            candidates,
            candidate_scores,
            k=k,
            group_key=lambda item: f"{item.sample_id}:{item.source}",
        )
        return [prediction for prediction in predictions if prediction.row_index in selected_candidates]
    if policy == "oracle_candidate_top":
        oracle_scores = {candidate.row_index: candidate.gt_iou for candidate in candidates}
        selected_candidates = select_candidate_ids(candidates, oracle_scores, k=k, group_key=lambda item: item.sample_id)
        return [prediction for prediction in predictions if prediction.row_index in selected_candidates]
    if policy == "prediction_top":
        return select_prediction_rows(predictions, k=k, group_key=lambda item: item.sample_id)
    if policy == "class_top":
        return select_prediction_rows(predictions, k=k, group_key=lambda item: f"{item.sample_id}:{item.class_id}")
    if policy == "source_class_top":
        return select_prediction_rows(
            predictions,
            k=k,
            group_key=lambda item: f"{item.sample_id}:{item.source}:{item.class_id}",
        )
    raise ValueError(f"Unknown policy: {policy}")


def select_candidate_ids(
    candidates: list[CandidateRecord],
    scores: dict[int, float],
    *,
    k: int,
    group_key: Any,
) -> set[int]:
    grouped: dict[str, list[CandidateRecord]] = {}
    for candidate in candidates:
        grouped.setdefault(str(group_key(candidate)), []).append(candidate)
    selected: set[int] = set()
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: (-scores[item.row_index], item.source_rank, item.source, item.row_index),
        )
        selected.update(item.row_index for item in ordered[:k])
    return selected


def select_prediction_rows(
    predictions: list[ScoredPrediction],
    *,
    k: int,
    group_key: Any,
) -> list[ScoredPrediction]:
    grouped: dict[str, list[ScoredPrediction]] = {}
    for prediction in predictions:
        grouped.setdefault(str(group_key(prediction)), []).append(prediction)
    selected: list[ScoredPrediction] = []
    for group in grouped.values():
        selected.extend(
            sorted(group, key=lambda item: (-item.score, item.class_id, item.source, item.row_index))[:k]
        )
    selected.sort(key=lambda item: (item.sample_id, item.class_id, -item.score, item.row_index))
    return selected


def summarize_localization(
    candidates: list[CandidateRecord],
    predictions: list[ScoredPrediction],
) -> dict[str, float | int]:
    sample_ids = sorted({candidate.sample_id for candidate in candidates})
    selected_ids = {prediction.row_index for prediction in predictions}
    candidate_by_id = {candidate.row_index: candidate for candidate in candidates}
    retained_by_sample: dict[str, list[CandidateRecord]] = {sample_id: [] for sample_id in sample_ids}
    for row_index in selected_ids:
        candidate = candidate_by_id[row_index]
        retained_by_sample[candidate.sample_id].append(candidate)
    best_ious = [
        max((candidate.gt_iou for candidate in retained_by_sample[sample_id]), default=0.0)
        for sample_id in sample_ids
    ]
    result: dict[str, float | int] = {
        "retained_candidates": len(selected_ids),
        "mean_retained_candidates_per_sample": len(selected_ids) / max(len(sample_ids), 1),
        "loc_mean_best_iou": float(np.mean(best_ious)) if best_ious else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        result[f"loc_recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in best_ious])) if best_ious else float("nan")
        )
    return result


def to_detection_predictions(predictions: Iterable[ScoredPrediction]) -> list[DetectionPrediction]:
    return [
        DetectionPrediction(prediction.sample_id, prediction.class_id, prediction.score, prediction.xyxy)
        for prediction in predictions
    ]


def write_markdown(path: Path, rows: list[dict[str, Any]], *, run_dir: Path) -> None:
    combined_rows = [row for row in rows if row["split"] == "valid_combined"]
    best_non_oracle = sorted(
        [row for row in combined_rows if row["policy"] != "oracle_candidate_top"],
        key=lambda row: float(row["mAP50_95"]),
        reverse=True,
    )[:15]
    best_oracle = sorted(
        [row for row in combined_rows if row["policy"] == "oracle_candidate_top"],
        key=lambda row: float(row["mAP50_95"]),
        reverse=True,
    )[:10]
    lines = [
        "# Task2 Stage2S Top-k Diagnostic",
        "",
        f"Run dir: `{repo_relative(run_dir, REPO_ROOT)}`",
        "",
        "## Best Non-Oracle Policies on valid_combined",
        "",
        "| Rank | Score mode | Policy | k | mAP50 | mAP50-95 | loc R@0.75 | predictions |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(best_non_oracle, start=1):
        lines.append(format_markdown_row(rank, row))
    lines.extend(
        [
            "",
            "## Oracle Candidate-Selection Upper Bound on valid_combined",
            "",
            "| Rank | Score mode | Policy | k | mAP50 | mAP50-95 | loc R@0.75 | predictions |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for rank, row in enumerate(best_oracle, start=1):
        lines.append(format_markdown_row(rank, row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_markdown_row(rank: int, row: dict[str, Any]) -> str:
    return (
        f"| {rank} | {row['score_mode']} | {row['policy']} | {row['k']} | "
        f"{float(row['mAP50']):.4f} | {float(row['mAP50_95']):.4f} | "
        f"{float(row['loc_recall_iou_0.75']):.4f} | {int(row['num_predictions'])} |"
    )


def parse_candidate_xyxy(raw: dict[str, str]) -> tuple[float, float, float, float]:
    if first_non_empty(raw, "x1", default="") != "":
        return parse_xyxy(raw, "x1", "y1", "x2", "y2")
    return parse_xyxy(raw, "candidate_x1", "candidate_y1", "candidate_x2", "candidate_y2")


def parse_xyxy(raw: dict[str, str], x1: str, y1: str, x2: str, y2: str) -> tuple[float, float, float, float]:
    return (
        float(first_non_empty(raw, x1)),
        float(first_non_empty(raw, y1)),
        float(first_non_empty(raw, x2)),
        float(first_non_empty(raw, y2)),
    )


def first_non_empty(raw: dict[str, str], *keys: str, default: str | None = None) -> str:
    for key in keys:
        value = raw.get(key, "")
        if value != "":
            return value
    if default is not None:
        return default
    raise KeyError(f"None of the requested keys are present/non-empty: {keys}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value, REPO_ROOT)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
