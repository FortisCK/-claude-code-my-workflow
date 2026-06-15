#!/usr/bin/env python3
"""Evaluate source-filter policies from saved Stage2U prediction rows.

This diagnostic is intentionally post-hoc: it does not run the ROI model again.
It reuses ``*_candidates_used.csv`` and ``*_eval_prediction_rows.csv`` emitted by
``train_stage2u_quality_ranker.py`` and recomputes Task2 mAP after filtering or
downweighting candidate sources.

The fast AP path uses the stored candidate-vs-GT IoU. This is exact for the
current Task2 evaluation files because each sample has one GT box and each
candidate row is already matched against that sample's GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CLASS_SPECS = (("normal", 0), ("collision", 1))
IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Stage2U eval/train run directory containing *_candidates_used.csv and *_prediction_rows.csv.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("valid_combined", "valid_phantom", "valid_animal"),
    )
    parser.add_argument("--score-mode", default="blend_rank_decay_roi")
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        help=(
            "Policy spec NAME:source1,source2:refined_mult:topk. "
            "Use '-' for all sources or no topk, e.g. yolo:yolo_stage2l:1:-."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def default_policies() -> list[tuple[str, set[str] | None, float, int | None]]:
    return [
        ("all", None, 1.0, None),
        ("original_sources", {"yolo_stage2l", "task1_geometry_rect"}, 1.0, None),
        ("yolo_only", {"yolo_stage2l"}, 1.0, None),
        ("refined_only", {"stage2p_refined"}, 1.0, None),
        ("all_refined_x0.2", None, 0.2, None),
        ("all_refined_x0.05", None, 0.05, None),
        ("all_refined_x0.2_top50", None, 0.2, 50),
        ("all_refined_x0.05_top50", None, 0.05, 50),
    ]


def parse_policy(spec: str) -> tuple[str, set[str] | None, float, int | None]:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"Policy must have four ':'-separated fields, got {spec!r}")
    name, sources_text, refined_mult_text, topk_text = parts
    sources = None if sources_text == "-" else set(filter(None, sources_text.split(",")))
    refined_mult = float(refined_mult_text)
    topk = None if topk_text == "-" else int(topk_text)
    return name, sources, refined_mult, topk


def load_rows(run_dir: Path, split: str) -> tuple[list[tuple[dict[str, str], dict[str, str]]], dict[int, int]]:
    candidates_path = run_dir / f"{split}_candidates_used.csv"
    predictions_path = run_dir / f"{split}_eval_prediction_rows.csv"
    if not predictions_path.exists():
        predictions_path = run_dir / f"{split}_best_prediction_rows.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    with candidates_path.open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    with predictions_path.open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(candidates) != len(predictions):
        raise ValueError(
            f"{split}: candidate/prediction row count mismatch: "
            f"{len(candidates)} vs {len(predictions)}"
        )

    sample_gt_class: dict[str, int] = {}
    for candidate in candidates:
        sample_gt_class.setdefault(candidate["sample_id"], int(candidate["gt_class"]))
    gt_counts: dict[int, int] = defaultdict(int)
    for class_id in sample_gt_class.values():
        gt_counts[class_id] += 1
    return list(zip(candidates, predictions)), dict(gt_counts)


def ap_101_point(tp_sequence: Iterable[bool], num_gt: int) -> float:
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


def source_multiplier(source: str, refined_mult: float) -> float:
    if source == "stage2p_refined":
        return refined_mult
    return 1.0


def apply_policy(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    source_keep: set[str] | None,
    refined_mult: float,
    topk: int | None,
    score_mode: str,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    if source_keep is not None:
        rows = [(candidate, prediction) for candidate, prediction in rows if candidate["source"] in source_keep]
    if topk is None:
        return rows

    grouped: dict[str, list[tuple[float, dict[str, str], dict[str, str]]]] = defaultdict(list)
    normal_key = f"{score_mode}_score_normal"
    collision_key = f"{score_mode}_score_collision"
    for candidate, prediction in rows:
        multiplier = source_multiplier(candidate["source"], refined_mult)
        score = max(float(prediction[normal_key]), float(prediction[collision_key])) * multiplier
        grouped[candidate["sample_id"]].append((score, candidate, prediction))
    return [
        (candidate, prediction)
        for items in grouped.values()
        for _, candidate, prediction in sorted(items, key=lambda item: item[0], reverse=True)[:topk]
    ]


def evaluate_policy(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    gt_counts: dict[int, int],
    *,
    source_keep: set[str] | None,
    refined_mult: float,
    topk: int | None,
    score_mode: str,
) -> dict[str, object]:
    policy_rows = apply_policy(
        rows,
        source_keep=source_keep,
        refined_mult=refined_mult,
        topk=topk,
        score_mode=score_mode,
    )
    ap_by_threshold: dict[float, list[float]] = {threshold: [] for threshold in IOU_THRESHOLDS}
    class_metrics: dict[str, dict[str, float]] = {}

    for class_name, class_id in CLASS_SPECS:
        score_key = f"{score_mode}_score_{class_name}"
        predictions = sorted(
            (
                float(prediction[score_key]) * source_multiplier(candidate["source"], refined_mult),
                candidate["sample_id"],
                int(candidate["gt_class"]),
                float(candidate["gt_iou"]),
            )
            for candidate, prediction in policy_rows
            if score_key in prediction
        )
        predictions.reverse()

        class_ap_values: list[float] = []
        for threshold in IOU_THRESHOLDS:
            matched: set[str] = set()
            tp_sequence = []
            for _, sample_id, gt_class, gt_iou in predictions:
                is_tp = gt_class == class_id and gt_iou >= threshold and sample_id not in matched
                if is_tp:
                    matched.add(sample_id)
                tp_sequence.append(is_tp)
            ap = ap_101_point(tp_sequence, gt_counts.get(class_id, 0))
            if not math.isnan(ap):
                ap_by_threshold[threshold].append(ap)
                class_ap_values.append(ap)
        class_metrics[str(class_id)] = {
            "ap50": ap_by_threshold[0.5][-1],
            "ap50_95": float(sum(class_ap_values) / len(class_ap_values)) if class_ap_values else float("nan"),
            "num_gt": float(gt_counts.get(class_id, 0)),
        }

    map50_values = [value for value in ap_by_threshold[0.5] if not math.isnan(value)]
    all_values = [
        value
        for values in ap_by_threshold.values()
        for value in values
        if not math.isnan(value)
    ]
    return {
        "mAP50": float(sum(map50_values) / len(map50_values)) if map50_values else float("nan"),
        "mAP50-95": float(sum(all_values) / len(all_values)) if all_values else float("nan"),
        "rows": len(policy_rows),
        "classes": class_metrics,
    }


def main() -> int:
    args = parse_args()
    policies = default_policies()
    policies.extend(parse_policy(spec) for spec in args.policy)

    output: dict[str, object] = {
        "run_dir": args.run_dir.as_posix(),
        "score_mode": args.score_mode,
        "splits": {},
    }
    for split in args.splits:
        rows, gt_counts = load_rows(args.run_dir, split)
        split_results = {}
        for name, sources, refined_mult, topk in policies:
            split_results[name] = evaluate_policy(
                rows,
                gt_counts,
                source_keep=sources,
                refined_mult=refined_mult,
                topk=topk,
                score_mode=args.score_mode,
            )
        output["splits"][split] = split_results

    text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
