#!/usr/bin/env python3
"""Sweep domain-aware score policies on optionally calibrated Task2 boxes."""

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

from cathaction.metrics.detection import DetectionGroundTruth, DetectionPrediction, compute_class_ap, compute_detection_map
from scripts.task2.export_clean_predictions import CLEAN_FIELDS, write_rows
from scripts.task2.export_stage2ab_domain_policy_predictions import (
    CLASS_SUFFIX,
    DEFAULT_SPLITS,
    apply_box_transform,
    load_box_transforms,
)
from scripts.task2.sweep_stage2v_domain_policy import DEFAULT_SCORE_MODES


DOMAINS = ("phantom", "animal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--box-transform-json", type=Path, default=None)
    parser.add_argument("--optimize-split", default="valid_combined")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--score-mode", action="append", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rows(run_dir: Path, split: str) -> tuple[list[DetectionGroundTruth], list[tuple[dict[str, str], dict[str, str]]]]:
    candidates = read_csv(run_dir / f"{split}_candidates_used.csv")
    predictions = read_csv(run_dir / f"{split}_eval_prediction_rows.csv")
    if len(candidates) != len(predictions):
        raise ValueError(f"{split}: row count mismatch: {len(candidates)} vs {len(predictions)}")
    for row_index, (candidate, prediction) in enumerate(zip(candidates, predictions), start=2):
        for field in ("sample_id", "source", "source_rank"):
            if str(candidate.get(field, "")) != str(prediction.get(field, "")):
                raise ValueError(
                    f"{split}: alignment mismatch row {row_index} for {field}: "
                    f"{candidate.get(field)!r} vs {prediction.get(field)!r}"
                )
    ground_truths: list[DetectionGroundTruth] = []
    seen: set[str] = set()
    for candidate in candidates:
        sample_id = candidate["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        ground_truths.append(
            DetectionGroundTruth(
                sample_id=sample_id,
                class_id=int(candidate["gt_class"]),
                box_xyxy=(
                    float(candidate["gt_x1"]),
                    float(candidate["gt_y1"]),
                    float(candidate["gt_x2"]),
                    float(candidate["gt_y2"]),
                ),
            )
        )
    return ground_truths, list(zip(candidates, predictions))


def available_score_modes(rows: list[tuple[dict[str, str], dict[str, str]]]) -> list[str]:
    if not rows:
        return []
    fields = set(rows[0][1])
    modes: list[str] = []
    for mode in DEFAULT_SCORE_MODES:
        if all(f"{mode}_score_{suffix}" in fields for suffix in CLASS_SUFFIX.values()):
            modes.append(mode)
    return modes


def policy_domain(domain: str) -> str:
    return domain if domain in DOMAINS else "animal"


def transformed_box(
    candidate: dict[str, str],
    *,
    class_id: int,
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> tuple[float, float, float, float]:
    domain = policy_domain(candidate.get("domain", "phantom"))
    transform = box_transforms.get(domain, {}).get(str(class_id))
    if transform is None:
        transform = {"dx_center": 0.0, "dy_center": 0.0, "log_w": 0.0, "log_h": 0.0}
    return apply_box_transform(candidate, transform)


def class_predictions_for_domain_policy(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    policy: dict[str, str],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> list[DetectionPrediction]:
    suffix = CLASS_SUFFIX[class_id]
    predictions: list[DetectionPrediction] = []
    for candidate, prediction in rows:
        domain = policy_domain(candidate.get("domain", "phantom"))
        score_mode = policy.get(domain, policy.get("default", "roi"))
        score_key = f"{score_mode}_score_{suffix}"
        predictions.append(
            DetectionPrediction(
                sample_id=candidate["sample_id"],
                class_id=class_id,
                score=float(prediction[score_key]),
                box_xyxy=transformed_box(candidate, class_id=class_id, box_transforms=box_transforms),
            )
        )
    return predictions


def prediction_rows_for_domain_policy(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    policy: dict[str, str],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    suffix = CLASS_SUFFIX[class_id]
    result: list[dict[str, Any]] = []
    for candidate, prediction in rows:
        domain = policy_domain(candidate.get("domain", "phantom"))
        score_mode = policy.get(domain, policy.get("default", "roi"))
        score_key = f"{score_mode}_score_{suffix}"
        x1, y1, x2, y2 = transformed_box(candidate, class_id=class_id, box_transforms=box_transforms)
        result.append(
            {
                "sample_id": candidate["sample_id"],
                "video_id": candidate.get("video_id", ""),
                "frame_index": candidate.get("frame_index", ""),
                "domain": candidate.get("domain", domain),
                "class_id": class_id,
                "score": float(prediction[score_key]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
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
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for phantom_mode in score_modes:
        for animal_mode in score_modes:
            policy = {"phantom": phantom_mode, "animal": animal_mode, "default": phantom_mode}
            predictions = class_predictions_for_domain_policy(
                rows,
                class_id=class_id,
                policy=policy,
                box_transforms=box_transforms,
            )
            ap_values = [
                compute_class_ap(
                    ground_truths,
                    predictions,
                    class_id=class_id,
                    iou_threshold=threshold,
                )["ap"]
                for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
            ]
            row = {
                "class_id": class_id,
                "policy": policy,
                "ap50": float(ap_values[0]),
                "ap50_95": float(sum(float(value) for value in ap_values) / len(ap_values)),
            }
            if best is None or float(row["ap50_95"]) > float(best["ap50_95"]):
                best = row
    if best is None:
        raise ValueError("No class policy was evaluated")
    return best


def evaluate_policy(
    run_dir: Path,
    split: str,
    *,
    class_policies: dict[int, dict[str, str]],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ground_truths, rows = load_rows(run_dir, split)
    predictions = [
        *class_predictions_for_domain_policy(
            rows, class_id=0, policy=class_policies[0], box_transforms=box_transforms
        ),
        *class_predictions_for_domain_policy(
            rows, class_id=1, policy=class_policies[1], box_transforms=box_transforms
        ),
    ]
    prediction_rows = [
        *prediction_rows_for_domain_policy(
            rows, class_id=0, policy=class_policies[0], box_transforms=box_transforms
        ),
        *prediction_rows_for_domain_policy(
            rows, class_id=1, policy=class_policies[1], box_transforms=box_transforms
        ),
    ]
    return compute_detection_map(ground_truths, predictions, class_ids=(0, 1)), prediction_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CLEAN_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    box_transforms = load_box_transforms(args.box_transform_json)
    optimize_ground_truths, optimize_rows = load_rows(args.run_dir, args.optimize_split)
    score_modes = args.score_mode or available_score_modes(optimize_rows)
    if not score_modes:
        raise ValueError("No score modes available")

    best_class = {
        class_id: search_class_policy(
            optimize_ground_truths,
            optimize_rows,
            class_id=class_id,
            score_modes=list(score_modes),
            box_transforms=box_transforms,
        )
        for class_id in (0, 1)
    }
    class_policies = {class_id: dict(best_class[class_id]["policy"]) for class_id in (0, 1)}

    split_metrics: dict[str, Any] = {}
    for split in args.splits:
        metrics, rows = evaluate_policy(
            args.run_dir,
            split,
            class_policies=class_policies,
            box_transforms=box_transforms,
        )
        split_metrics[split] = {"detection": metrics, "prediction_rows": len(rows)}
        if args.prediction_dir is not None:
            write_csv(args.prediction_dir / f"{split}_domain_policy_predictions.csv", rows)

    result = {
        "artifact_type": "task2_stage2ae_calibrated_policy_sweep",
        "run_dir": args.run_dir.as_posix(),
        "box_transform_json": args.box_transform_json.as_posix()
        if args.box_transform_json is not None
        else None,
        "optimize_split": args.optimize_split,
        "score_modes": list(score_modes),
        "best_class_policies": best_class,
        "metrics": split_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

