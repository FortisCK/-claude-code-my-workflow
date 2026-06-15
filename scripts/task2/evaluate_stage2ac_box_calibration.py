#!/usr/bin/env python3
"""Evaluate train-fitted box calibration on frozen Stage2AB predictions.

This is an internal diagnostic. It learns simple per-domain/per-class box
geometry transforms from train candidates with GT boxes, then evaluates the
frozen Stage2AB score policy on validation candidates after applying those
transforms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import DetectionGroundTruth, DetectionPrediction, compute_detection_map
from scripts.task2.export_stage2ab_domain_policy_predictions import CLASS_SUFFIX
from scripts.task2.verify_stage2ab_domain_policy import EXPECTED_POLICIES


DOMAINS = ("phantom", "animal")
CLASSES = (0, 1)


@dataclass(frozen=True)
class BoxTransform:
    n_fit: int
    dx_center: float
    dy_center: float
    log_w: float
    log_h: float


IDENTITY_TRANSFORM = BoxTransform(n_fit=0, dx_center=0.0, dy_center=0.0, log_w=0.0, log_h=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="valid_combined")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--min-fit-iou",
        type=float,
        default=0.30,
        help="Only train candidates with gt_iou >= this threshold are used to fit transforms.",
    )
    parser.add_argument(
        "--max-scale-log-abs",
        type=float,
        default=0.40,
        help="Clamp absolute log width/height scale to avoid pathological transforms.",
    )
    parser.add_argument(
        "--disable-class-specific",
        action="store_true",
        help="Fit one transform per domain only, shared across classes.",
    )
    parser.add_argument(
        "--calibrate-domain",
        action="append",
        choices=DOMAINS,
        default=None,
        help=(
            "Domain to calibrate. May be repeated. By default all domains are "
            "calibrated. Non-selected domains use identity transforms."
        ),
    )
    parser.add_argument(
        "--transform-strength",
        type=float,
        default=1.0,
        help=(
            "Multiply fitted dx/dy/log-scale corrections by this factor. "
            "1.0 uses the fitted transform; 0.0 is identity."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def box_from_row(row: dict[str, str], prefix: str = "") -> tuple[float, float, float, float]:
    return (
        float(row[f"{prefix}x1"]),
        float(row[f"{prefix}y1"]),
        float(row[f"{prefix}x2"]),
        float(row[f"{prefix}y2"]),
    )


def box_stats(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2), width, height)


def fit_transform(rows: list[dict[str, str]], *, max_scale_log_abs: float) -> BoxTransform:
    if not rows:
        return IDENTITY_TRANSFORM
    dxs: list[float] = []
    dys: list[float] = []
    log_ws: list[float] = []
    log_hs: list[float] = []
    for row in rows:
        cx, cy, width, height = box_stats(box_from_row(row))
        gcx, gcy, gt_width, gt_height = box_stats(
            (
                float(row["gt_x1"]),
                float(row["gt_y1"]),
                float(row["gt_x2"]),
                float(row["gt_y2"]),
            )
        )
        dxs.append((gcx - cx) / width)
        dys.append((gcy - cy) / height)
        log_ws.append(math.log(max(1e-6, gt_width / width)))
        log_hs.append(math.log(max(1e-6, gt_height / height)))
    log_w = max(-max_scale_log_abs, min(max_scale_log_abs, float(median(log_ws))))
    log_h = max(-max_scale_log_abs, min(max_scale_log_abs, float(median(log_hs))))
    return BoxTransform(
        n_fit=len(rows),
        dx_center=float(median(dxs)),
        dy_center=float(median(dys)),
        log_w=log_w,
        log_h=log_h,
    )


def fit_transforms(
    train_rows: list[dict[str, str]],
    *,
    min_fit_iou: float,
    max_scale_log_abs: float,
    class_specific: bool,
    calibrate_domains: set[str] | None = None,
    transform_strength: float = 1.0,
) -> dict[str, dict[str, BoxTransform]]:
    filtered = [
        row
        for row in train_rows
        if row.get("domain") in DOMAINS and float(row.get("gt_iou", 0.0)) >= min_fit_iou
    ]
    transforms: dict[str, dict[str, BoxTransform]] = {}
    for domain in DOMAINS:
        transforms[domain] = {}
        if calibrate_domains is not None and domain not in calibrate_domains:
            for class_id in CLASSES:
                transforms[domain][str(class_id)] = IDENTITY_TRANSFORM
            continue
        if class_specific:
            for class_id in CLASSES:
                rows = [
                    row
                    for row in filtered
                    if row.get("domain") == domain and int(row["gt_class"]) == class_id
                ]
                transforms[domain][str(class_id)] = fit_transform(
                    rows, max_scale_log_abs=max_scale_log_abs
                )
        else:
            rows = [row for row in filtered if row.get("domain") == domain]
            transform = fit_transform(rows, max_scale_log_abs=max_scale_log_abs)
            for class_id in CLASSES:
                transforms[domain][str(class_id)] = transform
    if transform_strength != 1.0:
        transforms = scale_transforms(transforms, transform_strength)
    return transforms


def scale_transform(transform: BoxTransform, strength: float) -> BoxTransform:
    return BoxTransform(
        n_fit=transform.n_fit,
        dx_center=transform.dx_center * strength,
        dy_center=transform.dy_center * strength,
        log_w=transform.log_w * strength,
        log_h=transform.log_h * strength,
    )


def scale_transforms(
    transforms: dict[str, dict[str, BoxTransform]],
    strength: float,
) -> dict[str, dict[str, BoxTransform]]:
    return {
        domain: {
            class_id: scale_transform(transform, strength)
            for class_id, transform in per_class.items()
        }
        for domain, per_class in transforms.items()
    }


def apply_transform(
    row: dict[str, str],
    transform: BoxTransform,
) -> tuple[float, float, float, float]:
    cx, cy, width, height = box_stats(box_from_row(row))
    new_cx = cx + transform.dx_center * width
    new_cy = cy + transform.dy_center * height
    new_w = width * math.exp(transform.log_w)
    new_h = height * math.exp(transform.log_h)
    x1 = new_cx - 0.5 * new_w
    y1 = new_cy - 0.5 * new_h
    x2 = new_cx + 0.5 * new_w
    y2 = new_cy + 0.5 * new_h
    image_width = float(row.get("image_width", 1e9) or 1e9)
    image_height = float(row.get("image_height", 1e9) or 1e9)
    x1 = max(0.0, min(image_width, x1))
    y1 = max(0.0, min(image_height, y1))
    x2 = max(0.0, min(image_width, x2))
    y2 = max(0.0, min(image_height, y2))
    if x2 <= x1:
        x2 = min(image_width, x1 + 1.0)
    if y2 <= y1:
        y2 = min(image_height, y1 + 1.0)
    return (x1, y1, x2, y2)


def load_eval_rows(run_dir: Path, split: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    candidates = read_csv(run_dir / f"{split}_candidates_used.csv")
    predictions = read_csv(run_dir / f"{split}_eval_prediction_rows.csv")
    if len(candidates) != len(predictions):
        raise ValueError(
            f"{split}: row count mismatch: {len(candidates)} candidates vs {len(predictions)} predictions"
        )
    for index, (candidate, prediction) in enumerate(zip(candidates, predictions), start=2):
        for field in ("sample_id", "source", "source_rank"):
            if candidate.get(field, "") != prediction.get(field, ""):
                raise ValueError(
                    f"{split}: alignment mismatch row {index} for {field}: "
                    f"{candidate.get(field)!r} vs {prediction.get(field)!r}"
                )
    return candidates, predictions


def ground_truths_from_candidates(candidates: list[dict[str, str]]) -> list[DetectionGroundTruth]:
    ground_truths: list[DetectionGroundTruth] = []
    seen: set[str] = set()
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
                    float(row["gt_x1"]),
                    float(row["gt_y1"]),
                    float(row["gt_x2"]),
                    float(row["gt_y2"]),
                ),
            )
        )
    return ground_truths


def predictions_for_policy(
    candidates: list[dict[str, str]],
    predictions: list[dict[str, str]],
    transforms: dict[str, dict[str, BoxTransform]] | None = None,
) -> list[DetectionPrediction]:
    output: list[DetectionPrediction] = []
    for class_id in CLASSES:
        suffix = CLASS_SUFFIX[class_id]
        for candidate, prediction in zip(candidates, predictions):
            domain = candidate.get("domain", "phantom")
            policy_domain = domain if domain in EXPECTED_POLICIES[str(class_id)] else "animal"
            score_mode = EXPECTED_POLICIES[str(class_id)][policy_domain]
            score_key = f"{score_mode}_score_{suffix}"
            if score_key not in prediction:
                raise KeyError(score_key)
            transform = (
                transforms.get(policy_domain, {}).get(str(class_id), IDENTITY_TRANSFORM)
                if transforms is not None
                else IDENTITY_TRANSFORM
            )
            output.append(
                DetectionPrediction(
                    sample_id=candidate["sample_id"],
                    class_id=class_id,
                    score=float(prediction[score_key]),
                    box_xyxy=apply_transform(candidate, transform),
                )
            )
    return output


def transform_manifest(transforms: dict[str, dict[str, BoxTransform]]) -> dict[str, dict[str, Any]]:
    return {
        domain: {class_id: asdict(transform) for class_id, transform in per_class.items()}
        for domain, per_class in transforms.items()
    }


def main() -> int:
    args = parse_args()
    train_rows = read_csv(args.run_dir / f"{args.train_split}_candidates_used.csv")
    transforms = fit_transforms(
        train_rows,
        min_fit_iou=float(args.min_fit_iou),
        max_scale_log_abs=float(args.max_scale_log_abs),
        class_specific=not bool(args.disable_class_specific),
        calibrate_domains=set(args.calibrate_domain) if args.calibrate_domain else None,
        transform_strength=float(args.transform_strength),
    )
    candidates, prediction_rows = load_eval_rows(args.run_dir, args.eval_split)
    ground_truths = ground_truths_from_candidates(candidates)
    baseline_predictions = predictions_for_policy(candidates, prediction_rows, transforms=None)
    calibrated_predictions = predictions_for_policy(candidates, prediction_rows, transforms=transforms)
    result = {
        "artifact_type": "task2_stage2ac_box_calibration_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir.as_posix(),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "min_fit_iou": float(args.min_fit_iou),
        "max_scale_log_abs": float(args.max_scale_log_abs),
        "class_specific": not bool(args.disable_class_specific),
        "calibrate_domains": sorted(args.calibrate_domain) if args.calibrate_domain else list(DOMAINS),
        "transform_strength": float(args.transform_strength),
        "frozen_policy": EXPECTED_POLICIES,
        "transforms": transform_manifest(transforms),
        "baseline": compute_detection_map(ground_truths, baseline_predictions, class_ids=CLASSES),
        "calibrated": compute_detection_map(ground_truths, calibrated_predictions, class_ids=CLASSES),
    }
    result["delta"] = {
        "mAP50": float(result["calibrated"]["mAP50"]) - float(result["baseline"]["mAP50"]),
        "mAP50-95": float(result["calibrated"]["mAP50-95"])
        - float(result["baseline"]["mAP50-95"]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["delta"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
