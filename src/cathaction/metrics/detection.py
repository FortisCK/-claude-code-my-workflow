"""Detection metrics for CATHACTION Task 2.

The challenge primary metric for Task 2 is mAP.  These helpers implement the
standard one-to-one matching protocol used by object detection metrics: for a
fixed class and IoU threshold, predictions are sorted by confidence, each
ground-truth box can be matched at most once, and AP is computed with the
101-point interpolated precision envelope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


BoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionGroundTruth:
    sample_id: str
    class_id: int
    box_xyxy: BoxXYXY


@dataclass(frozen=True)
class DetectionPrediction:
    sample_id: str
    class_id: int
    score: float
    box_xyxy: BoxXYXY


def iou_xyxy(box_a: BoxXYXY, box_b: BoxXYXY) -> float:
    """Return intersection-over-union for two ``xyxy`` boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def ap_101_point(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """Compute COCO-style 101-point interpolated AP."""

    recall = np.asarray(recalls, dtype=np.float64)
    precision = np.asarray(precisions, dtype=np.float64)
    if recall.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    ap = 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        eligible = mpre[mrec >= threshold]
        ap += float(eligible.max()) if eligible.size else 0.0
    return ap / 101.0


def compute_class_ap(
    ground_truths: Iterable[DetectionGroundTruth],
    predictions: Iterable[DetectionPrediction],
    *,
    class_id: int,
    iou_threshold: float,
) -> dict[str, float | int]:
    """Compute AP for one class at one IoU threshold."""

    class_ground_truths = [
        ground_truth for ground_truth in ground_truths if ground_truth.class_id == class_id
    ]
    class_predictions = [
        prediction for prediction in predictions if prediction.class_id == class_id
    ]
    class_predictions.sort(key=lambda prediction: prediction.score, reverse=True)

    num_gt = len(class_ground_truths)
    if num_gt == 0:
        return {
            "ap": float("nan"),
            "num_gt": 0,
            "num_predictions": len(class_predictions),
            "true_positives": 0,
            "false_positives": len(class_predictions),
        }

    gt_by_sample: dict[str, list[tuple[int, DetectionGroundTruth]]] = {}
    for gt_index, ground_truth in enumerate(class_ground_truths):
        gt_by_sample.setdefault(ground_truth.sample_id, []).append((gt_index, ground_truth))

    matched_gt: set[int] = set()
    true_positives = np.zeros(len(class_predictions), dtype=np.float64)
    false_positives = np.zeros(len(class_predictions), dtype=np.float64)

    for pred_index, prediction in enumerate(class_predictions):
        candidates = gt_by_sample.get(prediction.sample_id, [])
        best_iou = 0.0
        best_gt_index: int | None = None
        for gt_index, ground_truth in candidates:
            overlap = iou_xyxy(prediction.box_xyxy, ground_truth.box_xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
            and best_gt_index not in matched_gt
        ):
            true_positives[pred_index] = 1.0
            matched_gt.add(best_gt_index)
        else:
            false_positives[pred_index] = 1.0

    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recalls = cumulative_tp / max(num_gt, 1)
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    ap = ap_101_point(recalls, precisions)

    return {
        "ap": float(ap),
        "num_gt": num_gt,
        "num_predictions": len(class_predictions),
        "true_positives": int(cumulative_tp[-1]) if cumulative_tp.size else 0,
        "false_positives": int(cumulative_fp[-1]) if cumulative_fp.size else 0,
    }


def compute_detection_map(
    ground_truths: Sequence[DetectionGroundTruth],
    predictions: Sequence[DetectionPrediction],
    *,
    class_ids: Sequence[int] = (0, 1),
    iou_thresholds: Sequence[float] | None = None,
) -> dict[str, object]:
    """Compute mAP50 and mAP50-95 over the requested classes."""

    if iou_thresholds is None:
        iou_thresholds = tuple(float(round(value, 2)) for value in np.arange(0.5, 0.96, 0.05))

    class_metrics: dict[str, dict[str, object]] = {}
    ap_by_threshold: dict[float, list[float]] = {float(threshold): [] for threshold in iou_thresholds}

    for class_id in class_ids:
        per_threshold: dict[str, dict[str, float | int]] = {}
        ap_values: list[float] = []
        for threshold in iou_thresholds:
            threshold_float = float(threshold)
            result = compute_class_ap(
                ground_truths,
                predictions,
                class_id=class_id,
                iou_threshold=threshold_float,
            )
            per_threshold[f"{threshold_float:.2f}"] = result
            ap = float(result["ap"])
            if not np.isnan(ap):
                ap_by_threshold[threshold_float].append(ap)
                ap_values.append(ap)

        ap50 = per_threshold.get("0.50", {}).get("ap", float("nan"))
        class_metrics[str(class_id)] = {
            "num_gt": int(sum(1 for gt in ground_truths if gt.class_id == class_id)),
            "num_predictions": int(sum(1 for pred in predictions if pred.class_id == class_id)),
            "ap50": float(ap50),
            "ap50_95": float(np.mean(ap_values)) if ap_values else float("nan"),
            "ap_by_iou": per_threshold,
        }

    map50_values = ap_by_threshold.get(0.5, [])
    all_ap_values = [
        value
        for threshold_values in ap_by_threshold.values()
        for value in threshold_values
        if not np.isnan(value)
    ]
    return {
        "mAP50": float(np.mean(map50_values)) if map50_values else float("nan"),
        "mAP50-95": float(np.mean(all_ap_values)) if all_ap_values else float("nan"),
        "iou_thresholds": [float(threshold) for threshold in iou_thresholds],
        "classes": class_metrics,
    }


_CASE_RE = re.compile(r"(.+)_([0-9]+)")


def case_id_from_sample(sample_id: str) -> str:
    """Procedure/case id for a sample stem (drops the trailing frame index).

    Mirrors ``cathaction.data.task2.parse_task2_sample_stem`` so phantom
    ``video_0`` and animal ``video_2_animal`` map to distinct cases.
    """

    match = _CASE_RE.fullmatch(sample_id)
    return match.group(1) if match else sample_id


def compute_detection_map_per_case(
    ground_truths: Sequence[DetectionGroundTruth],
    predictions: Sequence[DetectionPrediction],
    *,
    class_ids: Sequence[int] = (0, 1),
    iou_thresholds: Sequence[float] | None = None,
    case_of: Callable[[str], str] = case_id_from_sample,
) -> dict[str, object]:
    """Per-case (per-procedure) mAP, macro-averaged across cases.

    The official CATHACTION wording (2026-04-22 PDF, Task 2 Assessment Methods)
    defines mAP as "averaging AP scores across all test cases". This computes
    mAP within each case (video/procedure) and averages the case-level mAPs with
    equal weight per case. In contrast, :func:`compute_detection_map` pools every
    box globally and is therefore frame-count weighted. Both readings are kept
    until the official evaluator is released; see
    ``quality_reports/decisions/2026-06-15_task2_metric_interpretation.md``.

    A case contributes a class's AP only when that class has at least one GT box
    in the case (handled by :func:`compute_class_ap`); a case with no GT for any
    requested class is skipped entirely so it neither rewards nor punishes.
    """

    cases: dict[str, dict[str, list]] = {}
    for ground_truth in ground_truths:
        cases.setdefault(case_of(ground_truth.sample_id), {"gt": [], "pred": []})[
            "gt"
        ].append(ground_truth)
    for prediction in predictions:
        cases.setdefault(case_of(prediction.sample_id), {"gt": [], "pred": []})[
            "pred"
        ].append(prediction)

    per_case: dict[str, dict[str, object]] = {}
    map50_values: list[float] = []
    map5095_values: list[float] = []
    for case_key in sorted(cases):
        case = cases[case_key]
        if not case["gt"]:
            # No ground truth in this case -> AP undefined; skip.
            continue
        case_result = compute_detection_map(
            case["gt"],
            case["pred"],
            class_ids=class_ids,
            iou_thresholds=iou_thresholds,
        )
        case_map50 = float(case_result["mAP50"])
        case_map5095 = float(case_result["mAP50-95"])
        per_case[case_key] = {
            "mAP50": case_map50,
            "mAP50-95": case_map5095,
            "num_gt": len(case["gt"]),
            "num_predictions": len(case["pred"]),
        }
        if not np.isnan(case_map50):
            map50_values.append(case_map50)
        if not np.isnan(case_map5095):
            map5095_values.append(case_map5095)

    if iou_thresholds is None:
        resolved_thresholds = [float(round(value, 2)) for value in np.arange(0.5, 0.96, 0.05)]
    else:
        resolved_thresholds = [float(threshold) for threshold in iou_thresholds]

    return {
        "aggregation": "per_case_macro_average",
        "num_cases": len(per_case),
        "mAP50": float(np.mean(map50_values)) if map50_values else float("nan"),
        "mAP50-95": float(np.mean(map5095_values)) if map5095_values else float("nan"),
        "iou_thresholds": resolved_thresholds,
        "per_case": per_case,
    }
