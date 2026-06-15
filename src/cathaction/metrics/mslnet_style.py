"""MSLNet-style segmentation metrics for thin tool masks.

The MSLNet paper evaluates binary catheter / guidewire segmentation with
standard Dice and IoU, plus tolerance-based precision, recall, F1, and average
Hausdorff distance (AHD). This module implements that evaluation for saved
CATHACTION Task 1 masks after collapsing non-background labels into foreground.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class MslnetStyleMetrics:
    dice: float
    iou: float
    ahd: float
    precision_by_radius: dict[int, float]
    recall_by_radius: dict[int, float]
    f1_by_radius: dict[int, float]
    prediction_pixels: int
    target_pixels: int

    def to_flat_dict(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "dice": self.dice,
            "iou": self.iou,
            "ahd": self.ahd,
            "prediction_pixels": self.prediction_pixels,
            "target_pixels": self.target_pixels,
        }
        for radius, value in self.precision_by_radius.items():
            result[f"precision_r{radius}"] = value
        for radius, value in self.recall_by_radius.items():
            result[f"recall_r{radius}"] = value
        for radius, value in self.f1_by_radius.items():
            result[f"f1_r{radius}"] = value
        return result


def mslnet_style_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    tolerance_radii: Iterable[int] = (0, 1, 2, 3, 4),
    background_label: int = 0,
) -> MslnetStyleMetrics:
    """Compute MSLNet-style binary segmentation metrics for one image."""

    pred = np.asarray(prediction) != background_label
    tgt = np.asarray(target) != background_label
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs target {tgt.shape}")

    radii = _normal_radii(tolerance_radii)
    pred_count = int(pred.sum())
    tgt_count = int(tgt.sum())
    intersection = int(np.logical_and(pred, tgt).sum())
    union = int(np.logical_or(pred, tgt).sum())

    dice = 1.0 if pred_count + tgt_count == 0 else float(2.0 * intersection / (pred_count + tgt_count))
    iou = 1.0 if union == 0 else float(intersection / union)

    precision_by_radius: dict[int, float] = {}
    recall_by_radius: dict[int, float] = {}
    f1_by_radius: dict[int, float] = {}

    if pred_count == 0 or tgt_count == 0:
        ahd = float("nan")
        for radius in radii:
            precision = 1.0 if pred_count == 0 and tgt_count == 0 else 0.0
            recall = 1.0 if pred_count == 0 and tgt_count == 0 else 0.0
            precision_by_radius[radius] = precision
            recall_by_radius[radius] = recall
            f1_by_radius[radius] = _f1(precision, recall)
        return MslnetStyleMetrics(
            dice=dice,
            iou=iou,
            ahd=ahd,
            precision_by_radius=precision_by_radius,
            recall_by_radius=recall_by_radius,
            f1_by_radius=f1_by_radius,
            prediction_pixels=pred_count,
            target_pixels=tgt_count,
        )

    distance_to_target = distance_transform_edt(~tgt)
    distance_to_prediction = distance_transform_edt(~pred)
    predicted_pixel_distances = distance_to_target[pred]
    target_pixel_distances = distance_to_prediction[tgt]
    ahd = float(0.5 * (predicted_pixel_distances.mean() + target_pixel_distances.mean()))

    for radius in radii:
        precision = float(np.count_nonzero(predicted_pixel_distances <= radius) / pred_count)
        recall = float(np.count_nonzero(target_pixel_distances <= radius) / tgt_count)
        precision_by_radius[radius] = precision
        recall_by_radius[radius] = recall
        f1_by_radius[radius] = _f1(precision, recall)

    return MslnetStyleMetrics(
        dice=dice,
        iou=iou,
        ahd=ahd,
        precision_by_radius=precision_by_radius,
        recall_by_radius=recall_by_radius,
        f1_by_radius=f1_by_radius,
        prediction_pixels=pred_count,
        target_pixels=tgt_count,
    )


def aggregate_mslnet_style_metrics(
    metrics: Iterable[MslnetStyleMetrics],
    *,
    tolerance_radii: Iterable[int] = (0, 1, 2, 3, 4),
) -> dict[str, float | int]:
    """Average per-image MSLNet-style metrics."""

    rows = list(metrics)
    radii = _normal_radii(tolerance_radii)
    result: dict[str, float | int] = {
        "samples": len(rows),
        "dice": _nanmean([row.dice for row in rows]),
        "iou": _nanmean([row.iou for row in rows]),
        "ahd": _nanmean([row.ahd for row in rows]),
    }
    for radius in radii:
        result[f"precision_r{radius}"] = _nanmean([row.precision_by_radius[radius] for row in rows])
        result[f"recall_r{radius}"] = _nanmean([row.recall_by_radius[radius] for row in rows])
        result[f"f1_r{radius}"] = _nanmean([row.f1_by_radius[radius] for row in rows])
    return result


def _normal_radii(tolerance_radii: Iterable[int]) -> list[int]:
    return sorted(set(max(0, int(radius)) for radius in tolerance_radii))


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _nanmean(values: Iterable[float]) -> float:
    finite_values = [float(value) for value in values if not math.isnan(float(value))]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))
