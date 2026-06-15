"""Segmentation metrics for CATHACTION Task 1."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def per_class_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    include_background: bool = False,
    background_label: int = 0,
    empty_score: float = 1.0,
) -> dict[int, float]:
    pred, tgt = _validated_arrays(prediction, target)
    return {
        label: _dice_for_label(pred, tgt, label, empty_score)
        for label in _metric_labels(pred, tgt, labels, include_background, background_label)
    }


def dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    include_background: bool = False,
    background_label: int = 0,
    empty_score: float = 1.0,
) -> float:
    scores = per_class_dice(
        prediction,
        target,
        labels=labels,
        include_background=include_background,
        background_label=background_label,
        empty_score=empty_score,
    )
    return _mean(scores.values(), empty_score)


def per_class_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    include_background: bool = False,
    background_label: int = 0,
    empty_score: float = 1.0,
) -> dict[int, float]:
    pred, tgt = _validated_arrays(prediction, target)
    return {
        label: _iou_for_label(pred, tgt, label, empty_score)
        for label in _metric_labels(pred, tgt, labels, include_background, background_label)
    }


def iou_score(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    include_background: bool = False,
    background_label: int = 0,
    empty_score: float = 1.0,
) -> float:
    scores = per_class_iou(
        prediction,
        target,
        labels=labels,
        include_background=include_background,
        background_label=background_label,
        empty_score=empty_score,
    )
    return _mean(scores.values(), empty_score)


def mean_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    labels: Iterable[int] | None = None,
    include_background: bool = False,
    background_label: int = 0,
    empty_score: float = 1.0,
) -> float:
    return iou_score(
        prediction,
        target,
        labels=labels,
        include_background=include_background,
        background_label=background_label,
        empty_score=empty_score,
    )


def pixel_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, tgt = _validated_arrays(prediction, target)
    if pred.size == 0:
        return 1.0
    return float(np.mean(pred == tgt))


def foreground_mask(mask: np.ndarray, *, background_label: int = 0) -> np.ndarray:
    return (np.asarray(mask) != background_label).astype(np.uint8)


def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def _validated_arrays(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction)
    tgt = np.asarray(target)
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs target {tgt.shape}")
    return pred, tgt


def _metric_labels(
    prediction: np.ndarray,
    target: np.ndarray,
    labels: Iterable[int] | None,
    include_background: bool,
    background_label: int,
) -> list[int]:
    if labels is None:
        values = set(np.unique(prediction).tolist()) | set(np.unique(target).tolist())
        labels_list = sorted(int(value) for value in values)
    else:
        labels_list = [int(label) for label in labels]
    if not include_background:
        labels_list = [label for label in labels_list if label != background_label]
    return labels_list


def _dice_for_label(prediction: np.ndarray, target: np.ndarray, label: int, empty_score: float) -> float:
    pred_mask = prediction == label
    tgt_mask = target == label
    denom = int(pred_mask.sum() + tgt_mask.sum())
    if denom == 0:
        return float(empty_score)
    intersection = int(np.logical_and(pred_mask, tgt_mask).sum())
    return float(2.0 * intersection / denom)


def _iou_for_label(prediction: np.ndarray, target: np.ndarray, label: int, empty_score: float) -> float:
    pred_mask = prediction == label
    tgt_mask = target == label
    union = int(np.logical_or(pred_mask, tgt_mask).sum())
    if union == 0:
        return float(empty_score)
    intersection = int(np.logical_and(pred_mask, tgt_mask).sum())
    return float(intersection / union)


def _mean(values: Iterable[float], empty_score: float) -> float:
    values = list(values)
    if not values:
        return float(empty_score)
    return float(np.mean(values))

