import numpy as np
import pytest

from cathaction.metrics.segmentation import (
    dice_score,
    foreground_mask,
    iou_score,
    mean_iou,
    normalize_binary_mask,
    per_class_dice,
    per_class_iou,
    pixel_accuracy,
)


def test_binary_foreground_metrics_known_values() -> None:
    target = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    pred = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)

    assert dice_score(pred, target, labels=[1]) == pytest.approx(2 / 3)
    assert iou_score(pred, target, labels=[1]) == pytest.approx(1 / 2)
    assert pixel_accuracy(pred, target) == pytest.approx(4 / 6)


def test_multiclass_metrics_exclude_background_by_default() -> None:
    target = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)
    pred = np.array([[0, 1, 2], [2, 0, 0]], dtype=np.uint8)

    assert per_class_dice(pred, target) == {
        1: pytest.approx(2 / 3),
        2: pytest.approx(1 / 2),
    }
    assert dice_score(pred, target) == pytest.approx(7 / 12)
    assert per_class_iou(pred, target) == {
        1: pytest.approx(1 / 2),
        2: pytest.approx(1 / 3),
    }
    assert mean_iou(pred, target) == pytest.approx(5 / 12)


def test_metrics_can_include_background() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    pred = np.array([[0, 0], [1, 0]], dtype=np.uint8)

    scores = per_class_dice(pred, target, include_background=True)

    assert set(scores) == {0, 1}
    assert scores[0] == pytest.approx(4 / 5)
    assert scores[1] == pytest.approx(2 / 3)


def test_empty_class_uses_empty_score() -> None:
    target = np.zeros((2, 2), dtype=np.uint8)
    pred = np.zeros((2, 2), dtype=np.uint8)

    assert dice_score(pred, target, labels=[1]) == pytest.approx(1.0)
    assert iou_score(pred, target, labels=[1]) == pytest.approx(1.0)
    assert dice_score(pred, target, labels=[1], empty_score=0.0) == pytest.approx(0.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        dice_score(np.zeros((2, 2)), np.zeros((2, 3)), labels=[1])


def test_mask_normalization_helpers() -> None:
    human_mask = np.array([[0, 255], [10, 0]], dtype=np.uint8)
    multiclass = np.array([[0, 1], [2, 0]], dtype=np.uint8)

    assert normalize_binary_mask(human_mask).tolist() == [[0, 1], [1, 0]]
    assert foreground_mask(multiclass).tolist() == [[0, 1], [1, 0]]
