"""Tests for the official-worded per-case mAP variant (Task 2 metric hedge).

The official PDF defines mAP as AP averaged across test cases; the existing
compute_detection_map pools boxes globally (frame-count weighted). These tests
pin that the two readings genuinely diverge and that the per-case variant
macro-averages with equal weight per case.
"""

import math

import pytest

from cathaction.metrics.detection import (
    DetectionGroundTruth,
    DetectionPrediction,
    case_id_from_sample,
    compute_detection_map,
    compute_detection_map_per_case,
)

BOX = (0.0, 0.0, 10.0, 10.0)
FAR = (100.0, 100.0, 110.0, 110.0)


def test_case_id_from_sample_phantom_and_animal() -> None:
    assert case_id_from_sample("video_0_00004090") == "video_0"
    assert case_id_from_sample("video_2_animal_00000022") == "video_2_animal"


def _scenario():
    # Case video_1: 3 perfect class-0 detections -> case mAP50 = 1.0
    # Case video_2: 1 class-0 GT, 1 missed prediction -> case mAP50 = 0.0
    gts = [
        DetectionGroundTruth("video_1_00000001", 0, BOX),
        DetectionGroundTruth("video_1_00000002", 0, BOX),
        DetectionGroundTruth("video_1_00000003", 0, BOX),
        DetectionGroundTruth("video_2_00000001", 0, BOX),
    ]
    preds = [
        DetectionPrediction("video_1_00000001", 0, 0.9, BOX),
        DetectionPrediction("video_1_00000002", 0, 0.9, BOX),
        DetectionPrediction("video_1_00000003", 0, 0.9, BOX),
        DetectionPrediction("video_2_00000001", 0, 0.5, FAR),  # false positive
    ]
    return gts, preds


def test_per_case_macro_average_equals_half() -> None:
    gts, preds = _scenario()
    result = compute_detection_map_per_case(gts, preds, class_ids=(0,))
    assert result["aggregation"] == "per_case_macro_average"
    assert result["num_cases"] == 2
    # (1.0 + 0.0) / 2 == 0.5, regardless of the 3:1 frame imbalance
    assert result["mAP50"] == pytest.approx(0.5)
    assert result["mAP50-95"] == pytest.approx(0.5)
    assert result["per_case"]["video_1"]["mAP50"] == pytest.approx(1.0)
    assert result["per_case"]["video_2"]["mAP50"] == pytest.approx(0.0)


def test_global_pool_diverges_from_per_case() -> None:
    gts, preds = _scenario()
    global_result = compute_detection_map(gts, preds, class_ids=(0,))
    per_case = compute_detection_map_per_case(gts, preds, class_ids=(0,))
    # Global pooling is frame-count weighted: 3 TP + 1 FP -> 76/101 interpolated.
    assert global_result["mAP50"] == pytest.approx(76 / 101, abs=1e-6)
    # The divergence is the whole point of keeping a second evaluator.
    assert global_result["mAP50"] - per_case["mAP50"] > 0.2


def test_case_with_no_gt_is_skipped() -> None:
    gts = [DetectionGroundTruth("video_1_00000001", 0, BOX)]
    preds = [
        DetectionPrediction("video_1_00000001", 0, 0.9, BOX),
        DetectionPrediction("video_9_00000001", 0, 0.9, BOX),  # case with no GT
    ]
    result = compute_detection_map_per_case(gts, preds, class_ids=(0,))
    assert result["num_cases"] == 1
    assert "video_9" not in result["per_case"]
    assert result["mAP50"] == pytest.approx(1.0)
    assert not math.isnan(result["mAP50-95"])
