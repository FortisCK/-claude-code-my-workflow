from cathaction.metrics.detection import (
    DetectionGroundTruth,
    DetectionPrediction,
    compute_detection_map,
    iou_xyxy,
)


def test_iou_xyxy_for_partially_overlapping_boxes() -> None:
    assert iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175


def test_detection_map_is_one_for_perfect_predictions() -> None:
    ground_truths = [
        DetectionGroundTruth("a", 0, (0, 0, 10, 10)),
        DetectionGroundTruth("b", 1, (20, 20, 30, 30)),
    ]
    predictions = [
        DetectionPrediction("a", 0, 0.9, (0, 0, 10, 10)),
        DetectionPrediction("b", 1, 0.8, (20, 20, 30, 30)),
    ]

    metrics = compute_detection_map(
        ground_truths,
        predictions,
        class_ids=(0, 1),
        iou_thresholds=(0.5, 0.75),
    )

    assert metrics["mAP50"] == 1.0
    assert metrics["mAP50-95"] == 1.0
    assert metrics["classes"]["0"]["ap50"] == 1.0
    assert metrics["classes"]["1"]["ap50"] == 1.0


def test_detection_map_counts_wrong_class_as_false_positive() -> None:
    ground_truths = [DetectionGroundTruth("a", 1, (0, 0, 10, 10))]
    predictions = [DetectionPrediction("a", 0, 0.9, (0, 0, 10, 10))]

    metrics = compute_detection_map(
        ground_truths,
        predictions,
        class_ids=(1,),
        iou_thresholds=(0.5,),
    )

    assert metrics["mAP50"] == 0.0
    assert metrics["classes"]["1"]["num_gt"] == 1
    assert metrics["classes"]["1"]["num_predictions"] == 0
