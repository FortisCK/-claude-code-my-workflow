from __future__ import annotations

from scripts.task2.export_clean_predictions import apply_topk, clean_row


def test_clean_row_removes_validation_fields() -> None:
    row = {
        "sample_id": "s1",
        "video_id": "v",
        "frame_index": "3",
        "domain": "phantom",
        "class_id": "1",
        "score": "0.7",
        "x1": "1",
        "y1": "2",
        "x2": "3",
        "y2": "4",
        "source": "yolo_stage2l",
        "source_rank": "5",
        "score_mode": "rank_decay_roi",
        "policy_name": "class1",
        "gt_class": "1",
        "gt_iou": "0.9",
    }

    cleaned = clean_row(row)

    assert "gt_class" not in cleaned
    assert "gt_iou" not in cleaned
    assert cleaned["class_id"] == 1
    assert cleaned["score"] == 0.7


def test_apply_topk_per_sample_class() -> None:
    rows = [
        {"sample_id": "a", "class_id": 0, "score": 0.1},
        {"sample_id": "a", "class_id": 0, "score": 0.9},
        {"sample_id": "a", "class_id": 1, "score": 0.2},
        {"sample_id": "a", "class_id": 1, "score": 0.8},
        {"sample_id": "b", "class_id": 0, "score": 0.3},
    ]

    kept = apply_topk(rows, 1)

    assert [(row["sample_id"], row["class_id"], row["score"]) for row in kept] == [
        ("a", 0, 0.9),
        ("a", 1, 0.8),
        ("b", 0, 0.3),
    ]
