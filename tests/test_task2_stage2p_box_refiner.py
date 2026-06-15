from __future__ import annotations

from pathlib import Path

import pytest

from scripts.task2.train_stage2p_box_refiner import (
    clip_xyxy,
    decode_box_delta,
    encode_box_delta,
    summarize_sample_group,
)
from cathaction.data.task2 import Task2Box, Task2Sample


def test_stage2p_box_delta_roundtrip() -> None:
    source = (10.0, 20.0, 30.0, 60.0)
    target = (12.0, 24.0, 34.0, 70.0)

    delta = encode_box_delta(source, target, clip=4.0)
    decoded = decode_box_delta(source, delta, image_width=100, image_height=100, clip=4.0)

    assert decoded == pytest.approx(target, abs=1e-5)


def test_stage2p_box_delta_clips_large_targets() -> None:
    source = (10.0, 20.0, 30.0, 60.0)
    target = (-100.0, -100.0, 400.0, 500.0)

    delta = encode_box_delta(source, target, clip=1.0)

    assert all(-1.0 <= value <= 1.0 for value in delta)


def test_stage2p_clip_xyxy_reorders_and_bounds() -> None:
    clipped = clip_xyxy((120.0, -10.0, -5.0, 80.0), image_width=100, image_height=50)

    assert clipped == (0.0, 0.0, 100.0, 50.0)


def test_stage2p_augmented_metrics_keep_original_and_refined_candidates() -> None:
    samples = [
        make_sample("phantom_case_001_000001"),
        make_sample("phantom_case_001_000002"),
    ]
    rows_by_sample = {
        samples[0].sample_id: [{"before_iou": 0.80, "after_iou": 0.20}],
        samples[1].sample_id: [{"before_iou": 0.40, "after_iou": 0.76}],
    }

    metrics = summarize_sample_group(samples, rows_by_sample)

    assert metrics["before_recall_iou_0.75"] == 0.5
    assert metrics["after_recall_iou_0.75"] == 0.5
    assert metrics["augmented_recall_iou_0.75"] == 1.0
    assert metrics["augmented_recall_iou_0.75_gain"] == 0.5


def make_sample(sample_id: str) -> Task2Sample:
    return Task2Sample(
        sample_id=sample_id,
        video_id="phantom_case_001",
        frame_index=1,
        image_path=Path(__file__),
        label_path=Path(__file__),
        box=Task2Box(class_id=0, x_center=0.5, y_center=0.5, width=0.1, height=0.1),
    )
