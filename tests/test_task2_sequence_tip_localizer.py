from __future__ import annotations

from pathlib import Path

import pytest

from cathaction.data.task2 import Task2Box, Task2Sample
from scripts.task2.train_sequence_tip_localizer import (
    LetterboxMeta,
    letterbox_xyxy_to_original,
    make_class_agnostic_tip_target,
)


def make_sample(class_id: int = 1) -> Task2Sample:
    return Task2Sample(
        sample_id="video_0_00000010",
        video_id="video_0",
        frame_index=10,
        image_path=Path("image.jpg"),
        label_path=Path("label.txt"),
        box=Task2Box(
            class_id=class_id,
            x_center=0.5,
            y_center=0.25,
            width=0.1,
            height=0.2,
        ),
    )


def test_class_agnostic_target_uses_single_heatmap_channel_for_both_classes() -> None:
    meta = LetterboxMeta(
        original_width=200,
        original_height=100,
        input_size=128,
        scale=0.64,
        pad_x=0,
        pad_y=32,
        resized_width=128,
        resized_height=64,
    )

    target0 = make_class_agnostic_tip_target(make_sample(class_id=0), meta=meta, input_size=128, sigma=2.0)
    target1 = make_class_agnostic_tip_target(make_sample(class_id=1), meta=meta, input_size=128, sigma=2.0)

    assert target0.heatmap.shape == (1, 128, 128)
    assert target1.heatmap.shape == (1, 128, 128)
    assert target0.center_index == target1.center_index
    assert float(target0.heatmap.max()) > 0.9
    assert float(target1.heatmap.max()) > 0.9


def test_letterbox_xyxy_to_original_inverts_padding_and_scale() -> None:
    meta = LetterboxMeta(
        original_width=200,
        original_height=100,
        input_size=128,
        scale=0.64,
        pad_x=0,
        pad_y=32,
        resized_width=128,
        resized_height=64,
    )

    original = letterbox_xyxy_to_original((64 - 6.4, 48 - 6.4, 64 + 6.4, 48 + 6.4), meta)

    assert original == pytest.approx((90.0, 15.0, 110.0, 35.0))
