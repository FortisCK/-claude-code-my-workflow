from __future__ import annotations

import pytest
import torch

from cathaction.data.task2 import Task2Box
from cathaction.data.task2_heatmap import decode_heatmap_prediction, make_heatmap_target


def test_make_heatmap_target_places_peak_at_center() -> None:
    box = Task2Box(class_id=1, x_center=0.5, y_center=0.25, width=0.1, height=0.2)

    target = make_heatmap_target(box, output_size=(32, 64), sigma=1.5)

    assert target.heatmap.shape == (2, 32, 64)
    assert target.size.shape == (2, 32, 64)
    assert target.offset.shape == (2, 32, 64)
    assert target.center_index == (8, 32)
    assert target.heatmap[1, 8, 32] == torch.max(target.heatmap[1])
    assert target.heatmap[0].sum() == 0
    assert target.size[0, 8, 32] == box.width
    assert target.size[1, 8, 32] == box.height


def test_decode_heatmap_prediction_recovers_center_and_size() -> None:
    heatmap_logits = torch.full((2, 16, 16), -10.0)
    heatmap_logits[1, 4, 12] = 10.0
    size_map = torch.zeros((2, 16, 16))
    size_map[0, 4, 12] = 0.15
    size_map[1, 4, 12] = 0.25
    offset_map = torch.zeros((2, 16, 16))
    offset_map[0, 4, 12] = 0.25
    offset_map[1, 4, 12] = 0.75

    decoded = decode_heatmap_prediction(heatmap_logits, size_map, offset_map)

    assert decoded.class_id == 1
    assert decoded.score > 0.99
    assert decoded.x_center == (12.25 / 16)
    assert decoded.y_center == (4.75 / 16)
    assert decoded.width == pytest.approx(0.15)
    assert decoded.height == 0.25
