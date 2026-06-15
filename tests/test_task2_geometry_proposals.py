from __future__ import annotations

import numpy as np

from scripts.task2.evaluate_task1_geometry_proposals import (
    ResizeMeta,
    generate_geometry_proposals,
    restore_prediction_to_original,
)
from cathaction.metrics.detection import iou_xyxy


def test_geometry_proposals_cover_class_overlap() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:81, 48:51] = 1
    mask[48:51, 20:81] = 2

    proposals = generate_geometry_proposals(
        mask,
        dilation_px=3,
        box_sizes=[20, 32, 48],
        max_proposals=20,
    )

    assert proposals
    assert proposals[0].source.startswith("overlap")
    gt = (36.0, 36.0, 64.0, 64.0)
    assert max(iou_xyxy(proposal.xyxy, gt) for proposal in proposals[:10]) > 0.7


def test_geometry_proposals_use_skeleton_fallback_for_single_class_line() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:71, 38:41] = 1

    proposals = generate_geometry_proposals(
        mask,
        dilation_px=3,
        box_sizes=[24, 40],
        max_proposals=40,
    )

    assert proposals
    assert any("skeleton" in proposal.source for proposal in proposals)
    gt = (24.0, 24.0, 56.0, 56.0)
    assert max(iou_xyxy(proposal.xyxy, gt) for proposal in proposals) > 0.3


def test_restore_prediction_crops_aspect_padding() -> None:
    prediction = np.zeros((10, 10), dtype=np.uint8)
    prediction[2:8, 3:7] = 2
    meta = ResizeMeta(
        original_width=4,
        original_height=6,
        target_width=10,
        target_height=10,
        resized_width=4,
        resized_height=6,
        left=3,
        top=2,
    )

    restored = restore_prediction_to_original(prediction, meta)

    assert restored.shape == (6, 4)
    assert np.all(restored == 2)
