from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.task2.train_stage2o_candidate_ranker import CandidateExample
from scripts.task2.train_stage2u_quality_ranker import (
    ALL_DETECTION_MODES,
    QUALITY_MODES,
    SOURCE_METADATA_DIM,
    SourceAwareResidualRanker,
    encode_quality_targets,
    encode_source_metadata,
    filter_candidates_by_source,
    parse_source_keep,
    quality_adjusted_score,
    quality_blend,
    split_quality_outputs,
)


def make_candidate(source: str = "yolo") -> CandidateExample:
    return CandidateExample(
        split="valid",
        sample_id="s",
        video_id="video_0",
        frame_index=0,
        domain="phantom",
        image_path=Path("image.jpg"),
        gt_class=0,
        verifier_label=1,
        xyxy=(0.0, 0.0, 1.0, 1.0),
        gt_xyxy=(0.0, 0.0, 1.0, 1.0),
        gt_iou=0.0,
        source_conf=0.5,
        source=source,
        source_priority=0,
        source_rank=4,
        row_index=0,
        image_width=10,
        image_height=10,
    )


def test_stage2u_quality_targets_threshold_and_clip() -> None:
    high = encode_quality_targets(0.76)
    mid = encode_quality_targets(0.60)
    clipped = encode_quality_targets(1.50)

    assert high.tolist() == pytest.approx([0.76, 1.0, 1.0])
    assert mid.tolist() == pytest.approx([0.60, 1.0, 0.0])
    assert clipped.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_stage2u_split_quality_outputs_shapes_and_ranges() -> None:
    outputs = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.0, -1.0, 1.0],
            [3.0, 2.0, 1.0, 2.0, 0.5, -0.5],
        ]
    )

    class_logits, pred_iou, iou50_logits, iou75_logits = split_quality_outputs(outputs)

    assert class_logits.shape == (2, 3)
    assert pred_iou.shape == (2,)
    assert iou50_logits.shape == (2,)
    assert iou75_logits.shape == (2,)
    assert torch.all((pred_iou >= 0.0) & (pred_iou <= 1.0))


def test_stage2u_quality_adjusted_score_uses_base_score() -> None:
    candidate = make_candidate()

    score = quality_adjusted_score(
        candidate,
        0.2,
        0.8,
        base_mode="source_roi",
        quality=0.25,
    )

    assert score == 0.25 * 0.5 * 0.8


def test_stage2u_quality_modes_and_detection_modes_include_blend() -> None:
    assert QUALITY_MODES == ("pred_iou", "prob_iou50", "prob_iou75", "blend")
    assert "blend_rank_decay_roi" in ALL_DETECTION_MODES
    assert quality_blend(0.2, 0.4, 0.8) == 0.4


def test_stage2u_parse_and_filter_valid_sources() -> None:
    assert parse_source_keep(None) is None
    assert parse_source_keep(" yolo_stage2l, task1_geometry_rect ") == {
        "yolo_stage2l",
        "task1_geometry_rect",
    }
    with pytest.raises(ValueError):
        parse_source_keep(" , ")

    candidates = [
        make_candidate("yolo_stage2l"),
        make_candidate("task1_geometry_rect"),
        make_candidate("stage2p_refined"),
    ]

    assert filter_candidates_by_source(candidates, None) is candidates
    assert [candidate.source for candidate in filter_candidates_by_source(candidates, {"yolo_stage2l"})] == [
        "yolo_stage2l"
    ]


def test_stage2u_source_metadata_is_fixed_width_and_safe_for_unknown_source() -> None:
    metadata = encode_source_metadata(make_candidate("stage2x_class1"))
    unknown = encode_source_metadata(make_candidate("new_source"))

    assert metadata.shape == (SOURCE_METADATA_DIM,)
    assert unknown.shape == (SOURCE_METADATA_DIM,)
    assert torch.isfinite(metadata).all()
    assert torch.isfinite(unknown).all()


def test_stage2u_source_aware_residual_starts_as_image_model() -> None:
    image_model = torch.nn.Linear(4, 6)
    model = SourceAwareResidualRanker(image_model, metadata_dim=SOURCE_METADATA_DIM, output_dim=6)
    images = torch.ones(2, 4)
    metadata = torch.randn(2, SOURCE_METADATA_DIM)

    assert torch.allclose(model(images, metadata), image_model(images))
