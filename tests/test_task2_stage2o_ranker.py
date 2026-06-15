from __future__ import annotations

from pathlib import Path

from scripts.task2.train_stage2o_candidate_ranker import (
    CandidateExample,
    apply_candidate_budget,
    candidate_score,
)


def make_candidate(
    *,
    sample_id: str = "s",
    source_priority: int,
    source_rank: int,
    source_conf: float,
    row_index: int,
) -> CandidateExample:
    return CandidateExample(
        split="valid",
        sample_id=sample_id,
        video_id="video_0",
        frame_index=row_index,
        domain="phantom",
        image_path=Path("image.jpg"),
        gt_class=0,
        verifier_label=0,
        xyxy=(0.0, 0.0, 1.0, 1.0),
        gt_xyxy=(0.0, 0.0, 1.0, 1.0),
        gt_iou=0.0,
        source_conf=source_conf,
        source=f"src{source_priority}",
        source_priority=source_priority,
        source_rank=source_rank,
        row_index=row_index,
        image_width=10,
        image_height=10,
    )


def test_stage2o_ranker_candidate_budget_source_order() -> None:
    candidates = [
        make_candidate(source_priority=1, source_rank=1, source_conf=0.9, row_index=0),
        make_candidate(source_priority=0, source_rank=2, source_conf=0.2, row_index=1),
        make_candidate(source_priority=0, source_rank=1, source_conf=0.1, row_index=2),
    ]

    selected = apply_candidate_budget(candidates, max_candidates_per_sample=2, sort_mode="source_order")

    assert [item.row_index for item in selected] == [1, 2]


def test_stage2o_ranker_candidate_budget_raw_conf() -> None:
    candidates = [
        make_candidate(source_priority=1, source_rank=1, source_conf=0.9, row_index=0),
        make_candidate(source_priority=0, source_rank=2, source_conf=0.2, row_index=1),
        make_candidate(source_priority=0, source_rank=1, source_conf=0.1, row_index=2),
    ]

    selected = apply_candidate_budget(candidates, max_candidates_per_sample=2, sort_mode="raw_conf")

    assert [item.row_index for item in selected] == [0, 1]


def test_stage2o_ranker_score_modes() -> None:
    candidate = make_candidate(source_priority=0, source_rank=4, source_conf=0.25, row_index=0)

    assert candidate_score(candidate, 0.2, 0.8, mode="roi") == 0.8
    assert candidate_score(candidate, 0.2, 0.8, mode="bg_suppressed_roi") == 0.6400000000000001
    assert candidate_score(candidate, 0.2, 0.8, mode="source_roi") == 0.2
    assert candidate_score(candidate, 0.2, 0.8, mode="sqrt_source_roi") == 0.4
    assert candidate_score(candidate, 0.2, 0.8, mode="rank_decay_roi") == 0.4
    assert candidate_score(candidate, 0.2, 0.8, mode="source_rank_decay_roi") == 0.1
