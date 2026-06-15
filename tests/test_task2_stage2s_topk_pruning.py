from __future__ import annotations

from scripts.task2.evaluate_stage2s_topk_pruning import (
    CandidateRecord,
    ScoredPrediction,
    select_predictions,
    summarize_localization,
)


def make_candidate(row_index: int, *, sample_id: str = "s", source: str = "a", gt_iou: float = 0.0) -> CandidateRecord:
    return CandidateRecord(
        row_index=row_index,
        split="valid",
        sample_id=sample_id,
        source=source,
        source_rank=row_index + 1,
        gt_class=0,
        gt_xyxy=(0.0, 0.0, 10.0, 10.0),
        xyxy=(0.0, 0.0, 10.0, 10.0),
        gt_iou=gt_iou,
    )


def make_prediction(row_index: int, *, sample_id: str = "s", source: str = "a", class_id: int, score: float) -> ScoredPrediction:
    return ScoredPrediction(
        row_index=row_index,
        sample_id=sample_id,
        source=source,
        class_id=class_id,
        score=score,
        xyxy=(0.0, 0.0, 10.0, 10.0),
        gt_iou=0.0,
    )


def test_stage2s_candidate_top_keeps_both_class_predictions_for_best_candidate() -> None:
    candidates = [
        make_candidate(0, gt_iou=0.2),
        make_candidate(1, gt_iou=0.8),
    ]
    predictions = [
        make_prediction(0, class_id=0, score=0.7),
        make_prediction(0, class_id=1, score=0.1),
        make_prediction(1, class_id=0, score=0.8),
        make_prediction(1, class_id=1, score=0.2),
    ]
    candidate_scores = {0: 0.7, 1: 0.8}

    selected = select_predictions(
        candidates=candidates,
        predictions=predictions,
        candidate_scores=candidate_scores,
        policy="candidate_top",
        k=1,
    )

    assert [(item.row_index, item.class_id) for item in selected] == [(1, 0), (1, 1)]


def test_stage2s_class_top_keeps_best_prediction_per_class() -> None:
    candidates = [make_candidate(0), make_candidate(1)]
    predictions = [
        make_prediction(0, class_id=0, score=0.7),
        make_prediction(1, class_id=0, score=0.8),
        make_prediction(0, class_id=1, score=0.9),
        make_prediction(1, class_id=1, score=0.2),
    ]

    selected = select_predictions(
        candidates=candidates,
        predictions=predictions,
        candidate_scores={0: 0.9, 1: 0.8},
        policy="class_top",
        k=1,
    )

    assert [(item.row_index, item.class_id) for item in selected] == [(1, 0), (0, 1)]


def test_stage2s_oracle_candidate_top_uses_gt_iou_for_diagnostic_upper_bound() -> None:
    candidates = [
        make_candidate(0, gt_iou=0.9),
        make_candidate(1, gt_iou=0.1),
    ]
    predictions = [
        make_prediction(0, class_id=0, score=0.1),
        make_prediction(0, class_id=1, score=0.1),
        make_prediction(1, class_id=0, score=0.9),
        make_prediction(1, class_id=1, score=0.9),
    ]

    selected = select_predictions(
        candidates=candidates,
        predictions=predictions,
        candidate_scores={0: 0.1, 1: 0.9},
        policy="oracle_candidate_top",
        k=1,
    )

    assert {item.row_index for item in selected} == {0}
    loc = summarize_localization(candidates, selected)
    assert loc["loc_recall_iou_0.75"] == 1.0
