from __future__ import annotations

from scripts.task2.train_stage2t_tabular_selector import (
    CandidatePredictionRecord,
    FeatureBuilder,
    add_rank_features,
    build_prediction_rows,
    select_candidate_indices,
)


def make_record(
    row_index: int,
    *,
    sample_id: str = "s",
    source: str = "yolo",
    source_conf: float = 0.5,
    source_rank: int = 1,
    gt_iou: float = 0.0,
) -> CandidatePredictionRecord:
    return CandidatePredictionRecord(
        split="valid",
        sample_id=sample_id,
        video_id="video_0",
        frame_index=row_index,
        domain="phantom",
        source=source,
        source_rank=source_rank,
        source_priority=0,
        source_conf=source_conf,
        row_index=row_index,
        image_width=100,
        image_height=80,
        gt_class=0,
        gt_xyxy=(10.0, 10.0, 20.0, 20.0),
        xyxy=(10.0 + row_index, 10.0, 20.0 + row_index, 20.0),
        gt_iou=gt_iou,
        prob_background=0.2,
        prob_normal=0.7,
        prob_collision=0.1,
        scores={
            "roi_score_normal": 0.7,
            "roi_score_collision": 0.1,
            "bg_suppressed_roi_score_normal": 0.56,
            "bg_suppressed_roi_score_collision": 0.08,
            "source_roi_score_normal": source_conf * 0.7,
            "source_roi_score_collision": source_conf * 0.1,
            "sqrt_source_roi_score_normal": source_conf**0.5 * 0.7,
            "sqrt_source_roi_score_collision": source_conf**0.5 * 0.1,
            "rank_decay_roi_score_normal": 0.7 / (source_rank**0.5),
            "rank_decay_roi_score_collision": 0.1 / (source_rank**0.5),
            "source_rank_decay_roi_score_normal": source_conf * 0.7 / (source_rank**0.5),
            "source_rank_decay_roi_score_collision": source_conf * 0.1 / (source_rank**0.5),
        },
    )


def test_stage2t_features_exclude_ground_truth_targets() -> None:
    records = [make_record(0, gt_iou=0.9), make_record(1, gt_iou=0.1)]
    add_rank_features(records)

    builder = FeatureBuilder.fit(records, include_domain=False)
    matrix = builder.transform(records)

    assert matrix.shape[0] == 2
    assert "gt_iou" not in builder.feature_names
    assert "candidate_iou" not in builder.feature_names
    assert "gt_class" not in builder.feature_names
    assert "verifier_label" not in builder.feature_names


def test_stage2t_select_candidate_indices_is_groupwise() -> None:
    records = [
        make_record(0, sample_id="a"),
        make_record(1, sample_id="a"),
        make_record(2, sample_id="b"),
        make_record(3, sample_id="b"),
    ]
    quality = [0.1, 0.9, 0.8, 0.2]

    selected = select_candidate_indices(records, quality, k=1)

    assert selected == [1, 2]


def test_stage2t_prediction_rows_emit_two_classes_per_candidate() -> None:
    records = [make_record(0, source_conf=0.5)]
    quality = [0.25]

    rows = build_prediction_rows(records, quality, score_mode="roi", k=1)

    assert [(row["candidate_row_index"], row["class_id"]) for row in rows] == [(0, 0), (0, 1)]
    assert rows[0]["score"] == 0.25 * 0.7
    assert rows[1]["score"] == 0.25 * 0.1
