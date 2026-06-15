from __future__ import annotations

from scripts.task2.build_stage2o_candidate_pool import (
    candidate_sort_key_raw_conf,
    candidate_sort_key_source_order,
    parse_candidate_xyxy,
    parse_top_k_values,
    verifier_label_from_iou,
)


def test_stage2o_parse_top_k_values_sorts_and_deduplicates() -> None:
    assert parse_top_k_values("50,1,10,1") == [1, 10, 50]


def test_stage2o_parse_candidate_xyxy_accepts_proposal_columns() -> None:
    row = {
        "proposal_x1": "1",
        "proposal_y1": "2",
        "proposal_x2": "3",
        "proposal_y2": "4",
    }

    assert parse_candidate_xyxy(row) == (1.0, 2.0, 3.0, 4.0)


def test_stage2o_parse_candidate_xyxy_accepts_plain_columns() -> None:
    row = {"x1": "5", "y1": "6", "x2": "7", "y2": "8"}

    assert parse_candidate_xyxy(row) == (5.0, 6.0, 7.0, 8.0)


def test_stage2o_verifier_label_thresholds() -> None:
    assert verifier_label_from_iou(gt_class=0, candidate_iou=0.60, positive_iou=0.50, background_iou=0.20) == 1
    assert verifier_label_from_iou(gt_class=1, candidate_iou=0.60, positive_iou=0.50, background_iou=0.20) == 2
    assert verifier_label_from_iou(gt_class=1, candidate_iou=0.10, positive_iou=0.50, background_iou=0.20) == 0
    assert verifier_label_from_iou(gt_class=1, candidate_iou=0.30, positive_iou=0.50, background_iou=0.20) == -1


def test_stage2o_candidate_sort_keys_are_deterministic() -> None:
    first = {"source_priority": 0, "source_rank": 2, "source_conf": 0.1, "source": "a"}
    second = {"source_priority": 1, "source_rank": 1, "source_conf": 0.9, "source": "b"}

    assert candidate_sort_key_source_order(first) < candidate_sort_key_source_order(second)
    assert candidate_sort_key_raw_conf(second) < candidate_sort_key_raw_conf(first)
