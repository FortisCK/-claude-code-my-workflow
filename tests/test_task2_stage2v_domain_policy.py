from __future__ import annotations

from scripts.task2.sweep_stage2v_domain_policy import available_score_modes


def test_available_score_modes_requires_both_class_columns() -> None:
    rows = [
        (
            {"sample_id": "s"},
            {
                "roi_score_normal": "0.9",
                "roi_score_collision": "0.1",
                "rank_decay_roi_score_normal": "0.8",
            },
        )
    ]

    assert available_score_modes(rows) == ["roi"]

