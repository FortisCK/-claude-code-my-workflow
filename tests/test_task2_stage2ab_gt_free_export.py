from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.task2.export_clean_predictions import CLEAN_FIELDS
from scripts.task2.export_stage2ab_domain_policy_predictions import (
    export_split,
    load_score_policies,
    make_prediction_rows,
)


def _candidate(
    sample_id: str,
    *,
    domain: str = "phantom",
    source: str = "yolo_stage2l",
    source_rank: str = "1",
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "video_id": "video_0",
        "frame_index": "3",
        "domain": domain,
        "x1": "1",
        "y1": "2",
        "x2": "11",
        "y2": "12",
        "source": source,
        "source_rank": source_rank,
        "gt_class": "1",
        "gt_iou": "0.9",
        "verifier_label": "1",
    }


def _prediction(sample_id: str, *, source: str = "yolo_stage2l", source_rank: str = "1") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "source": source,
        "source_rank": source_rank,
        "prob_iou75_source_rank_decay_roi_score_normal": "0.61",
        "rank_decay_roi_score_collision": "0.17",
        "prob_iou75_roi_score_collision": "0.73",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_make_prediction_rows_applies_frozen_domain_policy_without_gt() -> None:
    rows = make_prediction_rows(
        [_candidate("s1", domain="phantom"), _candidate("s2", domain="animal")],
        [_prediction("s1"), _prediction("s2")],
        split="valid_combined",
    )

    assert len(rows) == 4
    assert rows[0]["class_id"] == 0
    assert rows[0]["score"] == pytest.approx(0.61)
    assert rows[0]["score_mode"] == "prob_iou75_source_rank_decay_roi"
    assert rows[1]["class_id"] == 0
    assert rows[2]["class_id"] == 1
    assert rows[2]["domain"] == "phantom"
    assert rows[2]["score"] == pytest.approx(0.17)
    assert rows[2]["score_mode"] == "rank_decay_roi"
    assert rows[3]["class_id"] == 1
    assert rows[3]["domain"] == "animal"
    assert rows[3]["score"] == pytest.approx(0.73)
    assert rows[3]["score_mode"] == "prob_iou75_roi"
    assert "gt_iou" not in rows[0]
    assert "verifier_label" not in rows[0]


def test_make_prediction_rows_rejects_missing_domain_by_default() -> None:
    with pytest.raises(ValueError, match="missing/unknown domain"):
        make_prediction_rows(
            [_candidate("s1", domain="")],
            [_prediction("s1")],
            split="valid_combined",
        )


def test_make_prediction_rows_allows_explicit_fallback_domain() -> None:
    rows = make_prediction_rows(
        [_candidate("s1", domain="")],
        [_prediction("s1")],
        split="hidden",
        fallback_domain="phantom",
    )

    assert rows[0]["domain"] == "phantom"


def test_make_prediction_rows_keeps_human_domain_but_uses_animal_policy() -> None:
    rows = make_prediction_rows(
        [_candidate("human_case_0001", domain="human")],
        [_prediction("human_case_0001")],
        split="hidden",
    )

    assert rows[0]["class_id"] == 0
    assert rows[0]["domain"] == "human"
    assert rows[0]["score_mode"] == "prob_iou75_source_rank_decay_roi"
    assert rows[0]["policy_name"] == "class0_human_as_animal"
    assert rows[1]["class_id"] == 1
    assert rows[1]["domain"] == "human"
    assert rows[1]["score"] == pytest.approx(0.73)
    assert rows[1]["score_mode"] == "prob_iou75_roi"
    assert rows[1]["policy_name"] == "class1_human_as_animal"


def test_make_prediction_rows_can_map_human_to_phantom_policy_explicitly() -> None:
    rows = make_prediction_rows(
        [_candidate("human_case_0001", domain="human")],
        [_prediction("human_case_0001")],
        split="hidden",
        human_policy_domain="phantom",
    )

    assert rows[1]["class_id"] == 1
    assert rows[1]["domain"] == "human"
    assert rows[1]["score"] == pytest.approx(0.17)
    assert rows[1]["score_mode"] == "rank_decay_roi"
    assert rows[1]["policy_name"] == "class1_human_as_phantom"


def test_make_prediction_rows_rejects_alignment_mismatch() -> None:
    with pytest.raises(ValueError, match="alignment mismatch"):
        make_prediction_rows(
            [_candidate("s1")],
            [_prediction("s2")],
            split="valid_combined",
        )


def test_make_prediction_rows_applies_optional_box_transform() -> None:
    rows = make_prediction_rows(
        [_candidate("s1", domain="animal")],
        [_prediction("s1")],
        split="hidden",
        box_transforms={
            "animal": {
                "1": {
                    "dx_center": 0.1,
                    "dy_center": -0.2,
                    "log_w": 0.0,
                    "log_h": 0.0,
                }
            }
        },
    )

    normal_row, collision_row = rows
    assert normal_row["class_id"] == 0
    assert normal_row["x1"] == pytest.approx(1.0)
    assert normal_row["y1"] == pytest.approx(2.0)
    assert collision_row["class_id"] == 1
    assert collision_row["x1"] == pytest.approx(2.0)
    assert collision_row["x2"] == pytest.approx(12.0)
    assert collision_row["y1"] == pytest.approx(0.0)
    assert collision_row["y2"] == pytest.approx(10.0)


def test_load_score_policies_accepts_sweep_json(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "best_class_policies": {
                    "0": {
                        "policy": {
                            "phantom": "prob_iou75_source_rank_decay_roi",
                            "animal": "prob_iou75_source_rank_decay_roi",
                            "default": "prob_iou75_source_rank_decay_roi",
                        }
                    },
                    "1": {
                        "policy": {
                            "phantom": "pred_iou_source_rank_decay_roi",
                            "animal": "sqrt_source_roi",
                            "default": "pred_iou_source_rank_decay_roi",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    policies = load_score_policies(path)

    assert policies["1"]["phantom"] == "pred_iou_source_rank_decay_roi"
    assert policies["1"]["animal"] == "sqrt_source_roi"


def test_make_prediction_rows_uses_custom_score_policy() -> None:
    rows = make_prediction_rows(
        [_candidate("s1", domain="animal")],
        [
            {
                **_prediction("s1"),
                "sqrt_source_roi_score_collision": "0.44",
            }
        ],
        split="hidden",
        score_policies={
            "0": {
                "phantom": "prob_iou75_source_rank_decay_roi",
                "animal": "prob_iou75_source_rank_decay_roi",
                "default": "prob_iou75_source_rank_decay_roi",
            },
            "1": {
                "phantom": "rank_decay_roi",
                "animal": "sqrt_source_roi",
                "default": "rank_decay_roi",
            },
        },
    )

    assert rows[1]["class_id"] == 1
    assert rows[1]["score"] == pytest.approx(0.44)
    assert rows[1]["score_mode"] == "sqrt_source_roi"


def test_export_split_writes_clean_header(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    _write_csv(run_dir / "valid_combined_candidates_used.csv", [_candidate("s1")])
    _write_csv(run_dir / "valid_combined_eval_prediction_rows.csv", [_prediction("s1")])

    summary = export_split(run_dir, out_dir, "valid_combined")

    assert summary["rows"] == 2
    output = out_dir / "valid_combined_domain_policy_predictions.csv"
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(CLEAN_FIELDS)
        exported = list(reader)
    assert len(exported) == 2
    assert "gt_iou" not in exported[0]
