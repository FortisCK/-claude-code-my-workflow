from __future__ import annotations

import csv
from pathlib import Path
from argparse import Namespace

from scripts.task2.run_stage2_champion_pipeline import (
    DEFAULT_AE_TRANSFORM_JSON,
    DEFAULT_AI_POLICY_JSON,
    DEFAULT_AN_SCALE_JSON,
    apply_score_scales_to_csv,
    apply_stage2aq_replacement_to_csv,
    candidate_has_gt,
    load_score_scales,
    summarize_prediction_csv,
    variant_score_policy_json,
    variant_transform_json,
)


def test_candidate_has_gt_detects_required_fields() -> None:
    assert candidate_has_gt(
        [{"sample_id": "s1", "gt_class": "0", "gt_x1": "1", "gt_y1": "2", "gt_x2": "3", "gt_y2": "4"}]
    )
    assert not candidate_has_gt([{"sample_id": "s1"}])
    assert not candidate_has_gt([])


def test_stage2ai_variant_uses_calibrated_transform_and_custom_policy() -> None:
    args = Namespace(
        stage2ad_transform_json=Path("ad.json"),
        stage2ae_transform_json=DEFAULT_AE_TRANSFORM_JSON,
        stage2ai_transform_json=DEFAULT_AE_TRANSFORM_JSON,
        stage2ai_policy_json=DEFAULT_AI_POLICY_JSON,
        stage2an_transform_json=DEFAULT_AE_TRANSFORM_JSON,
        stage2an_scale_json=DEFAULT_AN_SCALE_JSON,
        stage2aq_transform_json=DEFAULT_AE_TRANSFORM_JSON,
    )

    assert variant_transform_json(args, "stage2ai") == DEFAULT_AE_TRANSFORM_JSON
    assert variant_score_policy_json(args, "stage2ai") == DEFAULT_AI_POLICY_JSON
    assert variant_score_policy_json(args, "stage2ae") is None
    assert variant_transform_json(args, "stage2an") == DEFAULT_AE_TRANSFORM_JSON
    assert variant_score_policy_json(args, "stage2an") is None
    assert variant_transform_json(args, "stage2aq") == DEFAULT_AE_TRANSFORM_JSON
    assert variant_score_policy_json(args, "stage2aq") is None


def test_load_score_scales_accepts_stage2an_best_policy(tmp_path: Path) -> None:
    path = tmp_path / "scale.json"
    path.write_text(
        '{"best_policy": {"class0_phantom": 1.5, "class1_animal": 0.5}}',
        encoding="utf-8",
    )

    scales = load_score_scales(path)

    assert scales == {("0", "phantom"): 1.5, ("1", "animal"): 0.5}


def test_summarize_prediction_csv_accepts_clean_schema(tmp_path: Path) -> None:
    path = tmp_path / "pred.csv"
    fields = [
        "sample_id",
        "video_id",
        "frame_index",
        "domain",
        "class_id",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "source",
        "source_rank",
        "score_mode",
        "policy_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "video_id": "v1",
                "frame_index": "1",
                "domain": "phantom",
                "class_id": "1",
                "score": "0.8",
                "x1": "1",
                "y1": "2",
                "x2": "3",
                "y2": "4",
                "source": "yolo",
                "source_rank": "1",
                "score_mode": "rank_decay_roi",
                "policy_name": "class1_phantom",
            }
        )

    summary = summarize_prediction_csv(path)

    assert summary["rows"] == 1
    assert summary["class_counts"] == {"1": 1}
    assert summary["domain_counts"] == {"phantom": 1}
    assert summary["min_score"] == 0.8
    assert summary["max_score"] == 0.8


def test_apply_score_scales_to_csv_preserves_clean_schema(tmp_path: Path) -> None:
    path = tmp_path / "pred.csv"
    fields = [
        "sample_id",
        "video_id",
        "frame_index",
        "domain",
        "class_id",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "source",
        "source_rank",
        "score_mode",
        "policy_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "video_id": "v1",
                "frame_index": "1",
                "domain": "phantom",
                "class_id": "0",
                "score": "0.2",
                "x1": "1",
                "y1": "2",
                "x2": "3",
                "y2": "4",
                "source": "yolo",
                "source_rank": "1",
                "score_mode": "rank_decay_roi",
                "policy_name": "class0_phantom",
            }
        )
        writer.writerow(
            {
                "sample_id": "s2",
                "video_id": "v2",
                "frame_index": "2",
                "domain": "animal",
                "class_id": "1",
                "score": "0.4",
                "x1": "5",
                "y1": "6",
                "x2": "7",
                "y2": "8",
                "source": "yolo",
                "source_rank": "1",
                "score_mode": "rank_decay_roi",
                "policy_name": "class1_animal",
            }
        )

    result = apply_score_scales_to_csv(path, {("0", "phantom"): 1.5})
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    assert result["scaled_rows"] == 1
    assert list(rows[0]) == fields
    assert abs(float(rows[0]["score"]) - 0.3) < 1e-12
    assert rows[0]["policy_name"] == "class0_phantom_scale1.5"
    assert float(rows[1]["score"]) == 0.4
    assert rows[1]["policy_name"] == "class1_animal"


def test_apply_stage2aq_replacement_only_swaps_phantom_class1(tmp_path: Path) -> None:
    fields = [
        "sample_id",
        "video_id",
        "frame_index",
        "domain",
        "class_id",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "source",
        "source_rank",
        "score_mode",
        "policy_name",
    ]
    prediction_csv = tmp_path / "valid_phantom_domain_policy_predictions.csv"
    with prediction_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "video_id": "v1",
                "frame_index": "1",
                "domain": "phantom",
                "class_id": "0",
                "score": "0.9",
                "x1": "0",
                "y1": "0",
                "x2": "10",
                "y2": "10",
                "source": "yolo_stage2l",
                "source_rank": "1",
                "score_mode": "base",
                "policy_name": "class0_phantom",
            }
        )
        writer.writerow(
            {
                "sample_id": "s1",
                "video_id": "v1",
                "frame_index": "1",
                "domain": "phantom",
                "class_id": "1",
                "score": "0.1",
                "x1": "1",
                "y1": "1",
                "x2": "11",
                "y2": "11",
                "source": "yolo_stage2l",
                "source_rank": "1",
                "score_mode": "base",
                "policy_name": "class1_phantom",
            }
        )
        writer.writerow(
            {
                "sample_id": "s2",
                "video_id": "v2",
                "frame_index": "2",
                "domain": "animal",
                "class_id": "1",
                "score": "0.8",
                "x1": "2",
                "y1": "2",
                "x2": "12",
                "y2": "12",
                "source": "yolo_stage2l",
                "source_rank": "1",
                "score_mode": "base",
                "policy_name": "class1_animal",
            }
        )

    multisource_dir = tmp_path / "multi"
    multisource_dir.mkdir()
    candidate_fields = [
        "sample_id",
        "video_id",
        "frame_index",
        "domain",
        "source",
        "source_rank",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    prediction_fields = ["sample_id", "source", "source_rank", "rank_decay_roi_score_collision"]
    with (multisource_dir / "valid_phantom_candidates_used.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "video_id": "v1",
                "frame_index": "1",
                "domain": "phantom",
                "source": "yolo_stage2l",
                "source_rank": "1",
                "x1": "3",
                "y1": "3",
                "x2": "13",
                "y2": "13",
            }
        )
        writer.writerow(
            {
                "sample_id": "s1",
                "video_id": "v1",
                "frame_index": "1",
                "domain": "phantom",
                "source": "stage2x_class1",
                "source_rank": "2",
                "x1": "4",
                "y1": "4",
                "x2": "14",
                "y2": "14",
            }
        )
        writer.writerow(
            {
                "sample_id": "s3",
                "video_id": "v3",
                "frame_index": "3",
                "domain": "animal",
                "source": "stage2x_class1",
                "source_rank": "1",
                "x1": "5",
                "y1": "5",
                "x2": "15",
                "y2": "15",
            }
        )
    with (multisource_dir / "valid_phantom_eval_prediction_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "source": "yolo_stage2l",
                "source_rank": "1",
                "rank_decay_roi_score_collision": "0.4",
            }
        )
        writer.writerow(
            {
                "sample_id": "s1",
                "source": "stage2x_class1",
                "source_rank": "2",
                "rank_decay_roi_score_collision": "0.6",
            }
        )
        writer.writerow(
            {
                "sample_id": "s3",
                "source": "stage2x_class1",
                "source_rank": "1",
                "rank_decay_roi_score_collision": "0.99",
            }
        )

    summary = apply_stage2aq_replacement_to_csv(
        prediction_csv,
        multisource_dir,
        "valid_phantom",
        source_policy="yolo_stage2x",
        score_mode="rank_decay_roi",
        topk=1,
        score_scale=0.5,
    )
    rows = list(csv.DictReader(prediction_csv.open(newline="", encoding="utf-8")))

    assert summary["baseline_rows"] == 3
    assert summary["kept_baseline_rows"] == 2
    assert summary["replacement_rows"] == 1
    assert len(rows) == 3
    assert list(rows[0]) == fields
    assert {row["policy_name"] for row in rows} == {
        "class0_phantom",
        "class1_animal",
        "stage2aq_rank_decay_roi_top1_x0.5",
    }
    replacement = [row for row in rows if row["policy_name"].startswith("stage2aq")][0]
    assert replacement["source"] == "stage2x_class1"
    assert abs(float(replacement["score"]) - 0.3) < 1e-12
