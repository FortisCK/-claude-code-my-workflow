from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.task2.verify_stage2aq_upstream import summarize_split


CLEAN_FIELDS = [
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


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_args(tmp_path: Path) -> Namespace:
    return Namespace(
        baseline_dir=tmp_path / "baseline",
        multisource_run_dir=tmp_path / "multi",
        stage2aq_dir=tmp_path / "aq",
        source_policy="yolo_stage2x",
        score_mode="rank_decay_roi",
        topk=5,
        score_scale=0.75,
        allow_missing_stage2aq=False,
    )


def write_minimal_stage2aq_split(tmp_path: Path, *, mismatch: bool = False) -> Namespace:
    args = make_args(tmp_path)
    clean_row = {
        "sample_id": "s1",
        "video_id": "v1",
        "frame_index": "1",
        "domain": "phantom",
        "class_id": "1",
        "score": "0.5",
        "x1": "1",
        "y1": "2",
        "x2": "3",
        "y2": "4",
        "source": "stage2x_class1",
        "source_rank": "1",
        "score_mode": "rank_decay_roi",
        "policy_name": "stage2aq",
    }
    write_csv(args.baseline_dir / "valid_phantom_domain_policy_predictions.csv", CLEAN_FIELDS, [clean_row])
    write_csv(args.stage2aq_dir / "valid_phantom_domain_policy_predictions.csv", CLEAN_FIELDS, [clean_row])

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
    write_csv(
        args.multisource_run_dir / "valid_phantom_candidates_used.csv",
        candidate_fields,
        [
            {
                "sample_id": "s0",
                "video_id": "v0",
                "frame_index": "0",
                "domain": "phantom",
                "source": "yolo_stage2l",
                "source_rank": "1",
                "x1": "0",
                "y1": "1",
                "x2": "2",
                "y2": "3",
            },
            {
                "sample_id": "s1",
                "video_id": "v1",
                "frame_index": "1",
                "domain": "phantom",
                "source": "stage2x_class1",
                "source_rank": "1",
                "x1": "1",
                "y1": "2",
                "x2": "3",
                "y2": "4",
            }
        ],
    )
    write_csv(
        args.multisource_run_dir / "valid_phantom_eval_prediction_rows.csv",
        prediction_fields,
        [
            {
                "sample_id": "s0",
                "source": "yolo_stage2l",
                "source_rank": "1",
                "rank_decay_roi_score_collision": "0.2",
            },
            {
                "sample_id": "s2" if mismatch else "s1",
                "source": "stage2x_class1",
                "source_rank": "1",
                "rank_decay_roi_score_collision": "0.8",
            }
        ],
    )
    return args


def test_stage2aq_upstream_verify_accepts_valid_minimal_split(tmp_path: Path) -> None:
    args = write_minimal_stage2aq_split(tmp_path)

    summary = summarize_split(args, "valid_phantom")

    assert summary["candidate_rows"] == 2
    assert summary["prediction_rows"] == 2
    assert summary["phantom_policy_candidate_rows"] == 2
    assert summary["candidate_has_gt"] is False


def test_stage2aq_upstream_verify_rejects_alignment_mismatch(tmp_path: Path) -> None:
    args = write_minimal_stage2aq_split(tmp_path, mismatch=True)

    with pytest.raises(ValueError, match="alignment mismatch"):
        summarize_split(args, "valid_phantom")
