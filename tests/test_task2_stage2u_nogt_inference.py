from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.task2.infer_stage2u_quality_ranker import (
    choose_backend_for_checkpoint,
    load_inference_candidate_csv,
    make_prediction_row,
)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_load_inference_candidate_csv_does_not_require_gt_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(image_path)
    candidate_csv = tmp_path / "hidden_candidates.csv"
    write_csv(
        candidate_csv,
        [
            "sample_id",
            "video_id",
            "frame_index",
            "domain",
            "image_path",
            "x1",
            "y1",
            "x2",
            "y2",
            "source_conf",
            "source",
            "source_rank",
        ],
        [
            {
                "sample_id": "hidden_000001",
                "video_id": "hidden",
                "frame_index": 1,
                "domain": "human",
                "image_path": image_path.as_posix(),
                "x1": 5,
                "y1": 6,
                "x2": 25,
                "y2": 26,
                "source_conf": 0.7,
                "source": "stage2x_class1",
                "source_rank": 2,
            }
        ],
    )

    candidates = load_inference_candidate_csv(candidate_csv, split="hidden")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.gt_class == -1
    assert candidate.verifier_label == -1
    assert candidate.gt_iou == 0.0
    assert candidate.image_width == 64
    assert candidate.image_height == 48
    assert candidate.xyxy == (5.0, 6.0, 25.0, 26.0)


def test_make_prediction_row_includes_stage2aq_score_columns(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(image_path)
    candidate_csv = tmp_path / "hidden_candidates.csv"
    write_csv(
        candidate_csv,
        ["sample_id", "image_path", "x1", "y1", "x2", "y2", "source", "source_rank"],
        [
            {
                "sample_id": "phantom_case_000001",
                "image_path": image_path.as_posix(),
                "x1": 5,
                "y1": 6,
                "x2": 25,
                "y2": 26,
                "source": "stage2x_class1",
                "source_rank": 2,
            }
        ],
    )
    candidate = load_inference_candidate_csv(candidate_csv, split="hidden")[0]

    row = make_prediction_row(
        candidate,
        probs=np.asarray([0.1, 0.2, 0.7], dtype=np.float32),
        pred_iou_value=0.6,
        prob_iou50=0.8,
        prob_iou75=0.4,
    )

    assert row["sample_id"] == "phantom_case_000001"
    assert row["gt_class"] == -1
    assert "rank_decay_roi_score_collision" in row
    assert "prob_iou75_rank_decay_roi_score_collision" in row
    assert row["rank_decay_roi_score_collision"] > 0


def test_choose_backend_for_checkpoint_detects_convnext_implementations() -> None:
    assert choose_backend_for_checkpoint("auto", {"stages.0.blocks.0.gamma": object()}) == "timm"
    assert choose_backend_for_checkpoint("auto", {"features.0.0.weight": object()}) == "torchvision"
    assert choose_backend_for_checkpoint("timm", {"features.0.0.weight": object()}) == "timm"
