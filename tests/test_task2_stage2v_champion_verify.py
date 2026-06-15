from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.task2.export_clean_predictions import CLEAN_FIELDS
from scripts.task2.verify_stage2v_champion_export import verify_export


def _write_csv(path: Path, *, extra_field: str | None = None) -> None:
    fields = list(CLEAN_FIELDS)
    if extra_field is not None:
        fields.append(extra_field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        row = {
            "sample_id": "s1",
            "video_id": "v1",
            "frame_index": "0",
            "domain": "phantom",
            "class_id": "1",
            "score": "0.75",
            "x1": "1",
            "y1": "2",
            "x2": "10",
            "y2": "12",
            "source": "yolo",
            "source_rank": "0",
            "score_mode": "roi",
            "policy_name": "class1",
        }
        if extra_field is not None:
            row[extra_field] = "leak"
        writer.writerow(row)


def _write_metrics(path: Path, *, map50: float = 0.2, map50_95: float = 0.05) -> None:
    metrics = {
        "class0_run_dir": "run",
        "class0_score_mode": "prob_iou75_source_rank_decay_roi",
        "class1_run_dir": "run",
        "class1_score_mode": "roi",
        "topk_per_sample_class": 0,
        "splits": {},
    }
    for split in ("valid_combined", "valid_phantom", "valid_animal"):
        metrics["splits"][split] = {
            "prediction_rows": 1,
            "detection": {
                "mAP50": map50 if split == "valid_combined" else 0.1,
                "mAP50-95": map50_95 if split == "valid_combined" else 0.01,
                "classes": {"0": {}, "1": {}},
            },
        }
    path.write_text(json.dumps(metrics), encoding="utf-8")


def _write_export_dir(path: Path) -> None:
    path.mkdir(exist_ok=True)
    for split in ("valid_combined", "valid_phantom", "valid_animal"):
        _write_csv(path / f"{split}_predictions.csv")
    _write_metrics(path / "eval_metrics.json")


def test_verify_export_accepts_clean_package(tmp_path: Path) -> None:
    _write_export_dir(tmp_path)

    manifest = verify_export(
        tmp_path,
        tmp_path / "eval_metrics.json",
        expected_valid_combined_map50=0.2,
        expected_valid_combined_map50_95=0.05,
    )

    assert manifest["artifact_type"] == "task2_stage2v_internal_champion_export"
    assert manifest["splits"]["valid_combined"]["csv"]["rows"] == 1
    assert manifest["class0_score_mode"] == "prob_iou75_source_rank_decay_roi"
    assert manifest["class1_score_mode"] == "roi"


def test_verify_export_rejects_forbidden_fields(tmp_path: Path) -> None:
    _write_export_dir(tmp_path)
    _write_csv(tmp_path / "valid_combined_predictions.csv", extra_field="gt_iou")

    with pytest.raises(ValueError, match="forbidden"):
        verify_export(
            tmp_path,
            tmp_path / "eval_metrics.json",
            expected_valid_combined_map50=0.2,
            expected_valid_combined_map50_95=0.05,
        )


def test_verify_export_rejects_metric_mismatch(tmp_path: Path) -> None:
    _write_export_dir(tmp_path)
    _write_metrics(tmp_path / "eval_metrics.json", map50=0.19, map50_95=0.05)

    with pytest.raises(ValueError, match="valid_combined mAP50"):
        verify_export(
            tmp_path,
            tmp_path / "eval_metrics.json",
            expected_valid_combined_map50=0.2,
            expected_valid_combined_map50_95=0.05,
        )


def test_verify_export_can_skip_frozen_metric_check(tmp_path: Path) -> None:
    _write_export_dir(tmp_path)
    _write_metrics(tmp_path / "eval_metrics.json", map50=0.19, map50_95=0.04)

    manifest = verify_export(
        tmp_path,
        tmp_path / "eval_metrics.json",
        expected_valid_combined_map50=0.2,
        expected_valid_combined_map50_95=0.05,
        check_frozen_metrics=False,
    )

    assert manifest["splits"]["valid_combined"]["mAP50"] == 0.19
