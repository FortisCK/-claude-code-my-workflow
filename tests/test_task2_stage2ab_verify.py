from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.task2.export_clean_predictions import CLEAN_FIELDS
from scripts.task2.verify_stage2ab_domain_policy import verify_stage2ab


def _write_csv(path: Path, *, score_mode: str = "rank_decay_roi", extra_field: str | None = None) -> None:
    fields = list(CLEAN_FIELDS)
    if extra_field is not None:
        fields.append(extra_field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        rows = [
            {
                "sample_id": "s1",
                "video_id": "v1",
                "frame_index": "0",
                "domain": "phantom",
                "class_id": "0",
                "score": "0.8",
                "x1": "1",
                "y1": "2",
                "x2": "10",
                "y2": "12",
                "source": "yolo",
                "source_rank": "1",
                "score_mode": "prob_iou75_source_rank_decay_roi",
                "policy_name": "class0_phantom",
            },
            {
                "sample_id": "s1",
                "video_id": "v1",
                "frame_index": "0",
                "domain": "phantom",
                "class_id": "1",
                "score": "0.7",
                "x1": "1",
                "y1": "2",
                "x2": "10",
                "y2": "12",
                "source": "yolo",
                "source_rank": "1",
                "score_mode": score_mode,
                "policy_name": "class1_phantom",
            },
        ]
        for row in rows:
            if extra_field is not None:
                row[extra_field] = "leak"
            writer.writerow(row)


def _write_manifest(path: Path, *, map50: float = 0.211, map50_95: float = 0.0506) -> None:
    manifest = {
        "artifact_type": "task2_stage2ab_domain_aware_internal_export",
        "base_run_dir": "run",
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
                    "phantom": "rank_decay_roi",
                    "animal": "prob_iou75_roi",
                    "default": "rank_decay_roi",
                }
            },
        },
        "splits": {},
    }
    for split in ("valid_combined", "valid_phantom", "valid_animal"):
        manifest["splits"][split] = {
            "rows": 2,
            "mAP50": map50 if split == "valid_combined" else 0.1,
            "mAP50-95": map50_95 if split == "valid_combined" else 0.01,
            "classes": {"0": {}, "1": {}},
        }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_artifact(path: Path) -> None:
    pred = path / "predictions"
    pred.mkdir(parents=True)
    for split in ("valid_combined", "valid_phantom", "valid_animal"):
        _write_csv(pred / f"{split}_domain_policy_predictions.csv")
    _write_manifest(path / "stage2ab_manifest.json")


def test_verify_stage2ab_accepts_clean_artifact(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    result = verify_stage2ab(
        tmp_path,
        tmp_path / "stage2ab_manifest.json",
        tmp_path / "predictions",
        expected_valid_combined_map50=0.211,
        expected_valid_combined_map50_95=0.0506,
    )

    assert result["artifact_type"] == "task2_stage2ab_domain_policy_verification"
    assert result["splits"]["valid_combined"]["csv"]["rows"] == 2


def test_verify_stage2ab_rejects_wrong_policy_score_mode(tmp_path: Path) -> None:
    _write_artifact(tmp_path)
    _write_csv(tmp_path / "predictions" / "valid_combined_domain_policy_predictions.csv", score_mode="roi")

    with pytest.raises(ValueError, match="expected score_mode"):
        verify_stage2ab(
            tmp_path,
            tmp_path / "stage2ab_manifest.json",
            tmp_path / "predictions",
            expected_valid_combined_map50=0.211,
            expected_valid_combined_map50_95=0.0506,
        )


def test_verify_stage2ab_rejects_forbidden_fields(tmp_path: Path) -> None:
    _write_artifact(tmp_path)
    _write_csv(
        tmp_path / "predictions" / "valid_combined_domain_policy_predictions.csv",
        extra_field="gt_iou",
    )

    with pytest.raises(ValueError, match="forbidden"):
        verify_stage2ab(
            tmp_path,
            tmp_path / "stage2ab_manifest.json",
            tmp_path / "predictions",
            expected_valid_combined_map50=0.211,
            expected_valid_combined_map50_95=0.0506,
        )

