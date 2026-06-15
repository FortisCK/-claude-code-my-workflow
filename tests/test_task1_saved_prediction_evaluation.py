import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cathaction.training.task1_baseline import (
    DEFAULT_CONFIG,
    evaluate_predictions_from_config,
    merge_config,
)


def test_evaluate_saved_predictions_against_manifest(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.npy"
    eval_manifest = tmp_path / "eval.csv"
    prediction_path = tmp_path / "pred.png"
    prediction_manifest = tmp_path / "predictions.csv"

    np.save(mask_path, np.array([[1, 1], [0, 0]], dtype=np.uint8))
    Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(prediction_path)
    _write_eval_manifest(eval_manifest, mask_path)
    _write_prediction_manifest(prediction_manifest, prediction_path)

    config = merge_config(
        DEFAULT_CONFIG,
        {
            "paths": {
                "repo_root": tmp_path.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
            }
        },
    )

    result = evaluate_predictions_from_config(
        config,
        predictions_csv=prediction_manifest,
    )

    assert result["matched_predictions"] == 1
    assert result["metrics"]["dice"] == pytest.approx(2 / 3)
    assert result["metrics"]["iou"] == pytest.approx(1 / 2)
    assert result["metrics"]["miou"] == pytest.approx(1 / 2)
    assert result["metrics"]["pixel_accuracy"] == pytest.approx(3 / 4)


def test_evaluate_saved_multiclass_predictions_against_manifest(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.npy"
    eval_manifest = tmp_path / "eval.csv"
    prediction_path = tmp_path / "pred.png"
    prediction_manifest = tmp_path / "predictions.csv"

    np.save(mask_path, np.array([[1, 2], [0, 2]], dtype=np.uint8))
    Image.fromarray(np.array([[1, 0], [0, 2]], dtype=np.uint8)).save(prediction_path)
    _write_eval_manifest(eval_manifest, mask_path)
    _write_prediction_manifest(prediction_manifest, prediction_path)

    config = merge_config(
        DEFAULT_CONFIG,
        {
            "paths": {
                "repo_root": tmp_path.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
            },
            "data": {"label_mode": "multiclass_012"},
            "model": {"out_channels": 3},
        },
    )

    result = evaluate_predictions_from_config(
        config,
        predictions_csv=prediction_manifest,
    )

    assert result["matched_predictions"] == 1
    assert result["metrics"]["dice"] == pytest.approx(5 / 6)
    assert result["metrics"]["iou"] == pytest.approx(3 / 4)
    assert result["metrics"]["miou"] == pytest.approx(3 / 4)
    assert result["metrics"]["pixel_accuracy"] == pytest.approx(3 / 4)


def _write_eval_manifest(path: Path, mask_path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split_role",
                "collection",
                "domain",
                "released_split",
                "sample_id",
                "case_id",
                "case_id_source",
                "image_path",
                "mask_path",
                "mask_encoding",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split_role": "released_eval",
                "collection": "animal_test",
                "domain": "animal",
                "released_split": "released_test",
                "sample_id": "ani_00000",
                "case_id": "",
                "case_id_source": "unavailable_from_filename",
                "image_path": "unused.png",
                "mask_path": mask_path.as_posix(),
                "mask_encoding": "npy_multiclass",
            }
        )


def _write_prediction_manifest(path: Path, prediction_path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "collection", "domain", "prediction_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "ani_00000",
                "collection": "animal_test",
                "domain": "animal",
                "prediction_path": prediction_path.as_posix(),
            }
        )
