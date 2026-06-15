import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cathaction.metrics.mslnet_style import mslnet_style_metrics
from scripts.task1.evaluate_mslnet_style import main as evaluate_mslnet_style_main


def test_mslnet_style_metrics_exact_match() -> None:
    target = np.zeros((3, 3), dtype=np.uint8)
    prediction = np.zeros((3, 3), dtype=np.uint8)
    target[1, 1] = 1
    prediction[1, 1] = 2

    metrics = mslnet_style_metrics(prediction, target, tolerance_radii=[0, 1])

    assert metrics.dice == pytest.approx(1.0)
    assert metrics.iou == pytest.approx(1.0)
    assert metrics.ahd == pytest.approx(0.0)
    assert metrics.precision_by_radius[0] == pytest.approx(1.0)
    assert metrics.recall_by_radius[0] == pytest.approx(1.0)
    assert metrics.f1_by_radius[0] == pytest.approx(1.0)


def test_mslnet_style_metrics_one_pixel_shift() -> None:
    target = np.zeros((3, 3), dtype=np.uint8)
    prediction = np.zeros((3, 3), dtype=np.uint8)
    target[1, 1] = 1
    prediction[1, 2] = 1

    metrics = mslnet_style_metrics(prediction, target, tolerance_radii=[0, 1])

    assert metrics.dice == pytest.approx(0.0)
    assert metrics.iou == pytest.approx(0.0)
    assert metrics.ahd == pytest.approx(1.0)
    assert metrics.f1_by_radius[0] == pytest.approx(0.0)
    assert metrics.precision_by_radius[1] == pytest.approx(1.0)
    assert metrics.recall_by_radius[1] == pytest.approx(1.0)
    assert metrics.f1_by_radius[1] == pytest.approx(1.0)


def test_evaluate_mslnet_style_saved_predictions(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.npy"
    prediction_path = tmp_path / "prediction.png"
    eval_manifest = tmp_path / "eval.csv"
    predictions_csv = tmp_path / "predictions.csv"
    output_json = tmp_path / "mslnet_eval.json"

    target = np.zeros((3, 3), dtype=np.uint8)
    target[1, 1] = 1
    np.save(mask_path, target)

    prediction = np.zeros((3, 3), dtype=np.uint8)
    prediction[1, 2] = 2
    Image.fromarray(prediction).save(prediction_path)
    _write_eval_manifest(eval_manifest, mask_path)
    _write_prediction_manifest(predictions_csv, prediction_path)

    exit_code = evaluate_mslnet_style_main(
        [
            "--repo-root",
            tmp_path.as_posix(),
            "--eval-manifest",
            eval_manifest.as_posix(),
            "--predictions-csv",
            predictions_csv.as_posix(),
            "--output-json",
            output_json.as_posix(),
            "--tolerance-radius",
            "0",
            "--tolerance-radius",
            "1",
        ]
    )

    assert exit_code == 0
    text = output_json.read_text(encoding="utf-8")
    assert '"dice": 0.0' in text
    assert '"f1_r1": 1.0' in text


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
