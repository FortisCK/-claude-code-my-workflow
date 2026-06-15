import csv
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.task1.make_prediction_overlays import create_overlays


def test_create_prediction_overlay(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    prediction_path = tmp_path / "pred.png"
    eval_manifest = tmp_path / "eval.csv"
    predictions_csv = tmp_path / "predictions.csv"
    output_dir = tmp_path / "overlays"

    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(image_path)
    Image.fromarray(np.array([[0, 1], [2, 0]], dtype=np.uint8)).save(prediction_path)
    _write_eval_manifest(eval_manifest, image_path)
    _write_prediction_manifest(predictions_csv, prediction_path)

    summary = create_overlays(
        eval_manifest=eval_manifest,
        predictions_csv=predictions_csv,
        output_dir=output_dir,
        repo_root=tmp_path,
        max_samples=None,
        alpha=0.5,
    )

    assert summary["overlays"] == 1
    assert (output_dir / "sample_000_overlay.png").is_file()
    assert (output_dir / "summary.json").is_file()


def _write_eval_manifest(path: Path, image_path: Path) -> None:
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
                "sample_id": "sample_000",
                "case_id": "",
                "case_id_source": "unavailable_from_filename",
                "image_path": image_path.as_posix(),
                "mask_path": "unused.npy",
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
                "sample_id": "sample_000",
                "collection": "animal_test",
                "domain": "animal",
                "prediction_path": prediction_path.as_posix(),
            }
        )
