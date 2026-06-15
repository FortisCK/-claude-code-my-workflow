"""Hermetic test for the Task 1 no-GT inference dataset path (hidden test).

Validates that Task1SegmentationDataset tolerates a manifest with no mask
(mask_encoding == "none" / empty mask_path) and returns a valid image tensor
plus an all-background dummy target, without importing the model stack (smp).
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from cathaction.training.task1_baseline import Task1SegmentationDataset


def _write_nogt_manifest(tmp_path: Path) -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 20, 3), 127, dtype=np.uint8)).save(
        img_dir / "video_x_00000001.png"
    )
    manifest = tmp_path / "nogt_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "mask_path", "mask_encoding", "sample_id", "collection", "domain"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "images/video_x_00000001.png",
                "mask_path": "",
                "mask_encoding": "none",
                "sample_id": "video_x_00000001",
                "collection": "hidden",
                "domain": "unknown",
            }
        )
    return manifest


def test_dataset_no_gt_inference_path(tmp_path: Path) -> None:
    manifest = _write_nogt_manifest(tmp_path)
    dataset = Task1SegmentationDataset(
        manifest_csv=manifest,
        repo_root=tmp_path,
        image_size=(8, 8),
        resize_mode="direct",
        label_mode="multiclass_012",
    )
    item = dataset[0]
    # Image tensor is C,H,W at the requested size; no mask file referenced.
    assert tuple(item["image"].shape) == (3, 8, 8)
    assert item["mask_path"] == ""
    assert item["sample_id"] == "video_x_00000001"
    # Dummy target is all-background (class 0) and never used for prediction.
    assert int(item["mask"].max()) == 0
