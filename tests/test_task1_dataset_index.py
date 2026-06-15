from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from cathaction.data.task1 import (
    build_task1_index,
    mask_to_binary_foreground,
    summarize_task1_index,
)


def test_build_task1_index_pairs_all_released_collections(tmp_path: Path) -> None:
    _write_animal_or_phantom_pair(tmp_path, "animal_train", "ani_00000")
    _write_animal_or_phantom_pair(tmp_path, "animal_test", "ani_00001")
    _write_animal_or_phantom_pair(tmp_path, "phantom_train", "pt_00000")
    _write_animal_or_phantom_pair(tmp_path, "phantom_test", "pt_00001")
    _write_human_pair(tmp_path, "JFQ_j3383201_img-00000-00000")

    samples = build_task1_index(tmp_path)

    assert summarize_task1_index(samples) == {
        "animal_test": 1,
        "animal_train": 1,
        "human_train": 1,
        "phantom_test": 1,
        "phantom_train": 1,
    }
    human = next(sample for sample in samples if sample.collection == "human_train")
    assert human.case_id == "JFQ_j3383201"
    assert human.case_id_source == "filename_before_img_marker"
    assert human.mask_path.name == "JFQ_j3383201_img-00000-00000_mask.png"


def test_build_task1_index_detects_missing_masks(tmp_path: Path) -> None:
    image_dir = tmp_path / "segmentation" / "animal_train" / "images"
    image_dir.mkdir(parents=True)
    _write_rgb(image_dir / "ani_00000.png")

    with pytest.raises(ValueError, match="pairing errors"):
        build_task1_index(tmp_path, collections=["animal_train"])


def test_load_mask_and_binary_foreground(tmp_path: Path) -> None:
    _write_animal_or_phantom_pair(tmp_path, "animal_train", "ani_00000")
    sample = build_task1_index(tmp_path, collections=["animal_train"])[0]

    mask = sample.load_mask()
    assert mask.tolist() == [[0, 1], [2, 0]]
    assert mask_to_binary_foreground(mask).tolist() == [[0, 1], [1, 0]]
    assert sample.load_mask(binary_foreground=True).tolist() == [[0, 1], [1, 0]]


def _write_animal_or_phantom_pair(root: Path, collection: str, stem: str) -> None:
    image_dir = root / "segmentation" / collection / "images"
    mask_dir = root / "segmentation" / collection / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(image_dir / f"{stem}.png")
    np.save(mask_dir / f"{stem}.npy", np.array([[0, 1], [2, 0]], dtype=np.uint8))


def _write_human_pair(root: Path, stem: str) -> None:
    image_dir = root / "human_dataset_train" / "img"
    mask_dir = root / "human_dataset_train" / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(image_dir / f"{stem}.jpg")
    Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(
        mask_dir / f"{stem}_mask.png"
    )


def _write_rgb(path: Path) -> None:
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)

