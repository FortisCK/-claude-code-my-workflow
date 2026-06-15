import csv
from pathlib import Path

import numpy as np
from PIL import Image

from cathaction.data.task1 import build_task1_index, task1_split_role
from scripts.task1.write_task1_split_manifests import write_manifests


def test_task1_split_role_policy() -> None:
    assert task1_split_role("animal_train") == "released_train"
    assert task1_split_role("phantom_train") == "released_train"
    assert task1_split_role("animal_test") == "released_eval"
    assert task1_split_role("phantom_test") == "released_eval"
    assert task1_split_role("human_train") == "human_holdout"


def test_write_manifests(tmp_path: Path) -> None:
    _write_pair(tmp_path, "animal_train", "ani_00000")
    _write_pair(tmp_path, "animal_test", "ani_00001")
    _write_pair(tmp_path, "phantom_train", "pt_00000")
    _write_pair(tmp_path, "phantom_test", "pt_00001")
    _write_human_pair(tmp_path, "JFQ_j3383201_img-00000-00000")

    output_dir = tmp_path / "configs" / "task1" / "splits"
    summary = write_manifests(tmp_path, output_dir, tmp_path)

    assert summary["total_samples"] == 5
    assert summary["roles"]["released_train"]["count"] == 2
    assert summary["roles"]["released_eval"]["count"] == 2
    assert summary["roles"]["human_holdout"]["count"] == 1

    released_train = _read_csv(output_dir / "released_train.csv")
    released_eval = _read_csv(output_dir / "released_eval.csv")
    human_holdout = _read_csv(output_dir / "human_holdout.csv")

    assert {row["collection"] for row in released_train} == {"animal_train", "phantom_train"}
    assert {row["collection"] for row in released_eval} == {"animal_test", "phantom_test"}
    assert human_holdout[0]["case_id"] == "JFQ_j3383201"
    assert human_holdout[0]["split_role"] == "human_holdout"
    assert human_holdout[0]["image_path"].startswith("human_dataset_train/img/")

    indexed = build_task1_index(tmp_path)
    all_manifest_ids = {row["sample_id"] for row in released_train + released_eval + human_holdout}
    assert all_manifest_ids == {sample.sample_id for sample in indexed}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_pair(root: Path, collection: str, stem: str) -> None:
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
