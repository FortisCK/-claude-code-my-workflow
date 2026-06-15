from pathlib import Path

import numpy as np
from PIL import Image

from cathaction.data.task2 import (
    build_task2_index,
    parse_task2_sample_stem,
    read_task2_split_label_names,
)
from scripts.task2.prepare_yolo_splits import prepare_splits


def test_build_task2_index_reads_one_box_per_image(tmp_path: Path) -> None:
    _write_task2_pair(tmp_path, "video_0_00000001", "1 0.5 0.25 0.1 0.2")

    samples = build_task2_index(tmp_path)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "video_0_00000001"
    assert sample.video_id == "video_0"
    assert sample.frame_index == 1
    assert sample.class_id == 1
    assert sample.box.xyxy_pixels(100, 200) == (45.0, 30.0, 55.00000000000001, 70.0)


def test_parse_task2_sample_stem_supports_animal_video_ids() -> None:
    assert parse_task2_sample_stem("video_2_animal_00000022") == ("video_2_animal", 22)


def test_read_task2_split_label_names_accepts_label_and_image_entries(tmp_path: Path) -> None:
    split = tmp_path / "split.txt"
    split.write_text("labels/video_0_00000001.txt\nimages/video_0_00000002.jpg\n")

    assert read_task2_split_label_names(tmp_path, split) == [
        "video_0_00000001.txt",
        "video_0_00000002.txt",
    ]


def test_prepare_splits_removes_whole_validation_video(tmp_path: Path) -> None:
    # video_0 is a validation video; ALL its train frames must drop (INV-2),
    # including video_0_00000001 which is not an exact frame-level duplicate.
    _write_task2_pair(tmp_path, "video_0_00000001", "0 0.5 0.5 0.1 0.1")
    _write_task2_pair(tmp_path, "video_0_00000002", "1 0.5 0.5 0.1 0.1")
    _write_task2_pair(tmp_path, "video_3_00000001", "1 0.5 0.5 0.1 0.1")
    _write_task2_pair(tmp_path, "video_1_animal_00000001", "1 0.5 0.5 0.1 0.1")
    (tmp_path / "train_phantom.txt").write_text(
        "labels/video_0_00000001.txt\n"
        "labels/video_0_00000002.txt\n"
        "labels/video_3_00000001.txt\n"
    )
    (tmp_path / "valid_phantom.txt").write_text("labels/video_0_00000002.txt\n")
    (tmp_path / "valid_animal.txt").write_text("labels/video_1_animal_00000001.txt\n")

    summary = prepare_splits(
        data_root=tmp_path,
        output_dir=tmp_path / "splits",
        config_dir=tmp_path / "configs",
        repo_root=tmp_path,
        class_names=("normal", "collision"),
    )

    assert summary["raw_split_overlap"]["train_phantom__valid_phantom"] == 1
    disjoint = summary["video_level_disjointness"]
    # video_0_00000001 is same-video-but-not-exact-frame: the residual leak.
    assert disjoint["residual_same_video_leak_frames_removed"] == 1
    assert disjoint["train_clean_video_overlap"] == []

    train_images = (tmp_path / "splits" / "train_clean_images.txt").read_text().splitlines()
    assert len(train_images) == 1
    assert train_images[0].endswith("images/video_3_00000001.jpg")


def _write_task2_pair(root: Path, stem: str, label: str) -> None:
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / f"{stem}.jpg")
    (label_dir / f"{stem}.txt").write_text(label + "\n")
