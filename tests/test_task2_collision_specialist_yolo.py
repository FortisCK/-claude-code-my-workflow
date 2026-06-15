from pathlib import Path

import numpy as np
from PIL import Image

from scripts.task2.prepare_collision_specialist_yolo import prepare_collision_specialist_yolo


def test_prepare_collision_specialist_yolo_rewrites_positive_and_empty_negative_labels(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    split_dir = tmp_path / "source_splits"
    specialist_root = tmp_path / "collision_only"
    output_split_dir = tmp_path / "collision_splits"
    config_dir = tmp_path / "configs"
    _write_task2_pair(source_root, "video_0_00000001", "1 0.5 0.25 0.1 0.2")
    _write_task2_pair(source_root, "video_0_00000002", "0 0.2 0.3 0.4 0.5")
    split_dir.mkdir()
    for split_name in (
        "train_clean",
        "valid_combined",
        "valid_phantom",
        "valid_animal",
        "train_smoke",
        "valid_smoke",
    ):
        (split_dir / f"{split_name}_labels.txt").write_text(
            "labels/video_0_00000001.txt\nlabels/video_0_00000002.txt\n",
            encoding="utf-8",
        )

    summary = prepare_collision_specialist_yolo(
        source_data_root=source_root,
        specialist_data_root=specialist_root,
        source_split_dir=split_dir,
        output_split_dir=output_split_dir,
        config_dir=config_dir,
        repo_root=tmp_path,
        class_name="collision_roi",
        collision_class_id=1,
        positive_repeat=1,
    )

    assert summary["labels_written"] == 2
    assert summary["positive_labels"] == 1
    assert summary["empty_negative_labels"] == 1
    assert (specialist_root / "labels" / "video_0_00000001.txt").read_text() == (
        "0 0.5 0.25 0.1 0.2\n"
    )
    assert (specialist_root / "labels" / "video_0_00000002.txt").read_text() == ""
    yaml_text = (config_dir / "collision_detection_collision_only_clean_combined.local.yaml").read_text()
    assert "0: collision_roi" in yaml_text
    assert "1:" not in yaml_text


def test_prepare_collision_specialist_yolo_repeats_train_positives_only(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    split_dir = tmp_path / "source_splits"
    specialist_root = tmp_path / "collision_only"
    output_split_dir = tmp_path / "collision_splits"
    config_dir = tmp_path / "configs"
    _write_task2_pair(source_root, "video_0_00000001", "1 0.5 0.25 0.1 0.2")
    _write_task2_pair(source_root, "video_0_00000002", "0 0.2 0.3 0.4 0.5")
    split_dir.mkdir()
    for split_name in (
        "train_clean",
        "valid_combined",
        "valid_phantom",
        "valid_animal",
        "train_smoke",
        "valid_smoke",
    ):
        (split_dir / f"{split_name}_labels.txt").write_text(
            "labels/video_0_00000001.txt\nlabels/video_0_00000002.txt\n",
            encoding="utf-8",
        )

    summary = prepare_collision_specialist_yolo(
        source_data_root=source_root,
        specialist_data_root=specialist_root,
        source_split_dir=split_dir,
        output_split_dir=output_split_dir,
        config_dir=config_dir,
        repo_root=tmp_path,
        class_name="collision_roi",
        collision_class_id=1,
        positive_repeat=3,
    )

    train_images = (output_split_dir / "train_clean_images.txt").read_text().splitlines()
    valid_images = (output_split_dir / "valid_combined_images.txt").read_text().splitlines()
    assert len(train_images) == 4
    assert sum(path.endswith("video_0_00000001.jpg") for path in train_images) == 3
    assert len(valid_images) == 2
    assert summary["split_summaries"]["train_clean"]["listed_collision_positive_labels"] == 3
    assert summary["split_summaries"]["valid_combined"]["listed_collision_positive_labels"] == 1


def _write_task2_pair(root: Path, stem: str, label: str) -> None:
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / f"{stem}.jpg")
    (label_dir / f"{stem}.txt").write_text(label + "\n", encoding="utf-8")
