from pathlib import Path

import numpy as np
from PIL import Image

from scripts.task2.prepare_class_agnostic_yolo import prepare_class_agnostic_yolo


def test_prepare_class_agnostic_yolo_rewrites_labels_and_yaml(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    split_dir = tmp_path / "source_splits"
    agnostic_root = tmp_path / "agnostic"
    output_split_dir = tmp_path / "agnostic_splits"
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

    summary = prepare_class_agnostic_yolo(
        source_data_root=source_root,
        agnostic_data_root=agnostic_root,
        source_split_dir=split_dir,
        output_split_dir=output_split_dir,
        config_dir=config_dir,
        repo_root=tmp_path,
        class_name="tool_roi",
    )

    assert summary["labels_written"] == 2
    assert (agnostic_root / "labels" / "video_0_00000001.txt").read_text() == (
        "0 0.5 0.25 0.1 0.2\n"
    )
    assert (agnostic_root / "labels" / "video_0_00000002.txt").read_text() == (
        "0 0.2 0.3 0.4 0.5\n"
    )
    yaml_text = (config_dir / "collision_detection_agnostic_clean_combined.local.yaml").read_text()
    assert "0: tool_roi" in yaml_text
    assert "1:" not in yaml_text
    train_images = (output_split_dir / "train_clean_images.txt").read_text().splitlines()
    assert train_images[0].endswith("agnostic/images/video_0_00000001.jpg")


def _write_task2_pair(root: Path, stem: str, label: str) -> None:
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / f"{stem}.jpg")
    (label_dir / f"{stem}.txt").write_text(label + "\n", encoding="utf-8")
