from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image

from scripts.task2.export_sequence_tip_nogt_proposals import (
    NoGtSequenceTipDataset,
    collate_nogt_sequence_tip,
    load_frames_from_csv,
    load_frames_from_dir,
    rows_for_sample,
)
from scripts.task2.train_sequence_tip_localizer import LetterboxMeta


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(1, 2, 3)).save(path)


def test_load_sequence_tip_nogt_frames_from_dir_and_csv(tmp_path: Path) -> None:
    image_path = tmp_path / "human_video_00000007.jpg"
    make_image(image_path)

    dir_frames = load_frames_from_dir(tmp_path, fallback_domain="unknown")

    assert len(dir_frames) == 1
    assert dir_frames[0].sample_id == "human_video_00000007"
    assert dir_frames[0].video_id == "human_video"
    assert dir_frames[0].frame_index == 7
    assert dir_frames[0].domain == "human"

    image_list = tmp_path / "images.csv"
    write_csv(
        image_list,
        ["sample_id", "image_path", "video_id", "frame_index", "domain"],
        [
            {
                "sample_id": "s1",
                "image_path": image_path.as_posix(),
                "video_id": "v1",
                "frame_index": 3,
                "domain": "phantom",
            }
        ],
    )

    csv_frames = load_frames_from_csv(image_list, fallback_domain="unknown")

    assert csv_frames[0].sample_id == "s1"
    assert csv_frames[0].video_id == "v1"
    assert csv_frames[0].frame_index == 3
    assert csv_frames[0].domain == "phantom"


def test_nogt_sequence_tip_dataset_uses_nearest_neighbors(tmp_path: Path) -> None:
    for index in (1, 3):
        make_image(tmp_path / f"case_0000000{index}.jpg")
    frames = load_frames_from_dir(tmp_path, fallback_domain="human")
    dataset = NoGtSequenceTipDataset(frames, input_size=32, frame_radius=2)

    item = dataset[0]

    assert item["image"].shape == (5, 32, 32)
    assert item["sample_id"] == "case_00000001"
    batch = collate_nogt_sequence_tip([item])
    assert batch["images"].shape == (1, 5, 32, 32)
    assert batch["sample_ids"] == ["case_00000001"]


def test_sequence_tip_rows_for_sample_have_no_gt_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "case_00000001.jpg"
    make_image(image_path)
    meta = LetterboxMeta(
        original_width=32,
        original_height=24,
        input_size=32,
        scale=1.0,
        pad_x=0,
        pad_y=4,
        resized_width=32,
        resized_height=24,
    )

    rows = rows_for_sample(
        split="hidden",
        sample_id="case_00000001",
        video_id="case",
        frame_index=1,
        domain="human",
        image_path=image_path,
        meta=meta,
        candidates=[],
    )

    assert len(rows) == 1
    assert rows[0]["has_proposal"] is False
    assert "gt_class" not in rows[0]
    assert "gt_x1" not in rows[0]
    assert rows[0]["proposal_rank"] == ""
