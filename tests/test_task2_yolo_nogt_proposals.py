from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image

from scripts.task2.export_yolo_nogt_proposals import (
    load_images_from_csv,
    load_images_from_dir,
    rows_for_item,
    select_proposals,
)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_load_yolo_nogt_images_from_dir_and_csv(tmp_path: Path) -> None:
    image_path = tmp_path / "human_case_00000007.jpg"
    Image.new("RGB", (32, 24), color=(1, 2, 3)).save(image_path)

    dir_items = load_images_from_dir(tmp_path, fallback_domain="unknown")

    assert len(dir_items) == 1
    assert dir_items[0].sample_id == "human_case_00000007"
    assert dir_items[0].video_id == "human_case"
    assert dir_items[0].frame_index == 7
    assert dir_items[0].domain == "human"

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

    csv_items = load_images_from_csv(image_list, fallback_domain="unknown")

    assert csv_items[0].sample_id == "s1"
    assert csv_items[0].video_id == "v1"
    assert csv_items[0].frame_index == 3
    assert csv_items[0].domain == "phantom"


def test_rows_for_item_writes_nogt_schema(tmp_path: Path) -> None:
    image_path = tmp_path / "case_00000001.jpg"
    Image.new("RGB", (32, 24), color=(1, 2, 3)).save(image_path)
    item = load_images_from_dir(tmp_path, fallback_domain="human")[0]

    rows = rows_for_item(item, split="hidden", image_width=32, image_height=24, proposals=[])

    assert len(rows) == 1
    row = rows[0]
    assert row["split"] == "hidden"
    assert row["has_proposal"] is False
    assert "gt_class" not in row
    assert "gt_x1" not in row
    assert row["proposal_rank"] == ""


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
        self.conf = torch.tensor([0.2, 0.9])
        self.cls = torch.tensor([0.0, 1.0])

    def __len__(self) -> int:
        return 2


class FakeResult:
    boxes = FakeBoxes()


def test_select_proposals_sorts_by_confidence() -> None:
    proposals = select_proposals(FakeResult())

    assert [proposal.rank for proposal in proposals] == [1, 2]
    assert abs(proposals[0].confidence - 0.9) < 1e-6
    assert proposals[0].class_id == 1
    assert abs(proposals[1].confidence - 0.2) < 1e-6
