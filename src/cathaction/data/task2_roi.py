"""ROI crop helpers for CATHACTION Task 2 collision classification."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

from PIL import Image

from cathaction.data.task2 import (
    Task2Sample,
    build_task2_index,
    label_name_to_sample,
    read_task2_split_label_names,
    samples_by_label_name,
)


@dataclass(frozen=True)
class RoiCropConfig:
    """Configuration for a square crop centered on the Task 2 bbox."""

    crop_scale: float = 8.0
    min_crop_size: int = 224
    max_crop_size: int = 512
    fill: int = 0


def load_task2_samples_from_split(
    data_root: Path | str,
    split_file: Path | str,
) -> list[Task2Sample]:
    """Load Task 2 samples referenced by a label/image split manifest."""

    root = Path(data_root)
    by_label = samples_by_label_name(build_task2_index(root))
    label_names = read_task2_split_label_names(root, split_file)
    return [label_name_to_sample(by_label, label_name) for label_name in label_names]


def class_counts(samples: Iterable[Task2Sample]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for sample in samples:
        counts[sample.class_id] = counts.get(sample.class_id, 0) + 1
    return dict(sorted(counts.items()))


def compute_roi_bounds(
    sample: Task2Sample,
    image_width: int,
    image_height: int,
    config: RoiCropConfig,
) -> tuple[int, int, int, int]:
    """Return square crop bounds centered on the bbox, allowing out-of-image padding."""

    box = sample.box
    center_x = box.x_center * image_width
    center_y = box.y_center * image_height
    box_width = box.width * image_width
    box_height = box.height * image_height
    side = max(box_width, box_height) * config.crop_scale
    side = max(side, float(config.min_crop_size))
    side = min(side, float(config.max_crop_size))
    side_int = max(1, int(ceil(side)))

    left = int(round(center_x - side_int / 2.0))
    top = int(round(center_y - side_int / 2.0))
    return left, top, left + side_int, top + side_int


def crop_roi_with_padding(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
    fill: int = 0,
) -> Image.Image:
    """Crop bounds from an image and pad outside regions with a constant value."""

    image = image.convert("RGB")
    left, top, right, bottom = bounds
    crop_width = right - left
    crop_height = bottom - top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"Invalid ROI bounds: {bounds}")

    canvas = Image.new("RGB", (crop_width, crop_height), color=(fill, fill, fill))
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, image.width)
    src_bottom = min(bottom, image.height)
    if src_right <= src_left or src_bottom <= src_top:
        return canvas

    patch = image.crop((src_left, src_top, src_right, src_bottom))
    dst_left = src_left - left
    dst_top = src_top - top
    canvas.paste(patch, (dst_left, dst_top))
    return canvas


def crop_task2_roi(
    image: Image.Image,
    sample: Task2Sample,
    config: RoiCropConfig,
) -> Image.Image:
    bounds = compute_roi_bounds(sample, image.width, image.height, config)
    return crop_roi_with_padding(image, bounds, fill=config.fill)

