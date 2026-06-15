#!/usr/bin/env python3
"""Prepare CATHACTION Task 2 data for the official YOLOV/YOLOV++ codebase."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import (  # noqa: E402
    Task2Sample,
    build_task2_index,
    label_name_to_sample,
    read_task2_split_label_names,
    repo_relative,
    samples_by_label_name,
)
from cathaction.data.task2_roi import class_counts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--train-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_train_labels.txt"),
    )
    parser.add_argument(
        "--valid-combined-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_combined_balanced_labels.txt"),
    )
    parser.add_argument(
        "--valid-phantom-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/valid_phantom_balanced_small_labels.txt"),
    )
    parser.add_argument(
        "--valid-animal-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_animal_labels.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/collision_detection_yolov"))
    parser.add_argument("--class-mode", choices=("agnostic", "two_class"), default="agnostic")
    parser.add_argument("--image-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument(
        "--limit-strategy",
        choices=("first", "stratified_temporal"),
        default="first",
        help="How to reduce train/valid splits when a limit is provided.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = build_task2_index(data_root)
    by_label = samples_by_label_name(all_samples)
    train_samples = maybe_limit(
        samples_from_split(data_root, resolve_path(args.train_split), by_label),
        args.train_limit,
        strategy=args.limit_strategy,
    )
    valid_sets = {
        "valid_combined": maybe_limit(
            samples_from_split(data_root, resolve_path(args.valid_combined_split), by_label),
            args.valid_limit,
            strategy=args.limit_strategy,
        ),
        "valid_phantom": maybe_limit(
            samples_from_split(data_root, resolve_path(args.valid_phantom_split), by_label),
            args.valid_limit,
            strategy=args.limit_strategy,
        ),
        "valid_animal": maybe_limit(
            samples_from_split(data_root, resolve_path(args.valid_animal_split), by_label),
            args.valid_limit,
            strategy=args.limit_strategy,
        ),
    }

    categories = make_categories(args.class_mode)
    train_summary = write_split_dataset(
        samples=train_samples,
        data_root=data_root,
        output_dir=output_dir,
        split_name="train",
        json_name="cathaction_train.json",
        categories=categories,
        class_mode=args.class_mode,
        image_mode=args.image_mode,
    )
    valid_summaries: dict[str, Any] = {}
    for valid_name, valid_samples in valid_sets.items():
        valid_summaries[valid_name] = write_split_dataset(
            samples=valid_samples,
            data_root=data_root,
            output_dir=output_dir,
            split_name=valid_name,
            json_name=f"cathaction_{valid_name}.json",
            categories=categories,
            class_mode=args.class_mode,
            image_mode=args.image_mode,
        )

    summary = {
        "data_root": repo_relative(data_root, REPO_ROOT),
        "output_dir": repo_relative(output_dir, REPO_ROOT),
        "class_mode": args.class_mode,
        "image_mode": args.image_mode,
        "categories": categories,
        "splits": {
            "train": train_summary,
            **valid_summaries,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2), flush=True)
    return 0


def write_split_dataset(
    *,
    samples: list[Task2Sample],
    data_root: Path,
    output_dir: Path,
    split_name: str,
    json_name: str,
    categories: list[dict[str, Any]],
    class_mode: str,
    image_mode: str,
) -> dict[str, Any]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted({sample.video_id for sample in samples})
    video_to_sid = {video_id: index for index, video_id in enumerate(videos)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 0
    annotation_id = 1
    per_video_frame_names: dict[str, list[str]] = {video_id: [] for video_id in videos}

    for sample in sorted(samples, key=lambda item: (item.video_id, item.frame_index, item.sample_id)):
        width, height = image_size(sample.image_path)
        image_name = f"{sample.sample_id}.jpg"
        link_or_copy_image(sample.image_path, split_dir / image_name, mode=image_mode)
        sid = video_to_sid[sample.video_id]
        images.append(
            {
                "id": image_id,
                "name": image_name,
                "file_name": image_name,
                "width": width,
                "height": height,
                "sid": sid,
                "video_id": sid,
                "frame_id": sample.frame_index,
            }
        )
        per_video_frame_names[sample.video_id].append(image_name)

        x1, y1, x2, y2 = sample.box.xyxy_pixels(width, height)
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id_for_sample(sample, class_mode=class_mode),
                "bbox": [x1, y1, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
                "segmentation": [],
            }
        )
        image_id += 1
        annotation_id += 1

    video_entries = [
        {
            "id": video_to_sid[video_id],
            "name": video_id,
            "file_names": per_video_frame_names[video_id],
            "length": len(per_video_frame_names[video_id]),
        }
        for video_id in videos
    ]
    annotation = {
        "info": {
            "description": "CATHACTION Task2 converted for YOLOV/YOLOV++",
            "class_mode": class_mode,
        },
        "licenses": [],
        "videos": video_entries,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    json_path = output_dir / json_name
    json_path.write_text(json.dumps(json_ready(annotation), indent=2) + "\n", encoding="utf-8")

    return {
        "samples": len(samples),
        "videos": len(videos),
        "class_counts_original": class_counts(samples),
        "class_counts_exported": exported_class_counts(samples, class_mode=class_mode),
        "domain_counts": dict(sorted(Counter(domain_for_video(sample.video_id) for sample in samples).items())),
        "image_dir": repo_relative(split_dir, REPO_ROOT),
        "annotation": repo_relative(json_path, REPO_ROOT),
    }


def samples_from_split(
    data_root: Path,
    split_file: Path,
    by_label: dict[str, Task2Sample],
) -> list[Task2Sample]:
    label_names = read_task2_split_label_names(data_root, split_file)
    return [label_name_to_sample(by_label, label_name) for label_name in label_names]


def maybe_limit(
    samples: list[Task2Sample],
    limit: int | None,
    *,
    strategy: str = "first",
) -> list[Task2Sample]:
    if limit is None or limit >= len(samples):
        return samples
    if strategy == "first":
        return samples[:limit]
    if strategy == "stratified_temporal":
        return stratified_temporal_limit(samples, limit)
    raise ValueError(f"Unsupported limit strategy: {strategy}")


def stratified_temporal_limit(samples: list[Task2Sample], limit: int) -> list[Task2Sample]:
    buckets: dict[tuple[str, int], list[Task2Sample]] = {}
    for sample in samples:
        key = (domain_for_video(sample.video_id), sample.class_id)
        buckets.setdefault(key, []).append(sample)

    selected: list[Task2Sample] = []
    keys = sorted(buckets)
    base_quota = limit // max(1, len(keys))
    remainder = limit % max(1, len(keys))
    for index, key in enumerate(keys):
        bucket = buckets[key]
        take = min(len(bucket), base_quota + int(index < remainder))
        selected.extend(even_temporal_sample(bucket, take))

    if len(selected) < limit:
        used = {sample.sample_id for sample in selected}
        remaining_pool = [sample for sample in samples if sample.sample_id not in used]
        selected.extend(even_temporal_sample(remaining_pool, limit - len(selected)))

    return sorted(selected[:limit], key=lambda item: (item.video_id, item.frame_index, item.sample_id))


def even_temporal_sample(samples: list[Task2Sample], count: int) -> list[Task2Sample]:
    ordered = sorted(samples, key=lambda item: (item.video_id, item.frame_index, item.sample_id))
    if count >= len(ordered):
        return ordered
    if count <= 0:
        return []
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        round(index * (len(ordered) - 1) / (count - 1))
        for index in range(count)
    }
    selected = [ordered[index] for index in sorted(indices)]
    if len(selected) < count:
        used = {sample.sample_id for sample in selected}
        for sample in ordered:
            if sample.sample_id in used:
                continue
            selected.append(sample)
            if len(selected) == count:
                break
    return sorted(selected, key=lambda item: (item.video_id, item.frame_index, item.sample_id))


def make_categories(class_mode: str) -> list[dict[str, Any]]:
    if class_mode == "agnostic":
        return [{"id": 0, "name": "tool_tip", "supercategory": "tool"}]
    if class_mode == "two_class":
        return [
            {"id": 0, "name": "normal", "supercategory": "collision_state"},
            {"id": 1, "name": "collision", "supercategory": "collision_state"},
        ]
    raise ValueError(f"Unsupported class mode: {class_mode}")


def category_id_for_sample(sample: Task2Sample, *, class_mode: str) -> int:
    if class_mode == "agnostic":
        return 0
    if class_mode == "two_class":
        return int(sample.class_id)
    raise ValueError(f"Unsupported class mode: {class_mode}")


def exported_class_counts(samples: list[Task2Sample], *, class_mode: str) -> dict[str, int]:
    counts = Counter(category_id_for_sample(sample, class_mode=class_mode) for sample in samples)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def link_or_copy_image(source: Path, target: Path, *, mode: str) -> None:
    if target.exists() or target.is_symlink():
        return
    if mode == "symlink":
        try:
            target.symlink_to(source.resolve())
            return
        except OSError:
            mode = "hardlink"
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            mode = "copy"
    if mode == "copy":
        shutil.copy2(source, target)
        return
    raise ValueError(f"Unsupported image mode: {mode}")


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def domain_for_video(video_id: str) -> str:
    return "animal" if "animal" in video_id.lower() else "phantom"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value, REPO_ROOT)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return value


if __name__ == "__main__":
    raise SystemExit(main())
