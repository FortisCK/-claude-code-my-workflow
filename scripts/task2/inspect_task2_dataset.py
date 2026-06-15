#!/usr/bin/env python3
"""Inspect the local CATHACTION Task 2 collision-detection dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from cathaction.data.task2 import (
    Task2Sample,
    build_task2_index,
    read_task2_split_label_names,
    repo_relative,
    samples_by_label_name,
)


def inspect_dataset(
    *,
    data_root: Path,
    output_json: Path | None,
    repo_root: Path,
    check_shapes: bool,
) -> dict[str, object]:
    samples = build_task2_index(data_root)
    by_label = samples_by_label_name(samples)
    split_names = ["train_phantom.txt", "valid_phantom.txt", "valid_animal.txt"]
    split_label_names = {
        name: read_task2_split_label_names(data_root, data_root / name) for name in split_names
    }

    split_summaries = {}
    for name, label_names in split_label_names.items():
        split_samples = [by_label[label_name] for label_name in label_names]
        split_summaries[name] = summarize_samples(split_samples)

    summary: dict[str, object] = {
        "data_root": repo_relative(data_root, repo_root),
        "images": len(list((data_root / "images").glob("*.jpg"))),
        "labels": len(list((data_root / "labels").glob("*.txt"))),
        "indexed_samples": len(samples),
        "all_samples": summarize_samples(samples),
        "split_summaries": split_summaries,
        "split_overlap": compute_split_overlap(split_label_names),
    }
    if check_shapes:
        summary["image_shapes"] = summarize_image_shapes(samples)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def summarize_samples(samples: list[Task2Sample]) -> dict[str, object]:
    class_counts = Counter(sample.class_id for sample in samples)
    video_counts = Counter(sample.video_id for sample in samples)
    box_counts = {
        "width": quantiles(sorted(sample.box.width for sample in samples)),
        "height": quantiles(sorted(sample.box.height for sample in samples)),
        "area": quantiles(sorted(sample.box.width * sample.box.height for sample in samples)),
    }
    return {
        "samples": len(samples),
        "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "videos": len(video_counts),
        "top_videos": [
            {"video_id": key, "samples": int(value)}
            for key, value in video_counts.most_common(10)
        ],
        "bbox": box_counts,
    }


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    n = len(values)
    return {
        "min": values[0],
        "p25": values[n // 4],
        "median": values[n // 2],
        "p75": values[(3 * n) // 4],
        "max": values[-1],
    }


def compute_split_overlap(split_label_names: dict[str, list[str]]) -> dict[str, object]:
    split_sets = {name: set(values) for name, values in split_label_names.items()}
    result: dict[str, object] = {}
    names = sorted(split_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = sorted(split_sets[left_name] & split_sets[right_name])
            result[f"{left_name}__{right_name}"] = {
                "count": len(overlap),
                "examples": overlap[:10],
            }
    return result


def summarize_image_shapes(samples: list[Task2Sample]) -> list[dict[str, object]]:
    counter: Counter[tuple[int, int]] = Counter()
    for sample in samples:
        with Image.open(sample.image_path) as image:
            counter[image.size] += 1
    return [
        {"width": width, "height": height, "count": int(count)}
        for (width, height), count in counter.most_common()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--check-shapes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    output_json = None
    if args.output_json is not None:
        output_json = (
            (repo_root / args.output_json).resolve()
            if not args.output_json.is_absolute()
            else args.output_json.resolve()
        )
    summary = inspect_dataset(
        data_root=data_root,
        output_json=output_json,
        repo_root=repo_root,
        check_shapes=args.check_shapes,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
