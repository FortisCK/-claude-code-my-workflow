#!/usr/bin/env python3
"""Prepare leakage-free YOLO split files for Task 2 collision detection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cathaction.data.task2 import (
    Task2Sample,
    build_task2_index,
    label_name_to_sample,
    read_task2_split_label_names,
    repo_relative,
    samples_by_label_name,
)


def prepare_splits(
    *,
    data_root: Path,
    output_dir: Path,
    config_dir: Path,
    repo_root: Path,
    class_names: tuple[str, str],
    smoke_train_size: int = 64,
    smoke_val_size: int = 64,
) -> dict[str, object]:
    samples = build_task2_index(data_root)
    by_label = samples_by_label_name(samples)

    train_raw = read_task2_split_label_names(data_root, data_root / "train_phantom.txt")
    valid_phantom = read_task2_split_label_names(data_root, data_root / "valid_phantom.txt")
    valid_animal = read_task2_split_label_names(data_root, data_root / "valid_animal.txt")

    valid_label_set = set(valid_phantom) | set(valid_animal)
    valid_video_ids = {by_label[label_name].video_id for label_name in valid_label_set}
    # Video-level (not frame-level) disjointness. Frame-level dedup alone leaves
    # same-video frames in train, an INV-2 case/procedure-level leak (the shipped
    # train_phantom.txt left 5 video_0 frames in train while video_0 is the entire
    # valid_phantom panel). Drop any train frame whose whole video is in validation.
    # See quality_reports/plans/2026-06-15_prevalidation_roadmap.md.
    train_clean = [
        label_name
        for label_name in train_raw
        if by_label[label_name].video_id not in valid_video_ids
    ]
    leaked_video_ids = sorted(
        {by_label[name].video_id for name in train_clean} & valid_video_ids
    )
    if leaked_video_ids:
        raise AssertionError(
            "Task 2 split leak: train_clean still shares videos with validation: "
            f"{leaked_video_ids}"
        )
    valid_combined = valid_phantom + valid_animal

    train_smoke = select_balanced_label_names(train_clean, by_label, smoke_train_size)
    valid_smoke = select_balanced_label_names(valid_combined, by_label, smoke_val_size)

    splits = {
        "train_clean": train_clean,
        "valid_phantom": valid_phantom,
        "valid_animal": valid_animal,
        "valid_combined": valid_combined,
        "train_smoke": train_smoke,
        "valid_smoke": valid_smoke,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, object] = {}
    image_lists: dict[str, Path] = {}
    label_lists: dict[str, Path] = {}
    for split_name, label_names in splits.items():
        split_samples = [label_name_to_sample(by_label, label_name) for label_name in label_names]
        image_list = output_dir / f"{split_name}_images.txt"
        label_list = output_dir / f"{split_name}_labels.txt"
        write_path_list(image_list, [sample.image_path.resolve() for sample in split_samples])
        write_path_list(label_list, [sample.label_path.resolve() for sample in split_samples])
        image_lists[split_name] = image_list
        label_lists[split_name] = label_list
        split_summaries[split_name] = summarize_samples(split_samples)

    yaml_paths = {
        "combined": config_dir / "collision_detection_clean_combined.local.yaml",
        "valid_phantom": config_dir / "collision_detection_clean_valid_phantom.local.yaml",
        "valid_animal": config_dir / "collision_detection_clean_valid_animal.local.yaml",
        "smoke": config_dir / "collision_detection_clean_smoke.local.yaml",
    }
    write_yolo_yaml(
        yaml_paths["combined"],
        data_root=data_root,
        train_images=image_lists["train_clean"],
        val_images=image_lists["valid_combined"],
        class_names=class_names,
    )
    write_yolo_yaml(
        yaml_paths["valid_phantom"],
        data_root=data_root,
        train_images=image_lists["train_clean"],
        val_images=image_lists["valid_phantom"],
        class_names=class_names,
    )
    write_yolo_yaml(
        yaml_paths["valid_animal"],
        data_root=data_root,
        train_images=image_lists["train_clean"],
        val_images=image_lists["valid_animal"],
        class_names=class_names,
    )
    write_yolo_yaml(
        yaml_paths["smoke"],
        data_root=data_root,
        train_images=image_lists["train_smoke"],
        val_images=image_lists["valid_smoke"],
        class_names=class_names,
    )

    overlap_train_valid_phantom = sorted(set(train_raw) & set(valid_phantom))
    overlap_train_valid_animal = sorted(set(train_raw) & set(valid_animal))
    train_raw_video_overlap = sorted(
        {by_label[name].video_id for name in train_raw} & valid_video_ids
    )
    residual_leak_frames = [
        name
        for name in train_raw
        if name not in valid_label_set and by_label[name].video_id in valid_video_ids
    ]
    summary: dict[str, object] = {
        "data_root": repo_relative(data_root, repo_root),
        "output_dir": repo_relative(output_dir, repo_root),
        "config_dir": repo_relative(config_dir, repo_root),
        "class_mapping_assumption": {
            "0": class_names[0],
            "1": class_names[1],
            "status": "working_assumption_pending_official_confirmation",
        },
        "all_samples": summarize_samples(samples),
        "raw_split_counts": {
            "train_phantom": len(train_raw),
            "valid_phantom": len(valid_phantom),
            "valid_animal": len(valid_animal),
        },
        "raw_split_overlap": {
            "train_phantom__valid_phantom": len(overlap_train_valid_phantom),
            "train_phantom__valid_animal": len(overlap_train_valid_animal),
            "examples": {
                "train_phantom__valid_phantom": overlap_train_valid_phantom[:10],
                "train_phantom__valid_animal": overlap_train_valid_animal[:10],
            },
        },
        "video_level_disjointness": {
            "valid_video_ids": sorted(valid_video_ids),
            "train_raw_video_overlap": train_raw_video_overlap,
            "residual_same_video_leak_frames_removed": len(residual_leak_frames),
            "residual_examples": residual_leak_frames[:10],
            "train_clean_video_overlap": leaked_video_ids,
            "guarantee": "train_clean is video-disjoint from validation (asserted)",
        },
        "clean_split_summaries": split_summaries,
        "image_lists": {
            name: repo_relative(path, repo_root) for name, path in image_lists.items()
        },
        "label_lists": {
            name: repo_relative(path, repo_root) for name, path in label_lists.items()
        },
        "yolo_yaml": {
            name: repo_relative(path, repo_root) for name, path in yaml_paths.items()
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def summarize_samples(samples: list[Task2Sample]) -> dict[str, object]:
    class_counts = Counter(sample.class_id for sample in samples)
    video_counts = Counter(sample.video_id for sample in samples)
    widths = sorted(sample.box.width for sample in samples)
    heights = sorted(sample.box.height for sample in samples)
    areas = sorted(sample.box.width * sample.box.height for sample in samples)
    return {
        "samples": len(samples),
        "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "videos": len(video_counts),
        "top_videos": [
            {"video_id": key, "samples": int(value)}
            for key, value in video_counts.most_common(10)
        ],
        "bbox_width": quantiles(widths),
        "bbox_height": quantiles(heights),
        "bbox_area": quantiles(areas),
    }


def select_balanced_label_names(
    label_names: list[str],
    by_label: dict[str, Task2Sample],
    max_samples: int,
) -> list[str]:
    if max_samples <= 0:
        return []

    buckets: dict[int, list[str]] = {}
    for label_name in label_names:
        class_id = by_label[label_name].class_id
        buckets.setdefault(class_id, []).append(label_name)

    selected: list[str] = []
    while len(selected) < max_samples and any(buckets.values()):
        for class_id in sorted(buckets):
            if buckets[class_id]:
                selected.append(buckets[class_id].pop(0))
                if len(selected) == max_samples:
                    break
    return selected


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


def write_path_list(path: Path, paths: list[Path]) -> None:
    path.write_text(
        "".join(f"{item.as_posix()}\n" for item in paths),
        encoding="utf-8",
    )


def write_yolo_yaml(
    path: Path,
    *,
    data_root: Path,
    train_images: Path,
    val_images: Path,
    class_names: tuple[str, str],
) -> None:
    text = "\n".join(
        [
            f"path: {data_root.resolve().as_posix()}",
            f"train: {train_images.resolve().as_posix()}",
            f"val: {val_images.resolve().as_posix()}",
            "names:",
            f"  0: {class_names[0]}",
            f"  1: {class_names[1]}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument("--output-dir", type=Path, default=Path("configs/task2/splits"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/task2"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--class-0-name", default="normal")
    parser.add_argument("--class-1-name", default="collision")
    parser.add_argument("--smoke-train-size", type=int, default=64)
    parser.add_argument("--smoke-val-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    summary = prepare_splits(
        data_root=(repo_root / args.data_root).resolve()
        if not args.data_root.is_absolute()
        else args.data_root.resolve(),
        output_dir=(repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve(),
        config_dir=(repo_root / args.config_dir).resolve()
        if not args.config_dir.is_absolute()
        else args.config_dir.resolve(),
        repo_root=repo_root,
        class_names=(args.class_0_name, args.class_1_name),
        smoke_train_size=args.smoke_train_size,
        smoke_val_size=args.smoke_val_size,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
