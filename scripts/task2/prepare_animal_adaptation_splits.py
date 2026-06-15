#!/usr/bin/env python3
"""Prepare video-level animal adaptation splits for Task 2."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument("--base-split-dir", type=Path, default=Path("configs/task2/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("configs/task2/splits_adapt_animal"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/task2"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--class-0-name", default="normal")
    parser.add_argument("--class-1-name", default="collision")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = resolve(repo_root, args.data_root)
    base_split_dir = resolve(repo_root, args.base_split_dir)
    output_dir = resolve(repo_root, args.output_dir)
    config_dir = resolve(repo_root, args.config_dir)
    summary = prepare_animal_adaptation_splits(
        data_root=data_root,
        base_split_dir=base_split_dir,
        output_dir=output_dir,
        config_dir=config_dir,
        repo_root=repo_root,
        class_names=(args.class_0_name, args.class_1_name),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def prepare_animal_adaptation_splits(
    *,
    data_root: Path,
    base_split_dir: Path,
    output_dir: Path,
    config_dir: Path,
    repo_root: Path,
    class_names: tuple[str, str],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    samples = build_task2_index(data_root)
    by_label = samples_by_label_name(samples)
    train_clean_labels = read_task2_split_label_names(data_root, base_split_dir / "train_clean_labels.txt")
    valid_phantom_labels = read_task2_split_label_names(data_root, base_split_dir / "valid_phantom_labels.txt")
    animal_labels = read_task2_split_label_names(data_root, data_root / "valid_animal.txt")
    animal_samples = [label_name_to_sample(by_label, label_name) for label_name in animal_labels]

    animal_by_video: dict[str, list[str]] = {}
    for sample in animal_samples:
        animal_by_video.setdefault(sample.video_id, []).append(sample.label_path.name)

    fold_specs = {
        "train_v1_val_v2": {
            "animal_train_videos": ["video_1_animal"],
            "animal_val_videos": ["video_2_animal"],
        },
        "train_v2_val_v1": {
            "animal_train_videos": ["video_2_animal"],
            "animal_val_videos": ["video_1_animal"],
        },
        "train_v1_v2_val_v0": {
            "animal_train_videos": ["video_1_animal", "video_2_animal"],
            "animal_val_videos": ["video_0_animal"],
        },
    }

    split_summaries: dict[str, object] = {}
    yaml_paths: dict[str, str] = {}
    for fold_name, spec in fold_specs.items():
        train_labels = list(train_clean_labels)
        for video_id in spec["animal_train_videos"]:
            train_labels.extend(animal_by_video[video_id])
        val_labels: list[str] = []
        for video_id in spec["animal_val_videos"]:
            val_labels.extend(animal_by_video[video_id])

        train_samples = [label_name_to_sample(by_label, label_name) for label_name in train_labels]
        val_samples = [label_name_to_sample(by_label, label_name) for label_name in val_labels]
        train_image_list = output_dir / f"{fold_name}_train_images.txt"
        train_label_list = output_dir / f"{fold_name}_train_labels.txt"
        val_image_list = output_dir / f"{fold_name}_val_images.txt"
        val_label_list = output_dir / f"{fold_name}_val_labels.txt"
        write_path_list(train_image_list, [sample.image_path.resolve() for sample in train_samples])
        write_path_list(train_label_list, [sample.label_path.resolve() for sample in train_samples])
        write_path_list(val_image_list, [sample.image_path.resolve() for sample in val_samples])
        write_path_list(val_label_list, [sample.label_path.resolve() for sample in val_samples])

        yaml_path = config_dir / f"collision_detection_adapt_animal_{fold_name}.local.yaml"
        write_yolo_yaml(
            yaml_path,
            data_root=data_root,
            train_images=train_image_list,
            val_images=val_image_list,
            class_names=class_names,
        )
        yaml_paths[fold_name] = repo_relative(yaml_path, repo_root)
        split_summaries[fold_name] = {
            "animal_train_videos": spec["animal_train_videos"],
            "animal_val_videos": spec["animal_val_videos"],
            "train": summarize_samples(train_samples),
            "val": summarize_samples(val_samples),
            "train_images": repo_relative(train_image_list, repo_root),
            "train_labels": repo_relative(train_label_list, repo_root),
            "val_images": repo_relative(val_image_list, repo_root),
            "val_labels": repo_relative(val_label_list, repo_root),
        }

    valid_phantom_samples = [label_name_to_sample(by_label, label_name) for label_name in valid_phantom_labels]
    summary = {
        "data_root": repo_relative(data_root, repo_root),
        "output_dir": repo_relative(output_dir, repo_root),
        "config_dir": repo_relative(config_dir, repo_root),
        "animal_videos": {
            video_id: summarize_samples(
                [label_name_to_sample(by_label, label_name) for label_name in labels]
            )
            for video_id, labels in sorted(animal_by_video.items())
        },
        "valid_phantom": summarize_samples(valid_phantom_samples),
        "folds": split_summaries,
        "yolo_yaml": yaml_paths,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


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
    path.write_text("".join(f"{item.as_posix()}\n" for item in paths), encoding="utf-8")


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


if __name__ == "__main__":
    raise SystemExit(main())
