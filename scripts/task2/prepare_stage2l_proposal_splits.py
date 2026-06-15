#!/usr/bin/env python3
"""Prepare Stage2L class-agnostic proposal splits with animal normal training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
from pathlib import Path
from typing import Iterable

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
    parser.add_argument("--source-data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--agnostic-data-root",
        type=Path,
        default=Path("datasets/collision_detection_agnostic"),
    )
    parser.add_argument("--base-split-dir", type=Path, default=Path("configs/task2/splits"))
    parser.add_argument(
        "--stage2j-split-dir",
        type=Path,
        default=Path("configs/task2/splits_stage2j_balanced"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs/task2"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--class-name", default="tool_roi")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    summary = prepare_stage2l_proposal_splits(
        source_data_root=resolve(repo_root, args.source_data_root),
        agnostic_data_root=resolve(repo_root, args.agnostic_data_root),
        base_split_dir=resolve(repo_root, args.base_split_dir),
        stage2j_split_dir=resolve(repo_root, args.stage2j_split_dir),
        output_dir=resolve(repo_root, args.output_dir),
        config_dir=resolve(repo_root, args.config_dir),
        repo_root=repo_root,
        class_name=args.class_name,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def prepare_stage2l_proposal_splits(
    *,
    source_data_root: Path,
    agnostic_data_root: Path,
    base_split_dir: Path,
    stage2j_split_dir: Path,
    output_dir: Path,
    config_dir: Path,
    repo_root: Path,
    class_name: str,
    seed: int,
) -> dict[str, object]:
    source_data_root = source_data_root.resolve()
    agnostic_data_root = agnostic_data_root.resolve()
    base_split_dir = base_split_dir.resolve()
    stage2j_split_dir = stage2j_split_dir.resolve()
    output_dir = output_dir.resolve()
    config_dir = config_dir.resolve()
    repo_root = repo_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    source_samples = build_task2_index(source_data_root)
    by_label = samples_by_label_name(source_samples)
    train_clean_labels = read_task2_split_label_names(
        source_data_root,
        base_split_dir / "train_clean_labels.txt",
    )
    valid_phantom_full_labels = read_task2_split_label_names(
        source_data_root,
        base_split_dir / "valid_phantom_labels.txt",
    )
    valid_phantom_balanced_labels = read_task2_split_label_names(
        source_data_root,
        stage2j_split_dir / "valid_phantom_balanced_small_labels.txt",
    )
    animal_labels = read_task2_split_label_names(source_data_root, source_data_root / "valid_animal.txt")
    animal_samples = [label_name_to_sample(by_label, label_name) for label_name in animal_labels]
    animal_by_video = group_labels_by_video(animal_samples)

    train_labels = list(train_clean_labels)
    for video_id in ("video_0_animal", "video_1_animal"):
        train_labels.extend(animal_by_video[video_id])
    valid_animal_labels = list(animal_by_video["video_2_animal"])
    valid_combined_balanced_labels = valid_animal_labels + list(valid_phantom_balanced_labels)

    rng = random.Random(seed)
    rng.shuffle(train_labels)

    panels = {
        "train_v0_v1_val_v2_train": train_labels,
        "train_v0_v1_val_v2_valid_animal": valid_animal_labels,
        "train_v0_v1_val_v2_valid_combined_balanced": valid_combined_balanced_labels,
        "valid_phantom_balanced_small": valid_phantom_balanced_labels,
        "valid_phantom_full": valid_phantom_full_labels,
    }

    list_paths: dict[str, dict[str, Path]] = {}
    summaries: dict[str, object] = {}
    for panel_name, label_names in panels.items():
        samples = [label_name_to_sample(by_label, label_name) for label_name in label_names]
        image_list = output_dir / f"{panel_name}_images.txt"
        label_list = output_dir / f"{panel_name}_labels.txt"
        write_path_list(
            image_list,
            [agnostic_data_root / "images" / sample.image_path.name for sample in samples],
        )
        write_path_list(
            label_list,
            [agnostic_data_root / "labels" / sample.label_path.name for sample in samples],
        )
        list_paths[panel_name] = {"images": image_list, "labels": label_list}
        summaries[panel_name] = summarize_samples(samples)

    yaml_specs = {
        "animal": (
            "collision_detection_stage2l_agnostic_train_v0_v1_val_v2_animal.local.yaml",
            list_paths["train_v0_v1_val_v2_valid_animal"]["images"],
        ),
        "valid_phantom_balanced_small": (
            "collision_detection_stage2l_agnostic_train_v0_v1_val_v2_valid_phantom_balanced_small.local.yaml",
            list_paths["valid_phantom_balanced_small"]["images"],
        ),
        "combined_balanced": (
            "collision_detection_stage2l_agnostic_train_v0_v1_val_v2_valid_combined_balanced.local.yaml",
            list_paths["train_v0_v1_val_v2_valid_combined_balanced"]["images"],
        ),
        "valid_phantom_full": (
            "collision_detection_stage2l_agnostic_train_v0_v1_val_v2_valid_phantom_full.local.yaml",
            list_paths["valid_phantom_full"]["images"],
        ),
    }
    yolo_yaml: dict[str, str] = {}
    for key, (filename, val_images) in yaml_specs.items():
        yaml_path = config_dir / filename
        write_yolo_yaml(
            yaml_path,
            data_root=agnostic_data_root,
            train_images=list_paths["train_v0_v1_val_v2_train"]["images"],
            val_images=val_images,
            class_name=class_name,
        )
        yolo_yaml[key] = repo_relative(yaml_path, repo_root)

    summary: dict[str, object] = {
        "source_data_root": repo_relative(source_data_root, repo_root),
        "agnostic_data_root": repo_relative(agnostic_data_root, repo_root),
        "output_dir": repo_relative(output_dir, repo_root),
        "config_dir": repo_relative(config_dir, repo_root),
        "class_mapping": {"0": class_name},
        "animal_train_videos": ["video_0_animal", "video_1_animal"],
        "animal_val_videos": ["video_2_animal"],
        "panels": summaries,
        "lists": {
            name: {
                "images": repo_relative(paths["images"], repo_root),
                "labels": repo_relative(paths["labels"], repo_root),
            }
            for name, paths in list_paths.items()
        },
        "yolo_yaml": yolo_yaml,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def group_labels_by_video(samples: Iterable[Task2Sample]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for sample in samples:
        grouped.setdefault(sample.video_id, []).append(sample.label_path.name)
    return {video_id: sorted(labels) for video_id, labels in grouped.items()}


def summarize_samples(samples: list[Task2Sample]) -> dict[str, object]:
    class_count = Counter(sample.class_id for sample in samples)
    video_count = Counter(sample.video_id for sample in samples)
    domain_count = Counter(domain_of(sample) for sample in samples)
    return {
        "samples": len(samples),
        "original_class_counts": {
            str(key): int(value) for key, value in sorted(class_count.items())
        },
        "agnostic_class_counts": {"0": len(samples)},
        "domain_counts": dict(sorted(domain_count.items())),
        "videos": len(video_count),
        "top_videos": [
            {"video_id": key, "samples": int(value)}
            for key, value in video_count.most_common(10)
        ],
    }


def domain_of(sample: Task2Sample) -> str:
    return "animal" if "animal" in sample.video_id else "phantom"


def write_path_list(path: Path, paths: list[Path]) -> None:
    # Use absolute() instead of resolve() so symlinked agnostic image paths do
    # not collapse back to the original two-class dataset. Ultralytics infers
    # label paths from image paths, so preserving the agnostic image root is
    # required for one-class proposal training.
    path.write_text("".join(f"{item.absolute().as_posix()}\n" for item in paths), encoding="utf-8")


def write_yolo_yaml(
    path: Path,
    *,
    data_root: Path,
    train_images: Path,
    val_images: Path,
    class_name: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                f"path: {data_root.resolve().as_posix()}",
                f"train: {train_images.resolve().as_posix()}",
                f"val: {val_images.resolve().as_posix()}",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
