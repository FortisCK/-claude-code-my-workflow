#!/usr/bin/env python3
"""Create collision-only YOLO labels for Task 2 specialist detection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
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


SPLIT_NAMES = (
    "train_clean",
    "valid_combined",
    "valid_phantom",
    "valid_animal",
    "train_smoke",
    "valid_smoke",
)


def prepare_collision_specialist_yolo(
    *,
    source_data_root: Path,
    specialist_data_root: Path,
    source_split_dir: Path,
    output_split_dir: Path,
    config_dir: Path,
    repo_root: Path,
    class_name: str,
    collision_class_id: int,
    positive_repeat: int = 1,
) -> dict[str, object]:
    if positive_repeat < 1:
        raise ValueError("positive_repeat must be >= 1")

    source_data_root = source_data_root.resolve()
    specialist_data_root = specialist_data_root.resolve()
    source_split_dir = source_split_dir.resolve()
    output_split_dir = output_split_dir.resolve()
    config_dir = config_dir.resolve()
    repo_root = repo_root.resolve()

    samples = build_task2_index(source_data_root)
    by_label = samples_by_label_name(samples)

    labels_dir = specialist_data_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    ensure_image_mirror(source_data_root / "images", specialist_data_root / "images")
    for sample in samples:
        write_collision_specialist_label(
            labels_dir / sample.label_path.name,
            sample,
            collision_class_id=collision_class_id,
        )

    output_split_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, object] = {}
    image_lists: dict[str, Path] = {}
    label_lists: dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        source_split = source_split_dir / f"{split_name}_labels.txt"
        label_names = read_task2_split_label_names(source_data_root, source_split)
        split_samples = [label_name_to_sample(by_label, label_name) for label_name in label_names]
        listed_samples = maybe_repeat_collision_samples(
            split_samples,
            split_name=split_name,
            collision_class_id=collision_class_id,
            positive_repeat=positive_repeat,
        )
        image_list = output_split_dir / f"{split_name}_images.txt"
        label_list = output_split_dir / f"{split_name}_labels.txt"
        write_path_list(
            image_list,
            [specialist_data_root / "images" / sample.image_path.name for sample in listed_samples],
        )
        write_path_list(
            label_list,
            [specialist_data_root / "labels" / sample.label_path.name for sample in listed_samples],
        )
        image_lists[split_name] = image_list
        label_lists[split_name] = label_list
        split_summaries[split_name] = summarize_split(
            split_samples,
            listed_samples,
            collision_class_id=collision_class_id,
        )

    yaml_paths = {
        "combined": config_dir / "collision_detection_collision_only_clean_combined.local.yaml",
        "valid_phantom": config_dir / "collision_detection_collision_only_clean_valid_phantom.local.yaml",
        "valid_animal": config_dir / "collision_detection_collision_only_clean_valid_animal.local.yaml",
        "smoke": config_dir / "collision_detection_collision_only_clean_smoke.local.yaml",
    }
    write_yolo_yaml(
        yaml_paths["combined"],
        data_root=specialist_data_root,
        train_images=image_lists["train_clean"],
        val_images=image_lists["valid_combined"],
        class_name=class_name,
    )
    write_yolo_yaml(
        yaml_paths["valid_phantom"],
        data_root=specialist_data_root,
        train_images=image_lists["train_clean"],
        val_images=image_lists["valid_phantom"],
        class_name=class_name,
    )
    write_yolo_yaml(
        yaml_paths["valid_animal"],
        data_root=specialist_data_root,
        train_images=image_lists["train_clean"],
        val_images=image_lists["valid_animal"],
        class_name=class_name,
    )
    write_yolo_yaml(
        yaml_paths["smoke"],
        data_root=specialist_data_root,
        train_images=image_lists["train_smoke"],
        val_images=image_lists["valid_smoke"],
        class_name=class_name,
    )

    original_counts = Counter(sample.class_id for sample in samples)
    summary: dict[str, object] = {
        "source_data_root": repo_relative(source_data_root, repo_root),
        "specialist_data_root": repo_relative(specialist_data_root, repo_root),
        "source_split_dir": repo_relative(source_split_dir, repo_root),
        "output_split_dir": repo_relative(output_split_dir, repo_root),
        "config_dir": repo_relative(config_dir, repo_root),
        "class_mapping": {"0": class_name},
        "collision_class_id": collision_class_id,
        "positive_repeat_train_only": positive_repeat,
        "labels_written": len(samples),
        "positive_labels": int(original_counts.get(collision_class_id, 0)),
        "empty_negative_labels": len(samples) - int(original_counts.get(collision_class_id, 0)),
        "original_class_counts": {
            str(key): int(value) for key, value in sorted(original_counts.items())
        },
        "split_summaries": split_summaries,
        "image_lists": {name: repo_relative(path, repo_root) for name, path in image_lists.items()},
        "label_lists": {name: repo_relative(path, repo_root) for name, path in label_lists.items()},
        "yolo_yaml": {name: repo_relative(path, repo_root) for name, path in yaml_paths.items()},
    }
    (output_split_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def ensure_image_mirror(source_images: Path, target_images: Path) -> None:
    if target_images.exists():
        if target_images.is_symlink() and target_images.resolve() == source_images.resolve():
            return
        if target_images.is_dir() and not target_images.is_symlink():
            return
        raise FileExistsError(f"Cannot reuse existing image mirror path: {target_images}")
    try:
        target_images.symlink_to(source_images.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(source_images, target_images)


def write_collision_specialist_label(
    path: Path,
    sample: Task2Sample,
    *,
    collision_class_id: int,
) -> None:
    if sample.class_id != collision_class_id:
        path.write_text("", encoding="utf-8")
        return
    box = sample.box
    path.write_text(
        f"0 {box.x_center:.10g} {box.y_center:.10g} {box.width:.10g} {box.height:.10g}\n",
        encoding="utf-8",
    )


def maybe_repeat_collision_samples(
    samples: list[Task2Sample],
    *,
    split_name: str,
    collision_class_id: int,
    positive_repeat: int,
) -> list[Task2Sample]:
    if split_name != "train_clean" or positive_repeat == 1:
        return samples
    repeated: list[Task2Sample] = []
    for sample in samples:
        repeated.append(sample)
        if sample.class_id == collision_class_id:
            repeated.extend([sample] * (positive_repeat - 1))
    return repeated


def summarize_split(
    base_samples: list[Task2Sample],
    listed_samples: list[Task2Sample],
    *,
    collision_class_id: int,
) -> dict[str, object]:
    base_counts = Counter(sample.class_id for sample in base_samples)
    listed_counts = Counter(sample.class_id for sample in listed_samples)
    video_counts = Counter(sample.video_id for sample in base_samples)
    positives = int(base_counts.get(collision_class_id, 0))
    listed_positives = int(listed_counts.get(collision_class_id, 0))
    return {
        "base_samples": len(base_samples),
        "listed_samples": len(listed_samples),
        "base_original_class_counts": {
            str(key): int(value) for key, value in sorted(base_counts.items())
        },
        "listed_original_class_counts": {
            str(key): int(value) for key, value in sorted(listed_counts.items())
        },
        "collision_positive_labels": positives,
        "empty_negative_labels": len(base_samples) - positives,
        "listed_collision_positive_labels": listed_positives,
        "listed_empty_negative_labels": len(listed_samples) - listed_positives,
        "specialist_class_counts": {"0": listed_positives},
        "videos": len(video_counts),
    }


def write_path_list(path: Path, paths: list[Path]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--specialist-data-root",
        type=Path,
        default=Path("datasets/collision_detection_collision_only"),
    )
    parser.add_argument("--source-split-dir", type=Path, default=Path("configs/task2/splits"))
    parser.add_argument(
        "--output-split-dir",
        type=Path,
        default=Path("configs/task2/splits_collision_only"),
    )
    parser.add_argument("--config-dir", type=Path, default=Path("configs/task2"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--class-name", default="collision_roi")
    parser.add_argument("--collision-class-id", type=int, default=1)
    parser.add_argument(
        "--positive-repeat",
        type=int,
        default=1,
        help="Repeat collision-positive images in train_clean only.",
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    summary = prepare_collision_specialist_yolo(
        source_data_root=resolve(repo_root, args.source_data_root),
        specialist_data_root=resolve(repo_root, args.specialist_data_root),
        source_split_dir=resolve(repo_root, args.source_split_dir),
        output_split_dir=resolve(repo_root, args.output_split_dir),
        config_dir=resolve(repo_root, args.config_dir),
        repo_root=repo_root,
        class_name=args.class_name,
        collision_class_id=args.collision_class_id,
        positive_repeat=args.positive_repeat,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
