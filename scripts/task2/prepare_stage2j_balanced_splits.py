#!/usr/bin/env python3
"""Prepare domain-balanced Task 2 split panels for Stage2J."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
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
    parser.add_argument("--output-dir", type=Path, default=Path("configs/task2/splits_stage2j_balanced"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/task2"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phantom-per-class", type=int, default=5000)
    parser.add_argument("--animal-repeat", type=int, default=50)
    parser.add_argument("--phantom-balanced-val-per-class", type=int, default=412)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--class-0-name", default="normal")
    parser.add_argument("--class-1-name", default="collision")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    summary = prepare_stage2j_balanced_splits(
        data_root=resolve(repo_root, args.data_root),
        base_split_dir=resolve(repo_root, args.base_split_dir),
        output_dir=resolve(repo_root, args.output_dir),
        config_dir=resolve(repo_root, args.config_dir),
        repo_root=repo_root,
        class_names=(args.class_0_name, args.class_1_name),
        phantom_per_class=args.phantom_per_class,
        animal_repeat=args.animal_repeat,
        phantom_balanced_val_per_class=args.phantom_balanced_val_per_class,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def prepare_stage2j_balanced_splits(
    *,
    data_root: Path,
    base_split_dir: Path,
    output_dir: Path,
    config_dir: Path,
    repo_root: Path,
    class_names: tuple[str, str],
    phantom_per_class: int,
    animal_repeat: int,
    phantom_balanced_val_per_class: int,
    seed: int,
) -> dict[str, object]:
    if phantom_per_class <= 0:
        raise ValueError("phantom_per_class must be positive")
    if animal_repeat <= 0:
        raise ValueError("animal_repeat must be positive")
    if phantom_balanced_val_per_class <= 0:
        raise ValueError("phantom_balanced_val_per_class must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    samples = build_task2_index(data_root)
    by_label = samples_by_label_name(samples)
    train_clean_labels = read_task2_split_label_names(data_root, base_split_dir / "train_clean_labels.txt")
    valid_phantom_labels = read_task2_split_label_names(data_root, base_split_dir / "valid_phantom_labels.txt")
    valid_animal_labels = read_task2_split_label_names(data_root, data_root / "valid_animal.txt")

    train_clean = [label_name_to_sample(by_label, label_name) for label_name in train_clean_labels]
    valid_phantom = [label_name_to_sample(by_label, label_name) for label_name in valid_phantom_labels]
    valid_animal = [label_name_to_sample(by_label, label_name) for label_name in valid_animal_labels]

    animal_by_video: dict[str, list[Task2Sample]] = {}
    for sample in valid_animal:
        animal_by_video.setdefault(sample.video_id, []).append(sample)
    for video_samples in animal_by_video.values():
        video_samples.sort(key=lambda sample: sample.label_path.name)

    phantom_train_subset = balanced_class_subset(
        train_clean,
        per_class=phantom_per_class,
        seed=seed,
        salt="phantom-train",
    )
    phantom_balanced_val = balanced_class_subset(
        valid_phantom,
        per_class=phantom_balanced_val_per_class,
        seed=seed,
        salt="phantom-val-balanced",
    )

    fold_specs = {
        "pilot_train_v1_val_v2": {
            "animal_train_videos": ["video_1_animal"],
            "animal_val_videos": ["video_2_animal"],
        },
        "pilot_train_v2_val_v1": {
            "animal_train_videos": ["video_2_animal"],
            "animal_val_videos": ["video_1_animal"],
        },
    }

    fold_summaries: dict[str, object] = {}
    yolo_yaml: dict[str, dict[str, str]] = {}
    shared_panels = {
        "valid_phantom_full": valid_phantom,
        "valid_phantom_balanced_small": phantom_balanced_val,
    }

    write_panel_files(
        output_dir=output_dir,
        prefix="valid_phantom_full",
        samples=valid_phantom,
        repo_root=repo_root,
    )
    write_panel_files(
        output_dir=output_dir,
        prefix="valid_phantom_balanced_small",
        samples=phantom_balanced_val,
        repo_root=repo_root,
    )

    for fold_name, spec in fold_specs.items():
        animal_train = [
            sample
            for video_id in spec["animal_train_videos"]
            for sample in animal_by_video[video_id]
        ]
        animal_val = [
            sample
            for video_id in spec["animal_val_videos"]
            for sample in animal_by_video[video_id]
        ]
        train_samples = list(phantom_train_subset) + list(animal_train) * animal_repeat

        train_image_list, train_label_list = write_panel_files(
            output_dir=output_dir,
            prefix=f"{fold_name}_train",
            samples=train_samples,
            repo_root=repo_root,
        )
        animal_val_image_list, animal_val_label_list = write_panel_files(
            output_dir=output_dir,
            prefix=f"{fold_name}_valid_animal",
            samples=animal_val,
            repo_root=repo_root,
        )

        fold_yaml: dict[str, str] = {}
        animal_yaml = config_dir / f"collision_detection_stage2j_{fold_name}_animal.local.yaml"
        write_yolo_yaml(
            animal_yaml,
            data_root=data_root,
            train_images=train_image_list,
            val_images=animal_val_image_list,
            class_names=class_names,
        )
        fold_yaml["valid_animal"] = repo_relative(animal_yaml, repo_root)

        for panel_name in shared_panels:
            panel_image_list = output_dir / f"{panel_name}_images.txt"
            panel_yaml = config_dir / f"collision_detection_stage2j_{fold_name}_{panel_name}.local.yaml"
            write_yolo_yaml(
                panel_yaml,
                data_root=data_root,
                train_images=train_image_list,
                val_images=panel_image_list,
                class_names=class_names,
            )
            fold_yaml[panel_name] = repo_relative(panel_yaml, repo_root)

        yolo_yaml[fold_name] = fold_yaml
        fold_summaries[fold_name] = {
            "animal_train_videos": spec["animal_train_videos"],
            "animal_val_videos": spec["animal_val_videos"],
            "animal_repeat": animal_repeat,
            "phantom_per_class": phantom_per_class,
            "train": summarize_samples(train_samples),
            "train_unique": summarize_samples(unique_samples(train_samples)),
            "valid_animal": summarize_samples(animal_val),
            "train_images": repo_relative(train_image_list, repo_root),
            "train_labels": repo_relative(train_label_list, repo_root),
            "valid_animal_images": repo_relative(animal_val_image_list, repo_root),
            "valid_animal_labels": repo_relative(animal_val_label_list, repo_root),
        }

    summary = {
        "data_root": repo_relative(data_root, repo_root),
        "output_dir": repo_relative(output_dir, repo_root),
        "config_dir": repo_relative(config_dir, repo_root),
        "seed": seed,
        "class_names": {"0": class_names[0], "1": class_names[1]},
        "base_splits": {
            "train_clean": summarize_samples(train_clean),
            "valid_phantom": summarize_samples(valid_phantom),
            "valid_animal": summarize_samples(valid_animal),
        },
        "shared_validation_panels": {
            panel_name: summarize_samples(panel_samples)
            for panel_name, panel_samples in shared_panels.items()
        },
        "animal_videos": {
            video_id: summarize_samples(video_samples)
            for video_id, video_samples in sorted(animal_by_video.items())
        },
        "folds": fold_summaries,
        "yolo_yaml": yolo_yaml,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def resolve(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def balanced_class_subset(samples: list[Task2Sample], *, per_class: int, seed: int, salt: str) -> list[Task2Sample]:
    grouped: dict[int, list[Task2Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.class_id, []).append(sample)

    result: list[Task2Sample] = []
    for class_id, class_samples in sorted(grouped.items()):
        shuffled = sorted(class_samples, key=lambda sample: sample.label_path.name)
        rng = random.Random(f"{seed}:{salt}:{class_id}")
        rng.shuffle(shuffled)
        result.extend(shuffled[: min(per_class, len(shuffled))])
    return sorted(result, key=lambda sample: (sample.class_id, sample.label_path.name))


def unique_samples(samples: list[Task2Sample]) -> list[Task2Sample]:
    by_label: dict[str, Task2Sample] = {}
    for sample in samples:
        by_label.setdefault(sample.label_path.name, sample)
    return [by_label[key] for key in sorted(by_label)]


def summarize_samples(samples: list[Task2Sample]) -> dict[str, object]:
    class_counts = Counter(sample.class_id for sample in samples)
    video_counts = Counter(sample.video_id for sample in samples)
    domain_counts = Counter("animal" if sample.video_id.endswith("_animal") else "phantom" for sample in samples)
    label_counts = Counter(sample.label_path.name for sample in samples)
    duplicate_occurrences = sum(count - 1 for count in label_counts.values() if count > 1)
    return {
        "samples": len(samples),
        "unique_samples": len(label_counts),
        "duplicate_occurrences": duplicate_occurrences,
        "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "domain_counts": {key: int(value) for key, value in sorted(domain_counts.items())},
        "videos": len(video_counts),
        "top_videos": [
            {"video_id": key, "samples": int(value)}
            for key, value in video_counts.most_common(10)
        ],
    }


def write_panel_files(
    *,
    output_dir: Path,
    prefix: str,
    samples: list[Task2Sample],
    repo_root: Path,
) -> tuple[Path, Path]:
    image_list = output_dir / f"{prefix}_images.txt"
    label_list = output_dir / f"{prefix}_labels.txt"
    write_path_list(image_list, [sample.image_path.resolve() for sample in samples])
    write_path_list(label_list, [sample.label_path.resolve() for sample in samples])
    return image_list, label_list


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
