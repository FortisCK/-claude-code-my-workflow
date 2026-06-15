#!/usr/bin/env python3
"""Prepare a non-leakage train subset for Stage2O multi-source ranker candidates."""

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

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import load_task2_samples_from_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--train-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_train_labels.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/task2/splits_stage2o_ranker"),
    )
    parser.add_argument("--name", default="train_v0_v1_val_v2_animal_all_phantom1000pc")
    parser.add_argument("--phantom-per-class", type=int, default=1000)
    parser.add_argument("--include-all-animal", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train_samples = load_task2_samples_from_split(data_root, resolve_path(args.train_split))
    selected = make_subset(
        train_samples,
        phantom_per_class=int(args.phantom_per_class),
        include_all_animal=bool(args.include_all_animal),
        rng=rng,
    )
    selected.sort(key=lambda sample: (sample.video_id, sample.frame_index, sample.sample_id))

    labels_path = output_dir / f"{args.name}_labels.txt"
    images_path = output_dir / f"{args.name}_images.txt"
    write_list(labels_path, [repo_relative(sample.label_path, REPO_ROOT) for sample in selected])
    write_list(images_path, [repo_relative(sample.image_path, REPO_ROOT) for sample in selected])

    summary = {
        "data_root": repo_relative(data_root, REPO_ROOT),
        "train_split": repo_relative(resolve_path(args.train_split), REPO_ROOT),
        "labels": repo_relative(labels_path, REPO_ROOT),
        "images": repo_relative(images_path, REPO_ROOT),
        "samples": len(selected),
        "domain_class_counts": domain_class_counts(selected),
        "video_counts": dict(sorted(Counter(sample.video_id for sample in selected).items())),
        "phantom_per_class": int(args.phantom_per_class),
        "include_all_animal": bool(args.include_all_animal),
        "seed": int(args.seed),
    }
    (output_dir / f"{args.name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def make_subset(
    samples: list[Task2Sample],
    *,
    phantom_per_class: int,
    include_all_animal: bool,
    rng: random.Random,
) -> list[Task2Sample]:
    selected: list[Task2Sample] = []
    if include_all_animal:
        selected.extend([sample for sample in samples if domain_of(sample) == "animal"])

    phantom_by_class: dict[int, list[Task2Sample]] = {}
    for sample in samples:
        if domain_of(sample) == "phantom":
            phantom_by_class.setdefault(sample.class_id, []).append(sample)
    for class_id in sorted(phantom_by_class):
        group = phantom_by_class[class_id]
        selected.extend(group if len(group) <= phantom_per_class else rng.sample(group, phantom_per_class))
    return selected


def domain_of(sample: Task2Sample) -> str:
    return "animal" if "animal" in sample.video_id.lower() else "phantom"


def domain_class_counts(samples: Iterable[Task2Sample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = f"{domain_of(sample)}:{sample.class_id}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
