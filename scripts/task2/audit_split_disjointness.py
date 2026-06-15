#!/usr/bin/env python3
"""Audit Task 2 splits for video-level (case/procedure-level) disjointness.

Enforces INV-2: training, validation, and test splits must be procedure/case
level, never frame-randomized across the same video. Exits non-zero if any
training video also appears in a validation/test split.

Two modes:

1. Directory auto-discovery (default convention used by prepare_yolo_splits.py):
     python scripts/task2/audit_split_disjointness.py --splits-dir configs/task2/splits
   Files matching ``train*_labels.txt`` / ``train*_images.txt`` form the train
   group; ``valid*`` / ``val*`` / ``test*`` form the held-out group.

2. Explicit list files (any split layout, repeatable):
     python scripts/task2/audit_split_disjointness.py \
       --train path/to/train_images.txt --valid path/to/valid_images.txt

Each list file contains one image or label path per line; the video id is parsed
from the filename stem with the same rule as the data loader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cathaction.data.task2 import parse_task2_sample_stem


def video_ids_from_list(list_file: Path) -> set[str]:
    """Parse the set of video ids referenced by a split list file."""
    video_ids: set[str] = set()
    for raw_line in list_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        stem = Path(line).stem
        video_id, _frame = parse_task2_sample_stem(stem)
        video_ids.add(video_id)
    return video_ids


def discover_groups(splits_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Split *_labels.txt (fallback *_images.txt) into train vs held-out groups."""
    candidates = sorted(splits_dir.glob("*_labels.txt"))
    if not candidates:
        candidates = sorted(splits_dir.glob("*_images.txt"))
    train: dict[str, Path] = {}
    held_out: dict[str, Path] = {}
    for path in candidates:
        name = path.name
        if name.startswith("train"):
            train[name] = path
        elif name.startswith(("valid", "val", "test")):
            held_out[name] = path
    return train, held_out


def audit(train_files: dict[str, Path], held_out_files: dict[str, Path]) -> int:
    if not train_files or not held_out_files:
        print(
            "ERROR: need at least one train list and one held-out list "
            f"(got train={list(train_files)} held_out={list(held_out_files)})",
            file=sys.stderr,
        )
        return 2

    train_videos: dict[str, set[str]] = {
        name: video_ids_from_list(path) for name, path in train_files.items()
    }
    held_out_videos: dict[str, set[str]] = {
        name: video_ids_from_list(path) for name, path in held_out_files.items()
    }

    train_union: set[str] = set().union(*train_videos.values())
    held_union: set[str] = set().union(*held_out_videos.values())

    print("Train lists:")
    for name, vids in sorted(train_videos.items()):
        print(f"  {name}: {len(vids)} videos")
    print("Held-out lists:")
    for name, vids in sorted(held_out_videos.items()):
        print(f"  {name}: {len(vids)} videos")

    overlap = sorted(train_union & held_union)
    if overlap:
        print()
        print(f"FAIL: {len(overlap)} video(s) appear in BOTH train and held-out:")
        for video_id in overlap:
            in_train = sorted(n for n, v in train_videos.items() if video_id in v)
            in_held = sorted(n for n, v in held_out_videos.items() if video_id in v)
            print(f"  {video_id}: train={in_train} held_out={in_held}")
        print("\nINV-2 violation: splits are NOT video/case-level disjoint.")
        return 1

    print(
        f"\nPASS: {len(train_union)} train videos disjoint from "
        f"{len(held_union)} held-out videos (INV-2 satisfied)."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        help="Directory of *_labels.txt / *_images.txt split lists to auto-discover.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        action="append",
        default=[],
        help="Explicit train list file (repeatable).",
    )
    parser.add_argument(
        "--valid",
        type=Path,
        action="append",
        default=[],
        help="Explicit held-out (valid/test) list file (repeatable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_files: dict[str, Path] = {}
    held_out_files: dict[str, Path] = {}

    if args.splits_dir is not None:
        if not args.splits_dir.is_dir():
            print(f"ERROR: --splits-dir not found: {args.splits_dir}", file=sys.stderr)
            return 2
        train_files, held_out_files = discover_groups(args.splits_dir)

    for path in args.train:
        train_files[str(path)] = path
    for path in args.valid:
        held_out_files[str(path)] = path

    if not train_files and not held_out_files:
        print("ERROR: provide --splits-dir or --train/--valid lists.", file=sys.stderr)
        return 2

    return audit(train_files, held_out_files)


if __name__ == "__main__":
    raise SystemExit(main())
