#!/usr/bin/env python3
"""Inspect the local CATHACTION Task 1 segmentation dataset.

The script is intentionally read-only. It validates image/mask pairing, records
shape and dtype information, and summarizes mask label values without assuming
that released folders correspond to the official hidden test set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    domain: str
    released_split: str
    image_dir: Path
    mask_dir: Path
    image_suffixes: tuple[str, ...]
    mask_suffixes: tuple[str, ...]
    mask_kind: str


def build_specs(data_root: Path) -> list[CollectionSpec]:
    seg_root = data_root / "segmentation"
    human_root = data_root / "human_dataset_train"
    return [
        CollectionSpec(
            name="animal_train",
            domain="animal",
            released_split="train",
            image_dir=seg_root / "animal_train" / "images",
            mask_dir=seg_root / "animal_train" / "masks",
            image_suffixes=(".png",),
            mask_suffixes=(".npy",),
            mask_kind="npy_multiclass",
        ),
        CollectionSpec(
            name="animal_test",
            domain="animal",
            released_split="released_test",
            image_dir=seg_root / "animal_test" / "images",
            mask_dir=seg_root / "animal_test" / "masks",
            image_suffixes=(".png",),
            mask_suffixes=(".npy",),
            mask_kind="npy_multiclass",
        ),
        CollectionSpec(
            name="phantom_train",
            domain="phantom",
            released_split="train",
            image_dir=seg_root / "phantom_train" / "images",
            mask_dir=seg_root / "phantom_train" / "masks",
            image_suffixes=(".png",),
            mask_suffixes=(".npy",),
            mask_kind="npy_multiclass",
        ),
        CollectionSpec(
            name="phantom_test",
            domain="phantom",
            released_split="released_test",
            image_dir=seg_root / "phantom_test" / "images",
            mask_dir=seg_root / "phantom_test" / "masks",
            image_suffixes=(".png",),
            mask_suffixes=(".npy",),
            mask_kind="npy_multiclass",
        ),
        CollectionSpec(
            name="human_train",
            domain="human",
            released_split="train",
            image_dir=human_root / "img",
            mask_dir=human_root / "mask",
            image_suffixes=(".jpg", ".jpeg", ".png"),
            mask_suffixes=(".png",),
            mask_kind="png_binary",
        ),
    ]


def list_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not root.is_dir():
        return []
    suffixes = tuple(s.lower() for s in suffixes)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def image_key(path: Path) -> str:
    return path.stem


def mask_key(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_mask"):
        stem = stem[: -len("_mask")]
    return stem


def choose_sample(paths: list[Path], sample_size: int) -> list[Path]:
    if sample_size <= 0 or len(paths) <= sample_size:
        return paths
    if sample_size == 1:
        return [paths[0]]
    last = len(paths) - 1
    indices = sorted({round(i * last / (sample_size - 1)) for i in range(sample_size)})
    return [paths[i] for i in indices]


def image_shape(path: Path) -> tuple[int, ...]:
    with Image.open(path) as image:
        arr_shape = np.asarray(image).shape
    return tuple(int(v) for v in arr_shape)


def mask_array(path: Path, mask_kind: str, mmap: bool = False) -> np.ndarray:
    if mask_kind == "npy_multiclass":
        if mmap:
            return np.load(path, mmap_mode="r")
        return np.load(path)
    with Image.open(path) as image:
        return np.asarray(image)


def sorted_counter(counter: Counter[Any]) -> list[dict[str, Any]]:
    return [
        {"value": stringify_key(key), "count": int(value)}
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    ]


def stringify_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "x".join(str(part) for part in key)
    return str(key)


def summarize_collection(
    spec: CollectionSpec, sample_size: int, full_pixels: bool, full_shapes: bool
) -> dict[str, Any]:
    images = list_files(spec.image_dir, spec.image_suffixes)
    masks = list_files(spec.mask_dir, spec.mask_suffixes)

    image_by_key = {image_key(path): path for path in images}
    mask_by_key = {mask_key(path): path for path in masks}
    image_keys = set(image_by_key)
    mask_keys = set(mask_by_key)
    paired_keys = sorted(image_keys & mask_keys)

    missing_masks = sorted(image_keys - mask_keys)
    orphan_masks = sorted(mask_keys - image_keys)

    shape_counter: Counter[tuple[int, ...]] = Counter()
    mask_shape_counter: Counter[tuple[int, ...]] = Counter()
    mask_dtype_counter: Counter[str] = Counter()
    mask_value_counter: Counter[int] = Counter()
    sampled_value_counter: Counter[int] = Counter()

    paired_images = [image_by_key[key] for key in paired_keys]
    sampled_images = choose_sample(paired_images, sample_size)
    sampled_keys = [image_key(path) for path in sampled_images]
    sampled_masks = [mask_by_key[key] for key in sampled_keys]

    shape_paths = paired_images if full_shapes else sampled_images
    shape_keys = paired_keys if full_shapes else sampled_keys

    for path in shape_paths:
        shape_counter[image_shape(path)] += 1

    for key in shape_keys:
        path = mask_by_key[key]
        arr = mask_array(path, spec.mask_kind, mmap=spec.mask_kind == "npy_multiclass")
        mask_shape_counter[tuple(int(v) for v in arr.shape)] += 1
        mask_dtype_counter[str(arr.dtype)] += 1

    pixel_paths = [mask_by_key[key] for key in paired_keys] if full_pixels else sampled_masks
    for path in pixel_paths:
        arr = np.asarray(mask_array(path, spec.mask_kind))
        values, counts = np.unique(arr, return_counts=True)
        target = mask_value_counter if full_pixels else sampled_value_counter
        for value, count in zip(values, counts):
            target[int(value)] += int(count)

    if not full_pixels:
        for path in sampled_masks:
            arr = np.asarray(mask_array(path, spec.mask_kind))
            values, counts = np.unique(arr, return_counts=True)
            for value, count in zip(values, counts):
                mask_value_counter[int(value)] += int(count)

    return {
        "name": spec.name,
        "domain": spec.domain,
        "released_split": spec.released_split,
        "image_dir": str(spec.image_dir),
        "mask_dir": str(spec.mask_dir),
        "mask_kind": spec.mask_kind,
        "image_count": len(images),
        "mask_count": len(masks),
        "paired_count": len(paired_keys),
        "missing_mask_count": len(missing_masks),
        "orphan_mask_count": len(orphan_masks),
        "missing_mask_examples": missing_masks[:10],
        "orphan_mask_examples": orphan_masks[:10],
        "image_shapes": sorted_counter(shape_counter),
        "mask_shapes": sorted_counter(mask_shape_counter),
        "mask_dtypes": sorted_counter(mask_dtype_counter),
        "mask_values": sorted_counter(mask_value_counter),
        "shape_summary_mode": "full" if full_shapes else f"sampled_{len(sampled_images)}",
        "pixel_summary_mode": "full" if full_pixels else f"sampled_{len(sampled_masks)}",
        "sampled_value_counts": sorted_counter(sampled_value_counter),
        "sampled_examples": [path.name for path in sampled_images[:10]],
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"Task 1 dataset root: {report['data_root']}")
    print(f"Sample size per collection: {report['sample_size']}")
    print(f"Shape summary mode: {'full' if report['full_shapes'] else 'sampled'}")
    print(f"Pixel summary mode: {'full' if report['full_pixels'] else 'sampled'}")
    print()

    for collection in report["collections"]:
        print(f"[{collection['name']}]")
        print(f"  domain/split: {collection['domain']} / {collection['released_split']}")
        print(
            "  pairs: "
            f"{collection['paired_count']} "
            f"(images={collection['image_count']}, masks={collection['mask_count']}, "
            f"missing_masks={collection['missing_mask_count']}, orphan_masks={collection['orphan_mask_count']})"
        )
        print(f"  mask kind: {collection['mask_kind']}")
        print(f"  image shapes: {format_counter(collection['image_shapes'], limit=5)}")
        print(f"  mask shapes: {format_counter(collection['mask_shapes'], limit=5)}")
        print(f"  mask dtypes: {format_counter(collection['mask_dtypes'], limit=5)}")
        print(f"  mask values: {format_counter(collection['mask_values'], limit=10)}")
        if collection["missing_mask_examples"]:
            print(f"  missing mask examples: {', '.join(collection['missing_mask_examples'])}")
        if collection["orphan_mask_examples"]:
            print(f"  orphan mask examples: {', '.join(collection['orphan_mask_examples'])}")
        print()

    totals = report["totals"]
    print("[totals]")
    print(f"  images: {totals['image_count']}")
    print(f"  masks: {totals['mask_count']}")
    print(f"  pairs: {totals['paired_count']}")
    print(f"  pairing issues: {totals['pairing_issue_count']}")


def format_counter(items: list[dict[str, Any]], limit: int) -> str:
    if not items:
        return "<none>"
    shown = items[:limit]
    text = ", ".join(f"{item['value']}={item['count']}" for item in shown)
    if len(items) > limit:
        text += f", ... (+{len(items) - limit} more)"
    return text


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    collections = [
        summarize_collection(
            spec,
            sample_size=args.sample_size,
            full_pixels=args.full_pixels,
            full_shapes=args.full_shapes,
        )
        for spec in build_specs(data_root)
    ]
    totals = {
        "image_count": sum(item["image_count"] for item in collections),
        "mask_count": sum(item["mask_count"] for item in collections),
        "paired_count": sum(item["paired_count"] for item in collections),
        "pairing_issue_count": sum(
            item["missing_mask_count"] + item["orphan_mask_count"] for item in collections
        ),
    }
    return {
        "data_root": str(data_root),
        "sample_size": args.sample_size,
        "full_shapes": args.full_shapes,
        "full_pixels": args.full_pixels,
        "collections": collections,
        "totals": totals,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--sample-size",
        type=int,
        default=64,
        help="Number of paired masks per collection used for value/pixel summaries unless --full-pixels is set.",
    )
    parser.add_argument(
        "--full-shapes",
        action="store_true",
        help="Scan every paired image/mask for shape summaries. Default uses the deterministic sample.",
    )
    parser.add_argument(
        "--full-pixels",
        action="store_true",
        help="Count mask values over every mask. This is slower but produces exact pixel totals.",
    )
    parser.add_argument("--json-out", type=Path, help="Optional path for a JSON report.")
    parser.add_argument(
        "--fail-on-pairing-errors",
        action="store_true",
        help="Exit non-zero if any image/mask pair is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    print_summary(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {args.json_out}")

    if args.fail_on_pairing_errors and report["totals"]["pairing_issue_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
