"""Task 1 segmentation dataset indexing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Task1CollectionSpec:
    name: str
    domain: str
    released_split: str
    image_dir: Path
    mask_dir: Path
    image_suffixes: tuple[str, ...]
    mask_suffixes: tuple[str, ...]
    mask_encoding: str


@dataclass(frozen=True)
class Task1Sample:
    collection: str
    domain: str
    released_split: str
    sample_id: str
    case_id: str | None
    case_id_source: str
    image_path: Path
    mask_path: Path
    mask_encoding: str

    def load_image(self) -> np.ndarray:
        with Image.open(self.image_path) as image:
            return np.asarray(image.convert("RGB"))

    def load_mask(self, *, binary_foreground: bool = False) -> np.ndarray:
        mask = load_task1_mask(self.mask_path, self.mask_encoding)
        if binary_foreground:
            return mask_to_binary_foreground(mask)
        return mask

    @property
    def split_role(self) -> str:
        return task1_split_role(self.collection)


def task1_collection_specs(data_root: Path | str) -> list[Task1CollectionSpec]:
    root = Path(data_root)
    seg_root = root / "segmentation"
    human_root = root / "human_dataset_train"
    return [
        Task1CollectionSpec(
            name="animal_train",
            domain="animal",
            released_split="train",
            image_dir=seg_root / "animal_train" / "images",
            mask_dir=seg_root / "animal_train" / "masks",
            image_suffixes=(".png",),
            mask_suffixes=(".npy",),
            mask_encoding="npy_multiclass",
        ),
        Task1CollectionSpec(
            name="animal_test",
            domain="animal",
            released_split="released_test",
            image_dir=seg_root / "animal_test" / "images",
            mask_dir=seg_root / "animal_test" / "masks",
            image_suffixes=(".png",),
            mask_suffixes=(".npy",),
            mask_encoding="npy_multiclass",
        ),
        Task1CollectionSpec(
            name="phantom_train",
            domain="phantom",
            released_split="train",
            image_dir=seg_root / "phantom_train" / "images",
            mask_dir=seg_root / "phantom_train" / "masks",
            image_suffixes=(".png",),
            mask_suffixes=(".npy",),
            mask_encoding="npy_multiclass",
        ),
        Task1CollectionSpec(
            name="phantom_test",
            domain="phantom",
            released_split="released_test",
            image_dir=seg_root / "phantom_test" / "images",
            mask_dir=seg_root / "phantom_test" / "masks",
            image_suffixes=(".png",),
            mask_suffixes=(".npy",),
            mask_encoding="npy_multiclass",
        ),
        Task1CollectionSpec(
            name="human_train",
            domain="human",
            released_split="train",
            image_dir=human_root / "img",
            mask_dir=human_root / "mask",
            image_suffixes=(".jpg", ".jpeg", ".png"),
            mask_suffixes=(".png",),
            mask_encoding="png_binary",
        ),
    ]


def build_task1_index(
    data_root: Path | str = "datasets",
    *,
    collections: Iterable[str] | None = None,
    require_complete_pairs: bool = True,
) -> list[Task1Sample]:
    wanted = set(collections) if collections is not None else None
    samples: list[Task1Sample] = []

    for spec in task1_collection_specs(data_root):
        if wanted is not None and spec.name not in wanted:
            continue

        images = _files_by_key(spec.image_dir, spec.image_suffixes, key_fn=_image_key)
        masks = _files_by_key(spec.mask_dir, spec.mask_suffixes, key_fn=_mask_key)
        image_keys = set(images)
        mask_keys = set(masks)
        missing_masks = sorted(image_keys - mask_keys)
        orphan_masks = sorted(mask_keys - image_keys)

        if require_complete_pairs and (missing_masks or orphan_masks):
            detail = (
                f"{spec.name}: missing_masks={missing_masks[:5]}, "
                f"orphan_masks={orphan_masks[:5]}"
            )
            raise ValueError(f"Image/mask pairing errors detected: {detail}")

        for key in sorted(image_keys & mask_keys):
            case_id, case_id_source = infer_case_id(key, spec.domain)
            samples.append(
                Task1Sample(
                    collection=spec.name,
                    domain=spec.domain,
                    released_split=spec.released_split,
                    sample_id=key,
                    case_id=case_id,
                    case_id_source=case_id_source,
                    image_path=images[key],
                    mask_path=masks[key],
                    mask_encoding=spec.mask_encoding,
                )
            )

    return samples


def load_task1_mask(path: Path | str, mask_encoding: str) -> np.ndarray:
    path = Path(path)
    if mask_encoding == "npy_multiclass":
        return np.load(path)
    if mask_encoding == "png_binary":
        with Image.open(path) as image:
            return np.asarray(image)
    raise ValueError(f"Unknown mask encoding: {mask_encoding}")


def mask_to_binary_foreground(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def infer_case_id(sample_id: str, domain: str) -> tuple[str | None, str]:
    if domain == "human" and "_img-" in sample_id:
        return sample_id.split("_img-", 1)[0], "filename_before_img_marker"
    return None, "unavailable_from_filename"


def summarize_task1_index(samples: Iterable[Task1Sample]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for sample in samples:
        summary[sample.collection] = summary.get(sample.collection, 0) + 1
    return dict(sorted(summary.items()))


def task1_split_role(collection: str) -> str:
    if collection in {"animal_train", "phantom_train"}:
        return "released_train"
    if collection in {"animal_test", "phantom_test"}:
        return "released_eval"
    if collection == "human_train":
        return "human_holdout"
    raise ValueError(f"Unknown Task 1 collection: {collection}")


def _files_by_key(
    directory: Path, suffixes: tuple[str, ...], *, key_fn
) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    suffixes = tuple(suffix.lower() for suffix in suffixes)
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = key_fn(path)
        if key in result:
            raise ValueError(f"Duplicate key {key!r} in {directory}")
        result[key] = path
    return result


def _image_key(path: Path) -> str:
    return path.stem


def _mask_key(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_mask"):
        stem = stem[: -len("_mask")]
    return stem
