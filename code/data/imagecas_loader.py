"""imagecas_loader.py — minimal loader for the ImageCAS dataset.

ImageCAS = 1000 high-resolution coronary CT angiography (CCTA) volumes,
single-phase prospective ECG-gated, with paired coronary segmentation labels.
Apache 2.0 license.  Source: Kaggle (xiaoweixu/imagecas).

This is a Week-1-2 SKELETON.  Concrete file-naming / directory-layout
assumptions have to be confirmed by `scripts/python/imagecas_inspect.py`
once the dataset is downloaded.  Update this module after Day 3 inspection.

Per `.claude/rules/python-code-conventions.md`:
    - pathlib.Path, no os.path.join
    - paths via `code.data.paths`
    - SimpleITK as primary IO (preserves voxel spacing + direction)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import SimpleITK as sitk

from . import paths

# --- Naming convention (verified 2026-04-30 LTSI session) ---
# ImageCAS Kaggle dataset extracts into 5 batch directories:
#   data/imagecas/raw/1-200/<id>.img.nii.gz       # CCTA volume
#   data/imagecas/raw/1-200/<id>.label.nii.gz     # coronary segmentation
#   data/imagecas/raw/201-400/...
#   ... (5 batches, 200 cases each)
# `<id>` is a numeric string like "1", "2", ..., "1000".

VOLUME_SUFFIX: str = ".img.nii.gz"
LABEL_SUFFIX: str = ".label.nii.gz"
BATCH_DIRS: tuple[str, ...] = ("1-200", "201-400", "401-600", "601-800", "801-1000")


def _find_case_file(case_id: str, suffix: str) -> Optional[Path]:
    """Locate a per-case file by scanning the 5 batch directories."""
    raw = paths.get("IMAGECAS_RAW")
    fname = f"{case_id}{suffix}"
    for batch in BATCH_DIRS:
        candidate = raw / batch / fname
        if candidate.exists():
            return candidate
    return None


def case_ids() -> list[str]:
    """Return sorted list of all ImageCAS case IDs (numeric strings)."""
    raw = paths.get("IMAGECAS_RAW")
    if not raw.exists():
        raise FileNotFoundError(
            f"ImageCAS raw directory not found at {raw}. "
            f"Either download + extract the dataset (see data/README.md) "
            f"or update data/.paths.local."
        )
    ids: set[str] = set()
    for batch in BATCH_DIRS:
        bdir = raw / batch
        if not bdir.exists():
            continue
        for img in bdir.glob(f"*{VOLUME_SUFFIX}"):
            cid = img.name[: -len(VOLUME_SUFFIX)]
            ids.add(cid)
    return sorted(ids, key=lambda s: int(s) if s.isdigit() else s)


def case_dir(case_id: str) -> Path:
    """Return the batch directory containing `case_id`. Raises if not found."""
    img = _find_case_file(case_id, VOLUME_SUFFIX)
    if img is None:
        raise FileNotFoundError(f"Case {case_id} not found in any batch directory.")
    return img.parent


def load_volume(case_id: str) -> sitk.Image:
    """Load the CCTA volume for a case as a SimpleITK image.

    Returns the volume in HU units, with original voxel spacing and direction.
    Raises FileNotFoundError if the case is missing or the file isn't there.
    """
    f = _find_case_file(case_id, VOLUME_SUFFIX)
    if f is None:
        raise FileNotFoundError(f"Volume file not found for case {case_id}")
    return sitk.ReadImage(str(f))


def load_label(case_id: str) -> Optional[sitk.Image]:
    """Load coronary-artery segmentation label, or None if not present."""
    f = _find_case_file(case_id, LABEL_SUFFIX)
    if f is None:
        return None
    return sitk.ReadImage(str(f))


def iterate_cases(
    limit: Optional[int] = None,
    require_label: bool = False,
) -> Iterator[tuple[str, sitk.Image, Optional[sitk.Image]]]:
    """Iterate (case_id, volume, label?) tuples.

    Args:
        limit: optionally cap the number of cases yielded (debug / dev).
        require_label: if True, skip cases without a segmentation label.
    """
    ids = case_ids()
    if limit is not None:
        ids = ids[:limit]
    for cid in ids:
        vol = load_volume(cid)
        lbl = load_label(cid)
        if require_label and lbl is None:
            continue
        yield cid, vol, lbl


def summarize_volume(img: sitk.Image) -> dict[str, object]:
    """Return a small dict describing one volume — for SUMMARY.md generation."""
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    return {
        "shape": tuple(int(s) for s in arr.shape),
        "spacing": tuple(float(s) for s in img.GetSpacing()),
        "origin": tuple(float(s) for s in img.GetOrigin()),
        "direction": tuple(float(d) for d in img.GetDirection()),
        "hu_min": float(arr.min()),
        "hu_max": float(arr.max()),
        "hu_mean": float(arr.mean()),
        "n_voxels": int(arr.size),
    }
