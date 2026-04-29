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

# --- Naming convention assumptions (TENTATIVE — verify with inspect script) ---
# ImageCAS Kaggle dataset typically ships as:
#   data/imagecas/raw/<case_id>/img.nii.gz        # CCTA volume
#   data/imagecas/raw/<case_id>/label.nii.gz      # coronary segmentation
# `<case_id>` is a numeric or alphanumeric directory name.
#
# If actual layout differs, update CASE_DIR_GLOB / VOLUME_FNAME / LABEL_FNAME.

CASE_DIR_GLOB: str = "*"
VOLUME_FNAME: str = "img.nii.gz"
LABEL_FNAME: str = "label.nii.gz"


def case_ids() -> list[str]:
    """Return sorted list of all ImageCAS case directory names."""
    raw = paths.get("IMAGECAS_RAW")
    if not raw.exists():
        raise FileNotFoundError(
            f"ImageCAS raw directory not found at {raw}. "
            f"Either download the dataset (see data/README.md) or update "
            f"data/.paths.local."
        )
    return sorted(p.name for p in raw.glob(CASE_DIR_GLOB) if p.is_dir())


def case_dir(case_id: str) -> Path:
    """Return the directory for `case_id`."""
    return paths.get("IMAGECAS_RAW") / case_id


def load_volume(case_id: str) -> sitk.Image:
    """Load the CCTA volume for a case as a SimpleITK image.

    Returns the volume in HU units, with original voxel spacing and direction.
    Raises FileNotFoundError if the case is missing or the file isn't there.
    """
    f = case_dir(case_id) / VOLUME_FNAME
    if not f.exists():
        raise FileNotFoundError(f"Volume file not found: {f}")
    return sitk.ReadImage(str(f))


def load_label(case_id: str) -> Optional[sitk.Image]:
    """Load coronary-artery segmentation label, or None if not present.

    Some cases in ImageCAS may lack labels — handle the None case explicitly.
    """
    f = case_dir(case_id) / LABEL_FNAME
    if not f.exists():
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
