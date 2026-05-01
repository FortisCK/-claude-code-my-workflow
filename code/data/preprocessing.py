"""preprocessing.py — ImageCAS volume preprocessing for KL-VAE / diffusion training.

Pipeline (per Stage-1 spec, plan: atomic-munching-naur.md Step 1):

    raw NIfTI (HU, native spacing)
        │
        ├─ resample to isotropic 1 mm³ via SimpleITK
        ├─ HU clip to [-1024, 3071]  (stay in standard CT range)
        ├─ crop a (192, 192, 192) box centered on heart_mask centroid
        │   (heart_mask comes from TotalSegmentator — pre-computed,
        │    not done here; this module assumes it is provided)
        ├─ normalize HU to [-1, 1] via linear rescale of clip range
        ▼
    .npz with: volume(192³ float32), heart_mask(192³ uint8), metadata(dict)

Why pre-process to disk:
    - VAE training reads the .npz directly (fast I/O, no SimpleITK overhead)
    - Multiple training runs share the same preprocessed cache
    - Augmentation (random flip / rotate / intensity jitter) happens on tensors
      in the dataloader, so this script outputs the deterministic backbone

Per `.claude/rules/python-code-conventions.md`:
    - pathlib.Path / no os.path.join
    - paths via `code.data.paths`
    - no float == ; use torch.allclose
    - relative paths only

Smoke test:
    python -m code.data.preprocessing --case-id 1 \\
        --heart-mask explorations/motion-day6/output_v2/heart_mask_1.nii.gz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from . import imagecas_loader, paths

log = logging.getLogger(__name__)

# Module-level constants (no magic numbers in function bodies)
DEFAULT_SPACING_MM: tuple[float, float, float] = (1.0, 1.0, 1.0)  # isotropic 1 mm³
DEFAULT_HU_CLIP: tuple[int, int] = (-1024, 3071)                  # standard CT HU range
DEFAULT_CROP_SIZE: tuple[int, int, int] = (192, 192, 192)         # voxels (post-resample)
HU_NORMALIZE_TO: tuple[float, float] = (-1.0, 1.0)                # VAE input target


# ============================================================
# Single-case preprocessing
# ============================================================


def resample_to_isotropic(
    image: sitk.Image,
    target_spacing: tuple[float, float, float],
    is_label: bool = False,
) -> sitk.Image:
    """Resample a SimpleITK image to target voxel spacing.

    Linear interpolation for HU volumes; nearest-neighbour for binary masks.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interp,
        image.GetOrigin(),
        target_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def heart_centroid_voxel(mask_arr: np.ndarray) -> tuple[int, int, int]:
    """Centroid (Z, Y, X) voxel index of a non-empty binary mask.

    Raises ValueError if mask is empty.
    """
    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        raise ValueError("heart_mask is empty — cannot compute centroid for cropping")
    return tuple(int(c) for c in coords.mean(axis=0))


def crop_centered(
    arr: np.ndarray,
    center: tuple[int, int, int],
    crop_size: tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Crop `arr` (3D) to `crop_size` centered on `center`.

    Pads with `pad_value` if the crop window extends outside the array.
    """
    out = np.full(crop_size, pad_value, dtype=arr.dtype)
    src_starts = [center[i] - crop_size[i] // 2 for i in range(3)]
    src_ends = [src_starts[i] + crop_size[i] for i in range(3)]
    dst_starts = [max(-src_starts[i], 0) for i in range(3)]
    src_starts_clip = [max(src_starts[i], 0) for i in range(3)]
    src_ends_clip = [min(src_ends[i], arr.shape[i]) for i in range(3)]
    dst_ends = [
        dst_starts[i] + (src_ends_clip[i] - src_starts_clip[i]) for i in range(3)
    ]
    out[
        dst_starts[0]:dst_ends[0],
        dst_starts[1]:dst_ends[1],
        dst_starts[2]:dst_ends[2],
    ] = arr[
        src_starts_clip[0]:src_ends_clip[0],
        src_starts_clip[1]:src_ends_clip[1],
        src_starts_clip[2]:src_ends_clip[2],
    ]
    return out


def normalize_hu(
    arr: np.ndarray,
    hu_clip: tuple[int, int],
    out_range: tuple[float, float] = HU_NORMALIZE_TO,
) -> np.ndarray:
    """Clip HU to [hu_clip] then linearly rescale to out_range."""
    lo, hi = hu_clip
    out_lo, out_hi = out_range
    arr_clipped = np.clip(arr, lo, hi).astype(np.float32)
    return (arr_clipped - lo) / (hi - lo) * (out_hi - out_lo) + out_lo


def preprocess_volume(
    case_id: str,
    heart_mask_path: Path,
    out_dir: Path,
    target_spacing: tuple[float, float, float] = DEFAULT_SPACING_MM,
    hu_clip: tuple[int, int] = DEFAULT_HU_CLIP,
    crop_size: tuple[int, int, int] = DEFAULT_CROP_SIZE,
    overwrite: bool = False,
) -> Path:
    """End-to-end preprocessing for a single ImageCAS case.

    Args:
        case_id: ImageCAS numeric case ID (e.g., "1").
        heart_mask_path: path to a binary heart mask NIfTI (from TotalSegmentator,
            in the same physical space as the raw volume).
        out_dir: output directory; case-named .npz written here.
        target_spacing: voxel size after resampling, default 1 mm³.
        hu_clip: HU clipping range before normalization.
        crop_size: voxel size of the heart-centered crop.
        overwrite: if False, skip cases already preprocessed.

    Returns:
        Path to the written .npz.

    Raises:
        FileNotFoundError if case or mask not on disk.
        ValueError if mask is empty.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_{case_id}.npz"

    if out_path.exists() and not overwrite:
        log.info("[skip] case %s — already at %s", case_id, out_path)
        return out_path

    log.info("Preprocessing case %s ...", case_id)

    # ---- Load raw volume + heart mask ----
    v_sitk = imagecas_loader.load_volume(case_id)
    if not heart_mask_path.exists():
        raise FileNotFoundError(f"heart_mask not found: {heart_mask_path}")
    mask_sitk = sitk.ReadImage(str(heart_mask_path))

    # ---- Resample both to target spacing ----
    v_resampled = resample_to_isotropic(v_sitk, target_spacing, is_label=False)
    # Mask was generated at the original resolution; resample using the volume's
    # post-resample geometry as reference (handles spacing mismatch from
    # TotalSegmentator's internal 3-mm resampling).
    mask_resampled = sitk.Resample(
        mask_sitk,
        v_resampled,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask_sitk.GetPixelID(),
    )

    v_arr = sitk.GetArrayFromImage(v_resampled)  # (Z, Y, X)
    mask_arr = sitk.GetArrayFromImage(mask_resampled).astype(np.uint8)

    # ---- Crop both centered on heart centroid ----
    centroid = heart_centroid_voxel(mask_arr)
    v_crop = crop_centered(v_arr, centroid, crop_size, pad_value=hu_clip[0])
    mask_crop = crop_centered(mask_arr, centroid, crop_size, pad_value=0).astype(
        np.uint8
    )

    # ---- HU clip + normalize to [-1, 1] ----
    v_norm = normalize_hu(v_crop, hu_clip, HU_NORMALIZE_TO)

    # ---- Save .npz ----
    np.savez_compressed(
        out_path,
        volume=v_norm.astype(np.float32),
        heart_mask=mask_crop,
        metadata=np.array(
            {
                "case_id": case_id,
                "target_spacing_mm": target_spacing,
                "hu_clip": hu_clip,
                "crop_size": crop_size,
                "hu_normalize_to": HU_NORMALIZE_TO,
                "centroid_voxel": centroid,
                "n_heart_voxels": int(mask_crop.sum()),
            },
            dtype=object,
        ),
    )
    log.info(
        "  wrote %s  (volume %s [%.3f, %.3f]  heart-voxels %d)",
        out_path,
        v_norm.shape,
        float(v_norm.min()),
        float(v_norm.max()),
        int(mask_crop.sum()),
    )
    return out_path


# ============================================================
# CLI for smoke test
# ============================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case-id", required=True, type=str)
    p.add_argument(
        "--heart-mask",
        required=True,
        type=Path,
        help="Path to binary heart-mask NIfTI (from TotalSegmentator)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: paths.IMAGECAS_PROCESSED)",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)
    out_dir = args.out_dir or paths.get("IMAGECAS_PROCESSED")
    out_path = preprocess_volume(
        case_id=args.case_id,
        heart_mask_path=args.heart_mask,
        out_dir=out_dir,
        overwrite=args.overwrite,
    )
    print(f"OK: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
