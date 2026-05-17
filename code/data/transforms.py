"""transforms.py — MONAI Compose pipelines for VAE / diffusion training.

Two pipelines:
    - `vae_train_transforms()`: for clean-volume VAE training.
        random flips (3 axes), random affine (small ±5° rotations + ±2 mm shifts),
        intensity jitter.
    - `diffusion_train_transforms()`: for paired (clean, corrupted) diffusion training.
        SAME geometric augmentation applied to BOTH volumes (so (z_clean, z_cond)
        stay aligned), but only clean side gets intensity jitter.

We keep augmentation light to avoid distorting the cardiac anatomy (downstream
TAVI evaluation is sensitive to geometry).

Per `.claude/rules/python-code-conventions.md` §4 — every preprocessing op
appears in a Compose pipeline for inspectability.
"""

from __future__ import annotations

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandShiftIntensityd,
    RandSpatialCropd,
    SpatialPadd,
)


def vae_train_transforms(
    p_flip: float = 0.5,
    rotate_range_deg: float = 5.0,
    shift_range_voxels: int = 2,
    intensity_shift_offset: float = 0.05,
) -> Compose:
    """Augmentation pipeline for VAE training (single volume)."""
    rotate_rad = rotate_range_deg * 3.141592653589793 / 180.0
    return Compose(
        [
            RandFlipd(keys=["volume"], prob=p_flip, spatial_axis=0),
            RandFlipd(keys=["volume"], prob=p_flip, spatial_axis=1),
            RandFlipd(keys=["volume"], prob=p_flip, spatial_axis=2),
            RandAffined(
                keys=["volume"],
                prob=0.5,
                rotate_range=(rotate_rad, rotate_rad, rotate_rad),
                translate_range=(shift_range_voxels,) * 3,
                padding_mode="border",
            ),
            RandShiftIntensityd(keys=["volume"], offsets=intensity_shift_offset, prob=0.5),
            RandAdjustContrastd(keys=["volume"], prob=0.3, gamma=(0.9, 1.1)),
            EnsureTyped(keys=["volume"], dtype="float32", track_meta=False),
        ]
    )


def diffusion_train_transforms(
    p_flip: float = 0.5,
    rotate_range_deg: float = 5.0,
    shift_range_voxels: int = 2,
    patch_size: tuple[int, int, int] | None = None,
) -> Compose:
    """Augmentation pipeline for paired (clean, corrupted) diffusion training.

    Geometric augmentations apply to BOTH `volume` and `corrupted` so they
    remain aligned in physical space. No intensity jitter (avoid distribution
    drift between train and inference). When `patch_size` is set, both volumes
    are cropped to the same fixed ROI, matching the frozen VAE's feasible
    128³ operating window.
    """
    rotate_rad = rotate_range_deg * 3.141592653589793 / 180.0
    transforms = [
        RandFlipd(keys=["volume", "corrupted"], prob=p_flip, spatial_axis=0),
        RandFlipd(keys=["volume", "corrupted"], prob=p_flip, spatial_axis=1),
        RandFlipd(keys=["volume", "corrupted"], prob=p_flip, spatial_axis=2),
        RandAffined(
            keys=["volume", "corrupted"],
            prob=0.5,
            rotate_range=(rotate_rad, rotate_rad, rotate_rad),
            translate_range=(shift_range_voxels,) * 3,
            padding_mode="border",
        ),
    ]
    if patch_size is not None:
        transforms.extend(
            [
                SpatialPadd(keys=["volume", "corrupted"], spatial_size=patch_size),
                RandSpatialCropd(
                    keys=["volume", "corrupted"],
                    roi_size=patch_size,
                    random_size=False,
                    random_center=True,
                ),
            ]
        )
    transforms.append(
        EnsureTyped(keys=["volume", "corrupted"], dtype="float32", track_meta=False)
    )
    return Compose(transforms)


def vae_v2_train_transforms(
    patch_size: tuple[int, int, int] = (64, 64, 64),
    p_flip: float = 0.5,
    rotate_range_deg: float = 5.0,
    shift_range_voxels: int = 2,
    intensity_shift_offset: float = 0.05,
) -> Compose:
    """Patch-based VAE training pipeline (MAISI v2 alignment).

    Adds `SpatialPadd → RandSpatialCropd` at the end so the network sees
    fixed-size `patch_size` cubes (default 64³) randomly cropped from the
    volume. This is the memory-efficient alternative to whole-volume training
    (192³ on 48 GB OOMs at bs=2; 64³ patches fit at bs=4 with ~15-20 GB peak).
    """
    rotate_rad = rotate_range_deg * 3.141592653589793 / 180.0
    return Compose(
        [
            RandFlipd(keys=["volume"], prob=p_flip, spatial_axis=0),
            RandFlipd(keys=["volume"], prob=p_flip, spatial_axis=1),
            RandFlipd(keys=["volume"], prob=p_flip, spatial_axis=2),
            RandAffined(
                keys=["volume"],
                prob=0.5,
                rotate_range=(rotate_rad, rotate_rad, rotate_rad),
                translate_range=(shift_range_voxels,) * 3,
                padding_mode="border",
            ),
            RandShiftIntensityd(keys=["volume"], offsets=intensity_shift_offset, prob=0.5),
            RandAdjustContrastd(keys=["volume"], prob=0.3, gamma=(0.9, 1.1)),
            # Pad first in case any spatial dim < patch_size (no-op for 192³ inputs).
            SpatialPadd(keys=["volume"], spatial_size=patch_size),
            RandSpatialCropd(
                keys=["volume"], roi_size=patch_size,
                random_size=False, random_center=True,
            ),
            EnsureTyped(keys=["volume"], dtype="float32", track_meta=False),
        ]
    )


def vae_v2_val_transforms() -> Compose:
    """Identity for VAE-v2 val: full 192³ goes through SlidingWindowInferer."""
    return Compose([EnsureTyped(keys=["volume"], dtype="float32", track_meta=False)])


def eval_transforms() -> Compose:
    """Identity-only pipeline for evaluation (no augmentation)."""
    return Compose([EnsureTyped(keys=["volume"], dtype="float32", track_meta=False)])
