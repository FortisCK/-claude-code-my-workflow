"""synth_motion_demo.py — Day 6 demo: 1 ImageCAS case → motion-corrupted via parametric DVF.

Pipeline:
    1. Load V_clean from one ImageCAS case (NIfTI via SimpleITK).
    2. Run TotalSegmentator to get whole-heart mask (heart_LV+RV+LA+RA union).
    3. Crop V_clean + heart_mask to a bounding box around the heart (saves
       GPU memory + time; the diffusion model will likely work on cropped
       ROI anyway).
    4. Run synthesize_motion_artifact() with default MotionParams + ScanParams.
    5. Save PNG side-by-side: clean / corrupted / difference at mid-axial,
       coronal, sagittal slices.

Usage:
    python scripts/python/synth_motion_demo.py --case-id 001 --out explorations/figures/motion-day6/
    python scripts/python/synth_motion_demo.py --case-id 001 --skip-segmentation \
            --heart-mask path/to/precomputed_mask.nii.gz

Per `.claude/rules/python-code-conventions.md`:
    - pathlib.Path
    - logging not print (best-effort given this is one-off demo)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

# Make `code` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from code.data import imagecas_loader as loader  # noqa: E402
from code.data.motion_synth import (  # noqa: E402
    MotionParams,
    ScanParams,
    synthesize_motion_artifact,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("synth_motion_demo")


# ============================================================
# Whole-heart segmentation via TotalSegmentator
# ============================================================


def segment_whole_heart(
    volume_sitk: sitk.Image,
    work_dir: Path,
) -> sitk.Image:
    """Run TotalSegmentator and return a binary {0, 1} whole-heart mask.

    Uses the `total` task which now (>=2.x) exposes a single `heart` ROI
    covering the whole heart. Older versions split into chambers/myocardium
    (now in `heartchambers_highres`); for the bbox-crop use case we don't
    need chamber-level detail.
    """
    from totalsegmentator.python_api import totalsegmentator

    work_dir.mkdir(parents=True, exist_ok=True)
    in_path = work_dir / "_input.nii.gz"
    seg_path = work_dir / "seg_total.nii.gz"
    sitk.WriteImage(volume_sitk, str(in_path))

    log.info("Running TotalSegmentator (this may take 30-60s on first run, downloads weights)...")
    t0 = time.time()
    # ml=True writes a single multi-label NIfTI; `output` is treated as a
    # file path (not a directory) in this mode.
    totalsegmentator(
        input=str(in_path),
        output=str(seg_path),
        task="total",
        roi_subset=["heart"],
        fast=True,  # 3mm res — much faster than 1.5mm; sufficient for masking
        ml=True,
        device="gpu",
    )
    log.info(f"TotalSegmentator done in {time.time() - t0:.1f}s")

    if not seg_path.exists():
        raise FileNotFoundError(f"Expected {seg_path} not produced by TotalSegmentator")

    seg_sitk = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(seg_sitk)
    binary = (arr > 0).astype(np.uint8)  # any heart label → 1
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(seg_sitk)
    return out


# ============================================================
# Heart-bbox cropping (saves GPU mem + time)
# ============================================================


def crop_to_heart_bbox(
    v_arr: np.ndarray,  # (Z, Y, X)
    mask_arr: np.ndarray,  # (Z, Y, X)
    pad_mm: float = 30.0,
    voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    """Crop volume + mask to a padded bbox around the heart mask.

    Returns (v_cropped, mask_cropped, slices) — slices to map back to original.
    """
    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        raise ValueError("Empty heart mask; cannot crop.")

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    pad_z = int(np.ceil(pad_mm / voxel_size_mm[0]))
    pad_y = int(np.ceil(pad_mm / voxel_size_mm[1]))
    pad_x = int(np.ceil(pad_mm / voxel_size_mm[2]))

    Z, Y, X = v_arr.shape
    sl = (
        slice(max(0, z_min - pad_z), min(Z, z_max + pad_z + 1)),
        slice(max(0, y_min - pad_y), min(Y, y_max + pad_y + 1)),
        slice(max(0, x_min - pad_x), min(X, x_max + pad_x + 1)),
    )
    return v_arr[sl], mask_arr[sl], sl


# ============================================================
# Visualization
# ============================================================


def save_comparison_pngs(
    v_clean: np.ndarray,
    v_corrupted: np.ndarray,
    out_dir: Path,
    case_id: str,
    hu_window: tuple[float, float] = (-200, 600),  # CCTA contrast window
) -> None:
    """Save 3-row × 3-col PNG: rows = axial / coronal / sagittal; cols = clean / corrupted / diff."""
    out_dir.mkdir(parents=True, exist_ok=True)

    z_mid = v_clean.shape[0] // 2
    y_mid = v_clean.shape[1] // 2
    x_mid = v_clean.shape[2] // 2

    slices = [
        ("axial", v_clean[z_mid], v_corrupted[z_mid]),
        ("coronal", v_clean[:, y_mid, :], v_corrupted[:, y_mid, :]),
        ("sagittal", v_clean[:, :, x_mid], v_corrupted[:, :, x_mid]),
    ]

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    vmin, vmax = hu_window
    diff_lim = 200

    for r, (name, c, k) in enumerate(slices):
        axs[r, 0].imshow(c, cmap="gray", vmin=vmin, vmax=vmax)
        axs[r, 0].set_title(f"clean — {name}")
        axs[r, 0].axis("off")

        axs[r, 1].imshow(k, cmap="gray", vmin=vmin, vmax=vmax)
        axs[r, 1].set_title(f"corrupted — {name}")
        axs[r, 1].axis("off")

        diff = k - c
        axs[r, 2].imshow(diff, cmap="seismic", vmin=-diff_lim, vmax=diff_lim)
        axs[r, 2].set_title(f"diff — {name}")
        axs[r, 2].axis("off")

    fig.suptitle(f"ImageCAS case {case_id} — parametric motion synthesis (Day 6 demo)")
    fig.tight_layout()
    out_path = out_dir / f"case_{case_id}_motion_synth.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-id", type=str, required=True, help="ImageCAS case ID (directory name)")
    p.add_argument("--out", type=Path, default=Path("explorations/figures/motion-day6"),
                   help="Output dir for PNG + heart_mask cache")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-phases", type=int, default=48)
    p.add_argument("--n-views", type=int, default=1000, help="N projection views (lower for speed)")
    p.add_argument("--no-crop", action="store_true", help="Skip heart-bbox cropping (slower)")
    p.add_argument("--skip-segmentation", action="store_true",
                   help="Use cached heart_mask if present; fail otherwise.")
    p.add_argument("--heart-mask", type=Path, default=None,
                   help="Optional pre-computed heart mask NIfTI; skip TotalSegmentator")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- Load V_clean ----
    log.info(f"Loading case {args.case_id} ...")
    v_sitk = loader.load_volume(args.case_id)
    v_arr = sitk.GetArrayFromImage(v_sitk)  # (Z, Y, X)
    spacing = v_sitk.GetSpacing()  # (x, y, z)
    voxel_size_mm = (float(spacing[2]), float(spacing[1]), float(spacing[0]))  # (z, y, x)
    log.info(f"  shape={v_arr.shape}  voxel={voxel_size_mm} mm  HU=[{v_arr.min()}, {v_arr.max()}]")

    # ---- Heart mask ----
    cache_mask_path = args.out / f"heart_mask_{args.case_id}.nii.gz"
    if args.heart_mask is not None:
        log.info(f"Loading provided heart mask: {args.heart_mask}")
        mask_sitk = sitk.ReadImage(str(args.heart_mask))
    elif args.skip_segmentation:
        if not cache_mask_path.exists():
            log.error(f"--skip-segmentation set but cache {cache_mask_path} not found.")
            return 1
        mask_sitk = sitk.ReadImage(str(cache_mask_path))
    else:
        if cache_mask_path.exists():
            log.info(f"Using cached heart mask: {cache_mask_path}")
            mask_sitk = sitk.ReadImage(str(cache_mask_path))
        else:
            log.info("Segmenting whole heart with TotalSegmentator ...")
            mask_sitk = segment_whole_heart(v_sitk, args.out / f"_seg_workdir_{args.case_id}")
            sitk.WriteImage(mask_sitk, str(cache_mask_path))
            log.info(f"Cached heart mask: {cache_mask_path}")

    mask_arr = sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)
    log.info(f"  heart mask: {int(mask_arr.sum())} voxels ({100 * mask_arr.mean():.2f}% of volume)")

    # ---- Crop to heart bbox ----
    if not args.no_crop:
        v_arr_c, mask_arr_c, sl = crop_to_heart_bbox(v_arr, mask_arr, pad_mm=30.0,
                                                    voxel_size_mm=voxel_size_mm)
        log.info(f"  cropped: {v_arr.shape} -> {v_arr_c.shape}")
    else:
        v_arr_c, mask_arr_c, sl = v_arr, mask_arr, (slice(None), slice(None), slice(None))

    # ---- Move to device + synthesize ----
    device = torch.device(args.device)
    v_clean_t = torch.from_numpy(v_arr_c.astype(np.float32)).to(device)
    heart_mask_t = torch.from_numpy(mask_arr_c.astype(np.float32)).to(device)

    scan_params = ScanParams(n_views=args.n_views)
    motion_params = MotionParams()

    log.info(f"Running motion synthesis: N_phases={args.n_phases}, N_views={args.n_views} ...")
    t0 = time.time()
    v_corr_t, debug = synthesize_motion_artifact(
        v_clean_t, heart_mask_t, voxel_size_mm,
        motion_params=motion_params, scan_params=scan_params,
        n_phases=args.n_phases, seed=args.seed, calibrate_hu=True,
    )
    log.info(f"Synthesis done in {time.time() - t0:.1f}s")
    log.info(f"  HU calibration: {debug['hu_calibration']}")
    log.info(f"  V_corrupted HU: [{v_corr_t.min().item():.0f}, {v_corr_t.max().item():.0f}]"
             f"   mean={v_corr_t.mean().item():.0f}")

    # ---- Save NIfTI ----
    v_corr_arr = v_corr_t.cpu().numpy()
    out_nii = args.out / f"case_{args.case_id}_corrupted.nii.gz"
    out_sitk = sitk.GetImageFromArray(v_corr_arr)
    out_sitk.SetSpacing(spacing)
    # Note: origin shift if cropped; for visualization we don't need exact world coords.
    sitk.WriteImage(out_sitk, str(out_nii))
    log.info(f"Saved: {out_nii}")

    # ---- Side-by-side PNG ----
    save_comparison_pngs(v_arr_c, v_corr_arr, args.out, args.case_id)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
