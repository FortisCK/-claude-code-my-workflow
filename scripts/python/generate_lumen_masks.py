"""generate_lumen_masks.py — align ImageCAS coronary GT labels into the 192³ frame.

Run card: experiments/runs/2026-06-16_*_lumen-gt-build.md

ImageCAS ships paired coronary segmentation labels (`<id>.label.nii.gz`) but the
processed `.npz` only carry `volume / corrupted / heart_mask`. This standalone
script maps each raw coronary label into the SAME 192³ frame as `volume`, using
the EXACT preprocessing transform (verified bit-exact by the 2026-06-16 scout):

    raw label --(NN resample onto v_resampled grid)--> --(crop_centered @ stored
    heart centroid, pad 0)--> 192³ uint8 lumen mask

It reads `metadata['centroid_voxel']` from the processed npz (== the
heart-centroid the pipeline used), so it needs NO TotalSegmentator re-run and
does NOT modify any existing npz. Masks are written as a non-destructive sidecar:

    data/imagecas/processed/lumen_masks/lumen_mask_<id>.npy   (uint8, 192³)

Usage:
    python -m scripts.python.generate_lumen_masks --split test
    python -m scripts.python.generate_lumen_masks --case-ids 21 32 41 50 51
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from code.data import imagecas_loader, paths
from code.data.preprocessing import (
    DEFAULT_CROP_SIZE,
    DEFAULT_SPACING_MM,
    crop_centered,
    resample_to_isotropic,
)

log = logging.getLogger(__name__)


def lumen_mask_for_case(case_id: str, processed_dir: Path) -> np.ndarray:
    """Build the 192³ aligned coronary lumen mask for one case.

    Reads the stored heart centroid from the processed npz and replays the exact
    resample(NN)+crop the pipeline applied to `volume`/`heart_mask`.

    Returns:
        uint8 (192,192,192) array, {0,1}, voxelwise-aligned to `volume`.

    Raises:
        FileNotFoundError if the raw label or processed npz is missing.
    """
    npz_path = processed_dir / f"case_{case_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"processed npz not found: {npz_path}")
    meta = np.load(npz_path, allow_pickle=True)["metadata"].item()
    centroid = tuple(int(c) for c in meta["centroid_voxel"])  # (Z, Y, X)

    lbl_sitk = imagecas_loader.load_label(case_id)
    if lbl_sitk is None:
        raise FileNotFoundError(f"coronary label not found for case {case_id}")
    v_sitk = imagecas_loader.load_volume(case_id)

    # Build the SAME resampled-volume grid the pipeline used, then land the label
    # on that exact lattice with nearest-neighbour (mirrors preprocessing.py:190-197).
    v_resampled = resample_to_isotropic(v_sitk, DEFAULT_SPACING_MM, is_label=False)
    lbl_resampled = sitk.Resample(
        lbl_sitk,
        v_resampled,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        lbl_sitk.GetPixelID(),
    )
    lbl_arr = sitk.GetArrayFromImage(lbl_resampled).astype(np.uint8)  # (Z, Y, X)

    # The raw label may have positive values > 1 in some encodings; binarize.
    lbl_arr = (lbl_arr > 0).astype(np.uint8)
    raw_voxels = int(lbl_arr.sum())

    lumen = crop_centered(lbl_arr, centroid, DEFAULT_CROP_SIZE, pad_value=0).astype(
        np.uint8
    )
    kept = int(lumen.sum())
    retained = (kept / raw_voxels) if raw_voxels > 0 else 0.0
    log.info(
        "case %s: lumen voxels @1mm=%d, kept-in-crop=%d (%.1f%%)",
        case_id,
        raw_voxels,
        kept,
        100.0 * retained,
    )
    return lumen


def _resolve_case_ids(args: argparse.Namespace) -> list[str]:
    if args.case_ids:
        return [str(c) for c in args.case_ids]
    split_file = paths.get("IMAGECAS_SPLITS") / "v1.json"
    splits = json.loads(split_file.read_text())
    ids = splits[args.split]
    if args.max_cases is not None:
        ids = ids[: args.max_cases]
    return [str(c) for c in ids]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--case-ids", nargs="+", default=None, help="explicit case ids (override --split)")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)
    processed_dir = args.processed_dir or paths.get("IMAGECAS_PROCESSED")
    out_dir = args.out_dir or (processed_dir.parent / "lumen_masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    case_ids = _resolve_case_ids(args)
    log.info("generating lumen masks for %d cases -> %s", len(case_ids), out_dir)

    stats: list[dict[str, object]] = []
    for cid in case_ids:
        out_path = out_dir / f"lumen_mask_{cid}.npy"
        if out_path.exists() and not args.overwrite:
            arr = np.load(out_path)
            stats.append({"case_id": cid, "voxels": int(arr.sum()), "skipped": True})
            continue
        lumen = lumen_mask_for_case(cid, processed_dir)
        np.save(out_path, lumen)
        stats.append({"case_id": cid, "voxels": int(lumen.sum()), "skipped": False})

    voxels = np.array([s["voxels"] for s in stats], dtype=np.int64)
    empties = [s["case_id"] for s in stats if s["voxels"] == 0]
    log.info(
        "DONE %d masks | lumen voxels: min=%d median=%d max=%d mean=%.0f | empty=%d",
        len(stats),
        int(voxels.min()),
        int(np.median(voxels)),
        int(voxels.max()),
        float(voxels.mean()),
        len(empties),
    )
    if empties:
        log.warning("EMPTY lumen masks (no coronary voxels in 192³ crop): %s", empties)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
