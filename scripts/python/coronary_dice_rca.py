"""coronary_dice_rca.py — gold-standard downstream coronary-segmentation metric.

Run card: experiments/runs/2026-06-16_*_coronary-dice-rca.md

Runs a FROZEN coronary segmenter (TotalSegmentator 2.13 `coronary_arteries`,
academic-licensed) on each restored volume and scores Dice — the TT U-Net
Dice-RCA protocol and the project's MUST-have downstream metric (novelty #3),
which had only ever been approximated by an HU-threshold proxy.

Two Dice variants per restored volume, vs:
  - TS(clean)  : PRIMARY, domain-gap-free (same segmenter both sides) — "does the
                 restoration recover the coronary structure the segmenter sees on
                 the clean image?". clean-vs-clean = 1.0 by construction.
  - GT lumen   : SECONDARY, absolute (vs ImageCAS GT), comparable to TT U-Net but
                 floored by the TS<->ImageCAS labelling-convention mismatch (~0.4
                 even for clean).

Sanity gate (printed): motion must degrade segmentability, i.e.
Dice(corrupted vs TS-clean) << 1.0; otherwise the metric is insensitive here.

Inputs: NIfTI volumes from `evaluate_residual_diffusion_full_volume.py
--save-volumes-dir` (per case: clean/corrupted/unet/diff_mean/diff_sample).

Usage:
    python -m scripts.python.coronary_dice_rca \
        --volumes-dir experiments/runs/dice_rca_pilot/volumes --case-ids 21 32 41 50 51
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from code.data import paths as path_registry

log = logging.getLogger("coronary_dice_rca")

VTYPES: tuple[str, ...] = ("clean", "corrupted", "unet", "diff_mean", "diff_sample")
TS_BIN = Path(sys.prefix) / "bin" / "TotalSegmentator"


def run_ts_coronary(nifti: Path, out_dir: Path) -> np.ndarray:
    """Run TotalSegmentator coronary_arteries; return binary (Z,Y,X) mask."""
    seg_file = out_dir / "coronary_arteries.nii.gz"
    if not seg_file.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(TS_BIN), "-i", str(nifti), "-o", str(out_dir), "-ta", "coronary_arteries"],
            check=True, capture_output=True, text=True,
        )
    return (sitk.GetArrayFromImage(sitk.ReadImage(str(seg_file))) > 0)


def dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    inter = float(np.logical_and(a, b).sum())
    return (2.0 * inter + eps) / (float(a.sum()) + float(b.sum()) + eps)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--volumes-dir", type=Path, required=True)
    p.add_argument("--case-ids", nargs="+", required=True)
    p.add_argument("--seg-dir", type=Path, default=None, help="cache dir for TS outputs")
    p.add_argument("--gt-dir", type=Path, default=None, help="lumen GT dir (default: processed/lumen_masks)")
    p.add_argument("--out-dir", type=Path, default=Path("experiments/runs/coronary_dice_rca"))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)
    seg_dir = args.seg_dir or (args.out_dir / "seg")
    gt_dir = args.gt_dir or (path_registry.get("IMAGECAS_PROCESSED").parent / "lumen_masks")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for cid in args.case_ids:
        gt = np.load(gt_dir / f"lumen_mask_{cid}.npy").astype(bool)
        segs: dict[str, np.ndarray] = {}
        for vt in VTYPES:
            nifti = args.volumes_dir / f"case_{cid}__{vt}.nii.gz"
            if not nifti.exists():
                log.warning("missing %s — skip", nifti)
                continue
            segs[vt] = run_ts_coronary(nifti, seg_dir / f"case_{cid}__{vt}")
        if "clean" not in segs:
            log.warning("case %s: no clean seg — skip", cid)
            continue
        seg_clean = segs["clean"]
        for vt, seg in segs.items():
            rows.append({
                "case_id": cid, "vtype": vt,
                "dice_vs_clean": dice(seg, seg_clean),
                "dice_vs_gt": dice(seg, gt),
                "seg_voxels": int(seg.sum()),
                "gt_voxels": int(gt.sum()),
            })
            log.info("case %s | %-12s dice_vs_clean=%.3f dice_vs_gt=%.3f (seg %d)",
                     cid, vt, rows[-1]["dice_vs_clean"], rows[-1]["dice_vs_gt"], rows[-1]["seg_voxels"])

    csv_path = args.out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    print("\n==== CORONARY Dice-RCA — per-vtype mean ====")
    print(f"{'vtype':<14}{'Dice vs TS(clean)':>20}{'Dice vs GT':>14}{'seg voxels':>12}")
    for vt in VTYPES:
        rr = [r for r in rows if r["vtype"] == vt]
        if not rr:
            continue
        def m(k): return float(np.mean([r[k] for r in rr]))
        print(f"{vt:<14}{m('dice_vs_clean'):>20.3f}{m('dice_vs_gt'):>14.3f}{m('seg_voxels'):>12.0f}")
    # sanity gate
    corr = [r for r in rows if r["vtype"] == "corrupted"]
    if corr:
        g = float(np.mean([r["dice_vs_clean"] for r in corr]))
        print(f"\nSanity gate: Dice(corrupted vs TS-clean) = {g:.3f} "
              f"({'OK — motion degrades segmentability' if g < 0.85 else 'WEAK — segmenter insensitive to motion'})")
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
