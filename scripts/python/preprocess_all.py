"""preprocess_all.py — batch TotalSegmentator + preprocess for all ImageCAS cases.

Phase 1 of the post-GPU-release pipeline. For each case_id under
data/imagecas/raw/:
    1. Run TotalSegmentator (`task='total'`, `roi_subset=['heart']`, fast=True).
       Output cached at `data/imagecas/heart_masks/heart_mask_<id>.nii.gz`.
    2. Run `code.data.preprocessing.preprocess_volume()` →
       `data/imagecas/processed/v1/case_<id>.npz` (192³, [-1, 1], heart-centered).

Resumable:
    - Skips cases whose `case_<id>.npz` already exists.
    - Skips TotalSegmentator if heart_mask cache already exists.

Failures: written to `experiments/runs/preprocess_all/failed.txt` (one line
per case). Pipeline does NOT abort on a single failure.

Usage:
    # All cases:
    python -m scripts.python.preprocess_all

    # First 50 cases (debug):
    python -m scripts.python.preprocess_all --limit 50

    # Specific cases:
    python -m scripts.python.preprocess_all --case-ids 1 5 17

    # Force CPU TotalSegmentator (useful when GPU is busy):
    python -m scripts.python.preprocess_all --device cpu

Per `.claude/rules/python-code-conventions.md`:
    - relative paths via `code.data.paths`
    - logging not print
    - pathlib.Path everywhere
    - structured failure log instead of crash-on-first-error
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data import imagecas_loader as loader  # noqa: E402
from code.data import paths as path_registry  # noqa: E402
from code.data.preprocessing import preprocess_volume  # noqa: E402

log = logging.getLogger(__name__)


# ============================================================
# TotalSegmentator wrapper
# ============================================================


def segment_whole_heart(
    volume_sitk: sitk.Image,
    work_dir: Path,
    out_path: Path,
    device: str = "gpu",
) -> sitk.Image:
    """Run TotalSegmentator (`heart` ROI from `total` task), return binary mask.

    Caches the input + segmentation to `work_dir`, writes the binary mask to
    `out_path`. Returns the SimpleITK binary mask in the original geometry.
    """
    from totalsegmentator.python_api import totalsegmentator

    work_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    in_path = work_dir / "_input.nii.gz"
    seg_path = work_dir / "seg_total.nii.gz"
    sitk.WriteImage(volume_sitk, str(in_path))

    totalsegmentator(
        input=str(in_path),
        output=str(seg_path),
        task="total",
        roi_subset=["heart"],
        fast=True,
        ml=True,
        device=device,
    )
    if not seg_path.exists():
        raise FileNotFoundError(f"TotalSegmentator did not produce {seg_path}")

    seg_sitk = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(seg_sitk)
    binary = (arr > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(seg_sitk)
    sitk.WriteImage(out, str(out_path))
    return out


# ============================================================
# Per-case driver
# ============================================================


def process_case(
    case_id: str,
    masks_dir: Path,
    processed_dir: Path,
    seg_workdir_root: Path,
    device: str,
    overwrite: bool,
) -> tuple[bool, str]:
    """Returns (ok, msg). On success msg is the .npz path; on failure, error trace."""
    npz_path = processed_dir / f"case_{case_id}.npz"
    if npz_path.exists() and not overwrite:
        return True, f"skip (cached): {npz_path}"

    mask_path = masks_dir / f"heart_mask_{case_id}.nii.gz"
    if not mask_path.exists() or overwrite:
        try:
            log.info("[case %s] segmenting...", case_id)
            t0 = time.time()
            v_sitk = loader.load_volume(case_id)
            segment_whole_heart(
                v_sitk,
                work_dir=seg_workdir_root / f"_seg_workdir_{case_id}",
                out_path=mask_path,
                device=device,
            )
            log.info("[case %s] segment done %.1fs", case_id, time.time() - t0)
        except Exception:
            return False, f"segmentation failed: {traceback.format_exc()}"
    else:
        log.info("[case %s] using cached mask %s", case_id, mask_path)

    try:
        log.info("[case %s] preprocessing...", case_id)
        t0 = time.time()
        out = preprocess_volume(
            case_id=case_id,
            heart_mask_path=mask_path,
            out_dir=processed_dir,
            overwrite=overwrite,
        )
        log.info("[case %s] preprocess done %.1fs → %s", case_id, time.time() - t0, out)
        return True, str(out)
    except Exception:
        return False, f"preprocess failed: {traceback.format_exc()}"


# ============================================================
# Entry-point
# ============================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch TotalSegmentator + preprocess for ImageCAS.")
    p.add_argument("--case-ids", nargs="+", default=None,
                   help="Specific case IDs to process. Default: all under IMAGECAS_RAW.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of cases (after filtering). For debug.")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                   help="TotalSegmentator device.")
    p.add_argument("--masks-dir", type=Path, default=None,
                   help="Heart-mask cache dir. Default: data/imagecas/heart_masks/")
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="Output dir for case_<id>.npz. Default: paths.IMAGECAS_PROCESSED.")
    p.add_argument("--seg-workdir-root", type=Path, default=None,
                   help="Root dir for TotalSegmentator scratch files. "
                        "Default: <masks-dir>/.workdir/")
    p.add_argument("--log-dir", type=Path,
                   default=Path("experiments/runs/preprocess_all"),
                   help="Where to write run.log + failed.txt + manifest.csv.")
    p.add_argument("--overwrite", action="store_true",
                   help="Reprocess cases whose .npz / mask already exist.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_dir / "run.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    processed_dir = args.processed_dir or path_registry.get("IMAGECAS_PROCESSED")
    masks_dir = args.masks_dir or (processed_dir.parent / "heart_masks")
    seg_workdir_root = args.seg_workdir_root or (masks_dir / ".workdir")
    masks_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    seg_workdir_root.mkdir(parents=True, exist_ok=True)

    if args.case_ids is None:
        all_ids = loader.case_ids()
    else:
        all_ids = list(args.case_ids)
    if args.limit is not None:
        all_ids = all_ids[: args.limit]

    log.info("=" * 60)
    log.info("preprocess_all | %d cases | device=%s", len(all_ids), args.device)
    log.info("masks_dir=%s", masks_dir)
    log.info("processed_dir=%s", processed_dir)
    log.info("=" * 60)

    failed: list[tuple[str, str]] = []
    n_done = 0
    n_skipped = 0
    t_start = time.time()

    for i, cid in enumerate(all_ids, start=1):
        log.info("--- [%d/%d] case %s ---", i, len(all_ids), cid)
        try:
            ok, msg = process_case(
                case_id=cid,
                masks_dir=masks_dir,
                processed_dir=processed_dir,
                seg_workdir_root=seg_workdir_root,
                device=args.device,
                overwrite=args.overwrite,
            )
        except Exception:
            ok, msg = False, f"unhandled: {traceback.format_exc()}"
        if ok:
            if msg.startswith("skip"):
                n_skipped += 1
            else:
                n_done += 1
        else:
            failed.append((cid, msg))
            log.error("[case %s] FAIL: %s", cid, msg.splitlines()[0][:200])

    fail_log = args.log_dir / "failed.txt"
    with fail_log.open("w") as fh:
        for cid, msg in failed:
            fh.write(f"{cid}\t{msg.splitlines()[0]}\n")

    log.info("=" * 60)
    log.info(
        "DONE  | processed=%d  skipped=%d  failed=%d  total=%d  wall=%.1fmin",
        n_done, n_skipped, len(failed), len(all_ids), (time.time() - t_start) / 60.0,
    )
    if failed:
        log.info("Failed-case list: %s", fail_log)
    return 0 if not failed else 2  # 2 = partial success


if __name__ == "__main__":
    sys.exit(main())
