"""generate_motion_pairs.py — batch-generate motion-corrupted training pairs.

Phase 2 of the post-GPU-release pipeline. For each preprocessed case
`case_<id>.npz`, generate K motion-corrupted variants:

    case_<id>__pair_<k>.npz  with arrays:
        volume     (192³ float32 in [-1, 1])     # clean target
        corrupted  (192³ float32 in [-1, 1])     # motion-corrupted condition
        heart_mask (192³ uint8)
        metadata   (object: case_id, variant_id, motion_params, scan_params,
                            seed, hu_calibration, ...)

Resumable:
    - Skips a (case, variant) if the file already exists.
    - Each variant uses a deterministic seed = hash(case_id || variant_id).

Failures: written to `experiments/runs/generate_motion_pairs/failed.txt`,
one line per (case_id, variant_id, error). Pipeline does NOT abort.

Usage:
    # All preprocessed cases, K=4 variants per case:
    python -m scripts.python.generate_motion_pairs --variants 4

    # Specific cases:
    python -m scripts.python.generate_motion_pairs --case-ids 1 5 17 --variants 2

    # Reduced sampling (for quick smoke):
    python -m scripts.python.generate_motion_pairs --variants 1 --n-views 500 --n-phases 24

Per `.claude/rules/python-code-conventions.md`:
    - relative paths via `code.data.paths`
    - logging not print
    - structured failure log
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data import paths as path_registry  # noqa: E402
from code.data.motion_synth import (  # noqa: E402
    MotionParams,
    ScanParams,
    synthesize_motion_artifact,
)

log = logging.getLogger(__name__)


# Default training-pair sampling distributions (per motion_synth.MotionParams docstring;
# also documented in quality_reports/specs/2026-04-30_motion-synthesis-pipeline.md).
DEFAULT_MOTION_STRENGTH_RANGE: tuple[float, float] = (0.5, 1.3)
DEFAULT_PERIOD_MS_RANGE: tuple[float, float] = (600.0, 1000.0)


# ============================================================
# HU normalization helpers
# ============================================================


def normalize_hu(arr_hu: np.ndarray, hu_clip: tuple[int, int]) -> np.ndarray:
    """Clip HU then linearly rescale to [-1, 1] (matches preprocessing.py)."""
    lo, hi = hu_clip
    arr = np.clip(arr_hu, lo, hi).astype(np.float32)
    return (arr - lo) / (hi - lo) * 2.0 - 1.0


def denormalize_hu(arr_norm: np.ndarray, hu_clip: tuple[int, int]) -> np.ndarray:
    """Inverse of normalize_hu: [-1, 1] → HU (clipped range)."""
    lo, hi = hu_clip
    return (arr_norm + 1.0) * 0.5 * (hi - lo) + lo


# ============================================================
# Motion-params sampler
# ============================================================


def sample_motion_params(rng: np.random.Generator) -> MotionParams:
    """Per-variant random sampling of motion-strength and cardiac period.

    Keeps the four physiological amplitudes at MotionParams() defaults
    (Stöhr-anchored) and varies the global strength + period. Reduces the
    augmentation hyperparameter space from 4D to 1D — see MotionParams
    docstring for rationale.
    """
    strength = float(rng.uniform(*DEFAULT_MOTION_STRENGTH_RANGE))
    period = float(rng.uniform(*DEFAULT_PERIOD_MS_RANGE))
    return MotionParams(
        motion_strength=strength,
        cardiac_period_ms=period,
    )


# ============================================================
# Per-variant driver
# ============================================================


def generate_one_pair(
    case_id: str,
    variant_id: int,
    npz_clean_path: Path,
    out_path: Path,
    n_views: int,
    n_phases: int,
    device: torch.device,
) -> tuple[bool, str]:
    """Generate one (V_clean, V_corrupted) pair file. Returns (ok, msg)."""
    if out_path.exists():
        return True, f"skip (cached): {out_path}"

    try:
        with np.load(npz_clean_path, allow_pickle=True) as data:
            v_norm = data["volume"]                  # (192, 192, 192) float32 in [-1, 1]
            heart_mask = data["heart_mask"]          # (192, 192, 192) uint8
            meta = data["metadata"].item() if "metadata" in data.files else {}

        hu_clip = tuple(meta.get("hu_clip", (-1024, 3071)))
        target_spacing = tuple(meta.get("target_spacing_mm", (1.0, 1.0, 1.0)))

        # Reverse normalize → HU domain for motion synth
        v_hu_np = denormalize_hu(v_norm, hu_clip)

        # Per-variant deterministic seed — combines case_id + variant_id so the
        # same (case, variant) always reproduces.
        seed = (abs(hash((case_id, variant_id))) & 0x7FFFFFFF)
        rng = np.random.default_rng(seed)
        motion_params = sample_motion_params(rng)
        scan_params = ScanParams(n_views=n_views)

        v_clean_t = torch.from_numpy(v_hu_np).to(device).float()
        mask_t = torch.from_numpy(heart_mask.astype(np.float32)).to(device)

        t0 = time.time()
        v_corrupted_t, debug = synthesize_motion_artifact(
            v_clean_t, mask_t, target_spacing,
            motion_params=motion_params,
            scan_params=scan_params,
            n_phases=n_phases,
            seed=seed,
            calibrate_hu=True,
        )
        synth_s = time.time() - t0

        v_corr_np = v_corrupted_t.detach().cpu().numpy()

        # Forward-normalize back to [-1, 1] for dataset consumption.
        v_corr_norm = normalize_hu(v_corr_np, hu_clip)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            volume=v_norm.astype(np.float32),
            corrupted=v_corr_norm.astype(np.float32),
            heart_mask=heart_mask.astype(np.uint8),
            metadata=np.array(
                {
                    "case_id": case_id,
                    "variant_id": variant_id,
                    "seed": int(seed),
                    "n_phases": int(n_phases),
                    "n_views": int(n_views),
                    "motion_params": dataclasses.asdict(motion_params),
                    "scan_params": dataclasses.asdict(scan_params),
                    "hu_clip": hu_clip,
                    "hu_calibration": debug.get("hu_calibration", {}),
                    "synth_seconds": synth_s,
                },
                dtype=object,
            ),
        )
        # Free GPU memory between variants
        del v_clean_t, mask_t, v_corrupted_t
        if device.type == "cuda":
            torch.cuda.empty_cache()  # explicit because we accumulate per variant
        return True, f"wrote {out_path} (synth {synth_s:.1f}s)"
    except Exception:
        return False, f"failed: {traceback.format_exc()}"


# ============================================================
# Entry-point
# ============================================================


def discover_clean_case_ids(processed_dir: Path) -> list[str]:
    """Cases with `case_<id>.npz` (no __pair_ marker)."""
    return sorted(
        p.stem.removeprefix("case_")
        for p in processed_dir.glob("case_*.npz")
        if "__pair_" not in p.name
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-generate motion pairs from preprocessed cases.")
    p.add_argument("--case-ids", nargs="+", default=None,
                   help="Specific case IDs. Default: all `case_*.npz` in processed dir.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of cases (for debug).")
    p.add_argument("--variants", type=int, default=4,
                   help="Number of motion variants per case (default 4).")
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="Where case_*.npz live AND where pair files are written. "
                        "Default: paths.IMAGECAS_PROCESSED.")
    p.add_argument("--device", default="cuda",
                   help="torch device for motion synth (default cuda).")
    p.add_argument("--n-phases", type=int, default=48)
    p.add_argument("--n-views", type=int, default=1000)
    p.add_argument("--log-dir", type=Path,
                   default=Path("experiments/runs/generate_motion_pairs"),
                   help="Where to write run.log + failed.txt.")
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
    if not processed_dir.exists():
        log.error("processed_dir %s does not exist — run preprocess_all.py first.", processed_dir)
        return 3

    if args.case_ids is None:
        all_ids = discover_clean_case_ids(processed_dir)
    else:
        all_ids = list(args.case_ids)
    if args.limit is not None:
        all_ids = all_ids[: args.limit]

    if not all_ids:
        log.error("No preprocessed cases at %s.", processed_dir)
        return 3

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    log.info("=" * 60)
    log.info("generate_motion_pairs | %d cases × %d variants | device=%s",
             len(all_ids), args.variants, device)
    log.info("processed_dir=%s", processed_dir)
    log.info("n_phases=%d  n_views=%d", args.n_phases, args.n_views)
    log.info("=" * 60)

    failed: list[tuple[str, int, str]] = []
    n_done = 0
    n_skipped = 0
    t_start = time.time()

    for ci, cid in enumerate(all_ids, start=1):
        npz_path = processed_dir / f"case_{cid}.npz"
        if not npz_path.exists():
            log.error("[case %s] missing %s — skipping all variants", cid, npz_path)
            for vid in range(args.variants):
                failed.append((cid, vid, "missing case_<id>.npz"))
            continue

        for vid in range(args.variants):
            out_path = processed_dir / f"case_{cid}__pair_{vid}.npz"
            log.info("--- [%d/%d case=%s variant=%d/%d] ---",
                     ci, len(all_ids), cid, vid + 1, args.variants)
            ok, msg = generate_one_pair(
                case_id=cid, variant_id=vid,
                npz_clean_path=npz_path, out_path=out_path,
                n_views=args.n_views, n_phases=args.n_phases,
                device=device,
            )
            if ok:
                if msg.startswith("skip"):
                    n_skipped += 1
                else:
                    n_done += 1
                log.info("[case %s v%d] %s", cid, vid, msg.splitlines()[0][:200])
            else:
                failed.append((cid, vid, msg))
                log.error("[case %s v%d] FAIL: %s", cid, vid, msg.splitlines()[0][:200])

    fail_log = args.log_dir / "failed.txt"
    with fail_log.open("w") as fh:
        for cid, vid, msg in failed:
            fh.write(f"{cid}\t{vid}\t{msg.splitlines()[0]}\n")

    log.info("=" * 60)
    log.info(
        "DONE | generated=%d  skipped=%d  failed=%d  total=%d  wall=%.1fmin",
        n_done, n_skipped, len(failed),
        len(all_ids) * args.variants, (time.time() - t_start) / 60.0,
    )
    if failed:
        log.info("Failed-(case,variant) list: %s", fail_log)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
