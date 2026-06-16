"""ood_motion_probe.py — does the U-Net genuinely correct motion, or did it
overfit the synthetic forward operator?

Run card: experiments/runs/2026-06-16_*_ood-motion-probe.md

The training pairs varied ONLY motion_strength ~ U(0.5,1.3) and
cardiac_period_ms ~ U(600,1000); the four physiological amplitudes
(contraction 10mm / twist 12° / long-axis 10mm) AND the scan geometry /
reconstruction phase (75%) were FIXED. So the corruption family the U-Net saw
is essentially 1-2 dimensional. This probe regenerates corrupted volumes for
test cases under regimes the U-Net NEVER saw and measures whether its error
explodes (memorized the operator) or stays low (genuinely robust):

    control     : in-distribution (strength 1.0, period 800, default amps, recon 75%)
    ood_amp     : amplitudes bumped (twist 22°, contraction 14mm, long-axis 14mm) — training never varied amps
    ood_strength: strength 2.2 (beyond the 1.3 training max)
    ood_phase   : reconstruct at 45% (end-systole) instead of trained 75% — forward-operator shift

Usage:
    python -m scripts.python.ood_motion_probe --test-cases 5
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from monai.utils import set_determinism
from omegaconf import OmegaConf

from code.data import paths as path_registry
from code.data.motion_synth import (
    MotionParams,
    ScanParams,
    make_random_smooth_dvf_fn,
    synthesize_motion_artifact,
)
from code.data.splits import load_split
from code.evaluation.artifact_metrics import load_heart_mask, load_lumen_mask, metric_row
from code.inference.unet_sliding_window import sliding_window_unet_correct
from code.training.train_residual_refiner import load_initializer

log = logging.getLogger("ood_motion_probe")

HU_CLIP: tuple[int, int] = (-1024, 3071)
N_PHASES: int = 48
N_VIEWS: int = 1000


def denorm(arr_norm: np.ndarray) -> np.ndarray:
    lo, hi = HU_CLIP
    return (arr_norm + 1.0) * 0.5 * (hi - lo) + lo


def norm(arr_hu: np.ndarray) -> np.ndarray:
    lo, hi = HU_CLIP
    return (np.clip(arr_hu, lo, hi).astype(np.float32) - lo) / (hi - lo) * 2.0 - 1.0


# Regime definitions. Each: (MotionParams, ScanParams, dvf_peak_mm).
# dvf_peak_mm=None -> the trained parametric 4-component motion model.
# dvf_peak_mm=float -> a structurally-DIFFERENT random smooth motion model
#   (no contraction/twist/long-axis structure), severity-matched via peak_mm.
# This is the HARD OOD: same forward operator (warp->project->FBP), different
# motion MODEL. NOTE: reconstruction_phase_pct is a dead param in motion_synth,
# and n_views 500 vs 1000 both oversample a 192^3 recon (verified ~no-op), so
# operator-knob OOD is uninformative; motion-MODEL OOD is the decisive test.
def regimes(rand_peak_mm: float) -> dict[str, tuple[MotionParams, ScanParams, float | None]]:
    base = MotionParams(motion_strength=1.0, cardiac_period_ms=800.0)  # in-distribution
    sp_train = ScanParams(n_views=N_VIEWS)  # 1000 views — training geometry
    return {
        "control": (base, sp_train, None),
        "ood_amp": (
            replace(base, contraction_amp_mm=14.0, twist_amp_deg=22.0, long_axis_amp_mm=14.0),
            sp_train, None,
        ),
        "ood_strength": (replace(base, motion_strength=2.2), sp_train, None),
        "ood_randdvf": (base, sp_train, rand_peak_mm),  # structurally-different motion model
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=Path("code/training/configs/diffusion_v2_residual.yaml"),
                   help="config providing the frozen initializer (U-Net) block")
    p.add_argument("--test-cases", type=int, default=5)
    p.add_argument("--case-ids", nargs="+", default=None)
    p.add_argument("--rand-peak-mm", type=float, default=12.0,
                   help="peak displacement (mm) of the structurally-different random DVF; "
                        "tune to severity-match the parametric control")
    p.add_argument("--roi-size", type=int, default=128)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path,
                   default=Path("experiments/runs/ood_motion_probe"))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)
    set_determinism(seed=args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = path_registry.get("IMAGECAS_PROCESSED")

    cfg = OmegaConf.load(args.config)
    unet = load_initializer(cfg.initializer, device)

    if args.case_ids:
        case_ids = [str(c) for c in args.case_ids]
    else:
        split = load_split(path_registry.get("IMAGECAS_SPLITS") / "v1.json")
        case_ids = [str(c) for c in split.test[: args.test_cases]]

    reg = regimes(args.rand_peak_mm)
    rows: list[dict[str, object]] = []
    for cid in case_ids:
        npz = np.load(processed_dir / f"case_{cid}.npz", allow_pickle=True)
        v_norm = npz["volume"]
        heart_np = npz["heart_mask"].astype(np.float32)
        spacing = tuple(npz["metadata"].item().get("target_spacing_mm", (1.0, 1.0, 1.0)))
        clean = torch.from_numpy(v_norm).unsqueeze(0).unsqueeze(0).to(device).float()
        heart_mask = load_heart_mask(cid, processed_dir, tuple(v_norm.shape))
        lumen_mask = load_lumen_mask(cid, processed_dir, tuple(v_norm.shape))
        v_hu = torch.from_numpy(denorm(v_norm)).to(device).float()
        mask_t = torch.from_numpy(heart_np).to(device)

        for label, (mp, sp, dvf_peak_mm) in reg.items():
            seed = (abs(hash((cid, label))) & 0x7FFFFFFF)
            dvf_fn = (
                make_random_smooth_dvf_fn(seed=seed, peak_mm=dvf_peak_mm)
                if dvf_peak_mm is not None
                else None
            )
            v_corr_hu, _ = synthesize_motion_artifact(
                v_hu, mask_t, spacing, motion_params=mp, scan_params=sp,
                n_phases=N_PHASES, seed=seed, calibrate_hu=True, dvf_fn=dvf_fn,
            )
            corr_norm = norm(v_corr_hu.detach().cpu().numpy())
            corrupted = torch.from_numpy(corr_norm).unsqueeze(0).unsqueeze(0).to(device).float()
            unet_out = sliding_window_unet_correct(
                corrupted, unet, roi_size=args.roi_size, overlap=args.overlap,
                sw_device=device, output_device="cpu",
            ).float()
            clean_cpu = clean.cpu()
            corr_cpu = corrupted.cpu()
            cm = metric_row("corrupted", corr_cpu, clean_cpu, heart_mask=heart_mask, lumen_mask=lumen_mask)
            um = metric_row("unet", unet_out, clean_cpu, heart_mask=heart_mask, lumen_mask=lumen_mask)
            removed = (cm["corrupted_mae_hu"] - um["unet_mae_hu"]) / max(cm["corrupted_mae_hu"], 1e-6)
            row = {
                "case_id": cid, "regime": label,
                "corrupted_mae_hu": cm["corrupted_mae_hu"],
                "unet_mae_hu": um["unet_mae_hu"],
                "unet_heart_mae_hu": um["unet_heart_mae_hu"],
                "unet_lumen_mae_hu": um.get("unet_lumen_mae_hu", float("nan")),
                "artifact_removed_frac": removed,
            }
            rows.append(row)
            log.info("case %s | %-12s corrupted %.1f -> unet %.1f HU (removed %.0f%%) | heart %.1f lumen %.1f",
                     cid, label, row["corrupted_mae_hu"], row["unet_mae_hu"],
                     100 * removed, row["unet_heart_mae_hu"], row["unet_lumen_mae_hu"])

    # write CSV + per-regime summary
    csv_path = args.out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print("\n==== OOD MOTION PROBE — per-regime mean (U-Net) ====")
    print(f"{'regime':<14}{'corrupt MAE':>12}{'unet MAE':>10}{'removed%':>10}{'heart MAE':>11}{'lumen MAE':>11}")
    for label in reg:
        rr = [r for r in rows if r["regime"] == label]
        def m(k): return float(np.mean([r[k] for r in rr]))
        print(f"{label:<14}{m('corrupted_mae_hu'):>12.1f}{m('unet_mae_hu'):>10.1f}"
              f"{100*m('artifact_removed_frac'):>9.0f}%{m('unet_heart_mae_hu'):>11.1f}{m('unet_lumen_mae_hu'):>11.1f}")
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
