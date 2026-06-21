"""probe_phaseoffset_prize.py — Stage-0 gate: "is operator estimation load-bearing?"

Generates synthetic motion with a known random cardiac PHASE OFFSET, then runs DPS
with operators whose phase offset is wrong by Delta in {0, 0.25, 0.5} cycle and
measures coronary Dice-RCA + heart-MAE vs the phase error. If wrong phase does NOT
degrade DPS, then NO operator estimator is load-bearing -> deploy a constant operator
and stop. If it degrades sharply, estimating phase is worth building.

Amplitudes/period are FIXED (defaults) so phase offset is the ONLY varied operator DoF
(it is the only DoF that re-SHAPES, not just scales, the artifact).

READS only ImageCAS processed pairs. WRITES only under cardiac-artifacts.

Usage:
    python scripts/python/probe_phaseoffset_prize.py 8        # N cases

Run card: experiments/runs/2026-06-19_phaseoffset-prize-probe.md
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "python"))

import numpy as np
import SimpleITK as sitk
import torch
from omegaconf import OmegaConf

sys.modules.pop("code", None)
from code.data import paths as P
from code.data.splits import load_split
from code.data.motion_synth import MotionParams, ScanParams, synthesize_motion_artifact
from code.inference.dps_forward import DifferentiableMotionForward
from code.inference.dps_sample import dps_residual_sample, _denorm
from code.models.residual_refiner import ResidualRefinerUNet3D
from code.training.train_residual_refiner import load_initializer
from evaluate_residual_diffusion_full_volume import load_residual_diffusion
from coronary_dice_rca import dice

TS = Path(sys.prefix) / "bin" / "TotalSegmentator"
OUT = REPO / "experiments" / "runs" / "phaseoffset_prize"
VOL, SEG = OUT / "volumes", OUT / "seg"
HU_SCALE = 2047.5
DELTAS = [0.0, 0.25, 0.5]  # operator phase error (cycles) relative to the true phase
ZETA, STEPS, NPHASES = 0.02, 24, 48
CFG = "code/training/configs/diffusion_v2_residual.yaml"
CKPT = "experiments/checkpoints/diffusion_v2_residual/longrun/epoch_080.pt"


def seg_cor(nii: Path, od: Path) -> np.ndarray:
    f = od / "coronary_arteries.nii.gz"
    if not f.exists():
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS), "-i", str(nii), "-o", str(od), "-ta", "coronary_arteries"],
                       check=True, capture_output=True, text=True)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0


def savehu(arr_hu: np.ndarray, p: Path) -> None:
    im = sitk.GetImageFromArray(arr_hu.astype(np.int16))
    im.SetSpacing((1.0, 1.0, 1.0))
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(im, str(p))


def main(ncases: int) -> int:
    dev = torch.device("cuda")
    cfg = OmegaConf.load(CFG)
    initializer = load_initializer(cfg.initializer, dev)
    engine, _ = load_residual_diffusion(CFG, CKPT, "ema", dev)
    proc = P.get("IMAGECAS_PROCESSED")
    cases = [str(c) for c in load_split(P.get("IMAGECAS_SPLITS") / "v1.json").test[:ncases]]
    rng = np.random.default_rng(0)
    rows = []
    for cid in cases:
        d = np.load(proc / f"case_{cid}__pair_0.npz", allow_pickle=True)
        clean = torch.from_numpy(d["volume"].astype(np.float32))[None, None].to(dev)
        heart = torch.from_numpy(d["heart_mask"].astype(np.float32)).to(dev)
        hb = heart > 0
        clean_hu = _denorm(clean[0, 0])
        phi_true = float(rng.uniform(0, 1))
        sp = ScanParams(n_views=1000)
        mp_true = MotionParams(phase_offset_frac=phi_true)  # fixed amps/period, strength 1
        corrupted_hu, dbg = synthesize_motion_artifact(clean_hu, heart, (1.0, 1.0, 1.0),
                                                       mp_true, sp, n_phases=NPHASES, seed=0)
        cal = dbg["hu_calibration"]
        corr_norm = torch.from_numpy(
            ((np.clip(corrupted_hu.cpu().numpy(), -1024.0, 3071.0) + 1024.0) / 4095.0 * 2.0 - 1.0)
        )[None, None].to(dev)
        mae_corr = float((corr_norm - clean).abs()[0, 0][hb].mean() * HU_SCALE)
        with torch.no_grad():
            initial = initializer(corr_norm)
        cond = ResidualRefinerUNet3D.make_condition(corr_norm.float(), initial.float())
        mae_u = float((initial - clean).abs()[0, 0][hb].mean() * HU_SCALE)

        savehu(_denorm(clean[0, 0]).cpu().numpy(), VOL / f"case_{cid}__clean.nii.gz")
        savehu(_denorm(initial[0, 0]).detach().cpu().numpy(), VOL / f"case_{cid}__unet.nii.gz")
        sc = seg_cor(VOL / f"case_{cid}__clean.nii.gz", SEG / f"case_{cid}__clean")
        su = seg_cor(VOL / f"case_{cid}__unet.nii.gz", SEG / f"case_{cid}__unet")
        row = {"case": cid, "phi_true": phi_true, "mae_corr": mae_corr, "mae_unet": mae_u,
               "unet_dice": dice(su, sc)}
        for delta in DELTAS:
            phi_op = (phi_true + delta) % 1.0
            mp_op = MotionParams(phase_offset_frac=phi_op)
            A = DifferentiableMotionForward(heart, (1.0, 1.0, 1.0), mp_op, sp, NPHASES,
                                            cal["scale"], cal["offset"], view_stride=1)
            x = dps_residual_sample(engine, cond, initial, _denorm(corr_norm[0, 0]), A,
                                    num_steps=STEPS, zeta=ZETA, seed=0)
            tag = f"d{int(delta*100):03d}"
            savehu(_denorm(x[0, 0]).detach().cpu().numpy(), VOL / f"case_{cid}__{tag}.nii.gz")
            sd = seg_cor(VOL / f"case_{cid}__{tag}.nii.gz", SEG / f"case_{cid}__{tag}")
            row[f"dice_{tag}"] = dice(sd, sc)
            row[f"mae_{tag}"] = float((x - clean).abs()[0, 0][hb].mean() * HU_SCALE)
        rows.append(row)
        msg = " ".join(f"d{int(dl*100)}={row[f'dice_d{int(dl*100):03d}']:.3f}" for dl in DELTAS)
        print(f"case {cid} | phi={phi_true:.2f} | corrMAE={mae_corr:.0f} U-Net dice={row['unet_dice']:.3f} | DPS {msg}", flush=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    print(f"\n==== STAGE-0 PHASE-OFFSET PRIZE PROBE (n={len(rows)}) ====")
    print(f"  U-Net Dice-RCA floor: {m('unet_dice'):.3f}  (corrupted heart-MAE {m('mae_corr'):.0f} HU)")
    for dl in DELTAS:
        t = f"d{int(dl*100):03d}"
        print(f"  DPS, phase error {dl:.2f} cycle: Dice-RCA {m(f'dice_{t}'):.3f}   heart-MAE {m(f'mae_{t}'):.0f} HU")
    prize = m(f"dice_d000") - m(f"dice_d050")
    print(f"\n  PRIZE (Dice drop, correct -> 0.5-cycle-wrong phase): {prize:+.3f}")
    if prize >= 0.04:
        print("  VERDICT: PRIZE EXISTS — wrong phase materially hurts DPS -> estimating phase is load-bearing -> proceed to Stage 1.")
    elif prize >= 0.02:
        print("  VERDICT: WEAK prize — phase matters a little; marginal case, inspect per-case spread.")
    else:
        print("  VERDICT: NO PRIZE — DPS is robust to phase error -> no operator estimator needed -> deploy a constant operator and STOP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
