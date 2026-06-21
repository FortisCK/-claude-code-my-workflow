"""r0b_positive_control.py — sensitivity check for the R0b motion-representability test.

Generates a SYNTHETIC case with KNOWN motion, then runs the exact R0b grid. If the
grid finds a motion operator that materially reduces the data-consistency residual
(vs no-motion), the test is SENSITIVE -> the real-058 +0.0% reduction is a true
negative (real motion not representable), not a dead test. Runs with two clean
estimates: x_hat=clean (idealized) and x_hat=U-Net(y) (realistic, matches R0b) to
also rule out "U-Net estimate is just bad" as the cause of the real negative.

Usage: python scripts/python/r0b_positive_control.py
Run card: experiments/runs/2026-06-19_r0-forward-model-mismatch.md
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "python"))

import numpy as np
import torch
from omegaconf import OmegaConf

sys.modules.pop("code", None)
from code.data.motion_synth import MotionParams, synthesize_motion_artifact
from code.inference.dps_forward import DifferentiableMotionForward
from code.inference.dps_sample import _denorm
from code.training.train_residual_refiner import load_initializer
from r0_forward_mismatch import calibrated_op, resid, load_synth, norm, SP, NPHASES, CFG

PHI_TRUE = 0.30  # known motion phase


def run_grid(label, xhat, y_hu, heart):
    A0, sc, off = calibrated_op(heart, MotionParams(motion_strength=0.0), y_hu)
    r_nomo = resid(A0, xhat, y_hu, heart)
    best = (r_nomo, "nomotion")
    for phase in (0.0, 0.25, 0.5, 0.75):
        for s in (0.5, 1.0, 1.5):
            mp = MotionParams(contraction_amp_mm=10.0 * s, twist_amp_deg=12.0 * s,
                              long_axis_amp_mm=10.0 * s, phase_offset_frac=phase)
            A = DifferentiableMotionForward(heart, (1., 1., 1.), mp, SP, NPHASES, sc, off, view_stride=1)
            r = resid(A, xhat, y_hu, heart)
            if r < best[0]:
                best = (r, f"phase={phase} amp x{s}")
    red = (r_nomo - best[0]) / max(r_nomo, 1e-6) * 100
    print(f"  [{label}] no-motion resid {r_nomo:.1f} HU -> best {best[0]:.1f} HU ({best[1]}) | reduction {red:+.1f}%", flush=True)
    return red


def main():
    dev = torch.device("cuda")
    clean_hu, heart = load_synth("21", dev)
    mp_true = MotionParams(phase_offset_frac=PHI_TRUE)
    y_hu, _ = synthesize_motion_artifact(clean_hu, heart, (1., 1., 1.), mp_true, SP, n_phases=NPHASES, seed=0)
    hb = heart > 0
    print(f"==== R0b POSITIVE CONTROL (synthetic case 21, known motion phase={PHI_TRUE}) ====", flush=True)
    print(f"  artifact magnitude ||corrupted - clean||_heart = {float((y_hu[hb]-clean_hu[hb]).abs().mean()):.1f} HU", flush=True)

    # idealized: x_hat = clean
    run_grid("xhat=clean", clean_hu, y_hu, heart)

    # realistic: x_hat = U-Net(y) (matches R0b on real)
    cfg = OmegaConf.load(CFG)
    initializer = load_initializer(cfg.initializer, dev)
    with torch.no_grad():
        xhat = _denorm(initializer(torch.from_numpy(norm(y_hu.cpu().numpy()))[None, None].to(dev))[0, 0])
    red_unet = run_grid("xhat=U-Net", xhat, y_hu, heart)

    print("\n  INTERPRETATION:")
    if red_unet >= 15:
        print("  TEST IS SENSITIVE (motion detected on synthetic via U-Net estimate)")
        print("  -> real-058 +0.0% is a TRUE NEGATIVE: real motion is not representable by our model.")
    elif red_unet >= 5:
        print("  TEST WEAKLY sensitive with U-Net estimate -> real negative is suggestive but partly confounded by xhat quality.")
    else:
        print("  TEST INSENSITIVE even on synthetic with U-Net estimate -> the real negative may be a test/xhat artifact,")
        print("     NOT proof that real motion is unrepresentable. Re-examine before pivoting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
