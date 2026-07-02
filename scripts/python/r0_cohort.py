"""r0_cohort.py — operator-fidelity budget across the real native-motion cohort (Act-I).

Scales the single-case R0 (058) to the worst-N Leo native-motion cases, each PAIRED with a
matched-severity synthetic positive control on the SAME anatomy. Pure forward operator
residuals (no DPS, no training, no TS segmentation). Produces the statistical Act-I claim:
real cardiac motion is model-class non-representable by our parametric forward operator,
controlling for difficulty.

READS ~/Code/Leo_anon (read-only) + leo_motion_screen/ranking.csv. WRITES only under cardiac-artifacts.

Usage: python scripts/python/r0_cohort.py 20
Run card: experiments/runs/2026-06-21_r0-cohort-operator-fidelity.md
"""
from __future__ import annotations

import csv
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
from r0_forward_mismatch import load_leo, calibrated_op, resid, norm, SP, NPHASES, CFG

RANKING = REPO / "experiments" / "runs" / "leo_motion_screen" / "ranking.csv"
PHASES = (0.0, 0.25, 0.5, 0.75)
AMPS = (0.5, 1.0, 1.5)
SEED_NS = 0x7FFFFFFF


def grid_best_residual(xhat, y_hu, heart, scale, off):
    """Lowest ||A_theta(xhat) - y||_heart over the parametric (phase x amp) grid."""
    best = float("inf")
    for phase in PHASES:
        for s in AMPS:
            mp = MotionParams(contraction_amp_mm=10.0 * s, twist_amp_deg=12.0 * s,
                              long_axis_amp_mm=10.0 * s, phase_offset_frac=phase)
            A = DifferentiableMotionForward(heart, (1., 1., 1.), mp, SP, NPHASES, scale, off, view_stride=1)
            best = min(best, resid(A, xhat, y_hu, heart))
    return best


def reduction(xhat, y_hu, heart):
    """(% residual reduction from best parametric operator vs no-motion, no-motion residual)."""
    A0, scale, off = calibrated_op(heart, MotionParams(motion_strength=0.0), y_hu)
    r_nomo = resid(A0, xhat, y_hu, heart)
    r_best = grid_best_residual(xhat, y_hu, heart, scale, off)
    return (r_nomo - r_best) / max(r_nomo, 1e-6) * 100.0, r_nomo, scale, off


def main(ncases: int) -> int:
    dev = torch.device("cuda")
    rows = list(csv.DictReader(open(RANKING)))
    rows.sort(key=lambda r: float(r["sharpness"]))  # lowest sharpness = most motion
    cids = [r["case"] for r in rows[:ncases]]
    cfg = OmegaConf.load(CFG)
    initializer = load_initializer(cfg.initializer, dev)
    print(f"R0 cohort: {len(cids)} worst-motion Leo cases", flush=True)

    out = []
    for cid in cids:
        try:
            y_hu, heart = load_leo(cid, dev)
        except Exception as e:
            print(f"  {cid} skip: {str(e)[:60]}", flush=True); continue
        hb = heart > 0
        if hb.sum() < 1000:
            print(f"  {cid} skip: heart too small", flush=True); continue
        with torch.no_grad():
            xhat = _denorm(initializer(torch.from_numpy(norm(y_hu.cpu().numpy()))[None, None].to(dev))[0, 0])
        m_real = float((xhat[hb] - y_hu[hb]).abs().mean())
        red_real, r_nomo_real, _, _ = reduction(xhat, y_hu, heart)

        # matched-severity positive control: inject synthetic motion onto xhat, scale amps ~ m_real
        rng = np.random.default_rng(abs(hash(("r0c", cid))) & SEED_NS)
        phi = float(rng.uniform(0, 1))
        mp_def = MotionParams(phase_offset_frac=phi)
        y_def, _ = synthesize_motion_artifact(xhat, heart, (1., 1., 1.), mp_def, SP, n_phases=NPHASES, seed=0)
        m_def = float((y_def[hb] - xhat[hb]).abs().mean())
        s = max(0.3, min(2.5, m_real / max(m_def, 1e-6)))  # one-shot severity match
        mp_match = MotionParams(contraction_amp_mm=10.0 * s, twist_amp_deg=12.0 * s,
                                long_axis_amp_mm=10.0 * s, phase_offset_frac=phi)
        y_synth, _ = synthesize_motion_artifact(xhat, heart, (1., 1., 1.), mp_match, SP, n_phases=NPHASES, seed=0)
        m_synth = float((y_synth[hb] - xhat[hb]).abs().mean())
        red_synth, _, _, _ = reduction(xhat, y_synth, heart)

        out.append({"cid": cid, "red_real": red_real, "red_synth": red_synth,
                    "m_real": m_real, "m_synth": m_synth})
        print(f"  {cid}: real artifact {m_real:.0f}HU red_real={red_real:+.1f}% | "
              f"synth(matched {m_synth:.0f}HU) red_synth={red_synth:+.1f}%", flush=True)

    if not out:
        print("no cases"); return 1
    rr = np.array([r["red_real"] for r in out])
    rs = np.array([r["red_synth"] for r in out])

    def boot_ci(a, n=5000):
        gen = np.random.default_rng(0)
        means = [gen.choice(a, size=len(a), replace=True).mean() for _ in range(n)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    rr_lo, rr_hi = boot_ci(rr); rs_lo, rs_hi = boot_ci(rs)
    print(f"\n==== R0 COHORT operator-fidelity budget (n={len(out)}) ====")
    print(f"  REAL native motion : mean residual reduction {rr.mean():+.1f}%  (95% CI [{rr_lo:+.1f}, {rr_hi:+.1f}])")
    print(f"  SYNTH matched-sev  : mean residual reduction {rs.mean():+.1f}%  (95% CI [{rs_lo:+.1f}, {rs_hi:+.1f}])")
    print(f"  mean artifact magnitude: real {np.mean([r['m_real'] for r in out]):.0f} HU vs synth {np.mean([r['m_synth'] for r in out]):.0f} HU (matched)")
    confirmed = rr.mean() < 8.0 and rr_hi < rs_lo
    print(f"\n  VERDICT: {'Act-I CONFIRMED — real motion model-class non-representable (real<<synth, CIs disjoint)' if confirmed else 'CHECK — real not clearly < synth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
