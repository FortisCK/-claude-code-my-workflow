"""uq_calibration_suite.py — full UQ calibration for the confirmed diffusion-unique win.

Computes, from the already-saved kill-gate uncertainty npz (no GPU rerun), the
publication-grade calibration of the posterior std sigma_r:
  - Pearson sigma vs |error| (the kill-gate number, re-confirmed)
  - reliability bins (mean sigma vs mean |error| per sigma-decile)
  - interval coverage at 50% / 90% nominal (is sigma a calibrated predictive std?)
  - sparsification / AUSE (does removing high-sigma voxels remove high-error voxels?
    THE property the variance-gated abstention method needs) vs oracle + random.

Heart-masked, per case + aggregate. Inputs: experiments/runs/uq_calibration_killgate/
uncertainty/test_case_*.npz (residual_std, corrected) + processed clean/heart.

Usage: python scripts/python/uq_calibration_suite.py
Run card: experiments/runs/2026-06-19_uq-calibration-killgate.md (extends it)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "python"))

import numpy as np

sys.modules.pop("code", None)
from code.data import paths as P

HU = 2047.5
UNC = REPO / "experiments" / "runs" / "uq_calibration_killgate" / "uncertainty"
OUT = REPO / "experiments" / "runs" / "uq_calibration_killgate"
Z = {0.50: 0.674489, 0.90: 1.644854}  # gaussian half-interval z for nominal coverage
SPARSE_FRACS = np.linspace(0.0, 0.95, 20)


def load_clean_heart(cid: str):
    proc = P.get("IMAGECAS_PROCESSED")
    f = sorted(proc.glob(f"case_{cid}__pair_*.npz"))[0]
    with np.load(f, allow_pickle=True) as d:
        return d["volume"].astype(np.float32), (d["heart_mask"] > 0)


def sparsification(sigma: np.ndarray, err: np.ndarray):
    """MAE of the retained voxels as we drop the top-frac by sigma / by true error / random."""
    n = err.size
    order_sig = np.argsort(-sigma)      # high sigma first
    order_err = np.argsort(-err)        # oracle: high error first
    err_by_sig = err[order_sig]
    err_by_err = err[order_err]
    cum = np.cumsum(err)                # for random ~ uniform: retained mean = total/remaining
    total = cum[-1]
    curve_sig, curve_orc, curve_rnd = [], [], []
    for t in SPARSE_FRACS:
        k = int(t * n)
        rem = n - k
        if rem <= 0:
            break
        curve_sig.append(err_by_sig[k:].mean())
        curve_orc.append(err_by_err[k:].mean())
        curve_rnd.append((total - 0) / n * 0 + err.mean())  # random keeps the mean
    curve_sig = np.array(curve_sig); curve_orc = np.array(curve_orc); curve_rnd = np.array(curve_rnd)
    fr = SPARSE_FRACS[: len(curve_sig)]
    # AUSE = area between sigma-sparsification and oracle (normalized by random gap); lower=better
    ause = np.trapz(curve_sig - curve_orc, fr)
    a_rand = np.trapz(curve_rnd - curve_orc, fr) + 1e-9
    return fr, curve_sig, curve_orc, curve_rnd, float(ause), float(ause / a_rand)


def main():
    files = sorted(glob.glob(str(UNC / "test_case_*.npz")))
    rows = []
    agg_relsig, agg_relerr = [], []
    for f in files:
        cid = f.split("test_case_")[-1].split(".npz")[0]
        d = np.load(f)
        sigma = d["residual_std"].astype(np.float32)
        corrected = d["corrected"].astype(np.float32)
        clean, heart = load_clean_heart(cid)
        if corrected.shape != clean.shape:
            print(f"case {cid}: shape mismatch, skip"); continue
        m = heart
        sig = (sigma[m] * HU).astype(np.float32)          # HU
        err = (np.abs(corrected[m] - clean[m]) * HU).astype(np.float32)  # HU
        pear = float(np.corrcoef(sig, err)[0, 1])
        # interval coverage
        cov = {p: float((err <= z * sig).mean()) for p, z in Z.items()}
        # reliability deciles (mean sigma vs mean err)
        q = np.quantile(sig, np.linspace(0, 1, 11))
        relsig, relerr = [], []
        for i in range(10):
            sel = (sig >= q[i]) & (sig <= q[i + 1] if i == 9 else sig < q[i + 1])
            if sel.sum() > 0:
                relsig.append(sig[sel].mean()); relerr.append(err[sel].mean())
        agg_relsig.append(relsig); agg_relerr.append(relerr)
        fr, c_s, c_o, c_r, ause, ause_n = sparsification(sig, err)
        rows.append({"cid": cid, "pearson": pear, "cov50": cov[0.50], "cov90": cov[0.90],
                     "ause_norm": ause_n, "mean_sigma_hu": float(sig.mean()), "mean_err_hu": float(err.mean())})
        print(f"case {cid}: r={pear:.3f} cov50={cov[0.50]:.2f} cov90={cov[0.90]:.2f} "
              f"AUSE/rand={ause_n:.3f} (0=oracle,1=random) sigma={sig.mean():.1f}HU err={err.mean():.1f}HU", flush=True)

    def m(k): return float(np.mean([r[k] for r in rows]))
    print(f"\n==== UQ CALIBRATION SUITE (n={len(rows)}, heart-masked, converged epoch_080) ====")
    print(f"  Pearson sigma vs |error|: {m('pearson'):.3f}")
    print(f"  interval coverage  50% nominal -> {m('cov50'):.3f}   90% nominal -> {m('cov90'):.3f}")
    print(f"  AUSE/random: {m('ause_norm'):.3f}  (0 = sigma ranks errors as well as the oracle; 1 = no better than random)")
    print(f"  mean sigma {m('mean_sigma_hu'):.1f} HU vs mean |err| {m('mean_err_hu'):.1f} HU")
    # verdict heuristics
    good_rank = m('ause_norm') < 0.6
    informative = m('pearson') >= 0.35
    print("\n  READING:")
    print(f"   - sigma is {'INFORMATIVE' if informative else 'WEAK'} (rank-correlates with error).")
    print(f"   - high-sigma abstention {'REMOVES high-error voxels well' if good_rank else 'is only weakly better than random'} (AUSE/rand={m('ause_norm'):.2f}).")
    under = m('cov90') < 0.85
    print(f"   - intervals are {'UNDER-confident/over-confident -> need temperature scaling' if (m('cov90')<0.85 or m('cov90')>0.95) else 'roughly calibrated'} "
          f"(90% nominal -> {m('cov90'):.2f} empirical).")

    # reliability + sparsification figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rs = np.array([r for r in agg_relsig if len(r) == 10]).mean(0)
        re = np.array([r for r in agg_relerr if len(r) == 10]).mean(0)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(rs, re, "o-"); lim = max(rs.max(), re.max())
        ax[0].plot([0, lim], [0, lim], "k--", alpha=0.5, label="ideal y=x")
        ax[0].set_xlabel("mean σ per decile (HU)"); ax[0].set_ylabel("mean |error| (HU)")
        ax[0].set_title(f"Reliability (Pearson {m('pearson'):.2f})"); ax[0].legend()
        fr, c_s, c_o, c_r, _, _ = sparsification(
            *(lambda d: ((d["residual_std"][load_clean_heart('21')[1]]*HU),
                         (np.abs(d["corrected"][load_clean_heart('21')[1]]-load_clean_heart('21')[0][load_clean_heart('21')[1]])*HU)))(np.load(files[0])))
        ax[1].plot(fr, c_s, "o-", label="by σ (ours)"); ax[1].plot(fr, c_o, "s-", label="oracle (by error)")
        ax[1].plot(fr, c_r, "--", label="random"); ax[1].set_xlabel("fraction abstained (highest σ)")
        ax[1].set_ylabel("MAE of retained (HU)"); ax[1].set_title("Sparsification (case 21)"); ax[1].legend()
        fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=130)
        print(f"\n  figure -> {OUT/'calibration.png'}")
    except Exception as e:
        print(f"  (figure skipped: {str(e)[:80]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
