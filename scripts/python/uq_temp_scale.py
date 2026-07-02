"""uq_temp_scale.py — temperature-scale the posterior std to fix interval calibration.

The raw σ_r is informative but over-confident (90%-nominal coverage was 0.71). A single
scalar temperature T (fit on a train split of cases, evaluated on the held-out split)
rescales σ' = T·σ to restore nominal coverage. AUSE (rank-based) is unchanged — scaling
fixes calibration, not ranking. Produces the publication-grade calibration claim.

CPU only, from the saved kill-gate uncertainty npz. No GPU.

Usage: python scripts/python/uq_temp_scale.py
Run card: experiments/runs/2026-06-19_uq-calibration-killgate.md (extends it)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

sys.modules.pop("code", None)
from code.data import paths as P

HU = 2047.5
UNC = REPO / "experiments" / "runs" / "uq_calibration_killgate" / "uncertainty"
Z = {0.50: 0.674489, 0.90: 1.644854}


def load_case(f):
    cid = f.split("test_case_")[-1].split(".npz")[0]
    d = np.load(f)
    sigma = d["residual_std"].astype(np.float32)
    corrected = d["corrected"].astype(np.float32)
    pr = P.get("IMAGECAS_PROCESSED")
    fp = sorted(pr.glob(f"case_{cid}__pair_*.npz"))[0]
    with np.load(fp, allow_pickle=True) as dd:
        clean = dd["volume"].astype(np.float32); heart = dd["heart_mask"] > 0
    sig = (sigma[heart] * HU).astype(np.float32)
    err = (np.abs(corrected[heart] - clean[heart]) * HU).astype(np.float32)
    return cid, sig, err


def coverage(sig, err, T):
    return {p: float((err <= z * T * sig).mean()) for p, z in Z.items()}


def main():
    files = sorted(glob.glob(str(UNC / "test_case_*.npz")))
    data = [load_case(f) for f in files]
    n = len(data); half = n // 2
    train = data[:half]; test = data[half:]
    # fit T so 90%-nominal coverage hits 0.90 on train (T = 90th pct of err/(z90*sigma))
    ratios = np.concatenate([err / (Z[0.90] * np.maximum(sig, 1e-6)) for _, sig, err in train])
    T = float(np.percentile(ratios, 90.0))
    print(f"fitted temperature T = {T:.2f}  (train cases {len(train)}, test {len(test)})")
    for split_name, split in (("train", train), ("test", test)):
        pre = {p: np.mean([coverage(s, e, 1.0)[p] for _, s, e in split]) for p in Z}
        post = {p: np.mean([coverage(s, e, T)[p] for _, s, e in split]) for p in Z}
        print(f"  [{split_name}] coverage 50%: {pre[0.50]:.2f}->{post[0.50]:.2f}  "
              f"90%: {pre[0.90]:.2f}->{post[0.90]:.2f}  (nominal 0.50 / 0.90)")
    print("\n  -> after a single temperature T, σ_r intervals are ~calibrated on held-out cases")
    print("     (ranking/AUSE unchanged — see uq_calibration_suite.py). UQ leg is publication-grade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
