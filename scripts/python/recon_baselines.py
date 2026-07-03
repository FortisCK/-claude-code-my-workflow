"""recon_baselines.py — classical sparse-view recon baselines (the bar DPS must beat).

FBP, SIRT (from sparse_view_sim) + ASD-POCS (TV-minimization, Sidky & Pan 2008) — the standard
strong iterative baseline for sparse-view CT. Run on the sparse-view sim to get baseline
reconstruction quality at the operating view counts, ready for the Stage-1 DPS-vs-ASD-POCS gate.

Usage: python scripts/python/recon_baselines.py 3
Run card: experiments/runs/2026-07-03_1500_sparse-view-sim-fbp-sirt.md (baselines leg)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "python"))

import numpy as np
import torch

sys.modules.pop("code", None)
from code.data import paths as P
from code.data.splits import load_split
from sparse_view_sim import build_ops, fbp, sirt, denorm, gradmag, HU_LO

VIEWS = [16, 24, 32]


def tv_grad(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV subgradient: d/dx sum|forward-diff|, per axis."""
    g = torch.zeros_like(x)
    for dim in range(3):
        s = torch.sign(torch.diff(x, dim=dim))
        g.narrow(dim, 0, x.shape[dim] - 1).sub_(s)
        g.narrow(dim, 1, x.shape[dim] - 1).add_(s)
    return g


@torch.no_grad()
def asd_pocs(fwd, bwd, y, shape, dev, n_iter=60, n_tv=15, beta=1.0, beta_red=0.995, tv_frac=0.2):
    eps = 1e-8
    R = 1.0 / fwd(torch.ones(shape, device=dev)).clamp_min(eps)
    C = 1.0 / bwd(torch.ones_like(y)).clamp_min(eps)
    x = torch.zeros(shape, device=dev)
    for _ in range(n_iter):
        x_prev = x
        x = (x + beta * C * bwd(R * (y - fwd(x)))).clamp_min(HU_LO)   # data-consistency (SIRT step) + positivity
        dd = float((x - x_prev).norm())
        for _ in range(n_tv):                                         # adaptive TV descent
            g = tv_grad(x)
            x = x - (tv_frac * dd) * g / (g.norm() + eps)
        beta *= beta_red
    return x


def main(ncases: int) -> int:
    dev = torch.device("cuda")
    proc = P.get("IMAGECAS_PROCESSED"); lumdir = proc.parent / "lumen_masks"
    cases = [str(c) for c in load_split(P.get("IMAGECAS_SPLITS") / "v1.json").test[:ncases]]
    agg = {n: {k: [] for k in ("sirt_lum", "asd_lum", "sirt_heart", "asd_heart", "asd_sharp")} for n in VIEWS}
    ref_sharp = []
    for cid in cases:
        d = np.load(proc / f"case_{cid}__pair_0.npz", allow_pickle=True)
        clean = denorm(torch.from_numpy(d["volume"].astype(np.float32))).to(dev)
        heart_b = torch.from_numpy(d["heart_mask"].astype(np.float32)).to(dev) > 0
        lg = lumdir / f"lumen_mask_{cid}.npy"
        lum_b = torch.from_numpy(np.load(lg).astype(np.float32)).to(dev) > 0 if lg.exists() else heart_b
        lum_np = lum_b.cpu().numpy()
        ref_sharp.append(float(gradmag(clean.cpu().numpy())[lum_np].mean()))
        shape = tuple(clean.shape)
        for n in VIEWS:
            fwd, bwd = build_ops(shape, n, dev)
            y = fwd(clean)
            rs = sirt(fwd, bwd, y, shape, dev, iters=60)
            ra = asd_pocs(fwd, bwd, y, shape, dev)
            agg[n]["sirt_lum"].append(float((rs - clean)[lum_b].abs().mean()))
            agg[n]["asd_lum"].append(float((ra - clean)[lum_b].abs().mean()))
            agg[n]["sirt_heart"].append(float((rs - clean)[heart_b].abs().mean()))
            agg[n]["asd_heart"].append(float((ra - clean)[heart_b].abs().mean()))
            agg[n]["asd_sharp"].append(float(gradmag(ra.cpu().numpy())[lum_np].mean()))
        print(f"case {cid} done", flush=True)

    def m(n, k): return float(np.mean(agg[n][k]))
    print(f"\n==== CLASSICAL BASELINES (n={len(cases)}), clean lumen-sharp ref {np.mean(ref_sharp):.0f} ====")
    print(f"{'views':>6}{'SIRT lumMAE':>12}{'ASD lumMAE':>12}{'SIRT heartMAE':>14}{'ASD heartMAE':>13}{'ASD lumSharp':>13}")
    for n in VIEWS:
        print(f"{n:>6}{m(n,'sirt_lum'):>12.1f}{m(n,'asd_lum'):>12.1f}{m(n,'sirt_heart'):>14.1f}{m(n,'asd_heart'):>13.1f}{m(n,'asd_sharp'):>13.1f}")
    print("\n  ASD-POCS is the STRONG classical baseline; Stage-1 gate = DPS must beat ASD on coronary Dice-RCA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
