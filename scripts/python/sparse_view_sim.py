"""sparse_view_sim.py — Stage-0 of the sparse-view CCTA pivot.

Forward-project clean ImageCAS CCTA through a REAL cone-beam operator (full 2pi), subsample
to N views, reconstruct with FBP + SIRT, and measure how badly the CORONARY LUMEN degrades
vs the clean volume (= ground truth). Sweeps N to fix the ultra-sparse operating point and
confirm classical recon fails there (so a diffusion prior has headroom).

Cone geometry is sized to FULLY cover the 192^3 / 1mm volume (mag 1.5, 400mm detector) to
avoid axial-truncation confounds. Reuses tomosipo (astra backend); ramp filter from motion_synth.

READS ImageCAS processed + lumen masks. WRITES only under cardiac-artifacts.
Usage: python scripts/python/sparse_view_sim.py 3
Run card: experiments/runs/2026-07-03_1500_sparse-view-sim-fbp-sirt.md
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import tomosipo as ts
from tomosipo.torch_support import to_autograd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.modules.pop("code", None)
from code.data import paths as P
from code.data.splits import load_split
from code.data.motion_synth import ramp_filter_sinogram

HU_LO, HU_HI, HU_SCALE = -1024.0, 3071.0, 2047.5
# cone geometry sized to cover a 192mm volume without truncation (mag 1.5)
SID, SDD = 1000.0, 1500.0
DET_ROWS, DET_COLS, DET_PITCH = 400, 400, 1.0
VIEW_SWEEP = [12, 16, 24, 32, 48, 64, 96]
SIRT_ITERS = 40
OUT = REPO / "experiments" / "runs" / "sparse_view_sim"


def denorm(a):
    return (a + 1.0) * 0.5 * (HU_HI - HU_LO) + HU_LO


def build_ops(shape, n_views, dev):
    z, y, x = shape
    vg = ts.volume(shape=(z, y, x), size=(float(z), float(y), float(x)))  # 1mm iso
    det = DET_ROWS * DET_PITCH
    angles = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    pg = ts.cone(angles=angles, shape=(DET_ROWS, DET_COLS), size=(det, det),
                 src_orig_dist=SID, src_det_dist=SDD)
    A = ts.operator(vg, pg)
    fwd = to_autograd(A, num_extra_dims=0)
    bwd = to_autograd(A.T, num_extra_dims=0)
    return fwd, bwd


@torch.no_grad()
def sirt(fwd, bwd, y, vol_shape, dev, iters=SIRT_ITERS):
    eps = 1e-6
    R = 1.0 / fwd(torch.ones(vol_shape, device=dev)).clamp_min(eps)
    C = 1.0 / bwd(torch.ones_like(y)).clamp_min(eps)
    x = torch.zeros(vol_shape, device=dev)
    for _ in range(iters):
        x = x + C * bwd(R * (y - fwd(x)))
    return x


@torch.no_grad()
def fbp(bwd, y, clean_hu, heart_b):
    rec = bwd(ramp_filter_sinogram(y))
    # HU-affine calibrate to clean inside heart (ramp+backproj is unscaled)
    ri, ci = rec[heart_b], clean_hu[heart_b]
    scale = (ci.std() / (ri.std() + 1e-6)); offset = ci.mean() - scale * ri.mean()
    return rec * scale + offset


def gradmag(a):
    gz, gy, gx = np.gradient(a)
    return np.sqrt(gz * gz + gy * gy + gx * gx)


def main(ncases: int) -> int:
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    proc = P.get("IMAGECAS_PROCESSED"); lumdir = proc.parent / "lumen_masks"
    cases = [str(c) for c in load_split(P.get("IMAGECAS_SPLITS") / "v1.json").test[:ncases]]
    print(f"cone geom: SID{SID}/SDD{SDD} det{DET_ROWS}x{DET_COLS}@{DET_PITCH}mm (mag {SDD/SID:.2f}); SIRT {SIRT_ITERS} it", flush=True)
    agg = {n: {"fbp_heart": [], "fbp_lum": [], "sirt_heart": [], "sirt_lum": [], "lum_sharp": []} for n in VIEW_SWEEP}
    clean_lum_sharp = []
    for cid in cases:
        d = np.load(proc / f"case_{cid}__pair_0.npz", allow_pickle=True)
        clean_hu = denorm(torch.from_numpy(d["volume"].astype(np.float32))).to(dev)
        heart_b = torch.from_numpy(d["heart_mask"].astype(np.float32)).to(dev) > 0
        lg = lumdir / f"lumen_mask_{cid}.npy"
        lum_b = torch.from_numpy(np.load(lg).astype(np.float32)).to(dev) > 0 if lg.exists() else heart_b
        vol_shape = tuple(clean_hu.shape)
        clean_np = clean_hu.cpu().numpy(); lum_np = lum_b.cpu().numpy()
        clean_lum_sharp.append(float(gradmag(clean_np)[lum_np].mean()))
        panels = {}
        for n in VIEW_SWEEP:
            fwd, bwd = build_ops(vol_shape, n, dev)
            y = fwd(clean_hu)
            rec_fbp = fbp(bwd, y, clean_hu, heart_b)
            rec_sirt = sirt(fwd, bwd, y, vol_shape, dev)
            for rec, tagh, tagl in ((rec_fbp, "fbp_heart", "fbp_lum"), (rec_sirt, "sirt_heart", "sirt_lum")):
                agg[n][tagh].append(float((rec - clean_hu)[heart_b].abs().mean()))
                agg[n][tagl].append(float((rec - clean_hu)[lum_b].abs().mean()))
            agg[n]["lum_sharp"].append(float(gradmag(rec_sirt.cpu().numpy())[lum_np].mean()))
            panels[n] = (rec_fbp.cpu().numpy(), rec_sirt.cpu().numpy())
        # render this case: coronary slice, clean vs FBP/SIRT at a few N
        zc = int(np.argwhere(lum_np).mean(0)[0]) if lum_np.any() else clean_np.shape[0] // 2
        showN = [12, 24, 64]
        fig, ax = plt.subplots(2, 1 + len(showN), figsize=(4 * (1 + len(showN)), 8))
        for r, (lbl, pick) in enumerate((("FBP", 0), ("SIRT", 1))):
            ax[r, 0].imshow(clean_np[zc], cmap="gray", vmin=-100, vmax=700); ax[r, 0].set_title("clean (GT)" if r == 0 else ""); ax[r, 0].axis("off")
            for j, n in enumerate(showN):
                ax[r, j + 1].imshow(panels[n][pick][zc], cmap="gray", vmin=-100, vmax=700)
                ax[r, j + 1].set_title(f"{lbl} {n} views"); ax[r, j + 1].axis("off")
        fig.suptitle(f"case {cid} — sparse-view classical recon (coronary slice)", fontsize=13)
        fig.tight_layout(); fig.savefig(OUT / f"case_{cid}__sweep.png", dpi=110); plt.close(fig)
        print(f"case {cid} rendered", flush=True)

    def m(n, k): return float(np.mean(agg[n][k]))
    print(f"\n==== SPARSE-VIEW CLASSICAL RECON SWEEP (n={len(cases)} cases) ====")
    print(f"clean in-lumen sharpness (ref) = {np.mean(clean_lum_sharp):.1f} HU/vox")
    print(f"{'views':>6}{'FBP heartMAE':>14}{'FBP lumMAE':>12}{'SIRT heartMAE':>15}{'SIRT lumMAE':>13}{'SIRT lumSharp':>14}")
    for n in VIEW_SWEEP:
        print(f"{n:>6}{m(n,'fbp_heart'):>14.1f}{m(n,'fbp_lum'):>12.1f}{m(n,'sirt_heart'):>15.1f}{m(n,'sirt_lum'):>13.1f}{m(n,'lum_sharp'):>14.1f}")
    print("\n  READING: pick N where SIRT lumen-MAE is high (lumen destroyed) but heart is roughly located")
    print("  -> that is the ultra-sparse operating point where a diffusion prior has headroom.")
    print(f"  panels -> {OUT}/case_*__sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
