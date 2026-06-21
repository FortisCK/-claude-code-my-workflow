"""r0_forward_mismatch.py — R0 gate: does our forward model represent REAL data at all?

Directly tests the inverse-crime worry, cheaply (forward-only, no DPS, no training):

R0a (static fidelity): round-trip residual ||A_nomotion(x) - x||_heart on real-clean
    Leo cases vs synthetic-clean ImageCAS cases. If our cone-beam+FDK handles real image
    statistics as well as synthetic, the two residuals are comparable.

R0b (motion representability on real native motion): for real Leo058, can ANY parametric
    motion operator (grid over phase x amplitude-scale) explain the observed artifact
    better than no-motion? x_hat = U-Net(y). Compare min ||A_theta(x_hat) - y||_heart to
    the no-motion residual. Large reduction -> real motion is parametric-capturable
    (promising). Tiny reduction -> real motion is outside our model family -> DPS-on-real
    is dead regardless of operator estimation.

READS ~/Code/Leo_anon (read-only) + ImageCAS processed. WRITES only under cardiac-artifacts.

Usage: python scripts/python/r0_forward_mismatch.py
Run card: experiments/runs/2026-06-19_r0-forward-model-mismatch.md
"""
from __future__ import annotations

import glob
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
from code.data.motion_synth import MotionParams, ScanParams
from code.inference.dps_forward import DifferentiableMotionForward
from code.inference.dps_sample import _denorm
from code.training.train_residual_refiner import load_initializer
from evaluate_residual_diffusion_full_volume import load_residual_diffusion  # noqa: F401 (engine not needed)

LEO = "/home/mingzhang/Code/Leo_anon"
OUT = REPO / "experiments" / "runs" / "r0_mismatch"
TS = Path(sys.prefix) / "bin" / "TotalSegmentator"
HU_LO, HU_HI = -1024.0, 3071.0
CROP, NPHASES = 192, 48
SP = ScanParams(n_views=1000)
CFG = "code/training/configs/diffusion_v2_residual.yaml"


def norm(hu):
    return (np.clip(hu, HU_LO, HU_HI) - HU_LO) / (HU_HI - HU_LO) * 2.0 - 1.0


def resample_iso(img, is_label):
    osp, osz = img.GetSpacing(), img.GetSize()
    nsz = [int(round(s * sp / 1.0)) for s, sp in zip(osz, osp)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing((1.0, 1.0, 1.0)); r.SetSize(nsz)
    r.SetOutputOrigin(img.GetOrigin()); r.SetOutputDirection(img.GetDirection())
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    r.SetDefaultPixelValue(0 if is_label else HU_LO)
    return r.Execute(img)


def crop_pad(arr, c, size, pad):
    out = np.full((size, size, size), pad, dtype=np.float32)
    st = [int(round(c[d])) - size // 2 for d in range(3)]
    s0 = [max(0, st[d]) for d in range(3)]; s1 = [min(arr.shape[d], st[d] + size) for d in range(3)]
    d0 = [s0[d] - st[d] for d in range(3)]; d1 = [d0[d] + (s1[d] - s0[d]) for d in range(3)]
    out[d0[0]:d1[0], d0[1]:d1[1], d0[2]:d1[2]] = arr[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]
    return out


def leo_heart_iso(cid, nii, ref_iso):
    od = OUT / "seg" / f"{cid}__heart"; hf = od / "heart.nii.gz"
    if not hf.exists():
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS), "-i", nii, "-o", str(od), "-ta", "total", "-rs", "heart", "--fast"],
                       check=True, capture_output=True, text=True)
    h = sitk.ReadImage(str(hf))
    return sitk.Resample(h, ref_iso, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def load_leo(cid, dev):
    nii = glob.glob(f"{LEO}/{cid}/*.nii")[0]
    vi = resample_iso(sitk.ReadImage(nii), False)
    hi = leo_heart_iso(cid, nii, vi)
    vol = sitk.GetArrayFromImage(vi).astype(np.float32)
    heart = (sitk.GetArrayFromImage(hi) > 0).astype(np.float32)
    c = np.argwhere(heart > 0).mean(0)
    vol_hu = torch.from_numpy(crop_pad(vol, c, CROP, HU_LO)).to(dev)
    heart_c = torch.from_numpy(crop_pad(heart, c, CROP, 0.0)).to(dev)
    return vol_hu, heart_c


def load_synth(cid, dev):
    d = np.load(P.get("IMAGECAS_PROCESSED") / f"case_{cid}__pair_0.npz", allow_pickle=True)
    clean_hu = _denorm(torch.from_numpy(d["volume"].astype(np.float32))).to(dev)
    heart = torch.from_numpy(d["heart_mask"].astype(np.float32)).to(dev)
    return clean_hu, heart


def calibrated_op(heart, mp, target_hu):
    """Build A with HU-affine fit so a no-motion baseline matches target heart stats."""
    A0 = DifferentiableMotionForward(heart, (1., 1., 1.), MotionParams(motion_strength=0.0),
                                     SP, NPHASES, 1.0, 0.0, view_stride=1)
    with torch.no_grad():
        base = A0(target_hu)
    hb = heart > 0
    bi, ti = base[hb], target_hu[hb]
    scale = float(ti.std() / (bi.std() + 1e-6)); offset = float(ti.mean() - scale * bi.mean())
    return DifferentiableMotionForward(heart, (1., 1., 1.), mp, SP, NPHASES, scale, offset, view_stride=1), scale, offset


def resid(A, x_hu, target_hu, heart):
    with torch.no_grad():
        a = A(x_hu)
    hb = heart > 0
    return float((a[hb] - target_hu[hb]).abs().mean())


def main():
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- R0a: static round-trip fidelity, real vs synthetic ----------
    print("==== R0a: static round-trip residual ||A_nomotion(x) - x||_heart (HU) ====", flush=True)
    nomo = MotionParams(motion_strength=0.0)
    for tag, cid, loader in [("synth", "21", load_synth), ("synth", "32", load_synth),
                             ("real ", "072", load_leo), ("real ", "067", load_leo)]:
        x_hu, heart = loader(cid, dev)
        A, sc, off = calibrated_op(heart, nomo, x_hu)
        r = resid(A, x_hu, x_hu, heart)
        print(f"  {tag} case {cid}: round-trip resid = {r:6.1f} HU  (calib scale={sc:.3f})", flush=True)

    # ---------- R0b: can parametric motion explain real 058's native artifact? ----------
    print("\n==== R0b: motion representability on REAL native-motion Leo058 ====", flush=True)
    y_hu, heart = load_leo("058", dev)
    hb = heart > 0
    cfg = OmegaConf.load(CFG)
    initializer = load_initializer(cfg.initializer, dev)
    with torch.no_grad():
        xhat = _denorm(initializer(torch.from_numpy(norm(y_hu.cpu().numpy()))[None, None].to(dev))[0, 0])
    # no-motion baseline residual (operator calibrated to y)
    A0, sc, off = calibrated_op(heart, MotionParams(motion_strength=0.0), y_hu)
    r_nomo = resid(A0, xhat, y_hu, heart)
    art_mag = float((xhat[hb] - y_hu[hb]).abs().mean())
    print(f"  artifact magnitude ||xhat - y||_heart = {art_mag:.1f} HU", flush=True)
    print(f"  no-motion residual  ||A_nomotion(xhat) - y||_heart = {r_nomo:.1f} HU", flush=True)
    best = (r_nomo, "nomotion")
    for phase in (0.0, 0.25, 0.5, 0.75):
        for s in (0.5, 1.0, 1.5):
            mp = MotionParams(contraction_amp_mm=10.0 * s, twist_amp_deg=12.0 * s,
                              long_axis_amp_mm=10.0 * s, phase_offset_frac=phase)
            A = DifferentiableMotionForward(heart, (1., 1., 1.), mp, SP, NPHASES, sc, off, view_stride=1)
            r = resid(A, xhat, y_hu, heart)
            if r < best[0]:
                best = (r, f"phase={phase} amp x{s}")
            print(f"    phase={phase:.2f} amp x{s:.1f}: resid {r:.1f} HU", flush=True)
    red = (r_nomo - best[0]) / max(r_nomo, 1e-6) * 100
    print(f"\n  BEST motion operator: {best[1]}  resid {best[0]:.1f} HU", flush=True)
    print(f"  residual reduction vs no-motion: {red:+.1f}%", flush=True)
    if red >= 20:
        print("  VERDICT: real artifact is PARTLY parametric-capturable -> operator estimation worth pursuing.")
    elif red >= 8:
        print("  VERDICT: WEAK -> parametric motion explains little of the real artifact; estimation marginal.")
    else:
        print("  VERDICT: real artifact NOT explained by our parametric motion -> DPS-on-real likely dead;")
        print("           the bottleneck is forward-model / motion-model fidelity, not operator estimation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
