"""blind_dps_native.py — best-effort recovery of NATIVE real motion (no known operator).

Takes a real motion-corrupted Leo_anon case (e.g. the dirtiest, case 058) and
attempts de-artifacting with the tools we have, HONESTLY:
  1. U-Net init (feed-forward)
  2. diffusion residual refiner (feed-forward, no measurement consistency)
  3. blind DPS — assumed population-mean motion operator + no-motion HU calibration.

There is NO clean GT (native corruption) -> visual + no-reference in-heart sharpness
only. Expected to be WEAK: our forward operator != the real scanner's image
formation, so the DPS data-consistency term cannot validly constrain real data.
This script makes that wall concrete rather than asserted.

READS ~/Code/Leo_anon (read-only). WRITES ONLY under cardiac-artifacts.

Usage:
    python scripts/python/blind_dps_native.py 058

Run card: experiments/runs/2026-06-19_blind-dps-native-058.md
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.modules.pop("code", None)
from code.data.motion_synth import MotionParams, ScanParams
from code.inference.dps_forward import DifferentiableMotionForward
from code.inference.dps_sample import dps_residual_sample, _denorm
from code.inference.residual_diffusion_sliding_window import sliding_window_residual_diffusion_correct
from code.models.residual_refiner import ResidualRefinerUNet3D
from code.training.train_residual_refiner import load_initializer
from evaluate_residual_diffusion_full_volume import load_residual_diffusion

LEO = "/home/mingzhang/Code/Leo_anon"
OUT = REPO / "experiments" / "runs" / "blind_dps_native"
SEG = OUT / "seg"
TS = Path(sys.prefix) / "bin" / "TotalSegmentator"
HU_LO, HU_HI, HU_SCALE = -1024.0, 3071.0, 2047.5
CROP = 192
ZETA, STEPS = 0.02, 24
# population-mean assumed motion (training dist midpoints; amps at defaults)
POP_STRENGTH, POP_PERIOD = 0.9, 800.0
CFG = "code/training/configs/diffusion_v2_residual.yaml"
CKPT = "experiments/checkpoints/diffusion_v2_residual/longrun/epoch_080.pt"


def norm(hu):
    return (np.clip(hu, HU_LO, HU_HI) - HU_LO) / (HU_HI - HU_LO) * 2.0 - 1.0


def resample_iso(img, spacing, is_label):
    osp, osz = img.GetSpacing(), img.GetSize()
    nsz = [int(round(s * sp / spacing)) for s, sp in zip(osz, osp)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing((spacing, spacing, spacing)); r.SetSize(nsz)
    r.SetOutputOrigin(img.GetOrigin()); r.SetOutputDirection(img.GetDirection())
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    r.SetDefaultPixelValue(0 if is_label else HU_LO)
    return r.Execute(img)


def crop_pad(arr, center, size, pad):
    out = np.full((size, size, size), pad, dtype=np.float32)
    st = [int(round(center[d])) - size // 2 for d in range(3)]
    s0 = [max(0, st[d]) for d in range(3)]; s1 = [min(arr.shape[d], st[d] + size) for d in range(3)]
    d0 = [s0[d] - st[d] for d in range(3)]; d1 = [d0[d] + (s1[d] - s0[d]) for d in range(3)]
    out[d0[0]:d1[0], d0[1]:d1[1], d0[2]:d1[2]] = arr[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]
    return out


def ts_mask(nii, cid, task, rs=None):
    od = SEG / f"case_{cid}__{task}{'_' + rs if rs else ''}"
    fname = f"{rs}.nii.gz" if rs else "coronary_arteries.nii.gz"
    f = od / fname
    if not f.exists():
        od.mkdir(parents=True, exist_ok=True)
        cmd = [str(TS), "-i", str(nii), "-o", str(od), "-ta", task]
        if rs:
            cmd += ["-rs", rs, "--fast"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0


def sharp_in(mask, vol_hu):
    if mask.sum() == 0:
        return 0.0
    gz, gy, gx = np.gradient(vol_hu)
    return float(np.sqrt(gz * gz + gy * gy + gx * gx)[mask].mean())


def main(cid):
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    nii = glob.glob(f"{LEO}/{cid}/*.nii")[0]

    # --- real corrupted -> 1mm iso, heart-centered 192^3 (y; no GT) ---
    vol_iso = resample_iso(sitk.ReadImage(nii), 1.0, is_label=False)
    heart_native = ts_mask(nii, cid, "total", rs="heart")  # native-res bool
    heart_img = sitk.GetImageFromArray(heart_native.astype(np.uint8))
    heart_img.CopyInformation(sitk.ReadImage(nii))
    heart_iso = sitk.Resample(heart_img, vol_iso, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    vol = sitk.GetArrayFromImage(vol_iso).astype(np.float32)
    heart = (sitk.GetArrayFromImage(heart_iso) > 0).astype(np.float32)
    center = np.argwhere(heart > 0).mean(0)
    y_hu = crop_pad(vol, center, CROP, HU_LO)
    heart_c = torch.from_numpy(crop_pad(heart, center, CROP, 0.0)).to(dev)
    hb = heart_c > 0
    corrupted = torch.from_numpy(norm(y_hu))[None, None].to(dev)
    corrupted_hu = _denorm(corrupted[0, 0])
    print(f"[blind {cid}] 1mm 192^3 heart-crop; heart vox {int(hb.sum())}", flush=True)

    # --- models ---
    cfg = OmegaConf.load(CFG)
    initializer = load_initializer(cfg.initializer, dev)
    engine, _ = load_residual_diffusion(CFG, CKPT, "ema", dev)
    with torch.no_grad():
        initial = initializer(corrupted)
    cond = ResidualRefinerUNet3D.make_condition(corrupted.float(), initial.float())

    # --- feed-forward diffusion refiner (no MC) ---
    ff, _, _ = sliding_window_residual_diffusion_correct(
        corrupted.cpu(), initializer, engine, roi_size=128, overlap=0.5, sw_batch_size=1,
        sw_device=dev, output_device="cpu", blend_mode="gaussian", sigma_scale=0.125,
        num_steps=32, n_samples=1, residual_scale=1.0, deterministic=True,
        progress=False, return_initial=True, return_single_sample=True)
    ff = ff.to(dev)

    # --- blind DPS: calibrate HU via no-motion baseline, assume pop-mean motion ---
    sp = ScanParams(n_views=1000)
    mp_cal = MotionParams(motion_strength=0.0, cardiac_period_ms=POP_PERIOD)
    A0 = DifferentiableMotionForward(heart_c, (1., 1., 1.), mp_cal, sp, 48, 1.0, 0.0, view_stride=1)
    with torch.no_grad():
        base = A0(corrupted_hu)
    bi, ci = base[hb], corrupted_hu[hb]
    scale = float(ci.std() / (bi.std() + 1e-6))
    offset = float(ci.mean() - scale * bi.mean())
    mp = MotionParams(motion_strength=POP_STRENGTH, cardiac_period_ms=POP_PERIOD)
    A = DifferentiableMotionForward(heart_c, (1., 1., 1.), mp, sp, 48, scale, offset, view_stride=1)
    dps = dps_residual_sample(engine, cond, initial, corrupted_hu, A, num_steps=STEPS, zeta=ZETA, seed=0)

    # --- save volumes ---
    def savehu(t, name):
        im = sitk.GetImageFromArray(_denorm(t[0, 0]).detach().cpu().numpy().astype(np.int16))
        im.SetSpacing((1., 1., 1.)); sitk.WriteImage(im, str(OUT / f"case_{cid}__{name}.nii.gz"))
    savehu(corrupted, "corrupted"); savehu(initial, "unet"); savehu(ff, "ff_diff"); savehu(dps, "blind_dps")

    # --- no-reference in-heart sharpness (higher = sharper) ---
    hbm = hb.cpu().numpy()
    sh = {n: sharp_in(hbm, _denorm(t[0, 0]).detach().cpu().numpy())
          for n, t in (("corrupted", corrupted), ("unet", initial), ("ff_diff", ff), ("blind_dps", dps))}
    print(f"[blind {cid}] in-heart sharpness: " + "  ".join(f"{k}={v:.1f}" for k, v in sh.items()), flush=True)

    # --- render ---
    co = _denorm(corrupted[0, 0]).cpu().numpy()
    cor_seg = ts_mask(OUT / f"case_{cid}__corrupted.nii.gz", cid, "coronary_arteries")
    if cor_seg.sum() > 0:
        zc = int(np.argwhere(cor_seg).mean(0)[0]); ys, xs = np.where(cor_seg.any(0)); m = 40
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, co.shape[1]); x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, co.shape[2])
    else:
        zc = co.shape[0] // 2; y0, y1, x0, x1 = 0, co.shape[1], 0, co.shape[2]
    panels = [("real corrupted (native motion)", corrupted, sh["corrupted"]),
              ("U-Net", initial, sh["unet"]),
              ("feed-forward diffusion", ff, sh["ff_diff"]),
              ("blind DPS (assumed operator)", dps, sh["blind_dps"])]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    for ax, (t, vol_t, s) in zip(axes, panels):
        a = _denorm(vol_t[0, 0]).detach().cpu().numpy()
        ax.imshow(a[zc, y0:y1, x0:x1], cmap="gray", vmin=-100, vmax=700)
        ax.set_title(f"{t}\nin-heart sharp={s:.0f}", fontsize=10); ax.axis("off")
    fig.suptitle(f"Leo_anon case {cid} — NATIVE real motion, NO GT. blind DPS expected weak (operator unknown)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"case_{cid}__panel.png", dpi=130, bbox_inches="tight")
    print(f"[blind {cid}] rendered -> {OUT / f'case_{cid}__panel.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "058"))
