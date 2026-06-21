"""leo_feedforward_demo.py — apply the feed-forward de-artifacter to a real CCTA case.

Runs the U-Net initializer + residual-EDM posterior refiner (NO DPS measurement
consistency — that needs the known motion operator we don't have on real data) on
a real Leo_anon clinical CCTA case and renders corrupted vs de-artifacted.

READS ~/Code/Leo_anon (read-only). WRITES ONLY under cardiac-artifacts.

Usage:
    python scripts/python/leo_feedforward_demo.py 058

Run card: experiments/runs/2026-06-19_real-leo058-feedforward-demo.md
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

# Put the repo root on sys.path BEFORE third-party imports: some deps `import code`
# (stdlib console helper) at import time, which would shadow our `code` package.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "python"))

import numpy as np
import SimpleITK as sitk
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.utils import set_determinism
from omegaconf import OmegaConf

sys.modules.pop("code", None)  # belt-and-suspenders: drop any stdlib `code` a dep imported
from code.inference.residual_diffusion_sliding_window import (  # noqa: E402
    sliding_window_residual_diffusion_correct,
)
from code.training.train_residual_refiner import load_initializer  # noqa: E402
from evaluate_residual_diffusion_full_volume import load_residual_diffusion  # noqa: E402

HU_LO, HU_HI = -1024.0, 3071.0
LEO = "/home/mingzhang/Code/Leo_anon"
OUT_ROOT = REPO / "experiments" / "runs" / "leo_dps_demo"
SEG_ROOT = REPO / "experiments" / "runs" / "leo_motion_screen" / "seg"
CFG = REPO / "code" / "training" / "configs" / "diffusion_v2_residual.yaml"
CKPT = REPO / "experiments" / "checkpoints" / "diffusion_v2_residual" / "longrun" / "epoch_080.pt"


def denorm(a: np.ndarray) -> np.ndarray:
    return (a + 1.0) * 0.5 * (HU_HI - HU_LO) + HU_LO


def resample_iso(img: sitk.Image, spacing: float = 1.0, is_label: bool = False) -> sitk.Image:
    osp, osz = img.GetSpacing(), img.GetSize()
    nsz = [int(round(s * sp / spacing)) for s, sp in zip(osz, osp)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing((spacing, spacing, spacing))
    r.SetSize(nsz)
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputDirection(img.GetDirection())
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    r.SetDefaultPixelValue(0 if is_label else HU_LO)
    return r.Execute(img)


def save_nifti(arr_norm: np.ndarray, path: Path) -> None:
    hu = denorm(arr_norm).astype(np.int16)
    img = sitk.GetImageFromArray(hu)
    img.SetSpacing((1.0, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def main(cid: str) -> int:
    set_determinism(seed=42)
    dev = torch.device("cuda")
    out = OUT_ROOT / cid
    out.mkdir(parents=True, exist_ok=True)

    # --- load real CCTA + its TS coronary seg, resample both to 1mm iso ---
    f = glob.glob(f"{LEO}/{cid}/*.nii")[0]
    vol_img = resample_iso(sitk.ReadImage(f), 1.0, is_label=False)
    seg_img = resample_iso(sitk.ReadImage(str(SEG_ROOT / cid / "coronary_arteries.nii.gz")), 1.0, is_label=True)
    vol = sitk.GetArrayFromImage(vol_img).astype(np.float32)  # (Z,Y,X) HU
    seg = sitk.GetArrayFromImage(seg_img) > 0
    if seg.shape != vol.shape:  # guard against off-by-one from independent resampling
        z = min(seg.shape[0], vol.shape[0]); y = min(seg.shape[1], vol.shape[1]); x = min(seg.shape[2], vol.shape[2])
        vol, seg = vol[:z, :y, :x], seg[:z, :y, :x]
    print(f"[leo {cid}] resampled vol {vol.shape} 1mm iso; coronary vox {int(seg.sum())}", flush=True)

    # --- normalize HU -> [-1,1]; corrupted stays on CPU (sliding window moves patches) ---
    v = np.clip(vol, HU_LO, HU_HI)
    v = (v - HU_LO) / (HU_HI - HU_LO) * 2.0 - 1.0
    corrupted = torch.from_numpy(v)[None, None].float()

    # --- load models ---
    cfg = OmegaConf.load(CFG)
    initializer = load_initializer(cfg.initializer, dev)
    engine, meta = load_residual_diffusion(CFG, CKPT, "ema", dev)
    print(f"[leo {cid}] models loaded (diffusion epoch {meta['checkpoint_epoch']})", flush=True)

    # --- feed-forward: U-Net init + residual-EDM posterior (mean+std over 4 samples) ---
    torch.cuda.reset_peak_memory_stats(dev)
    pred, initial, uncertainty, single = sliding_window_residual_diffusion_correct(
        corrupted, initializer, engine,
        roi_size=128, overlap=0.5, sw_batch_size=1,
        sw_device=dev, output_device="cpu",
        blend_mode="gaussian", sigma_scale=0.125,
        num_steps=32, n_samples=4, residual_scale=1.0,
        deterministic=True, progress=False,
        return_initial=True, return_uncertainty=True, return_single_sample=True,
    )
    print(f"[leo {cid}] done. peak GPU {torch.cuda.max_memory_allocated(dev)/1e9:.1f} GB", flush=True)

    corr_v = corrupted[0, 0].numpy()
    unet_v = initial[0, 0].numpy()
    diff_v = pred[0, 0].numpy()
    std_v = uncertainty[0, 0].numpy()

    # --- save volumes (under cardiac-artifacts) ---
    save_nifti(unet_v, out / f"case_{cid}__unet.nii.gz")
    save_nifti(diff_v, out / f"case_{cid}__diff_mean.nii.gz")

    # --- render: axial @ coronary centroid, zoomed to coronary bbox ---
    zc, yc, xc = [int(c) for c in np.argwhere(seg).mean(0)]
    ys, xs = np.where(seg.any(0)); m = 40
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, vol.shape[1])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, vol.shape[2])

    def crop(a):
        return denorm(a[zc, y0:y1, x0:x1])

    diff_signed = denorm(diff_v) - denorm(corr_v)  # what the model changed (HU)
    panels = [
        ("real corrupted", crop(corr_v), "gray", -100, 700),
        ("U-Net init", crop(unet_v), "gray", -100, 700),
        ("diffusion mean", crop(diff_v), "gray", -100, 700),
        ("diffusion - corrupted (HU)", diff_signed[zc, y0:y1, x0:x1], "bwr", -300, 300),
        ("posterior std (norm)", std_v[zc, y0:y1, x0:x1], "viridis", 0, float(np.quantile(std_v, 0.99))),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.4))
    for ax, (t, a, cm, lo, hi) in zip(axes, panels):
        im = ax.imshow(a, cmap=cm, vmin=lo, vmax=hi)
        ax.contour((seg[zc, y0:y1, x0:x1]).astype(float), levels=[0.5], colors="lime", linewidths=0.5, alpha=0.6)
        ax.set_title(t, fontsize=11); ax.axis("off")
    fig.suptitle(f"Leo_anon case {cid} — feed-forward de-artifacting (NO DPS; trained on synthetic motion). green=TS coronary", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"case_{cid}__panel.png", dpi=130, bbox_inches="tight")
    print(f"[leo {cid}] rendered -> {out / f'case_{cid}__panel.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "058"))
