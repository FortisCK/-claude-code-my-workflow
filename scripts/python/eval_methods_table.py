"""eval_methods_table.py — matched main comparison table (all methods, one test set).

For each ImageCAS test case, runs every feed-forward method through the SAME harness and
scores the paper's main metrics. Produces (a) the matched comparison TABLE and (b) the
per-output (sharpness, structural-Dice) pairs that feed the Act-II sharpness-trap analysis.

Methods: corrupted (floor), U-Net, TT U-Net, diffusion-mean, diffusion-sample.
Metrics : Dice-RCA (frozen TS coronary vs TS-clean), GT-lumen geometric Dice (INDEPENDENT
          judge), in-heart sharpness (grad-mag), heart-MAE (HU). DPS is cited separately
          (operator-dependent; not a feed-forward column).

Writes CSV + per-output volumes/segs under experiments/runs/methods_table/ (gitignored subdir).

Usage: python scripts/python/eval_methods_table.py 20
Run card: experiments/runs/2026-06-21_methods-table.md
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "python"))

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from omegaconf import OmegaConf

sys.modules.pop("code", None)
from code.data import paths as P
from code.data.splits import load_split
from code.inference.residual_diffusion_sliding_window import sliding_window_residual_diffusion_correct
from code.models.ttunet import TTUNet
from code.training.train_residual_refiner import load_initializer
from evaluate_residual_diffusion_full_volume import load_residual_diffusion
from coronary_dice_rca import dice

TS = Path(sys.prefix) / "bin" / "TotalSegmentator"
HU_LO, HU_HI, HU_SCALE = -1024.0, 3071.0, 2047.5
CFG = "code/training/configs/diffusion_v2_residual.yaml"
TT_CKPT = "experiments/checkpoints/ttunet_v1/epoch_100.pt"
DIFF_CKPT = "experiments/checkpoints/diffusion_v2_residual/longrun/epoch_080.pt"
OUT = REPO / "experiments" / "runs" / "methods_table"


def denorm(a):
    return (a + 1.0) * 0.5 * (HU_HI - HU_LO) + HU_LO


def savehu(arr_norm, p: Path):
    im = sitk.GetImageFromArray(denorm(arr_norm).astype(np.int16)); im.SetSpacing((1., 1., 1.))
    p.parent.mkdir(parents=True, exist_ok=True); sitk.WriteImage(im, str(p))


def seg(nii: Path, od: Path) -> np.ndarray:
    f = od / "coronary_arteries.nii.gz"
    if not f.exists():
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS), "-i", str(nii), "-o", str(od), "-ta", "coronary_arteries"],
                       check=True, capture_output=True, text=True)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0


def sharp_in(mask_np, vol_norm):
    hu = denorm(vol_norm)
    gz, gy, gx = np.gradient(hu)
    return float(np.sqrt(gz * gz + gy * gy + gx * gx)[mask_np].mean())


def main(ncases: int) -> int:
    set_determinism(seed=42)
    dev = torch.device("cuda")
    cfg = OmegaConf.load(CFG)
    initializer = load_initializer(cfg.initializer, dev)
    engine, _ = load_residual_diffusion(CFG, DIFF_CKPT, "ema", dev)
    tt_state = torch.load(TT_CKPT, map_location=dev)
    ttunet = TTUNet(cnum=int(tt_state.get("model_cfg", {}).get("cnum", 24))).to(dev)
    ttunet.load_state_dict(tt_state["ema_model"] if tt_state.get("ema_model") is not None else tt_state["model"])
    ttunet.eval()
    swi_unet = SlidingWindowInferer((128, 128, 128), 1, overlap=0.5, mode="gaussian", sw_device=dev, device="cpu")
    swi_tt = SlidingWindowInferer((64, 128, 128), 1, overlap=0.5, mode="gaussian", sw_device=dev, device="cpu")

    proc = P.get("IMAGECAS_PROCESSED"); lumdir = proc.parent / "lumen_masks"
    cases = [str(c) for c in load_split(P.get("IMAGECAS_SPLITS") / "v1.json").test[:ncases]]
    METHODS = ["corrupted", "unet", "ttunet", "diff_mean", "diff_sample"]
    rows = []
    for cid in cases:
        d = np.load(proc / f"case_{cid}__pair_0.npz", allow_pickle=True)
        clean = torch.from_numpy(d["volume"].astype(np.float32))[None, None]
        corrupted = torch.from_numpy(d["corrupted"].astype(np.float32))[None, None]
        heart = (d["heart_mask"] > 0); hb = torch.from_numpy(d["heart_mask"].astype(np.float32)) > 0
        with torch.inference_mode():
            unet = swi_unet(corrupted, initializer)
            tt = swi_tt(corrupted, ttunet)
        diff_mean, _, diff_sample = sliding_window_residual_diffusion_correct(
            corrupted, initializer, engine, roi_size=128, overlap=0.5, sw_batch_size=1,
            sw_device=dev, output_device="cpu", blend_mode="gaussian", sigma_scale=0.125,
            num_steps=32, n_samples=4, residual_scale=1.0, deterministic=True,
            return_initial=True, return_single_sample=True)
        outs = {"corrupted": corrupted, "unet": unet, "ttunet": tt, "diff_mean": diff_mean, "diff_sample": diff_sample}
        gt = np.load(lumdir / f"lumen_mask_{cid}.npy").astype(bool) if (lumdir / f"lumen_mask_{cid}.npy").exists() else None
        VOL, SEG = OUT / "vol", OUT / "seg"
        savehu(clean[0, 0].numpy(), VOL / f"{cid}__clean.nii.gz")
        sc = seg(VOL / f"{cid}__clean.nii.gz", SEG / f"{cid}__clean")
        row = {"cid": cid}
        for mname in METHODS:
            v = outs[mname]
            savehu(v[0, 0].numpy(), VOL / f"{cid}__{mname}.nii.gz")
            sm = seg(VOL / f"{cid}__{mname}.nii.gz", SEG / f"{cid}__{mname}")
            row[f"{mname}_dice"] = dice(sm, sc)
            row[f"{mname}_gt"] = dice(sm, gt) if gt is not None else float("nan")
            row[f"{mname}_sharp"] = sharp_in(heart, v[0, 0].numpy())
            row[f"{mname}_mae"] = float((v - clean).abs()[0, 0][hb].mean() * HU_SCALE)
        if gt is not None:
            row["clean_gt"] = dice(sc, gt)
        rows.append(row)
        print(f"{cid} | " + " ".join(f"{m}:{row[f'{m}_dice']:.3f}" for m in METHODS), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with (OUT / "methods_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(rows)

    def m(k):
        vals = [r[k] for r in rows if k in r and r[k] == r[k]]
        return float(np.mean(vals)) if vals else float("nan")
    print(f"\n==== MATCHED METHODS TABLE (n={len(rows)}) ====")
    print(f"{'method':<12} {'Dice-RCA':>9} {'GT-lumen':>9} {'sharpness':>10} {'MAE(HU)':>8}")
    for mn in METHODS:
        print(f"{mn:<12} {m(mn+'_dice'):>9.3f} {m(mn+'_gt'):>9.3f} {m(mn+'_sharp'):>10.1f} {m(mn+'_mae'):>8.1f}")
    print(f"{'clean(ref)':<12} {'1.000':>9} {m('clean_gt'):>9.3f}")
    print(f"\n  (oracle clean-paste Dice-RCA ~0.82; DPS known-op cited separately)")
    print(f"  CSV -> {OUT/'methods_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
