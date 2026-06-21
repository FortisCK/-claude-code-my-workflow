"""eval_ttunet_dice.py — evaluate the TT U-Net baseline on coronary Dice-RCA.

Sliding-window full-volume inference (roi = training patch) on ImageCAS test cases,
then frozen TotalSegmentator coronary Dice vs TS(clean) ("Dice-RCA") + the held-out
coronary_arteries_LEGACY (consistency) + the INDEPENDENT GT-lumen geometric Dice (the
real anti-cheat) + heart-MAE. Compares to U-Net 0.656 / diff_mean 0.654 / oracle 0.82.

Usage:
    python scripts/python/eval_ttunet_dice.py <ckpt.pt> <N_cases>
Run card: experiments/runs/2026-06-21_0035_ttunet-baseline.md
"""
from __future__ import annotations

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

sys.modules.pop("code", None)
from code.data import paths as P
from code.data.splits import load_split
from code.models.ttunet import TTUNet
from coronary_dice_rca import dice

TS = Path(sys.prefix) / "bin" / "TotalSegmentator"
HU_LO, HU_HI, HU_SCALE = -1024.0, 3071.0, 2047.5
ROI = (64, 128, 128)


def denorm(a):
    return (a + 1.0) * 0.5 * (HU_HI - HU_LO) + HU_LO


def savehu(arr_norm, p: Path):
    im = sitk.GetImageFromArray(denorm(arr_norm).astype(np.int16))
    im.SetSpacing((1.0, 1.0, 1.0))
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(im, str(p))


def seg(nii: Path, od: Path, task: str) -> np.ndarray:
    f = od / "coronary_arteries.nii.gz"
    if not f.exists():
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS), "-i", str(nii), "-o", str(od), "-ta", task], check=True, capture_output=True, text=True)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0


def main(ckpt_path: str, ncases: int) -> int:
    set_determinism(seed=42)
    dev = torch.device("cuda")
    state = torch.load(ckpt_path, map_location=dev)
    cnum = int(state.get("model_cfg", {}).get("cnum", 24))
    model = TTUNet(cnum=cnum).to(dev)
    model.load_state_dict(state["ema_model"] if state.get("ema_model") is not None else state["model"])
    model.eval()
    epoch = state.get("epoch", "?")
    inferer = SlidingWindowInferer(roi_size=ROI, sw_batch_size=1, overlap=0.5, mode="gaussian", sw_device=dev, device="cpu")

    proc = P.get("IMAGECAS_PROCESSED")
    lumdir = proc.parent / "lumen_masks"
    cases = [str(c) for c in load_split(P.get("IMAGECAS_SPLITS") / "v1.json").test[:ncases]]
    OUT = REPO / "experiments" / "runs" / "ttunet_eval" / f"epoch_{epoch}"
    VOL, SEG = OUT / "vol", OUT / "seg"
    rows = []
    for cid in cases:
        d = np.load(proc / f"case_{cid}__pair_0.npz", allow_pickle=True)
        clean = torch.from_numpy(d["volume"].astype(np.float32))[None, None]
        corrupted = torch.from_numpy(d["corrupted"].astype(np.float32))[None, None]
        heart = torch.from_numpy(d["heart_mask"].astype(np.float32)); hb = heart > 0
        with torch.inference_mode():
            pred = inferer(corrupted, model)  # cpu output
        mae = float((pred - clean).abs()[0, 0][hb].mean() * HU_SCALE)
        savehu(clean[0, 0].numpy(), VOL / f"{cid}__clean.nii.gz")
        savehu(pred[0, 0].numpy(), VOL / f"{cid}__ttunet.nii.gz")
        row = {"cid": cid, "mae": mae}
        for task, tag in (("coronary_arteries", ""), ("coronary_arteries_LEGACY", "_leg")):
            sc = seg(VOL / f"{cid}__clean.nii.gz", SEG / f"{cid}__clean{tag}", task)
            st = seg(VOL / f"{cid}__ttunet.nii.gz", SEG / f"{cid}__ttunet{tag}", task)
            row[f"dice{tag}"] = dice(st, sc)
        gt = lumdir / f"lumen_mask_{cid}.npy"
        if gt.exists():
            g = np.load(gt).astype(bool)
            sc0 = seg(VOL / f"{cid}__clean.nii.gz", SEG / f"{cid}__clean", "coronary_arteries")
            st0 = seg(VOL / f"{cid}__ttunet.nii.gz", SEG / f"{cid}__ttunet", "coronary_arteries")
            row["dice_gt"] = dice(st0, g)          # TT U-Net coronary vs GT lumen (independent judge)
            row["clean_dice_gt"] = dice(sc0, g)
        rows.append(row)
        print(f"case {cid} | MAE {mae:.1f} | Dice-RCA {row['dice']:.3f} | LEGACY {row['dice_leg']:.3f} | "
              f"GT-lumen {row.get('dice_gt', float('nan')):.3f}", flush=True)

    def m(k):
        vals = [r[k] for r in rows if k in r and r[k] == r[k]]
        return float(np.mean(vals)) if vals else float("nan")
    print(f"\n==== TT U-Net baseline (epoch {epoch}, n={len(rows)}) ====")
    print(f"  Dice-RCA (vs TS-clean): {m('dice'):.3f}   [U-Net 0.656, diff_mean 0.654, oracle 0.82]")
    print(f"  LEGACY (consistency):   {m('dice_leg'):.3f}")
    print(f"  GT-lumen Dice (independent judge): {m('dice_gt'):.3f}  [clean ref {m('clean_dice_gt'):.3f}]")
    print(f"  heart-MAE: {m('mae'):.1f} HU   [U-Net ~38]")
    return 0


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else "experiments/checkpoints/ttunet_v1/epoch_100.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    raise SystemExit(main(ck, n))
