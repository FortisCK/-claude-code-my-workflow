"""eval_lumen_refiner_dice.py — pilot verdict for the task-aware lumen refiner.

Run card: experiments/runs/2026-06-16_*_lumen-refiner-pilot.md
Plan: quality_reports/plans/2026-06-16_step2-task-aware-refiner.md

For test5, runs frozen U-Net + lumen refiner (full-volume sliding window),
segments U-Net-init / refiner / clean with the FROZEN TotalSegmentator
coronary_arteries (primary) and coronary_arteries_LEGACY (held-out anti-cheat),
and reports the pre-registered bar:
  PRIMARY  ΔDice-RCA(vs TS-clean) = refiner - unet  >= +0.03
  leash    ΔMAE(HU) = refiner - unet                <= +3
  anti-cheat ΔDice on LEGACY segmenter              >= 0
  hallucination: refiner coronary voxels <= 1.5x unet

Everything is recomputed in this run (U-Net + refiner segmented the same way) so
the comparison is internally consistent.

Usage:
    python -m scripts.python.eval_lumen_refiner_dice \
        --refiner-ckpt experiments/checkpoints/lumen_refiner_v1/train100_e30/epoch_030.pt --test-cases 5
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
from monai.utils import set_determinism
from omegaconf import OmegaConf

from code.data import paths as path_registry
from code.data.splits import load_split
from code.inference.residual_refiner_sliding_window import sliding_window_refiner_correct
from code.training.train_residual_refiner import build_refiner, load_initializer

log = logging.getLogger("eval_lumen_refiner_dice")
TS_BIN = Path(sys.prefix) / "bin" / "TotalSegmentator"
HU_SCALE = 2047.5
_HU_LO, _HU_HI = -1024.0, 3071.0


def save_hu(arr_norm: np.ndarray, path: Path) -> None:
    hu = ((arr_norm + 1.0) * 0.5 * (_HU_HI - _HU_LO) + _HU_LO).astype(np.int16)
    img = sitk.GetImageFromArray(hu); img.SetSpacing((1.0, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def run_ts(nifti: Path, out_dir: Path, task: str) -> np.ndarray:
    seg = out_dir / "coronary_arteries.nii.gz"
    if not seg.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS_BIN), "-i", str(nifti), "-o", str(out_dir), "-ta", task],
                       check=True, capture_output=True, text=True)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(seg))) > 0


def dice(a, b, eps=1e-8):
    return (2.0 * float(np.logical_and(a, b).sum()) + eps) / (float(a.sum()) + float(b.sum()) + eps)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--refiner-ckpt", type=Path, required=True)
    p.add_argument("--test-cases", type=int, default=5)
    p.add_argument("--features", nargs="+", type=int, default=[32, 32, 64, 128, 256, 32])
    p.add_argument("--out-dir", type=Path, default=Path("experiments/runs/lumen_refiner_v1/pilot_eval"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p


@torch.inference_mode()
def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)
    set_determinism(seed=args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processed_dir = path_registry.get("IMAGECAS_PROCESSED")
    lumen_dir = processed_dir.parent / "lumen_masks"
    vol_dir = args.out_dir / "volumes"; seg_dir = args.out_dir / "seg"

    initializer = load_initializer(OmegaConf.create({
        "config": "code/training/configs/unet_v1.yaml",
        "ckpt": "experiments/checkpoints/unet_v1/epoch_200.pt", "weights": "ema"}), device)
    refiner = build_refiner(OmegaConf.create({
        "in_channels": 3, "out_channels": 1, "features": list(args.features), "clamp_output": False})).to(device)
    state = torch.load(args.refiner_ckpt, map_location=device)
    refiner.load_state_dict(state["model"]); refiner.eval()
    log.info("[load] refiner %s @ epoch %s", args.refiner_ckpt, state.get("epoch"))

    cases = [str(c) for c in load_split(path_registry.get("IMAGECAS_SPLITS") / "v1.json").test[: args.test_cases]]
    rows: list[dict] = []
    for cid in cases:
        d = np.load(processed_dir / f"case_{cid}__pair_0.npz", allow_pickle=True)
        clean = d["volume"].astype(np.float32)
        # keep on CPU; SlidingWindowInferer moves patches to sw_device, and the
        # internal make_condition must match the cpu-resident initial output.
        corrupted = torch.from_numpy(d["corrupted"].astype(np.float32))[None, None]
        gt = np.load(lumen_dir / f"lumen_mask_{cid}.npy").astype(bool)

        final, initial = sliding_window_refiner_correct(
            corrupted, initializer, refiner, roi_size=128, overlap=0.5,
            sw_device=device, output_device="cpu", return_initial=True)
        final_np = final[0, 0].cpu().numpy(); init_np = initial[0, 0].cpu().numpy()
        mae_unet = float(np.abs(init_np - clean).mean() * HU_SCALE)
        mae_ref = float(np.abs(final_np - clean).mean() * HU_SCALE)

        save_hu(clean, vol_dir / f"case_{cid}__clean.nii.gz")
        save_hu(init_np, vol_dir / f"case_{cid}__unet.nii.gz")
        save_hu(final_np, vol_dir / f"case_{cid}__refiner.nii.gz")

        row = {"case_id": cid, "mae_unet": mae_unet, "mae_refiner": mae_ref, "dmae": mae_ref - mae_unet}
        for task, tag in [("coronary_arteries", ""), ("coronary_arteries_LEGACY", "_leg")]:
            sc = run_ts(vol_dir / f"case_{cid}__clean.nii.gz", seg_dir / f"case_{cid}__clean{tag}_{task}", task)
            su = run_ts(vol_dir / f"case_{cid}__unet.nii.gz", seg_dir / f"case_{cid}__unet{tag}_{task}", task)
            sr = run_ts(vol_dir / f"case_{cid}__refiner.nii.gz", seg_dir / f"case_{cid}__refiner{tag}_{task}", task)
            row[f"unet_dice{tag}"] = dice(su, sc)
            row[f"refiner_dice{tag}"] = dice(sr, sc)
            row[f"ddice{tag}"] = row[f"refiner_dice{tag}"] - row[f"unet_dice{tag}"]
            if tag == "":
                row["unet_dice_gt"] = dice(su, gt); row["refiner_dice_gt"] = dice(sr, gt)
                row["unet_seg_vox"] = int(su.sum()); row["refiner_seg_vox"] = int(sr.sum())
        rows.append(row)
        log.info("case %s | dMAE=%.2f | ddice=%.3f (unet %.3f->ref %.3f) | ddice_LEGACY=%.3f | vox %d->%d",
                 cid, row["dmae"], row["ddice"], row["unet_dice"], row["refiner_dice"],
                 row["ddice_leg"], row["unet_seg_vox"], row["refiner_seg_vox"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)

    def m(k): return float(np.mean([r[k] for r in rows]))
    ddice, dmae, ddice_leg = m("ddice"), m("dmae"), m("ddice_leg")
    vox_ratio = m("refiner_seg_vox") / max(m("unet_seg_vox"), 1)
    print("\n==== LUMEN REFINER PILOT — verdict vs pre-set bar ====")
    print(f"  Dice-RCA(vs TS-clean): U-Net {m('unet_dice'):.3f} -> refiner {m('refiner_dice'):.3f}  | ΔDice = {ddice:+.3f}  (bar >= +0.03)")
    print(f"  Dice vs GT           : U-Net {m('unet_dice_gt'):.3f} -> refiner {m('refiner_dice_gt'):.3f}")
    print(f"  MAE (HU)             : U-Net {m('mae_unet'):.2f} -> refiner {m('mae_refiner'):.2f}  | ΔMAE = {dmae:+.2f}  (bar <= +3)")
    print(f"  anti-cheat ΔDice(LEGACY) = {ddice_leg:+.3f}  (bar >= 0)")
    print(f"  hallucination: coronary voxels x{vox_ratio:.2f}  (bar <= 1.5x)")
    passed = (ddice >= 0.03) and (dmae <= 3.0) and (ddice_leg >= 0.0) and (vox_ratio <= 1.5)
    print(f"\n  VERDICT: {'PASS — scale to test100/full train' if passed else 'FAIL/marginal — Pilot-B (add GAN) or rethink'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
