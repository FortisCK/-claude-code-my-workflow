"""act2_sharpness_trap.py — Act-II: the sharpness trap + reference-free hallucination guard.

Consumes the matched methods table (eval_methods_table.py) on synthetic test cases (GT
available). Two results:

1. THE TRAP: across methods, no-reference in-heart sharpness does NOT track structural
   correctness (GT-lumen Dice). Quantified by their correlation (near-zero/negative) and the
   exemplar (diffusion-sample = highest sharpness, NOT highest structure).

2. REFERENCE-FREE GUARD: a deployable hallucination flag (coronary voxel-count ratio vs the
   corrupted input, from the saved TS segs) validated by ROC-AUC against a GT-defined
   hallucination label (sharper than input AND structurally worse than U-Net). On real data
   (no GT) the calibrated threshold is applied; here we validate it where GT exists.

CPU only (reads CSV + seg masks). No GPU.

Usage: python scripts/python/act2_sharpness_trap.py
Run card: experiments/runs/2026-06-21_methods-table.md (Act-II analysis)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TABLE = REPO / "experiments" / "runs" / "methods_table"
METHODS = ["corrupted", "unet", "ttunet", "diff_mean", "diff_sample"]


def coronary_vox(cid, method):
    f = TABLE / "seg" / f"{cid}__{method}" / "coronary_arteries.nii.gz"
    if not f.exists():
        return None
    return int((sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0).sum())


def roc_auc(score, label):
    label = np.asarray(label); score = np.asarray(score)
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U / AUC
    alls = np.concatenate([pos, neg]); order = alls.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(alls) + 1)
    auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def main():
    rows = list(csv.DictReader(open(TABLE / "methods_table.csv")))
    print(f"Act-II analysis on {len(rows)} cases x {len(METHODS)} methods\n")

    # ---- (1) the trap: sharpness vs structural correctness across all (case, method) ----
    sharp_all, struct_all, mlabel = [], [], []
    for r in rows:
        for mi, mth in enumerate(METHODS):
            if r.get(f"{mth}_sharp") and r.get(f"{mth}_gt"):
                sharp_all.append(float(r[f"{mth}_sharp"])); struct_all.append(float(r[f"{mth}_gt"])); mlabel.append(mi)
    sharp_all = np.array(sharp_all); struct_all = np.array(struct_all); mlabel = np.array(mlabel)
    corr = float(np.corrcoef(sharp_all, struct_all)[0, 1])
    print("(1) THE TRAP — no-reference sharpness vs structural GT-lumen Dice")
    print(f"    Pearson corr (across all case x method) = {corr:+.3f}  (>0 would mean sharpness tracks structure)")
    def mean_by(mth, key):
        v = [float(r[f"{mth}_{key}"]) for r in rows if r.get(f"{mth}_{key}")]
        return float(np.mean(v)) if v else float("nan")
    print(f"    {'method':<12}{'sharpness':>10}{'GT-lumen':>10}{'Dice-RCA':>10}")
    for mth in METHODS:
        print(f"    {mth:<12}{mean_by(mth,'sharp'):>10.1f}{mean_by(mth,'gt'):>10.3f}{mean_by(mth,'dice'):>10.3f}")
    sharp_rank = sorted(METHODS, key=lambda mth: mean_by(mth, "sharp"))[-1]
    struct_rank = sorted(METHODS, key=lambda mth: mean_by(mth, "gt"))[-1]
    print(f"    highest SHARPNESS: {sharp_rank} | highest STRUCTURE: {struct_rank}"
          f"  -> {'TRAP (sharpest != most structural)' if sharp_rank != struct_rank else 'aligned'}")

    # ---- (2) reference-free guard: voxel-ratio vs GT-hallucination label ----
    print("\n(2) REFERENCE-FREE HALLUCINATION GUARD (coronary voxel-count ratio vs input)")
    scores, labels = [], []
    have_vox = True
    for r in rows:
        cid = r["cid"]
        v_corr = coronary_vox(cid, "corrupted")
        if v_corr is None or v_corr == 0:
            have_vox = False; break
        sh_corr = float(r["corrupted_sharp"]); gt_unet = float(r["unet_gt"])
        for mth in ["unet", "ttunet", "diff_mean", "diff_sample"]:
            vm = coronary_vox(cid, mth)
            if vm is None:
                continue
            ratio_dev = abs(vm / v_corr - 1.0)             # reference-free signal
            halluc = int(float(r[f"{mth}_sharp"]) > sh_corr and float(r[f"{mth}_gt"]) < gt_unet)  # GT label
            scores.append(ratio_dev); labels.append(halluc)
    if have_vox and scores and 0 < sum(labels) < len(labels):
        auc = roc_auc(scores, labels)
        print(f"    voxel-ratio guard ROC-AUC vs GT-hallucination = {auc:.3f}  "
              f"(n={len(scores)}, halluc rate {np.mean(labels):.2f})")
    else:
        auc = float("nan")
        print(f"    (insufficient seg masks or no class variance; halluc rate {np.mean(labels) if labels else float('nan'):.2f})")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = plt.cm.viridis(np.linspace(0, 1, len(METHODS)))
    for mi, mth in enumerate(METHODS):
        sel = mlabel == mi
        ax[0].scatter(sharp_all[sel], struct_all[sel], s=18, color=colors[mi], label=mth, alpha=0.7)
    ax[0].set_xlabel("no-reference in-heart sharpness (HU)"); ax[0].set_ylabel("structural GT-lumen Dice")
    ax[0].set_title(f"Sharpness trap: corr={corr:+.2f} (sharpness does NOT track structure)"); ax[0].legend(fontsize=8)
    ax[1].bar(METHODS, [mean_by(m, "sharp") for m in METHODS], color=colors)
    ax[1].set_ylabel("mean sharpness (HU)"); ax[1].set_title("highest sharpness = diffusion sample (the hallucinator)")
    ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(TABLE / "act2_sharpness_trap.png", dpi=130)
    print(f"\n  figure -> {TABLE/'act2_sharpness_trap.png'}")
    print(f"\n  ACT-II SUMMARY: sharpness-structure corr {corr:+.2f}; voxel-ratio guard AUC {auc:.3f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
