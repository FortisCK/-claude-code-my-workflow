"""eval_dps_leo_realdata.py — external-validity DPS test on REAL clinical CCTA.

Takes clean (high-sharpness) real Leo_anon CCTA, injects KNOWN synthetic motion
(training distribution), runs frozen U-Net + DPS-MC with the known operator, and
measures frozen-TS coronary Dice-RCA recovery vs the real clean. Tests whether the
method generalizes to a SECOND, independent real dataset (cross-dataset external
validity). Motion is still our parametric model — native-motion operator estimation
is future work.

READS ~/Code/Leo_anon (read-only). WRITES ONLY under cardiac-artifacts.

Usage:
    python scripts/python/eval_dps_leo_realdata.py 072 031 067 012 160

Run card: experiments/runs/2026-06-19_dps-real-leo-external-validity.md
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.modules.pop("code", None)  # drop stdlib `code` if a dep imported it (shadow guard)
from code.data.motion_synth import MotionParams, ScanParams, synthesize_motion_artifact
from code.inference.dps_forward import DifferentiableMotionForward
from code.inference.dps_sample import dps_residual_sample, _denorm
from code.models.residual_refiner import ResidualRefinerUNet3D
from code.training.train_residual_refiner import load_initializer
from evaluate_residual_diffusion_full_volume import load_residual_diffusion
from coronary_dice_rca import dice

LEO = "/home/mingzhang/Code/Leo_anon"
OUT = REPO / "experiments" / "runs" / "dps_real_leo"
VOL, SEG = OUT / "volumes", OUT / "seg"
TS = Path(sys.prefix) / "bin" / "TotalSegmentator"
HU_LO, HU_HI = -1024.0, 3071.0
HU_SCALE = 2047.5
CROP = 192
ZETA, STEPS = 0.02, 24
# training motion distribution (generate_motion_pairs.py): amps fixed at MotionParams
# defaults; only strength + period sampled.
STRENGTH_RANGE, PERIOD_RANGE = (0.5, 1.3), (600.0, 1000.0)
CFG = "code/training/configs/diffusion_v2_residual.yaml"
CKPT = "experiments/checkpoints/diffusion_v2_residual/longrun/epoch_080.pt"


def norm(hu: np.ndarray) -> np.ndarray:
    return (np.clip(hu, HU_LO, HU_HI) - HU_LO) / (HU_HI - HU_LO) * 2.0 - 1.0


def resample_to_iso(img: sitk.Image, spacing: float, is_label: bool) -> sitk.Image:
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


def crop_pad_center(arr: np.ndarray, center, size: int, pad_val: float) -> np.ndarray:
    out = np.full((size, size, size), pad_val, dtype=np.float32)
    starts = [int(round(center[d])) - size // 2 for d in range(3)]
    s0 = [max(0, starts[d]) for d in range(3)]
    s1 = [min(arr.shape[d], starts[d] + size) for d in range(3)]
    d0 = [s0[d] - starts[d] for d in range(3)]
    d1 = [d0[d] + (s1[d] - s0[d]) for d in range(3)]
    out[d0[0]:d1[0], d0[1]:d1[1], d0[2]:d1[2]] = arr[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]
    return out


def heart_mask_native(nii: str, cid: str) -> sitk.Image:
    od = SEG / f"case_{cid}__heart"
    hf = od / "heart.nii.gz"
    if not hf.exists():
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS), "-i", nii, "-o", str(od), "-ta", "total", "-rs", "heart", "--fast"],
                       check=True, capture_output=True, text=True)
    return sitk.ReadImage(str(hf))


def savehu(arr_hu: np.ndarray, p: Path) -> None:
    im = sitk.GetImageFromArray(arr_hu.astype(np.int16))
    im.SetSpacing((1.0, 1.0, 1.0))
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(im, str(p))


def seg_coronary(nii: Path, od: Path, task: str) -> np.ndarray:
    f = od / "coronary_arteries.nii.gz"
    if not f.exists():
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS), "-i", str(nii), "-o", str(od), "-ta", task],
                       check=True, capture_output=True, text=True)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0


def prepare_clean(cid: str, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """Real Leo case -> (clean_hu 192^3 torch on dev, heart_mask 192^3 torch on dev)."""
    import glob
    nii = glob.glob(f"{LEO}/{cid}/*.nii")[0]
    vol_iso = resample_to_iso(sitk.ReadImage(nii), 1.0, is_label=False)
    heart_iso = sitk.Resample(heart_mask_native(nii, cid), vol_iso, sitk.Transform(),
                              sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    vol = sitk.GetArrayFromImage(vol_iso).astype(np.float32)
    heart = (sitk.GetArrayFromImage(heart_iso) > 0).astype(np.float32)
    center = np.argwhere(heart > 0).mean(0)
    clean_hu = crop_pad_center(vol, center, CROP, HU_LO)
    heart_c = crop_pad_center(heart, center, CROP, 0.0)
    return (torch.from_numpy(clean_hu).to(dev), torch.from_numpy(heart_c).to(dev))


def main(cids: list[str]) -> int:
    dev = torch.device("cuda")
    cfg = OmegaConf.load(CFG)
    initializer = load_initializer(cfg.initializer, dev)
    engine, _ = load_residual_diffusion(CFG, CKPT, "ema", dev)
    rows = []
    for cid in cids:
        clean_hu, heart = prepare_clean(cid, dev)
        hb = heart > 0
        if hb.sum() < 1000:
            print(f"case {cid}: heart mask too small ({int(hb.sum())}), skip", flush=True)
            continue
        # --- inject KNOWN synthetic motion (training distribution) ---
        rng = np.random.default_rng(abs(hash(("leo", cid))) & 0x7FFFFFFF)
        mp = MotionParams(motion_strength=float(rng.uniform(*STRENGTH_RANGE)),
                          cardiac_period_ms=float(rng.uniform(*PERIOD_RANGE)))
        sp = ScanParams(n_views=1000)
        corrupted_hu, dbg = synthesize_motion_artifact(clean_hu, heart, (1.0, 1.0, 1.0),
                                                       mp, sp, n_phases=48, seed=0, calibrate_hu=True)
        cal = dbg["hu_calibration"]
        clean = torch.from_numpy(norm(clean_hu.cpu().numpy()))[None, None].to(dev)
        corrupted = torch.from_numpy(norm(corrupted_hu.cpu().numpy()))[None, None].to(dev)
        with torch.no_grad():
            initial = initializer(corrupted)
        cond = ResidualRefinerUNet3D.make_condition(corrupted.float(), initial.float())
        A = DifferentiableMotionForward(heart, (1.0, 1.0, 1.0), mp, sp, 48,
                                        cal["scale"], cal["offset"], view_stride=1)
        x = dps_residual_sample(engine, cond, initial, _denorm(corrupted[0, 0]), A,
                                num_steps=STEPS, zeta=ZETA, seed=0)
        mae_u = float((initial - clean).abs()[0, 0][hb].mean() * HU_SCALE)
        mae_d = float((x - clean).abs()[0, 0][hb].mean() * HU_SCALE)
        savehu(_denorm(clean[0, 0]).cpu().numpy(), VOL / f"case_{cid}__clean.nii.gz")
        savehu(_denorm(corrupted[0, 0]).cpu().numpy(), VOL / f"case_{cid}__corrupted.nii.gz")
        savehu(_denorm(initial[0, 0]).detach().cpu().numpy(), VOL / f"case_{cid}__unet.nii.gz")
        savehu(_denorm(x[0, 0]).detach().cpu().numpy(), VOL / f"case_{cid}__dps.nii.gz")
        row = {"case": cid, "strength": mp.motion_strength, "period": mp.cardiac_period_ms,
               "mae_unet": mae_u, "mae_dps": mae_d}
        seg_arrs = {}
        for task, tag in (("coronary_arteries", ""), ("coronary_arteries_LEGACY", "_leg")):
            sc = seg_coronary(VOL / f"case_{cid}__clean.nii.gz", SEG / f"case_{cid}__clean{tag}", task)
            su = seg_coronary(VOL / f"case_{cid}__unet.nii.gz", SEG / f"case_{cid}__unet{tag}", task)
            sd = seg_coronary(VOL / f"case_{cid}__dps.nii.gz", SEG / f"case_{cid}__dps{tag}", task)
            row[f"unet_dice{tag}"] = dice(su, sc)
            row[f"dps_dice{tag}"] = dice(sd, sc)
            row[f"ddice{tag}"] = dice(sd, sc) - dice(su, sc)
            if tag == "":
                row["unet_vox"], row["dps_vox"] = int(su.sum()), int(sd.sum())
                seg_arrs["clean"] = sc
        rows.append(row)
        print(f"case {cid} | s={mp.motion_strength:.2f} T={mp.cardiac_period_ms:.0f} | "
              f"MAE u={mae_u:.1f} dps={mae_d:.1f} | Dice-RCA u={row['unet_dice']:.3f} "
              f"dps={row['dps_dice']:.3f} dDice={row['ddice']:+.3f} | LEGACY dDice={row['ddice_leg']:+.3f} | "
              f"vox {row['unet_vox']}->{row['dps_vox']}", flush=True)
        _render(cid, clean, corrupted, initial, x, seg_arrs["clean"], row)

    if not rows:
        print("no cases evaluated"); return 1

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    print(f"\n==== DPS-MC on REAL Leo CCTA (n={len(rows)}, zeta={ZETA}) ====")
    print(f"  Dice-RCA(vs TS-clean): U-Net {m('unet_dice'):.3f} -> DPS {m('dps_dice'):.3f}  ΔDice={m('ddice'):+.3f}")
    print(f"  heart-MAE (HU)       : U-Net {m('mae_unet'):.1f} -> DPS {m('mae_dps'):.1f}")
    print(f"  anti-cheat LEGACY ΔDice={m('ddice_leg'):+.3f} | voxels x{m('dps_vox')/max(m('unet_vox'),1):.2f}")
    passed = (m('dps_dice') - m('unet_dice') >= 0.05 and m('dps_dice') >= 0.70
              and m('ddice_leg') >= 0 and m('dps_vox') / max(m('unet_vox'), 1) <= 1.5)
    print(f"  VERDICT vs LOCKED bar: {'PASS — DPS generalizes to real anatomy' if passed else 'CHECK vs bar'}")
    return 0


def _render(cid, clean, corrupted, initial, x, sc, row) -> None:
    cl = _denorm(clean[0, 0]).cpu().numpy()
    co = _denorm(corrupted[0, 0]).cpu().numpy()
    un = _denorm(initial[0, 0]).detach().cpu().numpy()
    dp = _denorm(x[0, 0]).detach().cpu().numpy()
    if sc.sum() > 0:
        zc, yc, xc = [int(c) for c in np.argwhere(sc).mean(0)]
        ys, xs = np.where(sc.any(0)); m = 35
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, cl.shape[1])
        x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, cl.shape[2])
    else:
        zc = cl.shape[0] // 2; y0, y1, x0, x1 = 0, cl.shape[1], 0, cl.shape[2]
    panels = [("real clean (ref)", cl), ("+ synthetic motion", co),
              ("U-Net", un), (f"DPS-MC  Dice {row['dps_dice']:.2f} (U-Net {row['unet_dice']:.2f})", dp)]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    for ax, (t, a) in zip(axes, panels):
        ax.imshow(a[zc, y0:y1, x0:x1], cmap="gray", vmin=-100, vmax=700)
        ax.contour((sc[zc, y0:y1, x0:x1]).astype(float), levels=[0.5], colors="lime", linewidths=0.5, alpha=0.6)
        ax.set_title(t, fontsize=11); ax.axis("off")
    fig.suptitle(f"Leo_anon case {cid} (REAL CCTA) — DPS-MC recovery of known synthetic motion. green=TS coronary(clean)", fontsize=13)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"case_{cid}__panel.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    cids = sys.argv[1:] or ["072"]
    raise SystemExit(main(cids))
