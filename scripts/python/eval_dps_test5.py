"""eval_dps_test5.py — Stage-1 S4: DPS-MC test5 verdict on gold-standard Dice-RCA.

Runs frozen U-Net + DPS-MC (measurement-consistency posterior sampling, known
per-case DVF) on test5, segments U-Net / DPS / clean with frozen TS
coronary_arteries (+ LEGACY anti-cheat), and reports the LOCKED bar.
Plan: quality_reports/plans/2026-06-17_stage1-dps-mc-pilot.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, SimpleITK as sitk
from omegaconf import OmegaConf
from code.data import paths as P
from code.data.splits import load_split
from code.data.motion_synth import MotionParams, ScanParams
from code.inference.dps_forward import DifferentiableMotionForward
from code.inference.dps_sample import dps_residual_sample, _denorm
from code.models.residual_refiner import ResidualRefinerUNet3D
from code.training.train_residual_refiner import load_initializer
sys.path.insert(0, "scripts/python")
from evaluate_residual_diffusion_full_volume import load_residual_diffusion
from coronary_dice_rca import dice  # reuse

TS = Path(sys.prefix) / "bin" / "TotalSegmentator"
import subprocess
def seg(nifti, od, task):
    f = od / "coronary_arteries.nii.gz"
    if not f.exists():
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS), "-i", str(nifti), "-o", str(od), "-ta", task], check=True, capture_output=True, text=True)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0
def savehu(arr, p):
    im = sitk.GetImageFromArray(arr.astype(np.int16)); im.SetSpacing((1.,1.,1.)); p.parent.mkdir(parents=True, exist_ok=True); sitk.WriteImage(im, str(p))

ZETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
MODE = sys.argv[2] if len(sys.argv) > 2 else "main"
NCASES = int(sys.argv[3]) if len(sys.argv) > 3 else 5
WRONG = MODE == "wrongdvf"
WRONGAMP = MODE == "wrongamp"
STEPS = 24
dev = torch.device("cuda")
TAG = MODE if MODE!="main" else ("z0" if ZETA==0 else "main")
OUT = Path(f"experiments/runs/dps_pilot/test{NCASES}_{TAG}"); VOL = OUT/"volumes"; SEG = OUT/"seg"
proc = P.get("IMAGECAS_PROCESSED"); lumdir = proc.parent/"lumen_masks"
cfg = OmegaConf.load("code/training/configs/diffusion_v2_residual.yaml")
initializer = load_initializer(cfg.initializer, dev)
engine, _ = load_residual_diffusion("code/training/configs/diffusion_v2_residual.yaml",
    "experiments/checkpoints/diffusion_v2_residual/longrun/epoch_080.pt", "ema", dev)
cases = [str(c) for c in load_split(P.get("IMAGECAS_SPLITS")/"v1.json").test[:NCASES]]
rows = []
for cid in cases:
    d = np.load(proc/f"case_{cid}__pair_0.npz", allow_pickle=True); meta = d["metadata"].item()
    clean = torch.from_numpy(d["volume"].astype(np.float32))[None,None].to(dev)
    corrupted = torch.from_numpy(d["corrupted"].astype(np.float32))[None,None].to(dev)
    heart = torch.from_numpy(d["heart_mask"].astype(np.float32)).to(dev); hb = heart>0
    gt = np.load(lumdir/f"lumen_mask_{cid}.npy").astype(bool)
    with torch.no_grad():
        initial = initializer(corrupted)
    cond = ResidualRefinerUNet3D.make_condition(corrupted.float(), initial.float())
    mp = (MotionParams() if WRONG else MotionParams(contraction_amp_mm=20.0,twist_amp_deg=24.0,long_axis_amp_mm=20.0,motion_strength=1.0,cardiac_period_ms=800.0) if WRONGAMP else MotionParams(**meta["motion_params"])); sp = ScanParams(**meta["scan_params"]); cal = meta["hu_calibration"]
    A = DifferentiableMotionForward(heart,(1.,1.,1.),mp,sp,int(meta["n_phases"]),cal["scale"],cal["offset"],view_stride=1)
    x = dps_residual_sample(engine, cond, initial, _denorm(corrupted[0,0]), A, num_steps=STEPS, zeta=ZETA, seed=0)
    HU=2047.5
    mae_u=float((initial-clean).abs()[0,0][hb].mean()*HU); mae_d=float((x-clean).abs()[0,0][hb].mean()*HU)
    savehu(_denorm(clean[0,0]).cpu().numpy(), VOL/f"case_{cid}__clean.nii.gz")
    savehu(_denorm(initial[0,0]).detach().cpu().numpy(), VOL/f"case_{cid}__unet.nii.gz")
    savehu(_denorm(x[0,0]).detach().cpu().numpy(), VOL/f"case_{cid}__dps.nii.gz")
    row={"case":cid,"mae_unet":mae_u,"mae_dps":mae_d}
    for task,tag in (("coronary_arteries",""),("coronary_arteries_LEGACY","_leg")):
        sc=seg(VOL/f"case_{cid}__clean.nii.gz", SEG/f"case_{cid}__clean{tag}", task)
        su=seg(VOL/f"case_{cid}__unet.nii.gz", SEG/f"case_{cid}__unet{tag}", task)
        sd=seg(VOL/f"case_{cid}__dps.nii.gz", SEG/f"case_{cid}__dps{tag}", task)
        row[f"unet_dice{tag}"]=dice(su,sc); row[f"dps_dice{tag}"]=dice(sd,sc); row[f"ddice{tag}"]=dice(sd,sc)-dice(su,sc)
        if tag=="": row["unet_dice_gt"]=dice(su,gt); row["dps_dice_gt"]=dice(sd,gt); row["unet_vox"]=int(su.sum()); row["dps_vox"]=int(sd.sum())
    rows.append(row)
    print(f"case {cid} | MAE u={mae_u:.1f} dps={mae_d:.1f} | Dice-RCA u={row['unet_dice']:.3f} dps={row['dps_dice']:.3f} dDice={row['ddice']:+.3f} | LEGACY dDice={row['ddice_leg']:+.3f} | vox {row['unet_vox']}->{row['dps_vox']}", flush=True)
def m(k): return float(np.mean([r[k] for r in rows]))
print(f"\n==== DPS-MC PILOT test5 (zeta={ZETA}{'  WRONG-DVF control' if WRONG else ''}) ====")
print(f"  Dice-RCA(vs TS-clean): U-Net {m('unet_dice'):.3f} -> DPS {m('dps_dice'):.3f}  ΔDice={m('ddice'):+.3f}  (bar >=+0.044 -> >=0.70)")
print(f"  Dice vs GT           : U-Net {m('unet_dice_gt'):.3f} -> DPS {m('dps_dice_gt'):.3f}")
print(f"  heart-MAE (HU)       : U-Net {m('mae_unet'):.1f} -> DPS {m('mae_dps'):.1f}")
print(f"  anti-cheat LEGACY ΔDice={m('ddice_leg'):+.3f} (bar>=0) | voxels x{m('dps_vox')/max(m('unet_vox'),1):.2f} (bar<=1.5)")
passed = m('dps_dice')>=0.70 and m('ddice_leg')>=0 and m('dps_vox')/max(m('unet_vox'),1)<=1.5
print(f"\n  VERDICT: {'PASS -> Stage 2 (test100 + estimated-DVF)' if passed else 'check vs bar'}")
