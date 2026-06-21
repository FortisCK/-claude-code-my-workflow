"""eval_dps_varamp.py — variable-amplitude (OOD) motion robustness = real deployability test.

Training motion had FIXED amplitudes (10/12/10); population-mean operator was ~exact.
Here we synthesize corrupted with RANDOM per-case amplitudes (OOD), then compare:
  U-Net | DPS true-operator (upper bound) | DPS population-mean operator (deployable:
  amplitudes wrong per-case). Answers: does DPS still win when you DON'T know the
  per-case motion amplitudes? Dice-RCA via frozen TS coronary_arteries vs TS-clean.
"""
from __future__ import annotations
import sys, subprocess
from pathlib import Path
import numpy as np, torch, SimpleITK as sitk
from omegaconf import OmegaConf
from code.data import paths as P
from code.data.splits import load_split
from code.data.motion_synth import MotionParams, ScanParams, synthesize_motion_artifact
from code.inference.dps_forward import DifferentiableMotionForward
from code.inference.dps_sample import dps_residual_sample, _denorm
from code.models.residual_refiner import ResidualRefinerUNet3D
from code.training.train_residual_refiner import load_initializer
sys.path.insert(0, "scripts/python")
from evaluate_residual_diffusion_full_volume import load_residual_diffusion
from coronary_dice_rca import dice

TS = Path(sys.prefix)/"bin"/"TotalSegmentator"
def seg(n, od):
    f = od/"coronary_arteries.nii.gz"
    if not f.exists():
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS),"-i",str(n),"-o",str(od),"-ta","coronary_arteries"],check=True,capture_output=True,text=True)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(f)))>0
def savehu(a,p): im=sitk.GetImageFromArray(a.astype(np.int16)); im.SetSpacing((1.,1.,1.)); p.parent.mkdir(parents=True,exist_ok=True); sitk.WriteImage(im,str(p))
LO,HI=-1024.,3071.; norm=lambda a:(np.clip(a,LO,HI).astype(np.float32)-LO)/(HI-LO)*2-1

N = int(sys.argv[1]) if len(sys.argv)>1 else 20
ZETA=0.02; dev=torch.device("cuda")
OUT=Path("experiments/runs/dps_varamp/test%d"%N); VOL=OUT/"volumes"; SEG=OUT/"seg"
proc=P.get("IMAGECAS_PROCESSED"); lumdir=proc.parent/"lumen_masks"
cfg=OmegaConf.load("code/training/configs/diffusion_v2_residual.yaml")
initializer=load_initializer(cfg.initializer,dev)
engine,_=load_residual_diffusion("code/training/configs/diffusion_v2_residual.yaml","experiments/checkpoints/diffusion_v2_residual/longrun/epoch_080.pt","ema",dev)
cases=[str(c) for c in load_split(P.get("IMAGECAS_SPLITS")/"v1.json").test[:N]]
rows=[]
for cid in cases:
    d=np.load(proc/f"case_{cid}__pair_0.npz",allow_pickle=True); meta=d["metadata"].item()
    clean_n=d["volume"].astype(np.float32); clean=torch.from_numpy(clean_n)[None,None].to(dev)
    heart=torch.from_numpy(d["heart_mask"].astype(np.float32)).to(dev); hb=heart>0
    gt=np.load(lumdir/f"lumen_mask_{cid}.npy").astype(bool)
    rng=np.random.default_rng(abs(hash(("varamp",cid)))&0x7fffffff)
    # VARIABLE-amplitude OOD motion params
    va=MotionParams(contraction_amp_mm=float(rng.uniform(7,16)), twist_amp_deg=float(rng.uniform(8,24)),
                    long_axis_amp_mm=float(rng.uniform(7,16)), motion_strength=float(rng.uniform(0.7,1.3)),
                    cardiac_period_ms=float(rng.uniform(600,1000)))
    sp=ScanParams(n_views=1000)
    clean_hu=torch.from_numpy((clean_n+1)/2*(HI-LO)+LO).to(dev)
    corr_hu_t,dbg=synthesize_motion_artifact(clean_hu,heart,(1.,1.,1.),motion_params=va,scan_params=sp,n_phases=48,seed=0,calibrate_hu=True)
    corr_hu=corr_hu_t.detach()
    corrupted=torch.from_numpy(norm(corr_hu.cpu().numpy()))[None,None].to(dev)
    cal=dbg["hu_calibration"]
    with torch.no_grad(): initial=initializer(corrupted)
    cond=ResidualRefinerUNet3D.make_condition(corrupted.float(),initial.float())
    A_true=DifferentiableMotionForward(heart,(1.,1.,1.),va,sp,48,cal["scale"],cal["offset"],view_stride=1)
    A_pop =DifferentiableMotionForward(heart,(1.,1.,1.),MotionParams(),sp,48,cal["scale"],cal["offset"],view_stride=1)
    x_true=dps_residual_sample(engine,cond,initial,corr_hu,A_true,num_steps=24,zeta=ZETA,seed=0)
    x_pop =dps_residual_sample(engine,cond,initial,corr_hu,A_pop, num_steps=24,zeta=ZETA,seed=0)
    HU=2047.5
    savehu(_denorm(clean[0,0]).cpu().numpy(),VOL/f"case_{cid}__clean.nii.gz")
    savehu(_denorm(initial[0,0]).detach().cpu().numpy(),VOL/f"case_{cid}__unet.nii.gz")
    savehu(_denorm(x_true[0,0]).detach().cpu().numpy(),VOL/f"case_{cid}__dpstrue.nii.gz")
    savehu(_denorm(x_pop[0,0]).detach().cpu().numpy(),VOL/f"case_{cid}__dpspop.nii.gz")
    sc=seg(VOL/f"case_{cid}__clean.nii.gz",SEG/f"case_{cid}__clean")
    su=seg(VOL/f"case_{cid}__unet.nii.gz",SEG/f"case_{cid}__unet")
    st=seg(VOL/f"case_{cid}__dpstrue.nii.gz",SEG/f"case_{cid}__dpstrue")
    spp=seg(VOL/f"case_{cid}__dpspop.nii.gz",SEG/f"case_{cid}__dpspop")
    row=dict(case=cid,unet=dice(su,sc),dpstrue=dice(st,sc),dpspop=dice(spp,sc),
             mae_u=float((initial-clean).abs()[0,0][hb].mean()*HU),
             mae_t=float((x_true-clean).abs()[0,0][hb].mean()*HU),
             mae_p=float((x_pop-clean).abs()[0,0][hb].mean()*HU))
    rows.append(row)
    print(f"case {cid} (amp c{va.contraction_amp_mm:.0f}/t{va.twist_amp_deg:.0f}/l{va.long_axis_amp_mm:.0f}) | "
          f"Dice u={row['unet']:.3f} dps-true={row['dpstrue']:.3f} dps-pop={row['dpspop']:.3f}",flush=True)
def m(k): return float(np.mean([r[k] for r in rows]))
print(f"\n==== VARIABLE-AMPLITUDE (OOD) ROBUSTNESS test{N} ====")
print(f"  Dice-RCA: U-Net {m('unet'):.3f} | DPS true-op {m('dpstrue'):.3f} (+{m('dpstrue')-m('unet'):.3f}) | DPS pop-op {m('dpspop'):.3f} (+{m('dpspop')-m('unet'):.3f})")
print(f"  heart-MAE: U-Net {m('mae_u'):.1f} | DPS true {m('mae_t'):.1f} | DPS pop {m('mae_p'):.1f} HU")
print(f"  reading: DPS pop-op >> U-Net -> robust to unknown amplitudes (deployable); ~U-Net -> needs per-case DVF estimation")
