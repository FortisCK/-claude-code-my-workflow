"""screen_real_motion.py — find ImageCAS clean volumes with the worst (likely real)
coronary motion-artifact, by low in-lumen sharpness/CNR, and render slices to inspect."""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from code.data import paths as P
LO,HI=-1024.,3071.
proc=P.get("IMAGECAS_PROCESSED"); lumdir=proc.parent/"lumen_masks"
def hu(v): return (v+1)/2*(HI-LO)+LO
def gradmag(a):
    gz,gy,gx=np.gradient(a); return np.sqrt(gz*gz+gy*gy+gx*gx)
rows=[]
cases=sorted(int(p.stem.split('_')[-1]) for p in lumdir.glob("lumen_mask_*.npy"))
for cid in cases:
    npz=proc/f"case_{cid}.npz"
    if not npz.exists(): continue
    v=hu(np.load(npz,allow_pickle=True)["volume"].astype(np.float32))
    lum=np.load(lumdir/f"lumen_mask_{cid}.npy").astype(bool)
    nv=int(lum.sum())
    if nv<4000: continue   # need an adequately-sized vessel to judge blur
    ring=binary_dilation(lum,iterations=3)&~lum
    gm=gradmag(v)
    sharp=float(gm[lum].mean())
    cnr=float(v[lum].mean()-v[ring].mean())
    rows.append((cid,sharp,cnr,nv))
rows.sort(key=lambda r:r[1])  # ascending sharpness = most blurred first
print(f"screened {len(rows)} cases (lumen>=4000 vox)")
print(f"{'rank':<5}{'case':<7}{'lumen_sharp':>12}{'lumen_CNR':>11}{'voxels':>9}")
worst=rows[:9]
for i,(cid,s,c,nv) in enumerate(worst):
    print(f"{i+1:<5}{cid:<7}{s:>12.1f}{c:>11.1f}{nv:>9}")
# percentile context
allsharp=np.array([r[1] for r in rows])
print(f"\nsharpness: min={allsharp.min():.1f} p10={np.percentile(allsharp,10):.1f} median={np.median(allsharp):.1f} max={allsharp.max():.1f}")
# render montage: axial slice at lumen-centroid z, heart-bbox crop, coronary window, GT overlay
out=Path("experiments/runs/real_motion_screen"); out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(3,3,figsize=(13,13))
for ax,(cid,s,c,nv) in zip(axes.ravel(),worst):
    v=hu(np.load(proc/f"case_{cid}.npz",allow_pickle=True)["volume"].astype(np.float32))
    lum=np.load(lumdir/f"lumen_mask_{cid}.npy").astype(bool)
    zc=int(np.argwhere(lum).mean(0)[0])
    ys,xs=np.where(lum.any(0))
    y0,y1=max(ys.min()-20,0),min(ys.max()+20,191); x0,x1=max(xs.min()-20,0),min(xs.max()+20,191)
    sl=v[zc,y0:y1,x0:x1]; lm=lum[zc,y0:y1,x0:x1]
    ax.imshow(sl,cmap="gray",vmin=-100,vmax=600)
    ax.contour(lm,levels=[0.5],colors="r",linewidths=0.6)
    ax.set_title(f"case {cid}  sharp={s:.0f} CNR={c:.0f}",fontsize=10); ax.axis("off")
fig.suptitle("ImageCAS clean volumes — 9 lowest in-lumen sharpness (candidate real motion). red=GT lumen, axial @ lumen-centroid z",fontsize=11)
fig.tight_layout(); p=out/"worst9_axial.png"; fig.savefig(p,dpi=110,bbox_inches="tight")
print(f"\nrendered -> {p}")
