"""screen_leo_motion.py — find the most motion-blurred coronary cases in the Leo_anon
clinical CCTA set. READS ~/Code/Leo_anon (read-only); WRITES only under cardiac-artifacts.
Locates coronaries via frozen TS coronary_arteries, ranks by low in-coronary sharpness."""
from __future__ import annotations
import sys, glob, subprocess, csv
from pathlib import Path
import numpy as np, SimpleITK as sitk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TS=Path(sys.prefix)/"bin"/"TotalSegmentator"
OUT=Path("/home/mingzhang/cardiac-artifacts/experiments/runs/leo_motion_screen"); (OUT/"seg").mkdir(parents=True,exist_ok=True)
def gm(a):
    gz,gy,gx=np.gradient(a); return np.sqrt(gz*gz+gy*gy+gx*gx)
niis=sorted(glob.glob("/home/mingzhang/Code/Leo_anon/*/*.nii"))
print(f"screening {len(niis)} Leo_anon .nii cases (TS coronary localization)",flush=True)
rows=[]
for i,f in enumerate(niis):
    cid=f.split("/")[-2]; od=OUT/"seg"/cid
    try:
        sf=od/"coronary_arteries.nii.gz"
        if not sf.exists():
            subprocess.run([str(TS),"-i",f,"-o",str(od),"-ta","coronary_arteries"],check=True,capture_output=True,text=True,timeout=600)
        seg=sitk.GetArrayFromImage(sitk.ReadImage(str(sf)))>0
        if seg.sum()<3000: continue
        v=sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32)
        if v.shape!=seg.shape: continue
        sp=gm(v)[seg].mean()
        rows.append((cid,float(sp),int(seg.sum())))
    except Exception as e:
        print(f"  {cid} skip: {str(e)[:60]}",flush=True); continue
    if (i+1)%20==0: print(f"  ...{i+1}/{len(niis)}",flush=True)
rows.sort(key=lambda r:r[1])
alls=np.array([r[1] for r in rows])
print(f"\nscreened {len(rows)} (coronary>=3000 vox). sharpness min={alls.min():.0f} median={np.median(alls):.0f} max={alls.max():.0f}")
print("WORST 12 (lowest in-coronary sharpness = most blurred):")
worst=rows[:12]
for r,(cid,s,nv) in enumerate(worst): print(f"  {r+1:<3}case {cid}  sharp={s:.0f}  vox={nv}")
with (OUT/"ranking.csv").open("w",newline="") as fp:
    w=csv.writer(fp); w.writerow(["case","sharpness","coronary_vox"]); w.writerows(rows)
# render worst 12: axial @ coronary centroid, zoom to coronary bbox, hi-res
fig,axes=plt.subplots(3,4,figsize=(16,12))
for ax,(cid,s,nv) in zip(axes.ravel(),worst):
    f=glob.glob(f"/home/mingzhang/Code/Leo_anon/{cid}/*.nii")[0]
    v=sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32)
    seg=sitk.GetArrayFromImage(sitk.ReadImage(str(OUT/"seg"/cid/"coronary_arteries.nii.gz")))>0
    zc,yc,xc=[int(c) for c in np.argwhere(seg).mean(0)]
    ys,xs=np.where(seg.any(0)); m=50
    y0,y1=max(ys.min()-m,0),min(ys.max()+m,v.shape[1]); x0,x1=max(xs.min()-m,0),min(xs.max()+m,v.shape[2])
    ax.imshow(v[zc,y0:y1,x0:x1],cmap="gray",vmin=-100,vmax=700)
    ax.contour(seg[zc,y0:y1,x0:x1],levels=[0.5],colors="r",linewidths=0.5,alpha=0.6)
    ax.set_title(f"case {cid} sharp={s:.0f}",fontsize=9); ax.axis("off")
fig.suptitle("Leo_anon clinical CCTA — 12 lowest in-coronary sharpness (most motion-blurred candidates). red=TS coronary",fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"worst12.png",dpi=115,bbox_inches="tight")
print(f"\nrendered -> {OUT/'worst12.png'}")
