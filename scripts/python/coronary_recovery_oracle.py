"""coronary_recovery_oracle.py — is the downstream coronary Dice gap recoverable?

Ceiling test: take the U-Net output, paste the CLEAN HU into a dilated coronary
band (= "a method that perfectly fixed only the coronary region"), segment with
frozen TS coronary_arteries, and measure Dice-RCA vs TS(clean). If oracle Dice
approaches clean (high), the prize is reachable -> a generative method is worth
building. If it stays ~U-Net level, the segmenter/resolution/global-context caps
it -> chasing Dice is futile. Reuses dice_rca_pilot volumes + TS-clean segs.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import numpy as np, SimpleITK as sitk
from scipy.ndimage import binary_dilation
from code.data import paths as path_registry
from code.data.splits import load_split

TS = Path(sys.prefix)/"bin"/"TotalSegmentator"
VOL = Path("experiments/runs/dice_rca_pilot/volumes")
CLEAN_SEG = Path("experiments/runs/dice_rca_pilot/result/seg")
OUT = Path("experiments/runs/coronary_recovery_oracle"); OUT.mkdir(parents=True, exist_ok=True)
lumen_dir = path_registry.get("IMAGECAS_PROCESSED").parent/"lumen_masks"

def seg(nifti, od):
    f = od/"coronary_arteries.nii.gz"
    if not f.exists():
        od.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(TS),"-i",str(nifti),"-o",str(od),"-ta","coronary_arteries"],check=True,capture_output=True,text=True)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(f)))>0
def dice(a,b,e=1e-8): return (2*float(np.logical_and(a,b).sum())+e)/(float(a.sum())+float(b.sum())+e)
def save(arr,p): 
    im=sitk.GetImageFromArray(arr.astype(np.int16)); im.SetSpacing((1.,1.,1.)); sitk.WriteImage(im,str(p))

cases=[str(c) for c in load_split(path_registry.get("IMAGECAS_SPLITS")/"v1.json").test[:5]]
rows=[]
for cid in cases:
    clean=sitk.GetArrayFromImage(sitk.ReadImage(str(VOL/f"case_{cid}__clean.nii.gz")))
    unet=sitk.GetArrayFromImage(sitk.ReadImage(str(VOL/f"case_{cid}__unet.nii.gz")))
    lum=np.load(lumen_dir/f"lumen_mask_{cid}.npy").astype(bool)
    sc=seg(VOL/f"case_{cid}__clean.nii.gz", CLEAN_SEG/f"case_{cid}__clean")
    su=seg(VOL/f"case_{cid}__unet.nii.gz", CLEAN_SEG/f"case_{cid}__unet")
    row={"case":cid,"unet":dice(su,sc)}
    for r in (3,6):
        band=binary_dilation(lum,iterations=r)
        orac=unet.copy(); orac[band]=clean[band]
        p=OUT/f"case_{cid}__oracle_r{r}.nii.gz"; save(orac,p)
        so=seg(p, OUT/f"seg_case_{cid}__oracle_r{r}")
        row[f"oracle_r{r}"]=dice(so,sc)
    rows.append(row); print(f"case {cid}: unet={row['unet']:.3f}  oracle_r3={row['oracle_r3']:.3f}  oracle_r6={row['oracle_r6']:.3f}",flush=True)
import numpy as _np
print("\n==== CORONARY RECOVERY ORACLE (Dice-RCA vs TS-clean) ====")
for k in ("unet","oracle_r3","oracle_r6"):
    print(f"  {k:<12} {_np.mean([r[k] for r in rows]):.3f}")
print("\n  reading: oracle >> unet -> coronary region IS the bottleneck, prize reachable;")
print("           oracle ~ unet     -> capped by segmenter/resolution, chasing Dice futile.")
