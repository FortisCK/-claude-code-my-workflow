# Real-data demo — feed-forward de-artifacting on Leo058 clinical CCTA

Date: 2026-06-19 CEST
Author: MZ
Status: DONE — EXPLORATORY, negative-ish (feed-forward smooths, does not de-artifact)
Git SHA: be17e6b (working tree: leo_feedforward_demo.py uncommitted)
Config: diffusion_v2_residual.yaml + unet_v1 initializer
Seed: 42
Dataset: Leo_anon case 058 (real clinical CCTA, NOT ImageCAS) — READ ONLY from ~/Code
Planned GPU-hours: <0.2

## Intent (EXPLORATORY — qualitative, no GT)

The user picked Leo058 (one of the 12 lowest-in-coronary-sharpness real CCTA
candidates) and asked to SEE what our model does to it. **DPS proper cannot run
on real data** (needs the known per-case motion operator A = DVF+geometry+HU-cal,
which we do not have on real scans). So this runs the **feed-forward half** of the
pipeline only: U-Net initializer + residual-EDM posterior refiner (no
measurement-consistency guidance).

Falsifiable expectation: trained on ImageCAS *synthetic* motion @1mm iso, applied
to a real Leo CCTA (0.25/0.75mm, resampled to 1mm). Either (a) it produces a clean,
plausible coronary volume (encouraging generalization, paper-figure candidate with
caveats), or (b) domain shift wrecks it (streaks / blur / hallucination) — also
important to know. No GT → visual + no-reference only.

## Method

Leo058 .nii (168,512,512)@(0.25,0.25,0.75) → resample 1mm iso (linear; seg
nearest) → clip+normalize HU[-1024,3071]→[-1,1] → sliding_window_residual_diffusion
(roi128, overlap0.5, 32 steps, n_samples=4 → posterior mean + std). Render axial
@ coronary centroid, zoomed to coronary bbox, window -100..700 HU.
ALL outputs under experiments/runs/leo_dps_demo/058/ (cardiac-artifacts, NOT ~/Code).

## Result

Resampled to (126,128,128) 1mm iso; peak GPU 13.4 GB; n_samples=4.
Panel: experiments/runs/leo_dps_demo/058/case_058__panel.png
NIfTI: case_058__{unet,diff_mean}.nii.gz

- RUNS cleanly on real data: no streaks, no hallucinated vessels, no NaN. The
  trained nets generalize enough not to explode. (only clear positive.)
- But output is SMOOTHER than input — feed-forward smooths, does not sharpen /
  de-artifact. No targeted coronary correction visible.
- diffusion−corrupted diff map is large + DIFFUSE (global HU/contrast shift, blue
  rim at heart border, red in chambers) — domain-shift signature, not a localized
  motion fix.
- posterior std behaves sensibly (high at boundaries + bright structures).

**Resolution-gap caveat (important):** Leo058 is native 0.25mm in-plane; the model
lives at 1mm (ImageCAS processed at 1mm). Running it required a 4x downsample,
which itself blurs everything (coronary ~12px -> ~3px). So the "before" is already
a degraded 1mm image; this is a 1mm comparison, NOT clinical-resolution. The
training(1mm)/clinical(0.25mm) resolution gap is a real domain problem for any
real-data story with these weights.

## Decision

DISCARD as a paper figure (smoothing, not de-artifacting). KEEP as honest evidence
that (a) the feed-forward path ≈ U-Net-class smoothing on real data, and (b) the
DPS win cannot transfer to real scans without motion-operator estimation +
addressing the 1mm/0.25mm resolution gap. Informs the real-data strategy decision:
the real-data leg is not free; it needs either operator estimation or a reframing.


## Cross-references

- Screen: experiments/runs/leo_motion_screen/ (ranking.csv, worst12.png)
- DPS (synthetic, the real win): experiments/runs/2026-06-17_dps-mc-stage1-pilot.md
- Code: scripts/python/leo_feedforward_demo.py
