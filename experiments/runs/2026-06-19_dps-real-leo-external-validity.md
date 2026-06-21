# DPS-MC external validity — real clinical CCTA (Leo_anon) + known synthetic motion

Date: 2026-06-19 CEST
Author: MZ
Status: DONE — PASS (n=5, decisive; DPS win transfers to independent real dataset)
Git SHA: be17e6b (working tree: eval_dps_leo_realdata.py uncommitted)
Config: diffusion_v2_residual.yaml prior (longrun/epoch_080.pt) + U-Net init; zeta=0.02, 24 steps, full 1000 views, KNOWN per-case DVF
Seed: per-case rng(hash(cid)); DPS seed 0
Dataset: Leo_anon REAL clinical CCTA (independent of ImageCAS training set), 5 cleanest cases
Planned GPU-hours: ~0.5 (synth + DPS + 3x TS per case, serialized; coexist w/ user GPU proc)

## Intent — pre-registered cross-dataset external-validity test

The synthetic DPS win (ImageCAS test100: U-Net 0.626 -> DPS 0.815) was on the
TRAINING dataset's image statistics. Does the SAME method generalize to a SECOND,
independent REAL clinical CCTA set (Leo_anon — different scanner, patients, native
resolution)? We take the cleanest (highest in-coronary sharpness) real cases as
motion-free references, inject our KNOWN parametric motion (training distribution:
amps 10/12/10, strength~U(0.5,1.3), period~U(600,1000), 1000 views, 48 phases),
and run U-Net + DPS-MC with the known operator. Clean & corrupted both at 1mm iso
(removes the 0.25mm/1mm resolution confound of the feed-forward demo).

This tests image-generalization of the METHOD; the motion is still our model
(native-motion operator estimation remains future work). It is the strongest
real-data validation runnable today.

## LOCKED success bar (set before running)

On the selected real Leo cases (n>=5), mirroring the synthetic bar:
- PRIMARY: DPS Dice-RCA(vs TS-clean) >= U-Net + **0.05** (meaningful recovery)
- DPS Dice-RCA absolute >= **0.70**
- anti-cheat: held-out coronary_arteries_LEGACY ΔDice >= 0
- no hallucination: DPS coronary voxels <= 1.5x U-Net
PASS -> DPS generalizes to real anatomy; strong external-validity result; justifies
investing in native-motion operator estimation. FAIL/grey -> the win was
dataset-specific; reassess.

## Method

Per case: TS(total, roi_subset=heart) -> heart mask; resample vol(linear)+heart
(nearest) to 1mm iso; crop/pad 192^3 heart-centered (mirrors preprocessing) ->
clean_hu. sample_motion_params(rng) + synthesize_motion_artifact -> corrupted_hu +
hu_calibration. U-Net init -> DifferentiableMotionForward(known mp/sp/cal) ->
dps_residual_sample. TS coronary_arteries + LEGACY Dice-RCA vs clean. Render
clean|corrupted|U-Net|DPS. ALL outputs under experiments/runs/dps_real_leo/.

## Result — PASS on every locked criterion (n=5)

| case | strength | period | U-Net Dice | DPS Dice | ΔDice | LEGACY ΔDice | vox ratio |
|---|---|---|---|---|---|---|---|
| 072 | 0.70 | 775 | 0.744 | 0.881 | +0.137 | +0.149 | 1.01 |
| 031 | 0.53 | 748 | 0.831 | 0.896 | +0.065 | +0.063 | 1.07 |
| 067 | 0.97 | 693 | 0.649 | 0.837 | +0.188 | +0.187 | 1.17 |
| 012 | 1.17 | 865 | 0.738 | 0.899 | +0.160 | +0.158 | 1.17 |
| 160 | 0.90 | 992 | 0.692 | 0.786 | +0.094 | +0.075 | 1.06 |
| **mean** | | | **0.731** | **0.860** | **+0.129** | **+0.126** | **~1.10** |

- Dice-RCA(vs TS-clean): 0.731 -> 0.860, ΔDice **+0.129**, **5/5 improved**.
- heart-MAE: 80.7 -> 45.5 HU.
- anti-cheat held-out LEGACY: **+0.126**, 5/5 >= 0 -> not metric-gaming.
- no hallucination: voxels x1.10 (<= 1.5).
- DPS absolute Dice 0.860 >= 0.70. **ALL locked criteria met.**

Magnitude + direction match ImageCAS (test100 +0.19) -> the win is NOT
dataset-specific; the method generalizes to a second, independent real clinical
CCTA set's image statistics.

Panels: experiments/runs/dps_real_leo/case_{072,031,067,012,160}__panel.png

## Decision

KEEP. Cross-dataset external validity established for DPS-MC. This is the honest
"tested on real data" result for image-generalization. REMAINING gap: native real
motion (unknown operator) — addressed next by a motion-operator ESTIMATOR (trained
on synthetic labels) feeding DPS; the estimated-vs-known gap is measurable on
synthetic held-out, then qualitative/no-reference/downstream on real native-motion
cases. See blind-DPS probe (assumed operator): 2026-06-19_blind-dps-native-058.md.


## Cross-references

- Synthetic DPS win: experiments/runs/2026-06-17_dps-mc-stage1-pilot.md
- Feed-forward real demo (smooths): experiments/runs/2026-06-19_real-leo058-feedforward-demo.md
- Screen/refs: experiments/runs/leo_motion_screen/ranking.csv
- Code: scripts/python/eval_dps_leo_realdata.py
