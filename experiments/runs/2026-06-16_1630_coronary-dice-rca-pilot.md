# Coronary Dice-RCA pilot — gold-standard downstream metric (frozen TotalSegmentator)

Date: 2026-06-16 16:30 CEST
Author: MZ
Status: DONE (pilot, n=5)
Git SHA: be17e6b (working tree: lumen/dice-rca harness uncommitted)
Config: code/training/configs/diffusion_v2_residual.yaml; ckpt longrun/epoch_080.pt EMA
Seed: 42 (eval); TotalSegmentator 2.13 coronary_arteries (academic license, frozen)
Dataset version: ImageCAS-v1 test5 (cases 21,32,41,50,51) + aligned lumen GT
Plan: quality_reports/plans/2026-06-16_coronary-lumen-winning-result.md

## Purpose

Run the never-measured gold-standard downstream metric (TT U-Net's Dice-RCA
protocol; spec MUST-have novelty #3) to (a) SIZE THE PRIZE — how much coronary
segmentability the U-Net leaves unrecovered — and (b) test whether the diffusion
sample/mean beats U-Net on the downstream axis the HU-proxy could not see.

## Method

`evaluate_residual_diffusion_full_volume.py --save-volumes-dir` exports per-case
HU NIfTIs (clean/corrupted/unet/diff_mean/diff_sample, n_samples=4, 16 steps).
`coronary_dice_rca.py` runs frozen TotalSegmentator coronary_arteries on each and
scores Dice vs TS(clean) (PRIMARY, domain-gap-free) and vs ImageCAS GT lumen
(SECONDARY, TT-U-Net-style absolute; floored ~0.4 by TS<->GT convention).
De-risk: TS works on ImageCAS (clean_21: 94% of TS voxels in the GT band; Dice
vs GT 0.40 is a labelling-granularity floor, not a failure).

## Results (test5 mean)

| vtype | Dice vs TS(clean) | Dice vs GT | seg voxels |
|---|---:|---:|---:|
| clean | 1.000 | 0.435 | 1927 |
| corrupted | 0.573 | 0.427 | 2112 |
| **U-Net** | **0.656** | 0.418 | 1939 |
| diff_mean | 0.654 | 0.413 | 1856 |
| diff_sample | 0.619 | 0.399 | 1807 |

Sanity gate PASS: Dice(corrupted vs TS-clean) = 0.573 << 1.0 (segmenter is
sensitive to motion). Per-case consistent: U-Net ≈ diff_mean > diff_sample.

## What happened

Two findings:
1. **No diffusion win on the gold standard either.** diff_sample 0.619 < U-Net
   0.656; diff_mean 0.654 ties. The sharp sample's added high-frequency texture
   makes the segmenter produce a noisier, less-clean-like segmentation. So the
   naive distortion-trained diffusion does not beat U-Net even on the downstream
   axis — the proxy was not lying about that.
2. **The prize is large and was hidden by MAE.** Motion drops segmentability
   1.000 → 0.573; the U-Net recovers only 0.573 → 0.656 (i.e. +0.083 of the 0.427
   gap). **0.344 of downstream coronary Dice vs clean is unrecovered by every
   current model.** On MAE the U-Net looked ~90% recovered; on downstream
   segmentability it is only 0.656 vs clean 1.0. The downstream prize that
   novelty #3 targets is real and big — and far better-targeted than the ~0.08 HU
   global-MAE margin previously chased.

## Decision

KEEP as the gold-standard downstream instrument + the prize-sizing result.
**Conclusion: the path to a strong result is task-aware training (Step 2) — train
a model FOR this downstream/perceptual objective (differentiable segmenter-Dice
loss + adversarial + fidelity leash), judged on this Dice-RCA harness, against the
large 0.656→1.0 gap.** The distortion-trained diffusion is confirmed not the
vehicle. Caveat: part of the 0.344 gap is motion-destroyed (irreducible); n=5
pilot — firm up on test100. Guardrail Step 2 against vessel hallucination
(held-out segmenter anti-cheat, UQ check, report distortion floor).

## Cross-references

- Predecessor: `2026-06-16_1110_lumen-eval-e080.md` (HU-proxy, understated the gap)
- Re-examination that motivated this: workflow wf_3ccd3e10 (red-team / paths)
- Code: `scripts/python/coronary_dice_rca.py`, `evaluate_residual_diffusion_full_volume.py` (--save-volumes-dir)
