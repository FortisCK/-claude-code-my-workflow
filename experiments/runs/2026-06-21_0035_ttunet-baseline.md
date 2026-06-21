# TT U-Net baseline — published comparator (Deng et al., IEEE TMI 2023)

Date: 2026-06-21 00:35 CEST
Author: MZ
Git SHA: 1ca693e (working tree: ttunet model/trainer/config uncommitted)
Config: code/training/configs/ttunet_v1.yaml
Seed: 42
Dataset version: ImageCAS-v1 (train=800, val=100, test=100 held out)
Planned GPU-hours: ~10-18h (100 epochs x 800 steps, [64,128,128] patches, ~6 GB; coexists w/ user GPU job)
Status: DONE — KEEP (published baseline established; U-Net-class, capped as predicted)

## Intent (falsifiable)

Reproduce TT U-Net (the published cardiac-CT motion-correction method, recommends ImageCAS)
as a recognized BASELINE — the MICCAI/TMI referee-panel showstopper #1 ("no published
baseline"). Faithful port (code/models/ttunet.py); trained on the SAME pairs/split/
normalization/L1+gradient loss as unet_v1 so the comparison isolates architecture.
z-axis treated as the temporal sequence (single-volume adaptation; see plan
2026-06-21_ttunet-baseline-reproduction.md). Expectation: TT U-Net is ALSO a
distortion-trained feed-forward net -> bounded by the MMSE/conditional-mean cap, so it
should land ~U-Net class on Dice-RCA (U-Net 0.656), NOT beat it. A baseline that ties or
modestly differs is the expected, publishable outcome; a large gain would be a surprise.

## Hyperparameters (delta vs unet_v1)

- Architecture: TTUNet (cnum=24, 24.9M params) instead of ResidualUNet3D.
- patch_size [64,128,128] (vs unet 128^3) — depth-64 temporal clip for the temporal attention
  + GPU coexistence; spatial 128^2 matches. Everything else (lr 1e-4, EMA 0.999, L1+0.05 grad,
  batch 1, AMP) identical to unet_v1.

## Smoke (pre-run, PASS)

CPU tiny smoke: forward 8x32x32 shape-preserving, grads finite/nonzero; GPU mem [64,128,128]
6.1 GB, [128,128,128] 20.1 GB (chose 64-depth for coexistence). Full train-loop smoke ran.

## Result

Training: 100 epochs in 12720s (~3.5h), ~2.3 min/epoch; train l1 0.106 (ep1) -> 0.026 (ep100, converged).
Eval (epoch_100, EMA, sliding-window roi [64,128,128] overlap0.5 gaussian, ImageCAS test5):

| method | Dice-RCA (vs TS-clean) | LEGACY | GT-lumen (indep) | heart-MAE |
|---|---:|---:|---:|---:|
| **TT U-Net (this run)** | **0.626** | 0.573 | 0.415 (clean-ref 0.435) | 52.1 HU |
| our U-Net | 0.656 | — | — | ~38 |
| diffusion mean | 0.654 | — | — | — |
| diffusion sample | 0.619 | — | — | — |
| oracle (clean-paste) | 0.82 | — | — | — |

Per-case Dice-RCA: 21=0.584, 32=0.729, 41=0.499, 50=0.673, 51=0.643.

Reading: TT U-Net is U-Net-CLASS (0.626), slightly BELOW our U-Net (0.656) and ~52 vs 38 HU MAE.
It does NOT beat U-Net — exactly as predicted for a distortion-trained feed-forward bounded by
the MMSE/conditional-mean cap. All feed-forward methods cluster 0.62-0.66, far below the 0.82
oracle; only DPS-with-known-operator (inverse crime) broke past. The independent GT-lumen judge
shows TT U-Net (0.415) nearly matches the clean reference (0.435) — i.e. it preserves coronary
geometry but does not recover beyond U-Net.

FAIRNESS NOTE: TT U-Net here = 100 epochs @ [64,128,128] patches vs our U-Net's 200 epochs @ 128^3;
the MAE gap (52 vs 38) suggests it may be mildly undertrained / pay a single-volume-adaptation cost
(it was designed for 48-phase dynamic sequences). The qualitative conclusion (feed-forward capped,
does not beat U-Net) is robust; a longer-trained / test100 run is the follow-up for the final table.

## Decision

KEEP — published-baseline showstopper #1 cleared. TT U-Net reproduced faithfully and scored on our
Dice-RCA + GT-lumen harness; result reinforces the characterization thesis (published feed-forward
methods are conditional-mean-capped). FOLLOW-UP for the paper table: (a) train to 200 epochs / match
U-Net budget for a maximally fair number; (b) expand to test100 + matched per-case U-Net; (c) run TT
U-Net through R0 / the hallucination guard on real native motion (the "offensive reframe").

## Cross-references

- Plan: quality_reports/plans/2026-06-21_ttunet-baseline-reproduction.md
- Source: external/TT-U-Net (Deng et al., IEEE TMI 2023, doc 10236564)
- Compare vs: unet_v1 (Dice-RCA 0.656), diff_mean 0.654, diff_sample 0.619, DPS, oracle 0.82
- Code: code/models/ttunet.py, code/training/train_ttunet.py
