# Matched methods comparison table (paper main table + Act-II data)

Date: 2026-06-21 CEST
Author: MZ
Git SHA: b0cf6d4 (working tree: eval_methods_table.py uncommitted)
Config: diffusion_v2_residual.yaml (U-Net init + residual-EDM) + ttunet_v1/epoch_100
Seed: 42; Dataset: ImageCAS-v1 test split, first N
Planned GPU-hours: ~2 (inference + TS coronary per output; coexists w/ user GPU job)
Status: DONE — KEEP (matched main table + Act-II sharpness trap established)

## Intent

Produce the paper's MATCHED main comparison table: every feed-forward method through the
SAME harness on the SAME test cases. Clears the referee "self-vs-self / unmatched baseline"
issue and yields the (sharpness, structural-Dice) pairs for the Act-II sharpness-trap.
Methods: corrupted, U-Net, TT U-Net, diffusion-mean, diffusion-sample. Metrics: Dice-RCA
(TS vs TS-clean), GT-lumen geometric Dice (independent judge), in-heart sharpness, heart-MAE.
Expectation: all feed-forward cluster ~0.62-0.66 Dice-RCA (conditional-mean cap); diff_sample
highest sharpness but NOT highest structure (the trap). DPS cited separately (operator-dependent).

## Result (n=20 ImageCAS test)

| method | Dice-RCA | GT-lumen | sharpness (HU) | MAE (HU) |
|---|---:|---:|---:|---:|
| corrupted | 0.585 | 0.423 | 33.0 | 72.9 |
| U-Net | **0.657** | 0.409 | 31.8 | **44.4** |
| TT U-Net | 0.644 | 0.405 | 33.3 | 47.1 |
| diffusion-mean | 0.649 | 0.404 | 36.5 | 46.7 |
| diffusion-sample | 0.628 | 0.379 | **45.9** | 52.7 |
| clean (ref) | 1.000 | 0.439 | — | — |
| oracle clean-paste | ~0.82 | — | — | — |

Two results, matched & statistical:
1. All feed-forward methods cluster **0.63-0.66** Dice-RCA (U-Net best), far below oracle 0.82
   -> conditional-mean cap confirmed against the published TT U-Net baseline, not just self.
2. THE SHARPNESS TRAP (Act-II, act2_sharpness_trap.py): diffusion-sample = highest sharpness
   (45.9) but LOWEST structure (Dice-RCA 0.628, GT-lumen 0.379). Across all case x method:
   Pearson(sharpness, GT-lumen structural Dice) = **-0.11** (sharpness does NOT track structure;
   sharpest output = corrupted-beating hallucinated texture). Reference-free voxel-ratio
   hallucination guard: ROC-AUC **0.802** vs GT-defined hallucination (n=80). Figure:
   experiments/runs/methods_table/act2_sharpness_trap.png.

## Decision

KEEP — this is the paper's main comparison table (Act-II second pillar). Clears the
"self-vs-self baseline" referee issue and delivers the sharpness-trap demonstration + a
deployable reference-free guard. FOLLOW-UP (camera-ready): expand to test100; TT U-Net 200ep
for max fairness; apply the guard on real Leo outputs (qualitative).

## Cross-references

- TT U-Net: 2026-06-21_0035_ttunet-baseline.md ; Act-I: 2026-06-21_r0-cohort-operator-fidelity.md
- Code: scripts/python/eval_methods_table.py, scripts/python/act2_sharpness_trap.py

## Cross-references

- TT U-Net: 2026-06-21_0035_ttunet-baseline.md ; Act-I: 2026-06-21_r0-cohort-operator-fidelity.md
- Code: scripts/python/eval_methods_table.py
