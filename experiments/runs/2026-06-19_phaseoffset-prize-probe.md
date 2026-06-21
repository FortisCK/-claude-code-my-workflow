# Stage-0 — phase-offset "is there a prize?" probe

Date: 2026-06-19 CEST
Author: MZ
Status: DONE — PRIZE EXISTS (+0.088); but win is fragile to operator error
Git SHA: be17e6b (working tree: motion_synth phase_offset + probe_phaseoffset_prize.py uncommitted)
Config: diffusion_v2_residual.yaml prior (longrun/epoch_080.pt) + U-Net init; zeta=0.02, 24 steps, full 1000 views
Seed: 0 (per-case phi_true ~ U(0,1) via rng(0))
Dataset: ImageCAS-v1 test split, first 8 cases
Plan: quality_reports/plans/2026-06-19_dps-native-motion-operator-estimation.md (STAGE 0)

## Intent — the decisive cheapest gate

The DPS win on synthetic uses the EXACT generating operator (inverse crime). A
constant (pop-mean) operator already recovers 69-86% of the gain -> the only
operator DoF that RE-SHAPES (not just scales) the artifact is the cardiac PHASE
OFFSET. Question: does getting the phase WRONG materially hurt DPS? If NOT, no
operator estimator is load-bearing (deploy a constant operator, stop). If YES,
estimating phase is worth building (Stage 1).

Method: synth each case with a known random phase_offset phi_true (amps/period
FIXED -> phase is the only varied DoF); run DPS with operator phase error Delta in
{0, 0.25, 0.5} cycle; measure TS coronary Dice-RCA + heart-MAE vs Delta.

## LOCKED verdict rule (set before running)

PRIZE = mean Dice-RCA(Delta=0) - mean Dice-RCA(Delta=0.5).
- PRIZE >= 0.04 -> prize exists -> proceed to Stage 1 (build the ABS operator estimator).
- 0.02-0.04 -> weak, inspect per-case spread.
- < 0.02 -> NO prize -> DPS robust to phase -> deploy constant operator, STOP (publishable
  "DPS robust to operator misspecification within a realistic motion family").

## Result (n=8)

| arm | Dice-RCA | heart-MAE |
|---|---:|---:|
| U-Net floor | 0.652 | 75 |
| DPS phase err 0.00 (correct) | 0.815 | 31 |
| DPS phase err 0.25 | 0.660 | 66 |
| DPS phase err 0.50 | 0.727 | 56 |

PRIZE (Dice 0.00→0.50) = **+0.088** ≥ 0.04 → PRIZE EXISTS.
Sharper: **phase err 0.25 → DPS ≈ U-Net (0.660 vs 0.652)** — the win is FRAGILE to
operator error; the operator must be accurate to ~0.1 cycle to keep most of the gain.
Non-monotonic (0.25 worse than 0.50: anti-phase profile is partially symmetric →
data-consistency landscape has structure; grid search must be global, watch half-cycle aliasing).

## Decision

PROCEED — but reordered. Two consequences: (1) an operator estimator IS load-bearing
(prize confirmed) → Stage 1 justified; (2) the win depends ENTIRELY on accurate phase →
this strongly sharpens the inverse-crime concern. Because Gap-1 (estimate operator on
synthetic) is itself inverse-crime-favored (the data-consistency loss is minimized at the
true operator by construction → likely passes without proving real value), the decisive
cheap test of the user's worry is **R0 (real-data forward-model-mismatch budget)** —
recommend running R0 BEFORE building the Stage-1 estimator.

## Cross-references

- Plan: quality_reports/plans/2026-06-19_dps-native-motion-operator-estimation.md
- Code: code/data/motion_synth.py (phase_offset_frac), scripts/python/probe_phaseoffset_prize.py


## Cross-references

- Plan: quality_reports/plans/2026-06-19_dps-native-motion-operator-estimation.md
- Inverse-crime concern this addresses; external validity 2026-06-19_dps-real-leo-external-validity.md
- Code: code/data/motion_synth.py (phase_offset_frac), scripts/python/probe_phaseoffset_prize.py
