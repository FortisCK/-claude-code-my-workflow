# Session log — DPS on native real motion: operator-estimation line

Date: 2026-06-19
Branch: cardiac-artifacts

## Goal

Make the DPS-MC win usable on NATIVE real cardiac motion (not just synthetic /
injected motion), confronting the user's core concern that DPS may only work under
the inverse crime (same operator A used to generate AND invert synthetic data).

## What happened this session

1. Established cross-dataset external validity: DPS on real Leo anatomy + KNOWN
   injected motion = U-Net 0.731 -> DPS 0.860 (+0.129, 5/5), anti-cheat +0.126.
   (run: 2026-06-19_dps-real-leo-external-validity.md, KEEP)
2. Probed native real motion (Leo058) with blind DPS (assumed operator): FAILED —
   hallucinated texture, no-reference sharpness was a metric trap, blind DPS ≈
   feed-forward. (run: 2026-06-19_blind-dps-native-058.md, DISCARD-but-instructive)
3. User pressed the inverse-crime concern hard ("isn't DPS-with-known-A nonsense?").
   Answered honestly: NOT trivial (motion CT is ill-posed even with A; needs the
   prior) BUT the motion part of A is genuinely unknown in reality, and the synthetic
   numbers are inflated by inverse crime. The real-data value is unproven and at risk.
4. Ran a 9-agent design workflow -> plan
   (quality_reports/plans/2026-06-19_dps-native-motion-operator-estimation.md).
   KEY FINDINGS (verified vs code/logs): (a) a CONSTANT operator already recovers
   69-86% of the gain; (b) the only operator DoF that re-shapes the artifact is the
   cardiac PHASE OFFSET (was fixed at 0); (c) tomosipo A is differentiable only wrt
   image, NOT motion params, and period/phase enter via .long() (zero grad) -> learned
   regressors (strategies A/C) are broken as written. Recommended: analysis-by-synthesis
   (no training), amplitudes by Adam + period/phase by grid.

## Current task (in progress)

STAGE 0 "is there a prize?" probe (cheapest decisive gate). Added phase_offset_frac
to MotionParams (applied at the shared time->profile point in parametric_dvf, affects
synth + forward identically). Smoke confirmed phase changes the artifact (~29 HU at
0.5-cycle). Running probe_phaseoffset_prize.py (N=8): DPS Dice-RCA vs operator phase
error {0,0.25,0.5}. LOCKED rule: PRIZE = Dice(0)-Dice(0.5); >=0.04 -> build estimator
(Stage 1); <0.02 -> no estimator needed, deploy constant operator + stop.

## Honest stakes

~25-30% this whole direction yields a positive real-data result; ~70-75% a clean
honest negative/scoping result. Every branch publishable. Spend least effort first:
Stage 0 (~1.5h running) then Stage 1a self-check before any GPU-days.

## RESOLUTION (same session)

- STAGE 0: PRIZE EXISTS (+0.088) but win is FRAGILE — phase off 0.25 cycle -> DPS = U-Net.
  So the win depends entirely on an accurate operator.
- R0 (forward-model-mismatch, run BEFORE building the estimator — cheaper decisive test):
  on real Leo058 native motion, NO parametric operator beats no-motion (+0.0% residual
  reduction). Positive control confirms the test is sensitive (+18.9% U-Net / +37.5% clean
  on synthetic). => Real cardiac motion is NOT representable by our forward model; operator
  estimation cannot fix a model-class mismatch.

## CONCLUSION — DPS is a known-operator result, not a real-data de-artifacter

The user's inverse-crime instinct was correct and is now quantified. DPS-on-real-native-
motion is DEAD with this forward model. Stage 1 (ABS estimator) correctly NOT built (saved
~3 GPU-days). Reached the ~70% risk-register branch in ~1 GPU-day of gated experiments.

## PIVOT (awaiting user pick of main line)

Recommended spine = (C) characterization + UQ + eval-methodology paper, DPS reframed as the
recoverability ceiling. Assets ready: conditional-mean ceiling; task-aware Dice-RCA +
MAE-hides-downstream; metric-trap (no-ref sharpness fooled by hallucination); DPS known-op
ceiling + anti-cheat controls + R0 inverse-crime quantification; diffusion-native UQ (to build).
Optional (B) feed-forward deployable leg if clinical collaborator + reader study available.

## Uncommitted

motion_synth.py (phase_offset_frac) + scripts {probe_phaseoffset_prize, r0_forward_mismatch,
r0b_positive_control, eval_dps_leo_realdata, blind_dps_native, leo_feedforward_demo} + run
cards + plan + this log. Nothing committed yet this session.
