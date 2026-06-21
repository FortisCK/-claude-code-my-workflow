# R0 — forward-model-mismatch budget (does our A represent REAL data?)

Date: 2026-06-19 CEST
Author: MZ
Status: DONE — CONFIRMED NEGATIVE (positive-control validates test sensitivity)
Git SHA: be17e6b (working tree: r0_forward_mismatch.py + motion_synth phase_offset uncommitted)
Config: diffusion_v2_residual.yaml U-Net init; forward-only (no DPS, no training)
Dataset: real Leo_anon (072,067 clean; 058 native motion) + ImageCAS synth (21,32)
Plan: quality_reports/plans/2026-06-19_dps-native-motion-operator-estimation.md (Stage R0)

## Intent — cheapest direct test of the inverse-crime worry

DPS's win needs the operator to match the data's true degradation (Stage-0: phase
off 0.25 cycle -> DPS = U-Net). On real native motion the operator is unknown AND
our cone-beam+FDK + parametric motion may not match the real scanner. R0 tests
representability directly, forward-only:
- R0a: static round-trip ||A_nomotion(x)-x||_heart, real-clean vs synth-clean.
- R0b: can ANY parametric motion operator (grid phase x amp) explain real 058's
  native artifact better than no-motion? reduction = (r_nomo - r_best)/r_nomo.

## LOCKED reading (set before running)

R0b residual reduction vs no-motion:
- >= 20% -> real artifact partly parametric-capturable -> operator estimation worth building.
- 8-20% -> weak/marginal.
- < 8%  -> real motion NOT in our family -> DPS-on-real likely dead; bottleneck is
  forward/motion-model fidelity, not estimation -> pivot to feed-forward (B) or
  characterization+UQ+eval (C).

## Result

R0a static round-trip ||A_nomotion(x)-x||_heart: synth 21=43.3, 32=54.7 HU; real
072=64.1, 067=57.0 HU. -> real ~ synth (static model handles real comparably; but the
round-trip error itself is ~50-64 HU, large vs the artifact).

R0b real Leo058 native motion: artifact ||xhat-y||=102.4 HU; no-motion residual
||A_nomotion(xhat)-y||=120.9 HU; EVERY parametric motion operator (phase x amp grid)
gives >= 120.9 HU (121-145, monotonically worse with amplitude). BEST = no-motion.
**Residual reduction vs no-motion: +0.0%.**

Reading: no parametric motion operator explains real 058's artifact. The forward-model
round-trip mismatch (~60-120 HU) is >= the artifact (~102 HU), and our parametric motion
adds error rather than removing it. This QUANTIFIES the inverse crime: synthetic y=A(clean)
bakes the operator distortion into both y and A(x) (cancels at x=clean); real y has none of
our operator's distortion, so A(x) can never match it -> DPS data-consistency is inert on real.

CAVEAT: R0b residual is partly dominated by round-trip error, which could mask a weak motion
signal -> running synthetic positive-control (does the SAME grid detect motion when present?)
to confirm the test is sensitive before acting.

## Positive control (r0b_positive_control.py) — test IS sensitive

Synthetic case 21, known motion phase=0.3, SAME R0b grid:
- xhat=clean : no-motion 61.0 -> best 38.1 HU, reduction **+37.5%**
- xhat=U-Net : no-motion 62.3 -> best 50.5 HU, reduction **+18.9%**
vs real 058: **+0.0%**. The grid detects motion when present (incl. via a U-Net estimate,
matching R0b conditions) -> the real null is NOT a test/xhat artifact.

## Decision — CONFIRMED. DPS-on-real-native-motion is DEAD with this forward model.

The DPS win is a genuine KNOWN-OPERATOR result (inverse-problem contribution +
recoverability ceiling) but does NOT transfer to real native motion: real cardiac
motion is not representable by our cone-beam+FDK+parametric-DVF model, and operator
estimation cannot fix a model-class mismatch. This is the ~70% branch the plan's risk
register predicted (forward-model fidelity dominates), reached in ~1 GPU-day of gated
experiments instead of weeks of estimator-building.

CLOSE the operator-estimation line (Stage 1 NOT built — correctly avoided). PIVOT to:
(C) characterization + UQ + eval-methodology paper [recommended spine], with DPS reframed
as the recoverability ceiling; optionally (B) a feed-forward deployable leg if a clinical
collaborator + reader study become available. See session log for the synthesis.


## Cross-references

- Stage 0: experiments/runs/2026-06-19_phaseoffset-prize-probe.md
- Plan: quality_reports/plans/2026-06-19_dps-native-motion-operator-estimation.md
- Code: scripts/python/r0_forward_mismatch.py
