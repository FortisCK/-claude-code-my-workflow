# DPS-MC Stage-1 pilot — measurement-consistency posterior sampling, test5

Date: 2026-06-17 CEST
Author: MZ
Status: DONE — PASS (decisive)
Git SHA: 8ee8e7f (working tree: dps_forward/dps_sample/eval_dps_test5 uncommitted)
Config: diffusion_v2_residual.yaml prior (longrun/epoch_080.pt) + U-Net init; zeta=0.02, 24 steps, full 1000 views, KNOWN per-case DVF
Seed: 0; Dataset: ImageCAS-v1 test5 (21,32,41,50,51)
Plan: quality_reports/plans/2026-06-17_stage1-dps-mc-pilot.md

## Intent (pre-registered bar)

Does measurement-consistency posterior sampling (diffusion prior x known forward
operator) raise frozen-segmenter Dice-RCA above U-Net toward the recoverable
ceiling on test5? LOCKED bar: Dice-RCA ≥ 0.70, no streaks, LEGACY anti-cheat
ΔDice ≥ 0, coronary voxels ≤ 1.5x.

## Method (Stages 0-4, all green)

S0: tomosipo.to_autograd makes project/backproject differentiable; ∇‖A(x)−y‖²
flows, 5 GB @250 views / 22.5 GB @1000 views. S1: DifferentiableMotionForward
(warp per phase -> subset/full-view project -> Parker+ramp -> backproject ->
stored hu_calibration); A(clean)=corrupted EXACTLY (0.0 HU) at full 1000 views.
S2: DiffPIR-style guided sampler (denoise no-grad -> DC grad on Tweedie x̂₀ via A
backward only -> Euler step); runs, 23 GB. S3: zeta sweep -> sweet spot 0.02
(heart-MAE 48->25.5, dc_loss 5731->73; >=0.1 diverges). S4: test5 Dice-RCA.

## Result (test5, zeta=0.02) — PASS on every criterion

| metric | U-Net | DPS-MC | Δ |
|---|---:|---:|---:|
| Dice-RCA (vs TS-clean) | 0.632 | **0.808** | **+0.176** |
| Dice vs GT | 0.395 | 0.468 | +0.073 |
| heart-MAE (HU) | 57.1 | 29.6 | -27.5 |
| anti-cheat LEGACY ΔDice | — | — | +0.196 |
| coronary voxels ratio | — | — | x1.20 |

Per-case ΔDice: 21 +0.292, 32 +0.109, 41 +0.094, 50 +0.174, 51 +0.212 (5/5).
DPS reaches ~the clean-paste oracle ceiling (0.803-0.822). The held-out LEGACY
segmenter (in no loss) confirms the gain -> not metric-gaming.

## What happened

The measurement-consistency mechanism works decisively: with the exact per-case
DVF, ∇‖A(x)−y‖² concentrates corrective signal on the high-contrast coronary
lumen (highest projection residual), and the diffusion prior keeps the solution a
realistic clean vessel. This is the first method this project has produced that
beats U-Net on the gold-standard downstream metric — and it nearly hits the
absolute oracle.

## Decision

KEEP. DPS-MC validated as the headline method (novelty #1 strongest form:
physics-grounded diffusion posterior sampling; diffusion is the method, U-Net is
just the warm-start). PASS -> Stage 2.

**Critical honesty:** 0.808 is the KNOWN-DVF (perfect-operator) result. The
deployable setting has no true DVF; the estimated-DVF (semi-blind) number is
expected lower (~0.70-0.76 per the design panel) and is the key Stage-2 gate.
n=5 -> confirm on test100.

## Stage-2 update: test100 confirmation + mechanism controls

**test100 known-DVF (n=100) — the headline holds at scale:**
Dice-RCA(vs TS-clean) U-Net 0.626 -> DPS **0.815** (ΔDice **+0.189**, median +0.192,
std 0.092). **99/100 improved** (1 case -0.003 = noise); **94/100 reach >=0.70**;
heart-MAE 51.4 -> 27.2 HU; Dice vs GT 0.355 -> 0.449. Independent held-out
segmenter (coronary_arteries_LEGACY, in no loss): **100/100 improved, +0.217** ->
not metric-gaming, confirmed at scale.

**Mechanism controls (test5) — the gain is genuinely correct-operator data-consistency:**

| setting | Dice-RCA | reading |
|---|---:|---|
| U-Net | 0.632 | baseline |
| DPS zeta=0 (no guidance) | 0.610 | ~U-Net -> NOT the prior; guidance is essential |
| DPS wrong operator (2x amplitude) | 0.617 | ~U-Net -> wrong physics = no gain |
| DPS population-mean operator (~true, amps fixed) | 0.796 | exact DVF adds only +0.012 |
| DPS true DVF | 0.808 | the win |

Controls confirm: the +0.16-0.19 gain flows through data-consistency with a
correct(-ish) operator, NOT the prior, NOT a clean-leak (DPS never sees clean),
NOT generic sharpening. The exact per-case DVF contributes only +0.012 over a
population-mean operator -> the "privileged info" worry is small IN THIS narrow
synthetic motion distribution (amplitudes fixed). Deployability hinges on operator
specification; a 2x-wrong operator collapses to U-Net. Real (variable-amplitude)
motion + per-case DVF estimation is the open Stage-2 question.

## Cross-references

- Prize-sizing oracle: experiments/runs/2026-06-16_1630_coronary-dice-rca-pilot.md (+ coronary_recovery_oracle 0.82)
- Plan: quality_reports/plans/2026-06-17_stage1-dps-mc-pilot.md ; parent path-a plan
- Code: code/inference/dps_forward.py, code/inference/dps_sample.py, scripts/python/eval_dps_test5.py
- Next (Stage 2): test100 known-DVF + estimated-DVF deployable number + DiffPIR stability
