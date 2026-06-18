# Stage-1 plan — DPS-MC test5 pilot (measurement-consistency posterior sampling)

Status: ACTIVE (Stage 0 PASSED: differentiable forward works, ~5 GB @250 views)
Date: 2026-06-17
Parent: quality_reports/plans/2026-06-16_path-a-downstream-aware-generative.md (C1, DPS headline)

## Pre-registered question

Does measurement-consistency posterior sampling (diffusion prior x known forward
operator) recover coronary structure that raises frozen-segmenter Dice-RCA above
the U-Net (0.656) toward the recoverable ceiling, on test5 — WITHOUT streak
artifacts or hallucination? This is the only method with a mechanism to exceed
the conditional-mean cap; the pilot decides whether to scale it.

## LOCKED success bar (set before running, not movable)

On test5 (cases 21,32,41,50,51; oracle ceiling 0.822, U-Net 0.656):
- PRIMARY: Dice-RCA(vs TS-clean) ≥ **0.70** (meaningful: ~1/3 of recoverable gap)
- no streaks: high-frequency energy outside the heart not inflated vs U-Net
- anti-cheat: ΔDice on held-out coronary_arteries_LEGACY ≥ 0
- no hallucination: coronary voxel count ≤ 1.5x U-Net
- sanity: global MAE not catastrophically worse (≤ corrupted)
PASS → Stage 2 (test100 + DiffPIR stability + estimated-DVF deployable number).
0.66-0.70 grey → try DiffPIR/ΠGDM proximal DC (more stable) before deciding.
< 0.66 → measurement signal too weak (motion-destroyed) → pivot to UQ + the
characterization paper (eval protocol + MAE-hides-downstream + why-it-caps).

## Build (each sub-step gets a 1-case smoke before scaling)

S1. **Differentiable multi-phase forward** `A_diff(x, motion_params, scan_params)`
    in code/data/ (new fn or differentiable variant of synthesize_motion_artifact):
    warp x by each phase DVF (warp_volume, differentiable) → project subset views
    per phase via to_autograd(ts.operator) → Parker + ramp → to_autograd backproject.
    Use stored per-case motion_params/scan_params; FREEZE stored hu_calibration
    (affine, no second baseline pass). Subset views (~250) ± subset phases for
    memory. Smoke: grad d||A_diff(x)-y||^2/dx finite+nonzero, mem < ~15 GB.
S2. **DPS guidance sampler** (new, e.g. code/inference/dps_sample.py): reuse the
    ResidualEDM/EDM Heun reverse loop; per step compute Tweedie x̂₀ = D_θ(x,σ),
    then x ← x − ζ_t·∇_x ||A_diff(x̂₀) − y||² (y = observed corrupted volume);
    warm-start from U-Net output; anneal ζ_t (small at high σ), clamp guidance
    norm per step (anti-streak). Prior = converged residual-EDM longrun/epoch_080.
    Smoke (1 case): runs, recon finite, no NaN, visually not pure streaks.
S3. **Quick tune** (1-2 cases): ζ_t scale, #steps, subset size — get a sane recon.
S4. **test5 eval**: DPS on 5 cases → save HU volumes → coronary_dice_rca.py
    (coronary_arteries + LEGACY) vs the LOCKED bar → verdict. Run card.

## Stop points

- Any hard blocker (OOM that subsetting can't fix; sampler diverges irreparably).
- The test5 pilot verdict (PASS / grey / FAIL) — STOP and report; do NOT auto-scale
  to test100 or the full build before user OK.

## Constraints / risks

- Coexist with the running GPU process (subset-view ~5-8 GB regime; never crowd it).
- DPS instability (streaks) — mitigate: clamp guidance norm, σ-anneal ζ, fallback
  to DiffPIR/ΠGDM proximal data-consistency if vanilla DPS is twitchy.
- Honest ceiling ~0.74-0.79 (known-DVF); bar 0.70 is a meaningful-signal threshold,
  not the ceiling. n=5 is noisy — confirm on test100 in Stage 2 if it passes.
- The deployable (estimated-DVF) number is a Stage-2 concern, not a pilot blocker.

## Reuse / new

Reuse: motion_synth (warp/Parker/ramp/geometry), to_autograd (Stage-0 verified),
residual-EDM prior, U-Net init, coronary_dice_rca harness, dice/TS plumbing.
New: A_diff (differentiable multi-phase forward), dps_sample (guided sampler),
a Stage-1 run card.
