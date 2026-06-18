# Lumen refiner Pilot-A — task-aware (lumen soft-Dice) refiner, test5 verdict

Date: 2026-06-16 17:32 CEST
Author: MZ
Status: DONE — FAIL (catastrophic / degenerate)
Git SHA: 8ee8e7f (working tree: train_lumen_refiner + eval uncommitted)
Config: code/training/train_lumen_refiner.py (CLI: train-cases 100, epochs 30,
  lumen-weight 1.0, l1-weight 0.5, grad-weight 0.02)
Seed: 42; Dataset: ImageCAS-v1 train100 + aligned lumen GT
Plan: quality_reports/plans/2026-06-16_step2-task-aware-refiner.md (pre-set bar)

## Intent (pre-registered)

Does optimizing a residual refiner for a differentiable lumen soft-Dice objective
(+ L1 leash) capture downstream Dice-RCA the U-Net misses, judged by frozen TS
coronary_arteries? Bar: ΔDice-RCA ≥ +0.03, ΔMAE ≤ +3 HU, LEGACY ΔDice ≥ 0, no
vessel-voxel blowup.

## Result (test5) — FAIL on every criterion

| metric | U-Net | refiner | Δ | bar |
|---|---:|---:|---:|---|
| Dice-RCA (vs TS-clean) | 0.656 | **0.000** | **-0.656** | ≥ +0.03 |
| Dice vs GT | 0.418 | 0.012 | -0.406 | — |
| MAE (HU) | 40.61 | **71.73** | **+31.12** | ≤ +3 |
| ΔDice LEGACY (anti-cheat) | — | — | **-0.621** | ≥ 0 |
| coronary voxels | ~1939 | ~559 | ×0.29 | ≤1.5× |

Per-case identical pattern (refiner Dice ≈ 0.000 on all 5).

## What happened

The lumen-dominant loss (lumen 1.0 vs L1 leash 0.5) drove a DEGENERATE solution:
the refiner wrecked the global image (MAE 40.6 → 71.7 HU) chasing the
differentiable contrast-soft-Dice surrogate, and the resulting volume is so far
from a normal CCTA that the FROZEN segmenter finds essentially NO coronary
(Dice-RCA 0.000, voxels ×0.29). The L1 leash was far too weak to anchor fidelity.

Crucially this is NOT a clean refutation of the task-aware idea — it is a
loss-imbalance / surrogate-gaming failure, and **the independent judge worked**:
the differentiable lumen surrogate (in the loss) was gamed, but the frozen TS
segmenter (NOT in the loss) refused to reward the degenerate output → Dice 0.
The anti-cheat guardrail did its job.

## Decision

DISCARD this config. Do NOT proceed to Pilot-B (GAN) on top of a degenerate
setup — a GAN won't fix loss imbalance. The honest next options (user's call,
goal stop point reached):
1. Rebalance: fidelity-dominant (L1 anchor >> lumen, gentle lumen term operating
   ONLY inside GT lumen, not penalizing the band), re-pilot.
2. PMRF-style: anchor generation to the U-Net posterior mean by construction
   (transport, not free refinement) so fidelity cannot collapse, + perceptual.
3. Step back and reconsider whether the downstream gap is loss-addressable at all.

## Cross-references

- Prize-sizing: `2026-06-16_1630_coronary-dice-rca-pilot.md` (gap U-Net 0.656 vs clean 1.0)
- Plan: `quality_reports/plans/2026-06-16_step2-task-aware-refiner.md`
- Code: `code/training/train_lumen_refiner.py`, `scripts/python/eval_lumen_refiner_dice.py`,
  `code/evaluation/metrics.py::lumen_soft_dice_loss`
