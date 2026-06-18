# Step-2 plan — task-aware (lumen) refiner pilot

Status: ACTIVE (executing under /goal autonomous run)
Date: 2026-06-16
Parent: quality_reports/plans/2026-06-16_path-a-downstream-aware-generative.md (C1)

## Question the pilot answers (pre-registered)

Does optimizing a model for the coronary-lumen objective (instead of distortion)
capture a meaningful slice of the measured downstream gap (U-Net Dice-RCA 0.656 vs
clean 1.000) WITHOUT hallucinating — judged on an INDEPENDENT frozen segmenter?

## Pre-set success bar (LOCKED — set before running, not movable)

On test5, refiner vs frozen U-Net:
- PRIMARY: ΔDice-RCA(vs TS-clean) ≥ **+0.03**
- fidelity leash: global MAE worsens by ≤ **+3 HU**
- anti-cheat: ΔDice on a HELD-OUT segmenter (coronary_arteries_LEGACY) ≥ **0**
- no hallucination: segmented coronary voxel count not inflated > ~1.5× vs U-Net
PASS → report + ask before full train/test100. FAIL/marginal → Pilot-B (add GAN);
still fail → stop, report (gap likely not loss-addressable / needs different route).

## Model & loss

- Model: `ResidualRefinerUNet3D` on the frozen U-Net initializer (reuse
  `train_residual_refiner.py` data + load_initializer). x_final = x_u + Δ.
- Differentiable lumen loss (NO segmenter in the loss → no cheating): within a
  dilated lumen band B = dilate(GT_lumen, r),
  `L_lumen = 1 - softDice( sigmoid((x_HU - τ)/T) , GT_lumen )` over B,
  τ ≈ 200 HU (contrast), T a temperature. Rewards contrast inside GT vessels,
  penalizes outside — directly targets segmentability, fully differentiable.
- Fidelity leash: small global L1 (keep distortion bounded; report Pareto).
- Loss = λ_l1 · L1 + λ_lumen · L_lumen   (Pilot-A). Pilot-B adds λ_adv · 3D
  PatchGAN (reuse train_vae_v2 PatchDiscriminator) — the generative element that
  restores vessel structure if the deterministic objective alone is insufficient.

## Execution (autonomous)

1. lumen GT for train/val (running).
2. Implement `lumen_soft_dice_loss` in code/evaluation/metrics.py (differentiable).
3. Extend `train_residual_refiner.py` (or a thin variant config) with the lumen
   loss + lumen-centered patch sampling; config `residual_refiner_lumen_v1.yaml`.
4. CPU/1-step smoke; then Pilot-A train (small: ~train100, lumen-centered patches,
   ~30-40 epochs) on the frozen U-Net.
5. Eval: export volumes (test5) → coronary_dice_rca (TS coronary_arteries +
   LEGACY) → ΔDice-RCA + MAE + hallucination check vs the bar.
6. If FAIL/marginal → Pilot-B (add PatchGAN), re-eval. Then STOP + report verdict.

## Guardrails

- Independent judge: TS coronary_arteries (loss uses only the differentiable
  surrogate, never TS) → no metric-gaming.
- Report distortion (MAE) alongside Dice (perception-distortion Pareto).
- Coexist with the running GPU process; keep refiner small (patch 128³, bs 1).
- Stop at the pilot verdict; full train / test100 only after user OK.
