# Plan: AAAI-First Roadmap With CVPR Fallback

**Status:** ACTIVE
**Date:** 2026-05-18
**Scope:** Convert the current cardiac CT motion-artifact correction work from a strong engineering baseline into a paper-ready research story. AAAI is the stretch target; CVPR is the fallback if the evidence chain is not ready in time.

## Current Frozen Baseline

The current Stage-1/Stage-2 pipeline is strong enough to serve as the baseline for all next experiments.

**Frozen VAE**

- Checkpoint: `experiments/checkpoints/vae_v2/best_val.pt`
- Role: frozen Stage-1 encoder/decoder for latent-space correction.
- Full reconstruction quality is good enough that Stage-2 should not be blocked on VAE changes.

**Frozen diffusion v1**

- Checkpoint: `experiments/checkpoints/diffusion_v1/epoch_200.pt`
- Model family: conditional latent EDM, not flow matching.
- Inference protocol: deterministic 50-step full-volume sliding-window inference.
- Full held-out test split, 100 cases:

| Input or model | MAE HU | PSNR | SSIM | NRMSE |
|---|---:|---:|---:|---:|
| Corrupted input | 436.96 | 16.42 | 0.8131 | 0.18942 |
| Diffusion v1 det50 | 72.94 | 27.95 | 0.8814 | 0.05131 |

Additional facts:

- `100/100` test volumes improved in MAE.
- Relative MAE reduction is about `83.3%`.
- This is a usable baseline, not just a smoke result.

## Paper Positioning

The paper should not be framed as "we slightly improved diffusion metrics." The stronger story is:

> Single-volume 3D latent generative correction for cardiac CT motion artifacts, without requiring a multi-phase cardiac CT sequence.

This separates our setting from TT U-Net-style dynamic CT methods. TT U-Net is a strong related work, but it solves a different setting:

- TT U-Net uses a 48-phase 2D+time cardiac CT video clip.
- Our current setting uses one corrupted 3D volume.
- TT U-Net reports local RMSE/SSIM, RCA Dice, and clinical no-reference artifact metrics.
- We currently report full-volume MAE/PSNR/SSIM/NRMSE on a 100-case held-out test split.

Therefore, TT U-Net should be used as a related-work anchor and motivation for motion-aware evaluation, not as a directly comparable numeric target.

Recommended claim:

> Existing dynamic cardiac CT artifact-reduction methods exploit temporal phase sequences. We study a weaker-input single-volume setting and use conditional latent diffusion to model the ambiguity left by missing temporal information.

## Target Contributions

The AAAI version should aim for these contributions.

1. **Single-volume formulation**

   Define cardiac CT motion-artifact correction from a single 3D corrupted volume, rather than assuming access to a 48-phase dynamic sequence.

2. **Physics-inspired paired artifact benchmark**

   Present the ImageCAS-based paired synthesis protocol as a controlled benchmark for supervised and generative correction.

3. **Posterior residual diffusion**

   Introduce diffusion v2 as a U-Net-conditioned posterior residual model:

   ```text
   x_u = frozen_unet(corrupted)
   residual = clean - x_u
   p(residual | corrupted, x_u, corrupted - x_u)
   ```

   Hypothesis: a deterministic U-Net provides a strong point estimate, but
   single-volume motion correction still leaves ambiguous residual errors,
   especially around cardiac boundaries. Posterior residual modeling should
   focus generative capacity on those remaining artifact-specific errors while
   preserving reliable anatomy.

4. **Uncertainty-aware correction**

   Use stochastic posterior samples to estimate voxelwise uncertainty. Validate whether uncertainty correlates with error and concentrates in artifact-heavy or ambiguous regions.

5. **Artifact-aware evaluation**

   Add heart-region, boundary-region, and severity-stratified metrics so the evaluation is not dominated by easy background or stable anatomy.

## Minimum Evidence Chain For AAAI

P0 items are required for a credible AAAI submission.

| Priority | Item | Purpose |
|---|---|---|
| P0 | `diffusion_v2` residual EDM | Main model improvement and clearer hypothesis |
| P0 | Supervised 3D residual U-Net baseline | Answer "why diffusion instead of U-Net?" |
| P0 | Same test100 full-volume evaluation | Preserve apples-to-apples comparison |
| P0 | Heart and boundary metrics | Show artifact-local quality, not only global SSIM |
| P0 | Qualitative figures for best/median/worst cases | Explain what is fixed and what remains hard |
| P1 | Posterior mean and uncertainty maps | Justify the generative model beyond deterministic metrics |
| P1 | Uncertainty-error correlation | Quantify whether posterior variance is meaningful |
| P1 | Severity-stratified evaluation | Show robustness across mild/medium/severe artifacts |
| P2 | Flow matching ablation | Optional engine comparison, only if it does not distract |
| P2 | Real clinical qualitative/no-reference validation | Strongly helpful if data become available |

## Execution Plan

### Step 1: Paper-Oriented Evaluation Suite

Add metrics that better match the cardiac artifact story:

- Full-volume: MAE HU, PSNR, SSIM, NRMSE.
- Heart-region: MAE HU, RMSE HU, SSIM.
- Boundary band: MAE HU, RMSE HU, gradient error.
- Artifact severity strata: mild, medium, severe.
- Optional uncertainty hooks: posterior std, uncertainty-error correlation.

Implementation notes:

- Heart metrics can use the existing heart masks.
- Boundary band can be computed as `dilate(mask) - erode(mask)`.
- Severity can initially be defined by corrupted-input heart MAE or full-volume MAE.

### Step 2: Supervised 3D Residual U-Net Baseline

Train a deterministic supervised baseline on the same paired data:

```text
input:  corrupted volume or patch
target: clean volume or patch
output: residual or clean volume
loss:   L1 + optional SSIM/gradient loss
eval:   same full-volume sliding-window protocol
```

Main comparison table should include:

```text
Corrupted input
3D residual U-Net
Diffusion v1
Diffusion v2
Diffusion v2 posterior mean, if feasible
```

This is more defensible than directly comparing to TT U-Net published numbers.

### Step 3: Posterior Residual Diffusion v2

Keep `diffusion_v1` frozen. Add a new `diffusion_v2` config and checkpoint namespace.

First v2 should be tightly scoped:

- Frozen initializer unchanged: `experiments/checkpoints/unet_v1/epoch_200.pt`
- Data split unchanged: `data/imagecas/splits/v1.json`
- Patch size unchanged initially.
- Engine remains EDM initially.
- Prediction target changes to U-Net residual: `clean - unet_pred`.
- Conditioning uses `[corrupted, unet_pred, corrupted - unet_pred]`.
- The first implementation can be voxel-space. Latent residual diffusion should
  be treated as an ablation, not the main path, unless voxel-space memory blocks
  progress.
- Add heart/boundary weighting or gating before any long run.

Suggested training sequence:

1. CPU or small-GPU smoke test for loss and checkpointing.
2. 10-20 epoch sanity run.
3. 5-20 full-volume cases vs U-Net v1 and deterministic residual refiner.
4. Add gate or heart/boundary losses if the sanity result only matches U-Net.
5. Long run only after the short run beats U-Net on at least heart or boundary metrics.
6. Full test100 deterministic/posterior-mean evaluation.
7. Posterior uncertainty evaluation on selected cases or full test100 if compute allows.

### Step 4: Uncertainty Experiments

Use stochastic sampling to produce multiple corrections per case:

```text
N = 4 or 8 posterior samples
correction = posterior mean
uncertainty = voxelwise posterior std
```

Evaluate:

- Does posterior mean improve MAE/SSIM compared with one deterministic sample?
- Does uncertainty correlate with absolute error?
- Does uncertainty localize to heart boundary, streaks, or weak cases?
- Are weak cases more uncertain than easy cases?

This is the main reason to keep diffusion in the story if a deterministic U-Net is competitive.

### Step 5: Figures And Writing Assets

Prepare paper figures early:

- Pipeline figure: synthetic paired data -> VAE latent -> residual diffusion -> correction + uncertainty.
- Main result panel: corrupted, U-Net, v1, v2, target, error maps.
- Worst-case panel: where all methods still struggle.
- Uncertainty panel: posterior std overlaid with error.
- Metric plots: heart/boundary/severity strata.

## AAAI vs CVPR Decision Rule

AAAI target is reasonable if the following are ready in time:

- v2 beats v1 or provides a clear tradeoff.
- U-Net baseline is included.
- Artifact-aware metrics are included.
- Uncertainty evidence is credible.
- The story can be written as an AI contribution, not only an application result.

Move to CVPR if one of these remains missing:

- No supervised baseline.
- No artifact-local metrics.
- No uncertainty story.
- v2 is only a small metric improvement without a clear modeling contribution.
- No time for clean figures and ablations.

CVPR fallback is not a failure mode. It gives time to strengthen visual results, broader baselines, and possibly real clinical qualitative validation.

## Immediate Next Actions

1. Keep the paper-oriented evaluation suite frozen as the shared metric protocol.
2. Keep `unet_v1` as the controlled deterministic initializer.
3. Keep `diffusion_v2_residual/pilot5` as the first posterior-residual proposal
   model for gate development; do not long-run ungated residual diffusion.
4. Implement `residual_gate_v1`: a lightweight voxel/local reliability gate
   that consumes posterior mean/std residual features and outputs
   `x_final = x_u + g * mu_r`.
5. Prefer cached posterior features for gate training so gate iterations are not
   bottlenecked by online diffusion sampling.
6. Evaluate learned gate on test5, then test20, against U-Net, fixed gates,
   uncertainty shrinkage, and oracle voxel/block upper bounds.
7. Continue to test100 only if learned gate improves heart/boundary metrics
   while preserving global MAE.

## Progress Log

- 2026-05-18: Started Step 1. Added reusable masked/boundary metrics and wired
  full-volume diffusion evaluation to emit heart-region, heart-boundary,
  boundary-gradient, and artifact-severity fields. Verified with a 1-case,
  2-step smoke run only; no full 100-case rerun yet.
- 2026-05-18: Ran a 5-case deterministic 50-step sanity check with the new
  metrics. Diffusion v1 improved global, heart-region, boundary, and
  boundary-gradient metrics on the subset. Run card:
  `experiments/runs/2026-05-18_1629_diffusion-v1-test5-artifact-metrics.md`.
- 2026-05-18: Started Step 2. Added a supervised `ResidualUNet3D` baseline,
  `unet_v1` config, training entry point, and full-volume evaluation entry
  point. Verified with CPU training smoke and 1-case full-volume eval smoke;
  no real U-Net baseline training has been launched yet.
- 2026-05-18: Fixed the U-Net training limit-batch guard and ran a real CUDA
  4-batch sanity check for `unet_v1`. The model uses a `128^3` patch,
  batch size 1, and 5.75M parameters; it completed without OOM and exited
  after the requested four batches. The generated `epoch_001.pt` sanity
  checkpoint was removed before the pilot run to avoid confusing it with a
  trained baseline.
- 2026-05-18: Ran a 10-epoch `unet_v1` pilot and evaluated the checkpoint on
  the same five held-out test cases used for the diffusion v1 artifact-metric
  sanity check. The U-Net pilot strongly improves over corrupted input but does
  not yet match frozen diffusion v1 on heart and boundary metrics. Run card:
  `experiments/runs/2026-05-18_1714_unet-v1-pilot10-test5.md`.
- 2026-05-18: Launched the full `unet_v1` baseline training run from
  `epoch_010.pt` to the configured 200 epochs. Host PID: `4179056`. Log:
  `experiments/runs/unet_v1/train_full_from_epoch010.log`.
- 2026-05-19: Completed `unet_v1` training and evaluated `epoch_200.pt` EMA
  on the full held-out test100 split. U-Net v1 reached MAE `38.07 HU`, PSNR
  `33.23`, SSIM `0.9446`, and NRMSE `0.02839`, outperforming frozen diffusion
  v1 deterministic 50-step global metrics. Run card:
  `experiments/runs/2026-05-19_1511_unet-v1-epoch200-test100.md`.
- 2026-05-19: Re-ran frozen diffusion v1 deterministic 50-step full test100
  with the new artifact-aware metric suite. Diffusion v1 reached global MAE
  `72.94 HU`, heart MAE `47.55 HU`, boundary MAE `69.36 HU`, and boundary
  gradient L1 `53.44 HU`. The U-Net baseline remains stronger on global,
  heart, and boundary MAE/RMSE; diffusion v1 is slightly better on boundary
  gradient L1 than U-Net. Run card:
  `experiments/runs/2026-05-19_1600_diffusion-v1-test100-artifact-metrics.md`.
- 2026-05-20: Ran full test100 U-Net residual diagnosis with residual defined
  as `clean - unet(corrupted)`. U-Net residuals reached global MAE `38.07 HU`,
  heart MAE `43.17 HU`, and boundary MAE `56.06 HU`. Boundary residuals are
  denser than their voxel fraction, with boundary error enrichment `1.52x` and
  boundary p95 absolute residual `177.37 HU`. This supports a posterior
  residual model that targets cardiac boundary errors instead of global
  full-volume refinement. Run card:
  `experiments/runs/2026-05-20_1515_unet-v1-residual-diagnosis-test100.md`.
- 2026-05-20: Added and piloted a deterministic residual refiner baseline on
  top of frozen U-Net v1. The refiner uses input
  `[corrupted, unet_pred, corrupted - unet_pred]` and target
  `clean - unet_pred`. After 10 epochs, test20 full-volume evaluation was
  essentially tied with U-Net: global MAE delta `+0.019 HU`, heart MAE delta
  `-0.031 HU`, boundary MAE delta `-0.004 HU`. This indicates that a plain
  second U-Net refiner is not enough; v2 should use posterior residual
  diffusion, uncertainty/gating, and artifact-aware losses. Run card:
  `experiments/runs/2026-05-20_1548_residual-refiner-v1-epoch010-test20.md`.
- 2026-05-20: Added the first `diffusion_v2_residual` implementation without
  touching frozen diffusion v1. New components include a generic voxel-space
  conditional denoiser, residual EDM wrapper, training config/entry point,
  sliding-window residual-diffusion inference, and full-volume evaluation.
  The model predicts `clean - unet_pred` conditioned on
  `[corrupted, unet_pred, corrupted - unet_pred]`; EDM uses residual-scale
  defaults (`sigma_data=0.05`, `sigma_max=1.0`) plus image-domain L1/gradient
  auxiliary losses. CPU training smoke and 1-case full-volume 2-step evaluation
  both passed. The smoke checkpoint is intentionally untrained and should not
  be interpreted numerically. Smoke outputs:
  `experiments/checkpoints/diffusion_v2_residual/smoke/epoch_000.pt` and
  `experiments/runs/diffusion_v2_residual/eval_smoke_test1/`.
- 2026-05-20: Ran the first real `diffusion_v2_residual` pilot. The initial
  attention-enabled voxel model OOMed on 128^3 patches, so v2 was switched to
  a no-attention 9.47M-parameter denoiser with channels `[24, 48, 96, 96]`.
  A 5-epoch pilot trained successfully: final-image L1 improved from `0.0135`
  at epoch 1 to `0.0099` at epoch 5. Full-volume test5 evaluation with EMA,
  16-step, single-sample inference was worse than U-Net v1: global MAE delta
  `+21.14 HU`, heart MAE delta `+19.72 HU`, boundary MAE delta `+16.30 HU`.
  Fixed-gate test5 follow-ups showed alpha `0.25` reduces the global delta to
  `+3.01 HU` and improves boundary-gradient error by `-0.69 HU`; alpha `0.10`
  nearly preserves U-Net MAE with global delta `+0.80 HU` and boundary-gradient
  delta `-0.18 HU`. Diagnostics on case 21 show online weights are better than
  EMA, conservative residual scaling reduces the damage, 4-sample posterior
  mean helps further, negative scaling is not a sign fix, and lowering sampling
  `sigma_max` alone does not solve the issue. Current conclusion: v2 is
  technically working but must add gate/residual-magnitude control and stronger
  artifact-aware losses before another long run. Run cards:
  `experiments/runs/2026-05-20_1702_diffusion-v2-residual-pilot5-test5.md` and
  `experiments/runs/2026-05-20_1704_diffusion-v2-residual-pilot5.md`.
- 2026-05-20: Ran the first real `diffusion_v2_residual` pilot for 5 epochs.
  The initial attention-based voxel config OOMed at `128^3`, so the pilot uses
  a no-attention 9.47M-parameter denoiser. Training was stable, with mean
  final-image L1 dropping from `0.0135` to `0.0099`. Test5 full-volume
  evaluation shows raw residual addition is too aggressive: MAE `61.75 HU`
  versus U-Net `40.61 HU`. Constant gates reduce the damage: alpha `0.25`
  gives MAE `43.62 HU` and alpha `0.10` gives MAE `41.41 HU`. Importantly,
  both gated variants slightly improve boundary-gradient L1 over U-Net
  (`61.46` and `61.97 HU` vs U-Net `62.16 HU`). Decision: do not long-run the
  ungated residual diffusion; implement a learned/local gate or
  uncertainty-conditioned gate before the next serious run. Run card:
  `experiments/runs/2026-05-20_1702_diffusion-v2-residual-pilot5-test5.md`.
- 2026-05-21: Recorded the literature-driven method decision in
  `quality_reports/plans/reliability-gated-posterior-residual-diffusion.md`.
  The synthesis is that paired HU-preserving restoration favors a strong
  deterministic point estimate, while diffusion is best used as posterior
  residual proposal plus posterior-mean and reliability calibration. The method
  should not be framed as generic post-processing, but as reliability-gated
  posterior residual diffusion.
- 2026-05-21: Added a residual-diffusion gate ablation script and ran test5
  with 16 denoising steps and 4 posterior samples. Full posterior-mean release
  worsens MAE from U-Net `40.61 HU` to `51.92 HU`, but voxel oracle improves
  every case, reaching mean MAE `37.27 HU`, heart MAE `43.84 HU`, boundary MAE
  `59.65 HU`, and boundary-gradient L1 `59.70 HU`. Block32/scalar oracle gains
  are tiny, so useful residual corrections are local and require a fine-grained
  learned gate. Run card:
  `experiments/runs/2026-05-21_1651_diffusion-v2-residual-gate-test5.md`.
- 2026-05-22: Implemented and trained the learned reliability gate
  (`residual_gate_v1`): `ResidualGateNet3D` releasing `x_final = x_u + g·μ_r`
  with `g ≤ 0.25`, trained on cached posterior features. train100 epoch20 beat
  all fixed alphas; resuming to epoch60 improved monotonically (test5 -0.0746,
  val20 -0.0775 HU, 20/20 cases). Two oracle-supervised variants were
  DISCARDed: dense BCE oracle regressed to +0.014 HU (worse than U-Net), sparse
  oracle reached -0.007 HU (better than U-Net but worse than v1). Lesson:
  per-voxel oracle matching over-releases; v1's conservative final-image loss is
  the anchor. Run cards: `2026-05-22_1711_residual-gate-v1-train100-e060-test5-val20.md`,
  `2026-05-22_1301`, `2026-05-22_1322`.
- 2026-06-10/11: Evaluated `residual_gate_v1/train100_e060/epoch_060.pt` on the
  full held-out **test100**. Result KEEP: global MAE -0.0762 HU vs frozen U-Net,
  heart -0.6535, boundary -0.7219, boundary-grad -0.2275, **100/100 cases
  improved on all four metrics**, gains grow with severity. Direct posterior
  mean residual is much worse (49.55 HU), confirming the gate is necessary.
  Calibration analysis confirmed the gate is genuinely calibrated (high gate →
  monotonically higher improvement) but the global gain is diluted because ~55%
  of (non-heart) voxels are suppressed. This satisfied the Step-3 Go/No-Go but
  the win is "too small to be the standalone paper story." Run cards:
  `2026-06-11_1058_residual-gate-v1-e060-test100.md`,
  `2026-06-11_1615_residual-gate-v1-test100-calibration.md`.
- 2026-06-13: Trained `residual_gate_v3a` to epoch60 — the v1 gate plus three
  structure/reliability input channels recommended by the calibration analysis
  (`heart_mask`, `boundary_band`, `residual_snr = |μ_r|/σ_r`; 9ch total), same
  conservative v1 loss and `g_max=0.25`. Config
  `code/training/configs/residual_gate_v3a.yaml`.
- 2026-06-15: Launched the v3a evaluation (val20 + full test100, reusing the v1
  posterior caches — no diffusion re-sampling) to close the open loop, since
  v3a had been trained but never evaluated. Run card:
  `experiments/runs/2026-06-15_1511_residual-gate-v3a-e060-test100.md`.
  **Verdict: DISCARD.** v3a is 100/100-consistent but does not beat v1: test100
  global MAE Δ -0.0662 (v1 -0.0762), heart Δ -0.5624 (v1 -0.6535), boundary tie,
  and only boundary-gradient L1 clearly better (-0.327 vs -0.228). Gate mean
  actually dropped (0.0067 vs 0.0076) despite the heart_mask input. The
  added-channels hypothesis is falsified at this budget: the conservative v1 loss
  gives the gate no reason to exploit the new channels. `residual_gate_v1` remains
  the anchor. Next model step should be **v3b** (v3a inputs + a weak sparse
  high-confidence oracle auxiliary loss), not more inputs.

## Non-Goals For The Next Sprint

- Do not replace the VAE unless v2 evidence clearly points to VAE bottlenecking.
- Do not make flow matching the main path yet.
- Do not claim numeric superiority over TT U-Net published results.
- Do not add many architecture changes at once.
- Do not evaluate only global full-volume metrics for the paper version.
- Do not continue long-training the current deterministic residual refiner
  unless a specific ablation requires it; its 10-epoch pilot already shows that
  plain deterministic refinement is not the main path.
