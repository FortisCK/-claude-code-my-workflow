# Plan — Coronary-lumen / perceptual fidelity: the path to a winning result

Status: DRAFT (awaiting go)
Date: 2026-06-16
Direction chosen by user (2026-06-15): **coronary lumen + perceptual fidelity**,
with calibrated UQ (now real, uncertainty-error corr 0.42) as a second leg.

## Why this plan exists

Performance on global MAE is structurally capped (the residual-diffusion long run
confirmed: posterior mean → U-Net conditional mean, oracle headroom *shrinks* with
convergence). But we have **never measured the axis where diffusion should win**:
coronary-lumen geometry / perceptual sharpness. The U-Net conditional mean is
blurry by construction (regression-to-mean); diffusion samples are sharp. The
distortion-perception tradeoff says diffusion can win on lumen/perceptual even at
tied MAE. ImageCAS ships paired coronary GT labels — we just never plumbed them in.

Goal: a genuinely good, defensible result = **diffusion recovers coronary lumen
geometry better than the U-Net where it clinically matters**, + calibrated UQ.
Headline becomes "better coronary recovery + calibrated uncertainty," not "lower MAE."

## Verified enablers (from 2026-06-16 scout workflow)

- Coronary GT alignable bit-exact: raw `.label.nii.gz` → NN resample onto
  `v_resampled` grid → `crop_centered` with the stored `metadata['centroid_voxel']`
  → 192³ lumen mask aligned to `volume`. No TotalSegmentator re-run needed.
- Env `cardiac-diffusion` has TotalSegmentator 2.13 (`coronary_arteries` task),
  nnunetv2, monai.
- Single-sample (sharp) eval is currently blocked: `residual_diffusion_sliding_window.py:116`
  averages samples inside each patch (`residual_mean = residual_samples.mean(dim=1)`),
  so only the blurry posterior mean is stitched. Must emit a coherent single sample
  (fixed seed across patches) to test the sharp-sample hypothesis.

## Phase 1 — lumen GT + the decisive first signal (~1-2 days, deterministic)

1. **Lumen-mask generator** (`scripts/python/generate_lumen_masks.py`, new):
   standalone, reuses `metadata['centroid_voxel']` from each processed npz →
   no re-preprocess, no TotalSegmentator. Output aligned 192³ uint8 `lumen_mask`
   per case (start: the test100 cases + the 5-case pilot). **Verify coverage**:
   report lumen voxel counts at 1mm across test100 (risk: thin distal vessels
   shrink ~14× at 1mm; known samples have 5k-7.8k voxels = fine, confirm at scale).
2. **Real lumen + sharpness metrics** (`code/evaluation/metrics.py`,
   `code/evaluation/artifact_metrics.py`): add `dice_lumen_masked` (HU-domain
   threshold within GT lumen mask, ~300-500 HU), lumen-ROI `masked_mae/rmse`
   (reuse existing), lumen CNR/contrast (lumen vs surrounding myocardium ring via
   `make_boundary_band`), and sharpness (`gradient_magnitude` variance / tenengrad
   in-mask). Thread `lumen_mask` through `metric_row` (parallel to `heart_mask`);
   `summarize` auto-includes new floats. `load_lumen_mask` tolerates missing key
   (returns None) so old caches don't break.
3. **Single-sample fix** (`code/inference/residual_diffusion_sliding_window.py`):
   route A — fix the seed in the patch predictor and emit a coherent single sample
   alongside the posterior mean (+std). Add `single_sample` option; thread to
   `evaluate_residual_diffusion_full_volume.py` as a 4th `metric_row`
   (`diffusion_v2_sample`).
4. **Evaluate** {corrupted, U-Net, diffusion mean, diffusion single-sample} on the
   converged checkpoint `diffusion_v2_residual/longrun/epoch_080.pt` vs lumen GT,
   test5 → test100: lumen Dice, lumen-HU fidelity, CNR, sharpness, SSIM, MAE.

**Decisive read / go-no-go:** does the sharp diffusion sample beat U-Net on
lumen Dice / contrast / sharpness (even at tied or slightly worse MAE)?
- WIN → Phase 2a (scale + frozen-segmenter Dice + figure).
- TIE → Phase 2b (lumen-aware / perceptual training loss to win on that axis directly).
- LOSE on lumen too → reassess honestly (UQ-led paper).

## Phase 2 (contingent on Phase 1)

- **2a (if sharp sample already wins lumen):** scale to test100, add the rigorous
  frozen-segmenter Dice (TotalSegmentator `coronary_arteries` on
  clean/corrupted/U-Net/diffusion vs GT) — directly comparable to TT U-Net's
  Dice-RCA 0.81. Pilot on 5 cases first (domain-gap check). Build the killer figure.
- **2b (if tied):** implement the spec's lumen-aware differentiable loss; fine-tune
  a refiner/diffusion to optimize lumen fidelity directly (the axis we can win).

## Risks

- 1mm NN downsampling thins distal coronaries → verify voxel counts test100-wide.
- Frozen segmenter may fail on motion-corrupted images / have ImageCAS domain gap →
  5-case pilot before scaling (Phase 2a).
- Single-sample stitching seams → fixed seed across patches.
- Eval-behavior changes → new run cards (experiments-protocol); old summary.json
  unaffected (additive metrics, missing-key-tolerant loader).

## Run cards needed

- `experiments/runs/<ts>_lumen-gt-build.md` (mask generation + coverage stats)
- `experiments/runs/<ts>_lumen-eval-e080-test100.md` (the decisive eval)

## Not now

- Re-running full preprocessing (standalone generator avoids it).
- LPIPS / heavy perceptual deps (use gradient/MS-SSIM proxies first).
- Retraining the gate for accuracy (capped — settled by the long run).
