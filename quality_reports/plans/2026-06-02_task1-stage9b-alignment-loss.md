# Task 1 Stage 9B Alignment-Aware Loss Plan

Date: 2026-06-02

Status: stopped

## Goal

Improve exact Dice for thin catheter / guidewire masks by training a
shape-aware ConvNeXt-Small 640 model that is useful either as a standalone model
or, more realistically, as a complementary ensemble member.

The current local fallback champion is:

- seven-model `add_both_010_010` ensemble;
- fixed `remove_small_min32` postprocess;
- released-eval mean Dice: `0.6551964758686204`.

The tolerance diagnostic showed that the model is often close but not
pixel-exact:

- exact Dice: about `0.65`;
- 1px tolerant Dice: about `0.80`;
- 2px tolerant Dice: about `0.88`.

So Stage 9B should prioritize alignment, continuity, and boundary shape rather
than another generic backbone swap.

## Constraints

- Keep the released train/eval split unchanged.
- Do not use eval masks for training, ROI generation, pseudo-labeling, or
  postprocess tuning beyond already-fixed local validation decisions.
- Use existing public/pretrained model permissions only.
- Keep the current seven-model + postprocess pipeline as a fallback.

## Main Design

Train one new `ConvNeXt-Small 640 aspect_pad` model with the same architecture
family as Stage 7:

- SMP `FPN`;
- encoder `tu-convnext_small`;
- ImageNet weights;
- output channels: `4`;
- first three channels: `background / label_1 / label_2`;
- fourth channel: binary toolness auxiliary.

Use the Stage 7 checkpoint as warm start where possible.

Add a combined loss:

```text
L =
  DiceCE(label_0/1/2)
  + lambda_toolness * toolness_aux_loss
  + lambda_cb(epoch) * cbDice(foreground_union)
  + lambda_hd(epoch) * MONAI_HausdorffDT(foreground_union)
```

Rationale:

- DiceCE keeps region overlap stable.
- Toolness auxiliary preserves foreground localization.
- cbDice targets centerline, boundary, and radius/diameter balance on the
  foreground union.
- Hausdorff distance-transform loss targets boundary / endpoint displacement.

MONAI `1.5.2` is available locally and provides `HausdorffDTLoss`, so the
Hausdorff term should use MONAI rather than a custom implementation. cbDice is
not provided by MONAI, so implement a small project-local wrapper using the
official cbDice repository's foreground differentiable binarization and
morphological skeletonization/distance-transform weighting strategy.

## Implementation Steps

1. Add a combined loss wrapper in `src/cathaction/training/task1_baseline.py`.

   Candidate config name:

   `monai_dice_ce_toolness_cbdice_hd`

   It should reuse existing helpers:

   - `_MonaiMulticlassLossWrapper`
   - `_DiceCeToolnessAuxLoss` logic
   - `primary_logits_for_label_mode`

   New behavior:

   - apply DiceCE / cbDice / Hausdorff only to primary logits `[:, :3]`;
   - apply toolness only to auxiliary channel `[:, 3:4]`;
   - support `set_epoch(epoch)` for warmup schedules;
   - compute cbDice on `foreground_union = label_1 union label_2`;
   - compute Hausdorff on a two-channel background/foreground target;
   - keep `lambda_hd=0` by default so the wrapper can be smoke-tested
     incrementally.

2. Add a Stage 9B config.

   Proposed path:

   `configs/task1/smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b.yaml`

   Initial conservative settings:

   - `lambda_dice: 1.0`
   - `lambda_ce: 0.5`
   - `class_weights: [0.02, 1.0, 2.0]`
   - `lambda_toolness: 0.4`
   - `lambda_toolness_bce: 0.5`
   - `lambda_toolness_dice: 1.0`
   - `lambda_cbdice: 0.12`
   - `cbdice_iterations: 10`
   - `cbdice_warmup_epochs: 8`
   - `cbdice_threshold: 0.5`
   - `lambda_hd: 0.03`
   - `hd_warmup_epochs: 15`
   - `hd_include_background: false`
   - `epochs: 80`
   - `batch_size: 8`
   - `learning_rate: 0.00008`
   - `domain_balanced: true`

3. Run a smoke test before overnight training.

   Smoke criteria:

   - one or two epochs complete without NaN;
   - loss decreases or is numerically stable;
   - balanced eval writes metrics;
   - no shape errors from the 4-channel auxiliary output.

4. Run a full Stage 9B training job.

   Use the approved outside-sandbox GPU path because normal Codex shell may not
   expose `/dev/nvidia*`.

   Save logs under:

   - `quality_reports/logs/task1_stage9b_alignment_loss_master.log`
   - `quality_reports/logs/task1_smp_fpn_convnext_small_640_toolness_cldice_hd_stage9b.log`

5. Evaluate the best checkpoint.

   Required metrics:

   - full released eval raw hflip original-space Dice;
   - full released eval after fixed `remove_small_min32`;
   - label_1 / label_2 Dice;
   - animal / phantom Dice;
   - optional diagnostic tolerance r1/r2 to confirm whether exact alignment
     improved rather than only small-fragment removal.

6. Ensemble probe.

   Add the Stage 9B model to the current seven-model champion with small weights:

   - `0.05`
   - `0.075`
   - `0.10`
   - `0.125`

   Always evaluate both:

   - raw ensemble;
   - raw ensemble + fixed `remove_small_min32`.

## Promotion Criteria

Promote Stage 9B if either condition holds:

- standalone postprocessed full released-eval Dice beats `0.6551964758686204`;
  or
- ensemble with Stage 9B beats the current champion by at least `+0.0015`
  without hurting phantom.

If the gain is below `+0.001`, keep the decision note but stop this route and
move to Stage 9C ROI/alignment refinement.

## Stop Note

The Stage 9B GPU training process was stopped on 2026-06-03 at the user's
request after the run reached epoch 36 and showed no evidence of improving over
the current Stage 9A champion. The GPU was verified free with `nvidia-smi`
after termination.

## Expected Result

This is not expected to jump directly to `0.8x`; the diagnostic says strict
pixel-level exactness is the bottleneck. A realistic win is:

- `+0.001` to `+0.004` if used as an ensemble member;
- potentially stronger label_2 / endpoint behavior if the Hausdorff term is
  stable.

## Risks

- HausdorffDTLoss can destabilize early training if weighted too high.
- cbDice and Hausdorff may over-penalize noisy annotations.
- Warm-starting from Stage 7 may constrain exploration but is safer under the
  current deadline.
- If the new model is too similar to Stage 6/7, ensemble gain may be small.
