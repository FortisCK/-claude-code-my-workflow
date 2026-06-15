# Task 1 Stage 7: ConvNeXt-Small 640 Toolness Auxiliary Head

Date: 2026-05-31
Status: Running full Stage 7 train+eval

## Motivation

The current best validation result is still the original five-model ensemble at mean Dice
0.651651, with the Stage 6 clDice model adding only a small ensemble-tail gain to
0.652372. Full-frame clDice did not improve the single ConvNeXt-Small 640 model enough to
justify more tuning in the same direction.

Stage 7 tests a different high-signal idea from the roadmap: make the model learn a coarse
binary tool-vs-background foreground map alongside the three-class mask. This is a minimal
version of a coarse-to-fine/toolness auxiliary route, without yet adding ROI crops,
post-processing, pseudo-labels, or case-level consistency.

## Scope

1. Extend the Task 1 training code to allow multiclass models with one auxiliary output
   channel.
2. Add a loss that uses:
   - the first 3 logits for multiclass background/label_1/label_2 DiceCE,
   - the 4th logit for binary toolness supervision, where toolness is `target > 0`.
3. Ensure prediction, TTA evaluation, original-space evaluation, and ensemble code use only
   the first 3 logits for multiclass probabilities.
4. Add a Stage 7 ConvNeXt-Small 640 config and runner.
5. Run a smoke test before launching a full train+original-space eval job.

## Success Criteria

- Existing 3-channel checkpoints still evaluate unchanged.
- A 4-channel auxiliary checkpoint can train/evaluate without shape errors.
- Original-space TTA evaluation reports only 3-class probabilities.
- Full Stage 7 result is compared against:
  - best single ConvNeXt-Small 640: 0.637311 original-space Dice,
  - Stage 6 clDice single: 0.633660 original-space Dice,
  - current ensemble champion: 0.651651 original-space Dice,
  - six-model Stage 6 low-weight probe: 0.652372 original-space Dice.

## Progress

- Implemented 4-channel auxiliary model support.
- Smoke training passed on 32 train / 16 eval samples using CUDA.
- Original-space hflip TTA smoke eval passed on 8 released-eval samples using the 4-channel
  checkpoint, with probabilities sliced to the first 3 task classes before softmax.
- Full 80-epoch run launched as:
  `cathaction-task1-stage7-convnext-small640-toolness.service`.

## Non-Goals

- No ROI/patch mining in this step.
- No pseudo-labeling or external data.
- No post-processing tuned on evaluation results.
- No hidden-test feedback or split changes.
