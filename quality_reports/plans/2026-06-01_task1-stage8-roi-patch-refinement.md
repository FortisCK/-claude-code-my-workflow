# Task 1 Stage 8 ROI/Patch Refinement Plan

Date: 2026-06-01

## Goal

Move beyond full-frame backbone/loss tweaks by adding a coarse-to-fine
high-resolution refinement path for thin catheter/guidewire segmentation.

Current local champion:

- seven-model ensemble candidate: `add_both_010_010`
- mean Dice: `0.6536723455148790`
- label_1 Dice: `0.6327836065361310`
- label_2 Dice: `0.6745610844936268`
- animal Dice: `0.7394416932476355`
- phantom Dice: `0.6302574299601563`

## Motivation

Full-frame 512/640 training is now giving small gains only. Stage 6 clDice and
Stage 7 toolness auxiliary improved the ensemble by about `+0.002`, but they did
not change the main bottleneck: the instruments are only a few pixels wide and
phantom hard cases remain weak.

Stage 8 tests whether high-resolution local context can improve:

- phantom tool localization;
- thin label_2 continuity;
- label_1/label_2 class confusion near long structures;
- missed or shifted fine segments.

## Scope

Implement the smallest useful coarse-to-fine system:

1. Add a foreground-centered patch dataset mode.
2. Train a patch model using released-train masks for crop sampling only.
3. Add coarse ROI inference that uses an existing full-frame prediction/ensemble
   to propose boxes on eval/test images.
4. Stitch patch probabilities back into original image space and fuse with the
   full-frame result.

This stage must not use released-eval ground truth to generate inference ROIs.
Released-eval masks may only be used for evaluation.

## Design

### Patch Training

- Input patches are cropped from original images before resizing.
- Foreground-centered crops are sampled from training masks.
- Include a minority of random/background crops to control false positives.
- Preserve multiclass labels.
- Start with `ConvNeXt-Small/FPN` or a lighter `ConvNeXt-Tiny/FPN` patch model.
- Use `multiclass_012` DiceCE first; add toolness auxiliary only after the basic
  patch path is stable.

### ROI Inference

- Use current champion predictions as coarse masks.
- Build connected-component or bounding-box proposals from coarse foreground.
- Enlarge boxes with margin so the patch model sees context.
- Run patch model on each ROI.
- Map patch probabilities back to original image coordinates.
- Fuse with the full-frame ensemble probability.

### First Success Criteria

- Smoke training completes on GPU with patch crops.
- Patch model produces original-space predictions through ROI inference.
- Full released eval beats `0.6536723455148790`.

Useful intermediate signals:

- phantom Dice increases;
- binary foreground Dice increases;
- worst-case overlays show fewer large foreground misses.

## Risks

- Patch crops can lose global context and worsen label_1/label_2 semantics.
- Coarse ROI misses cannot be recovered by refinement.
- Patch stitching can introduce boundary artifacts.
- Inference can become too slow for Docker submission if the proposal count is
  not capped.

## Planned Artifacts

- patch-aware dataset code in `src/cathaction/training/task1_baseline.py`;
- Stage 8 config under `configs/task1/`;
- ROI/stitching script under `scripts/task1/`;
- smoke log under `quality_reports/logs/`;
- decision note after first full evaluation.
