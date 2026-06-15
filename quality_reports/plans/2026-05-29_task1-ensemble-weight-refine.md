# Task 1 Five-Model Ensemble Weight Refinement Plan

Date: 2026-05-29

## Goal

Refine the current five-model Task 1 champion without additional training.

Current champion:

- `ConvNeXt-Tiny 640`: `0.35`
- `EfficientNet-B3 512`: `0.35`
- `ConvNeXtV2-Base 512`: `0.10`
- `ConvNeXt-Small 512`: `0.10`
- `ConvNeXt-Small 640`: `0.10`
- released eval mean Dice: `0.6500661421073305`

## Method

Run a narrow weight grid around the current champion. To avoid repeated model
inference, evaluate all candidate weight vectors in one pass:

- load all five checkpoints once;
- compute original-mask-space probabilities once per sample/model;
- score every candidate weight vector from the same probabilities;
- use `none + hflip` TTA and the released eval manifest.

## Candidate Region

Search around:

- main model total weight: `0.65` to `0.75`;
- tail model total weight: `0.25` to `0.35`;
- keep both `ConvNeXt-Small 512` and `ConvNeXt-Small 640` unless the refined
  search proves one is redundant.

## Decision Rule

- Promote any candidate that beats `0.6500661421073305` on full released
  original-mask-space Dice.
- If no candidate beats it, keep the current five-model champion.

## Verification

The run must produce:

- a full JSON result under `outputs/task1/stage5_weight_refine/`;
- a reproducible command/script;
- ranked candidate metrics with mean Dice, label_1, label_2, animal, and
  phantom.

## Outcome

Completed on 2026-05-29.

Best candidate:

- `more_tail_equal_0325_0325_0117_0117_0116`
- `ConvNeXt-Tiny 640`: `0.325`
- `EfficientNet-B3 512`: `0.325`
- `ConvNeXtV2-Base 512`: `0.117`
- `ConvNeXt-Small 512`: `0.117`
- `ConvNeXt-Small 640`: `0.116`

Full released original-mask-space result:

- mean Dice: `0.6516511688167698`
- label_1 Dice: `0.6305547443921731`
- label_2 Dice: `0.6727475932413666`
- animal Dice: `0.7373800564302811`
- phantom Dice: `0.6282472988197025`

This is now the local Task 1 champion.
