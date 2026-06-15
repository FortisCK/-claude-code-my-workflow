# Plan: Task 1 ConvNeXt-Tiny Rescue

**Date:** 2026-05-22
**Status:** COMPLETED
**Scope:** One low-cost rescue attempt for the unstable ConvNeXt-Tiny FPN
architecture screen.

## Context

The first ConvNeXt-Tiny FPN screen used:

- SMP `FPN`
- `tu-convnext_tiny` ImageNet encoder
- 512 direct resize
- DiceCE loss
- fixed AdamW learning rate `0.0003`
- AMP enabled
- no ImageNet mean/std normalization

It diverged numerically:

- best balanced-monitor epoch: `7`
- best balanced-monitor Dice: `0.5100636033468431`
- loss became `NaN` from epoch `8`
- best checkpoint full released Dice: `0.4801980713555774`

Current winner remains EfficientNet-B3 UNet:

- full released Dice: `0.627019953686694`

## Rescue Hypothesis

The ConvNeXt run may have failed because the screening recipe is too aggressive
for this backbone:

- learning rate may be too high;
- AMP may be unstable;
- pretrained ConvNeXt may expect ImageNet normalization;
- gradients may need clipping.

## Rescue Configuration

Run one stabilized ConvNeXt-Tiny FPN variant:

- keep 512 direct resize for comparability;
- keep SMP `FPN` + `tu-convnext_tiny`;
- use ImageNet mean/std normalization;
- lower learning rate to `0.0001`;
- use ConvNeXt-style AdamW weight decay `0.05` after checking the official
  semantic segmentation configs;
- disable AMP;
- clip gradient norm at `1.0`;
- keep DiceCE and the same balanced monitor split.

Official ConvNeXt reference cloned for this check:

- `external_repos/ConvNeXt` (ignored by git)
- official segmentation recipe uses UPerNet, ImageNet normalization,
  AdamW `lr=0.0001`, `weight_decay=0.05`, warmup/poly schedule, and layer-wise
  LR decay.

## Decision Rule

- If full released Dice is below `0.53`, drop ConvNeXt for now.
- If full released Dice is `0.53-0.60`, keep as a secondary comparison only.
- If full released Dice approaches or beats `0.627`, then consider a more
  official ConvNeXt recipe with warmup/cosine/layer-wise LR decay or a different
  decoder.

## Verification

- add targeted code support for optional normalization and gradient clipping;
- run py_compile and component tests;
- smoke train on a small subset;
- run the full stabilized ConvNeXt training if smoke passes;
- evaluate the best checkpoint on full `released_eval.csv`.

## Completed Result

Full rescue training completed normally:

- end time: `2026-05-22T22:11:55+02:00`
- exit status: `0`
- epochs: `50`
- no NaN recurrence

Best balanced-monitor checkpoint:

- epoch: `46`
- balanced monitor Dice: `0.6654413140383311`
- `label_1` Dice: `0.6633248874055389`
- `label_2` Dice: `0.6675577406711233`
- animal Dice: `0.7319691487454341`
- phantom Dice: `0.5989134793312281`

Full released-eval result for the best checkpoint:

- output: `outputs/task1/smp_fpn_convnext_tiny_512_rescue_stable/eval_best_full_released.json`
- samples: `4691`
- mean Dice: `0.6311257683876099`
- `label_1` Dice: `0.6087852931686027`
- `label_2` Dice: `0.653466243606617`
- animal Dice: `0.728101265942059`
- phantom Dice: `0.6046515891366531`

The final epoch checkpoint was also evaluated on full released eval:

- output: `outputs/task1/smp_fpn_convnext_tiny_512_rescue_stable/eval_final_full_released.json`
- mean Dice: `0.6305445227190412`

Decision: keep ConvNeXt-Tiny FPN rescue as the current Stage 2 top single-model
checkpoint by mean full released Dice. It narrowly beats EfficientNet-B3 UNet
(`0.6311257683876099` vs `0.627019953686694`), mostly through animal-domain and
`label_1` gains. The margin is small enough that the next step should be
stability/seed or recipe refinement rather than assuming a decisive architecture
win.
