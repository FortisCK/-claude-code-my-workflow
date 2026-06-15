# Task 1 ConvNeXt Champion Deepening Plan

Date: 2026-05-28

## Goal

Improve the current Task 1 champion while keeping the search focused on the
observed strongest family: ConvNeXt-style encoders with FPN/UNet-compatible
multi-scale decoders.

Current champion:

- `ConvNeXt-Tiny 640 rescue + EfficientNet-B3 512`
- hflip TTA
- original-mask-space evaluation
- released eval mean Dice: `0.6438677932546133`

## Rationale

The Stage 4A architecture shootout did not find a new single-model champion,
but it did confirm that ConvNeXt-family models are the strongest single-model
direction under the current recipe:

- `FPN + ConvNeXtV2-Base 512`: `0.6340868616268381`
- `FPN + ConvNeXt-Small 512`: `0.6330665499725219`
- `SegFormer MiT-B3/B4`: below those under the same recipe

This suggests the next highest-value step is not another broad architecture
search, but a focused ConvNeXt-family improvement pass.

## Stage A: No-Training Ensemble Search

Search softmax-average ensembles over existing checkpoints:

- current `ConvNeXt-Tiny 640 rescue`;
- current `EfficientNet-B3 512` complement;
- Stage 4A `ConvNeXt-Small 512`;
- Stage 4A `ConvNeXtV2-Base 512`;
- optionally `EfficientNet-B5 512` only if it shows complementary behavior.

Evaluation discipline:

- use `configs/task1/splits/released_eval.csv`;
- evaluate in original mask space;
- use the same `none + hflip` TTA as the current champion;
- report mean Dice, per-class Dice, animal Dice, and phantom Dice.

Decision rule:

- If a no-training ensemble beats `0.6438677932546133`, promote it as the new
  local champion.
- If it does not beat the champion, keep the current champion and use the
  results to decide which ConvNeXt variant deserves further training.

## Stage B: Targeted ConvNeXt Training

After Stage A, run one focused training experiment rather than another broad
shootout:

- prefer the best ConvNeXt-family member from Stage A;
- increase spatial fidelity (`640` or `768`, aspect-pad);
- keep ImageNet normalization and stable AdamW-style regularization;
- add thin-structure loss or sampling only one factor at a time.

Candidate first training variants:

1. `ConvNeXtV2-Base 640`, aspect-pad, same stable loss as Stage 4A.
2. `ConvNeXt-Tiny/Small 640`, add class-wise soft-clDice after warmup.
3. `ConvNeXt-family 640`, foreground-centered crop/ROI sampling.

## Stage C: Timeboxed Mamba Check

Mamba/VMamba is not the mainline, but it remains an architecture-risk check.
After the first ConvNeXt deepening pass, run at most one VMamba smoke/short run
if the official dependencies are tractable. It must meet a predefined threshold
to enter the mainline.

## Verification

Every promoted result must have:

- a full released-eval JSON under `outputs/task1/`;
- a log under `quality_reports/logs/`;
- a comparison against the current `0.6438677932546133` champion;
- no hidden-test feedback or frame-level split leakage.

## Stage A Outcome

Completed on 2026-05-28. The best no-training ensemble was:

- `ConvNeXt-Tiny 640`: `0.40`
- `EfficientNet-B3 512`: `0.40`
- `ConvNeXtV2-Base 512`: `0.10`
- `ConvNeXt-Small 512`: `0.10`

Full released original-mask-space result:

- JSON: `outputs/task1/stage5_convnext_ensemble_search_full/w040_040_010_010_convnextv2_small.json`
- mean Dice: `0.648019210540978`
- label_1 Dice: `0.628068019483919`
- label_2 Dice: `0.667970401598037`
- animal Dice: `0.7326960974347608`
- phantom Dice: `0.6249025353129875`

This is now the local Task 1 champion. The next training experiment should be
`ConvNeXt-Small 640 aspect-pad`.

## Stage B Launch

Launched on 2026-05-28 as a user systemd service:

- unit: `cathaction-task1-convnext-small-640-stage5.service`
- config: `configs/task1/smp_fpn_convnext_small_640_stage5.yaml`
- master log: `quality_reports/logs/task1_stage5_convnext_small_640_master.log`
- training log: `quality_reports/logs/task1_smp_fpn_convnext_small_640_stage5.log`
- full released eval JSON after training:
  `outputs/task1/stage5_convnext_small_640_eval/convnext_small_640_hflip_original_full.json`

Smoke check passed before launch with 16 train samples, 8 eval samples, and 1
epoch. The full run started successfully and reached epoch 1 training progress.

## Stage B Outcome

Completed on 2026-05-29. `ConvNeXt-Small 640` did not become the single-model
champion, but it improved the ensemble:

- single-model full released Dice:
  `0.6373111721520722`
- best five-model ensemble full released Dice:
  `0.6500661421073305`
- best ensemble JSON:
  `outputs/task1/stage5_convnext_small640_ensemble_search/five_model_equal_tail_w035_035_010_010_010.json`

The new local champion weights are:

- `ConvNeXt-Tiny 640`: `0.35`
- `EfficientNet-B3 512`: `0.35`
- `ConvNeXtV2-Base 512`: `0.10`
- `ConvNeXt-Small 512`: `0.10`
- `ConvNeXt-Small 640`: `0.10`
