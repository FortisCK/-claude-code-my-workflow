# Task 1 Stage 6 clDice Result

Date: 2026-05-31

## Context

Stage 6 tested whether a `ConvNeXt-Small 640` model with adjusted class
weights and class-wise soft-clDice could fix the current champion's class
confusion and thin-structure errors.

## Configuration

- Config:
  `configs/task1/smp_fpn_convnext_small_640_cldice_stage6.yaml`
- Checkpoint:
  `outputs/task1/smp_fpn_convnext_small_640_cldice_stage6/best_checkpoint.pt`
- Full eval:
  `outputs/task1/stage6_convnext_small_640_cldice_eval/convnext_small_640_cldice_hflip_original_full.json`
- Six-model ensemble probe:
  `outputs/task1/stage6_convnext_small_640_cldice_eval/six_model_cldice_weight_probe_full.json`

## Result

Standalone Stage 6 result on full released eval:

| Metric | Value |
| --- | ---: |
| mean Dice | `0.633660055514658` |
| label_1 Dice | `0.612316012945537` |
| label_2 Dice | `0.655004098083779` |
| animal Dice | `0.7258733496359073` |
| phantom Dice | `0.6084859513393589` |

Previous `ConvNeXt-Small 640` single-model result:

| Metric | Value |
| --- | ---: |
| mean Dice | `0.6373111721520722` |
| label_1 Dice | `0.6161710981745443` |
| label_2 Dice | `0.6584512461296002` |
| animal Dice | `0.7281588395844082` |
| phantom Dice | `0.612509882209893` |

The Stage 6 standalone model is therefore worse than the previous
`ConvNeXt-Small 640` model.

## Ensemble Probe

Best six-model candidate:

| Model | Weight |
| --- | ---: |
| `ConvNeXt-Tiny 640` | `0.2925` |
| `EfficientNet-B3 512` | `0.2925` |
| `ConvNeXtV2-Base 512` | `0.1053` |
| `ConvNeXt-Small 512` | `0.1053` |
| `ConvNeXt-Small 640` | `0.1044` |
| `ConvNeXt-Small 640 clDice` | `0.10` |

Best six-model result:

| Metric | Value |
| --- | ---: |
| mean Dice | `0.6523720412288835` |
| label_1 Dice | `0.6313554715131297` |
| label_2 Dice | `0.6733886109446374` |
| animal Dice | `0.7370546254918469` |
| phantom Dice | `0.629253810626837` |

Previous five-model champion:

| Metric | Value |
| --- | ---: |
| mean Dice | `0.6516511688167698` |
| label_1 Dice | `0.6305547443921731` |
| label_2 Dice | `0.6727475932413666` |
| animal Dice | `0.7373800564302811` |
| phantom Dice | `0.6282472988197025` |

The six-model gain is `+0.0007208724121137` mean Dice. It is too small to
justify more full-frame clDice exploration, but the checkpoint can remain as a
low-weight ensemble tail.

## Decision

- Keep the current main direction unchanged: `ConvNeXt`/`EfficientNet`
  full-frame ensemble remains the champion family.
- Do not continue broad full-frame clDice tuning.
- For the next meaningful Task 1 improvement, move to toolness auxiliary
  training and ROI/patch refinement.
