# Decision: Promote ConvNeXt-Family Four-Model Ensemble

Date: 2026-05-28

## Context

The previous Task 1 local champion was:

- `ConvNeXt-Tiny 640 rescue + EfficientNet-B3 512`
- weights `0.50 / 0.50`
- hflip TTA
- original-mask-space evaluation
- released eval mean Dice: `0.6438677932546133`

Stage 4A showed that ConvNeXt-family single models were the strongest new
architecture candidates, but no single model beat the existing champion
ensemble.

## Action

Ran Stage 5 no-training ensemble search over existing checkpoints:

- `ConvNeXt-Tiny 640 rescue`
- `EfficientNet-B3 512`
- `ConvNeXtV2-Base 512`
- `ConvNeXt-Small 512`
- one low-priority `EfficientNet-B5 512` complement check

Command:

```bash
scripts/task1/run_stage5_convnext_ensemble_search.sh \
  outputs/task1/stage5_convnext_ensemble_search_full
```

Evaluation:

- manifest: `configs/task1/splits/released_eval.csv`
- samples: `4691`
- TTA: `none + hflip`
- metric space: original mask space

## Result

Best Stage 5 result:

- JSON: `outputs/task1/stage5_convnext_ensemble_search_full/w040_040_010_010_convnextv2_small.json`
- weights:
  - `ConvNeXt-Tiny 640`: `0.40`
  - `EfficientNet-B3 512`: `0.40`
  - `ConvNeXtV2-Base 512`: `0.10`
  - `ConvNeXt-Small 512`: `0.10`
- mean Dice: `0.648019210540978`
- label_1 Dice: `0.628068019483919`
- label_2 Dice: `0.667970401598037`
- animal Dice: `0.7326960974347608`
- phantom Dice: `0.6249025353129875`

This improves over the previous local champion by:

- mean Dice: `+0.0041514172863647`
- label_1 Dice: `+0.0041782150291756`
- label_2 Dice: `+0.0041246195435540`
- animal Dice: `+0.0050718335743387`
- phantom Dice: `+0.0039001448886167`

## Ranked Stage 5 Candidates

| Candidate | Mean Dice | label_1 | label_2 | animal | phantom |
| --- | ---: | ---: | ---: | ---: | ---: |
| `w040_040_010_010_convnextv2_small` | `0.648019210540978` | `0.628068019483919` | `0.667970401598037` | `0.7326960974347608` | `0.6249025353129875` |
| `w040_040_020_convnext_small` | `0.6474637438802847` | `0.6272425704383731` | `0.6676849173221967` | `0.7321037666860658` | `0.6243571324982997` |
| `w040_040_020_convnextv2` | `0.6471099178449295` | `0.6273457124358794` | `0.6668741232539797` | `0.7311899568271593` | `0.6241561812869586` |
| `w045_045_010_convnext_small` | `0.6468132645941737` | `0.6262862685333506` | `0.6673402606549969` | `0.7304905857032354` | `0.6239694694691490` |
| `w045_045_010_efficientnet_b5` | `0.6464961086431567` | `0.6268432893989886` | `0.6661489278873246` | `0.7292955543530825` | `0.6238919723109488` |
| `w045_045_010_convnextv2` | `0.6463267988438620` | `0.6265703347683460` | `0.6660832629193781` | `0.7302676930307131` | `0.6234110486262304` |
| `w050_040_010_convnextv2` | `0.6452714587808476` | `0.6254766319034966` | `0.6650662856581985` | `0.7309775374275870` | `0.6218738156007608` |
| `w040_050_010_convnextv2` | `0.6451927136747281` | `0.6249246910296457` | `0.6654607363198105` | `0.7289557249372742` | `0.6223255252540709` |

## Decision

Promote `w040_040_010_010_convnextv2_small` as the new local Task 1 champion.

Next ConvNeXt-family training should prioritize `ConvNeXt-Small 640
aspect-pad` before `ConvNeXtV2-Base 640`, because:

- `ConvNeXt-Small 512` showed stronger ensemble complementarity than expected;
- `ConvNeXtV2-Base 640` is likely much slower based on the Stage 4A 512 runtime;
- high-resolution ConvNeXt-Small is the better next overnight experiment under
  the current time budget.
