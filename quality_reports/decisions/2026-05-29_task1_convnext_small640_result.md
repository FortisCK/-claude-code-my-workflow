# Decision: Add ConvNeXt-Small 640 to the Task 1 Ensemble

Date: 2026-05-29

## Context

The previous local Task 1 champion was the Stage 5 four-model ensemble:

- `ConvNeXt-Tiny 640`: `0.40`
- `EfficientNet-B3 512`: `0.40`
- `ConvNeXtV2-Base 512`: `0.10`
- `ConvNeXt-Small 512`: `0.10`

Full released original-mask-space result:

- mean Dice: `0.648019210540978`
- label_1 Dice: `0.628068019483919`
- label_2 Dice: `0.667970401598037`
- animal Dice: `0.7326960974347608`
- phantom Dice: `0.6249025353129875`

## ConvNeXt-Small 640 Training

Training run:

- config: `configs/task1/smp_fpn_convnext_small_640_stage5.yaml`
- service: `cathaction-task1-convnext-small-640-stage5.service`
- master log: `quality_reports/logs/task1_stage5_convnext_small_640_master.log`
- train log: `quality_reports/logs/task1_smp_fpn_convnext_small_640_stage5.log`
- full eval log: `quality_reports/logs/task1_smp_fpn_convnext_small_640_stage5_full_eval.log`

The balanced 1024-sample monitor peaked at epoch 58:

- balanced Dice: `0.6690365418488515`
- label_1 Dice: `0.6637175049246149`
- label_2 Dice: `0.6743555787730879`

Full released single-model evaluation:

- JSON: `outputs/task1/stage5_convnext_small_640_eval/convnext_small_640_hflip_original_full.json`
- mean Dice: `0.6373111721520722`
- label_1 Dice: `0.6161710981745443`
- label_2 Dice: `0.6584512461296002`
- animal Dice: `0.7281588395844082`
- phantom Dice: `0.612509882209893`

Interpretation: `ConvNeXt-Small 640` is not a new single-model champion, but it
is stronger than the `ConvNeXt-Small 512` single-model full released result and
has useful ensemble complementarity.

## Ensemble Search with ConvNeXt-Small 640

Command:

```bash
scripts/task1/run_stage5_convnext_small640_ensemble_search.sh \
  outputs/task1/stage5_convnext_small640_ensemble_search
```

Best result:

- JSON:
  `outputs/task1/stage5_convnext_small640_ensemble_search/five_model_equal_tail_w035_035_010_010_010.json`
- weights:
  - `ConvNeXt-Tiny 640`: `0.35`
  - `EfficientNet-B3 512`: `0.35`
  - `ConvNeXtV2-Base 512`: `0.10`
  - `ConvNeXt-Small 512`: `0.10`
  - `ConvNeXt-Small 640`: `0.10`
- mean Dice: `0.6500661421073305`
- label_1 Dice: `0.6301199947397932`
- label_2 Dice: `0.6700122894748679`
- animal Dice: `0.7352773195377783`
- phantom Dice: `0.6268036062877836`

Improvement over previous four-model champion:

- mean Dice: `+0.0020469315663525`
- label_1 Dice: `+0.0020519752558742`
- label_2 Dice: `+0.0020418878768300`
- animal Dice: `+0.0025812221030175`
- phantom Dice: `+0.0019010709747961`

Improvement over original two-model champion:

- previous two-model mean Dice: `0.6438677932546133`
- new five-model mean Dice: `0.6500661421073305`
- delta: `+0.0061983488527172`

## Ranked Small640 Ensemble Candidates

| Candidate | Mean Dice | label_1 | label_2 | animal | phantom |
| --- | ---: | ---: | ---: | ---: | ---: |
| `five_model_equal_tail_w035_035_010_010_010` | `0.6500661421073305` | `0.6301199947397932` | `0.6700122894748679` | `0.7352773195377783` | `0.6268036062877836` |
| `small640_heavy_tail_w035_035_010_020` | `0.6493583024141275` | `0.6294871851017293` | `0.6692294197265259` | `0.7339478490735596` | `0.6262654709516070` |
| `two_small_tail_w035_035_010_020` | `0.6493226372962445` | `0.6299486012258428` | `0.6686966733666463` | `0.7351207729514637` | `0.6258998626777503` |
| `replace_small512_w040_040_010_010` | `0.6479664985241619` | `0.6285879143293850` | `0.6673450827189387` | `0.7325748382383183` | `0.6248685365832009` |
| `small640_w040_040_020` | `0.6477660423350107` | `0.6283682092764731` | `0.6671638753935484` | `0.7320551906689525` | `0.6247552192077528` |

## Decision

Promote the five-model equal-tail ensemble as the new local Task 1 champion.

Next useful step is a narrow ensemble-weight refinement around:

- main pair total: `0.65` to `0.75`
- tail models total: `0.25` to `0.35`
- keep both `ConvNeXt-Small 512` and `ConvNeXt-Small 640` unless a refined
  search proves one is redundant.
