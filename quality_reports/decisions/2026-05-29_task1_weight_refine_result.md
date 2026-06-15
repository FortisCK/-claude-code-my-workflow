# Decision: Promote Refined Five-Model Task 1 Ensemble

Date: 2026-05-29

## Context

The previous local Task 1 champion was:

- `ConvNeXt-Tiny 640`: `0.35`
- `EfficientNet-B3 512`: `0.35`
- `ConvNeXtV2-Base 512`: `0.10`
- `ConvNeXt-Small 512`: `0.10`
- `ConvNeXt-Small 640`: `0.10`
- mean Dice: `0.6500661421073305`

## Action

Ran a narrow five-model weight refinement around the current champion using a
single inference pass for all candidate weight vectors.

Command:

```bash
scripts/task1/run_stage5_weight_refine.sh outputs/task1/stage5_weight_refine
```

Output:

- JSON: `outputs/task1/stage5_weight_refine/five_model_weight_refine_full.json`
- log: `outputs/task1/stage5_weight_refine/five_model_weight_refine_full.log`
- candidates: `17`
- TTA: `none + hflip`
- eval manifest: `configs/task1/splits/released_eval.csv`
- metric space: original mask space

## Best Result

Promoted candidate:

- name: `more_tail_equal_0325_0325_0117_0117_0116`
- `ConvNeXt-Tiny 640`: `0.325`
- `EfficientNet-B3 512`: `0.325`
- `ConvNeXtV2-Base 512`: `0.117`
- `ConvNeXt-Small 512`: `0.117`
- `ConvNeXt-Small 640`: `0.116`

Metrics:

- mean Dice: `0.6516511688167698`
- label_1 Dice: `0.6305547443921731`
- label_2 Dice: `0.6727475932413666`
- animal Dice: `0.7373800564302811`
- phantom Dice: `0.6282472988197025`

Improvement over previous five-model champion:

- mean Dice: `+0.0015850267094393`
- label_1 Dice: `+0.0004347496523799`
- label_2 Dice: `+0.0027353037664987`
- animal Dice: `+0.0021027368925028`
- phantom Dice: `+0.0014436925319189`

Improvement over original two-model champion:

- original two-model mean Dice: `0.6438677932546133`
- refined five-model mean Dice: `0.6516511688167698`
- delta: `+0.0077833755621565`

## Ranked Candidates

| Candidate | Mean Dice | label_1 | label_2 | animal | phantom |
| --- | ---: | ---: | ---: | ---: | ---: |
| `more_tail_equal_0325_0325_0117_0117_0116` | `0.6516511688167698` | `0.6305547443921731` | `0.6727475932413666` | `0.7373800564302811` | `0.6282472988197025` |
| `smalls_plus_034_034_008_012_012` | `0.6509598724080684` | `0.6303480784463564` | `0.6715716663697804` | `0.7362247628937054` | `0.6276826729973355` |
| `main_c640_plus2_037_033_010_010_010` | `0.6506519399320451` | `0.6301651896241356` | `0.6711386902399545` | `0.7357868893722047` | `0.6274102142504167` |
| `main_effb3_plus2_033_037_010_010_010` | `0.6503205034911815` | `0.6298522166753261` | `0.6707887903070372` | `0.7365245985694052` | `0.6267869025010342` |
| `main_effb3_plus_034_036_010_010_010` | `0.6502628191596337` | `0.6300171204510884` | `0.6705085178681789` | `0.7361790878146212` | `0.6268077943924919` |
| `main_c640_plus_036_034_010_010_010` | `0.6501744781202128` | `0.6301558968232448` | `0.6701930594171807` | `0.7352848476249370` | `0.6269394627276070` |
| `tail_v2_plus_035_035_012_009_009` | `0.6500854322487668` | `0.6300316723727569` | `0.6701391921247765` | `0.7350407983044378` | `0.6268927325874356` |
| `tail_s512_plus_035_035_009_012_009` | `0.6500777949944976` | `0.6299914686509950` | `0.6701641213380001` | `0.7352231503818277` | `0.6268332285034110` |
| `incumbent_035_035_010_010_010` | `0.6500661421073305` | `0.6301199947397932` | `0.6700122894748679` | `0.7352773195377783` | `0.6268036062877836` |

## Decision

Promote `more_tail_equal_0325_0325_0117_0117_0116` as the new local Task 1
champion.

The result suggests the ensemble benefits from slightly more total tail weight:

- previous main pair total: `0.70`
- refined main pair total: `0.65`
- previous tail total: `0.30`
- refined tail total: `0.35`

Next ensemble search, if run, should explore tail total around `0.34` to `0.38`
with small perturbations among the three tail models.
