# Decision: Keep Stage 7 Toolness Auxiliary as an Ensemble Member, Not a Standalone Champion

Date: 2026-06-01

## Context

The previous Task 1 champion was the refined five-model ensemble:

- `ConvNeXt-Tiny 640`: `0.325`
- `EfficientNet-B3 512`: `0.325`
- `ConvNeXtV2-Base 512`: `0.117`
- `ConvNeXt-Small 512`: `0.117`
- `ConvNeXt-Small 640`: `0.116`
- mean Dice: `0.6516511688167698`

The weekend wrap-up plan called for testing whether a structure-aware
ConvNeXt-640 variant could improve the current ensemble or justify the next
method stage.

## Stage 6: ConvNeXt-Small 640 clDice

Run:

- service: `cathaction-task1-stage6-convnext-small640-cldice.service`
- config: `configs/task1/smp_fpn_convnext_small_640_cldice_stage6.yaml`
- master log: `quality_reports/logs/task1_stage6_convnext_small_640_cldice_master_systemd.log`
- full eval JSON:
  `outputs/task1/stage6_convnext_small_640_cldice_eval/convnext_small_640_cldice_hflip_original_full.json`

Single-model metrics:

- mean Dice: `0.6336600555146580`
- label_1 Dice: `0.6123160129455370`
- label_2 Dice: `0.6550040980837790`
- animal Dice: `0.7258733496359073`
- phantom Dice: `0.6084859513393589`

This did not beat the previous ConvNeXt-Small 640 single model
(`0.6373111721520722`), but a small ensemble weight was useful.

Best Stage 6 ensemble probe:

- JSON:
  `outputs/task1/stage6_convnext_small_640_cldice_eval/six_model_cldice_weight_probe_full.json`
- candidate: `add_cldice_010`
- weights:
  - `ConvNeXt-Tiny 640`: `0.2925`
  - `EfficientNet-B3 512`: `0.2925`
  - `ConvNeXtV2-Base 512`: `0.1053`
  - `ConvNeXt-Small 512`: `0.1053`
  - `ConvNeXt-Small 640`: `0.1044`
  - `ConvNeXt-Small 640 clDice`: `0.1000`

Metrics:

- mean Dice: `0.6523720412288835`
- label_1 Dice: `0.6313554715131297`
- label_2 Dice: `0.6733886109446374`
- animal Dice: `0.7370546254918469`
- phantom Dice: `0.6292538106268370`

## Stage 7: ConvNeXt-Small 640 Toolness Auxiliary

Run:

- service: `cathaction-task1-stage7-convnext-small640-toolness-aux.service`
- config: `configs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7.yaml`
- master log: `quality_reports/logs/task1_stage7_convnext_small_640_toolness_aux_master_systemd.log`
- full eval JSON:
  `outputs/task1/stage7_convnext_small_640_toolness_aux_eval/convnext_small_640_toolness_aux_hflip_original_full.json`

Single-model metrics:

- mean Dice: `0.6380117832841237`
- label_1 Dice: `0.6178994159397034`
- label_2 Dice: `0.6581241506285440`
- animal Dice: `0.7285606480626882`
- phantom Dice: `0.6132920660609931`

This is the best ConvNeXt-Small 640 single-model result so far, but it remains
well below the multi-model ensemble.

## Seven-Model Probe

Action:

Evaluated the refined five-model champion plus Stage 6 clDice and Stage 7
toolness auxiliary models in one full released-eval inference pass.

Output:

- JSON:
  `outputs/task1/stage7_convnext_small_640_toolness_aux_eval/seven_model_stage7_weight_probe_full.json`
- eval manifest: `configs/task1/splits/released_eval.csv`
- samples: `4691`
- TTA: `none + hflip`
- metric space: original mask space

Best candidate:

- name: `add_both_010_010`
- weights:
  - `ConvNeXt-Tiny 640`: `0.2600`
  - `EfficientNet-B3 512`: `0.2600`
  - `ConvNeXtV2-Base 512`: `0.0936`
  - `ConvNeXt-Small 512`: `0.0936`
  - `ConvNeXt-Small 640`: `0.0928`
  - `ConvNeXt-Small 640 clDice`: `0.1000`
  - `ConvNeXt-Small 640 toolness auxiliary`: `0.1000`

Metrics:

- mean Dice: `0.6536723455148790`
- label_1 Dice: `0.6327836065361310`
- label_2 Dice: `0.6745610844936268`
- animal Dice: `0.7394416932476355`
- phantom Dice: `0.6302574299601563`

Improvement over previous five-model champion:

- mean Dice: `+0.0020211766981092`
- label_1 Dice: `+0.0022288621439579`
- label_2 Dice: `+0.0018134912522602`
- animal Dice: `+0.0020616368173544`
- phantom Dice: `+0.0020101311404538`

## Decision

Do not promote Stage 6 or Stage 7 as standalone replacements. Promote the
seven-model `add_both_010_010` ensemble as the current local Task 1 champion.

The result supports the method direction:

- clDice and toolness auxiliary learning are not strong enough alone;
- they are complementary enough to improve the ensemble;
- the remaining bottleneck is likely still high-resolution ROI/patch
  refinement and phantom hard-case handling, not another full-frame backbone
  swap.

## Next Step

Proceed to a real method-stage improvement:

1. Build ROI/patch mining around foreground/skeleton/hard cases.
2. Keep the seven-model ensemble as the current submission fallback.
3. Use the current error-analysis overlays to drive patch sampling and phantom
   hard mining.
