# Plan: Task 1 EfficientNet-B3 ImageNet-Normalized Check

**Date:** 2026-05-24
**Status:** COMPLETED
**Scope:** One controlled follow-up run to make the EfficientNet-B3 comparison
fairer after the ConvNeXt rescue result.

## Context

Current top two single-model full released-eval results:

- ConvNeXt-Tiny FPN rescue: `0.6311257683876099`
- EfficientNet-B3 UNet original screen: `0.627019953686694`

The ConvNeXt rescue changed more than architecture:

- added ImageNet normalization;
- lowered learning rate;
- raised weight decay;
- disabled AMP;
- added gradient clipping.

The original EfficientNet-B3 screen was stable but did not use ImageNet
normalization despite using `encoder_weights: imagenet`.

## Question

Does EfficientNet-B3 close or exceed the ConvNeXt rescue result if it receives
the same pretrained-input normalization, while keeping its previously stable
training recipe otherwise unchanged?

## Configuration

Create:

- `configs/task1/smp_unet_efficientnet_b3_512_imagenet_norm.yaml`

Relative to `configs/task1/smp_unet_efficientnet_b3_512_shootout.yaml`:

- keep `architecture: Unet`;
- keep `encoder_name: efficientnet-b3`;
- keep 512 direct resize;
- keep DiceCE loss and class weights;
- keep `learning_rate: 0.0003`;
- keep `weight_decay: 0.0001`;
- keep `use_amp: true`;
- add ImageNet mean/std normalization.

This isolates the normalization effect rather than combining several recipe
changes at once.

## Verification

- YAML/config parse through the training script;
- one small CUDA smoke train;
- if smoke passes, launch full 50-epoch run;
- after completion, evaluate best checkpoint on full `released_eval.csv`;
- compare against both EfficientNet original and ConvNeXt rescue.

## Run Status

Static and smoke checks completed on 2026-05-24:

- `py_compile`: passed for the Task 1 training/eval scripts;
- config parse: passed;
- `git diff --check`: passed;
- CUDA smoke: passed with finite train/eval loss and checkpoint output.

Full 50-epoch run started:

- start time: `2026-05-24T17:02:20+02:00`
- config: `configs/task1/smp_unet_efficientnet_b3_512_imagenet_norm.yaml`
- log: `quality_reports/logs/task1_smp_unet_efficientnet_b3_512_imagenet_norm.log`
- output: `outputs/task1/smp_unet_efficientnet_b3_512_imagenet_norm`
- launcher PID: `827548`
- main training PID: `827583`
- early loss values were finite and decreasing through startup.

Full 50-epoch run completed:

- end time: `2026-05-24T19:50:00+02:00`
- exit status: `0`
- epochs: `50`

Best balanced-monitor checkpoint:

- epoch: `48`
- balanced monitor Dice: `0.6472360054722643`
- `label_1` Dice: `0.6474628763110862`
- `label_2` Dice: `0.6470091346334426`
- animal Dice: `0.6981806139852355`
- phantom Dice: `0.5962913969592933`

Full released-eval result for the best checkpoint:

- output: `outputs/task1/smp_unet_efficientnet_b3_512_imagenet_norm/eval_best_full_released.json`
- samples: `4691`
- mean Dice: `0.6212636066694707`
- `label_1` Dice: `0.6060514024182626`
- `label_2` Dice: `0.6364758109206787`
- animal Dice: `0.6998183110530815`
- phantom Dice: `0.5998182789598608`

Full released-eval result for the final checkpoint:

- output: `outputs/task1/smp_unet_efficientnet_b3_512_imagenet_norm/eval_final_full_released.json`
- samples: `4691`
- mean Dice: `0.6156795062202012`
- `label_1` Dice: `0.5960811248746367`
- `label_2` Dice: `0.6352778875657655`
- animal Dice: `0.6934233674687718`
- phantom Dice: `0.594455537586263`

Compared with EfficientNet-B3 original:

- mean Dice: `-0.005756347017223384`
- `label_1` Dice: `+0.0059322444483089765`
- `label_2` Dice: `-0.017444938482755745`
- animal Dice: `-0.007267719045394916`
- phantom Dice: `-0.005343744504240799`

Compared with ConvNeXt-Tiny FPN rescue:

- mean Dice: `-0.00986216171813925`
- `label_1` Dice: `-0.00273389075034014`
- `label_2` Dice: `-0.01699043268593825`
- animal Dice: `-0.028282954888977585`
- phantom Dice: `-0.004833310176792227`

Conclusion: adding ImageNet normalization to the previously strong
EfficientNet-B3 UNet recipe is harmful under this exact recipe. It slightly
improves `label_1` versus the original EfficientNet run, but the loss in
`label_2` and both domains lowers mean released Dice. Keep the original
EfficientNet-B3 checkpoint as the EfficientNet reference.

## Decision Rule

- If full released Dice beats `0.6311257683876099`, EfficientNet returns to the
  top single-model slot.
- If it remains within roughly `0.003-0.005`, keep both as co-best candidates
  and prioritize ensemble/TTA.
- If it drops materially, keep the original EfficientNet checkpoint as the
  stronger EfficientNet reference and treat normalization as harmful under this
  exact recipe.
