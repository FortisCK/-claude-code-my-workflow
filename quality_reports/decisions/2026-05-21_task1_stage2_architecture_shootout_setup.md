# Decision: Task 1 Stage 2 Architecture Shootout Setup

**Date:** 2026-05-21
**Status:** Active

## Purpose

Move from UNet training-strategy ablations to a stronger architecture screen
while keeping the first comparison controlled:

- 512 x 512 direct resize
- multiclass `0/1/2` masks
- MONAI DiceCE loss
- normal train shuffling
- balanced released-eval monitor set

## Added Candidates

Configured first-wave SMP candidates:

- `configs/task1/smp_unet_mit_b2_512_shootout.yaml`
- `configs/task1/smp_unet_efficientnet_b3_512_shootout.yaml`
- `configs/task1/smp_fpn_convnext_tiny_512_shootout.yaml`
- `configs/task1/smp_fpn_resnet50_512_shootout.yaml`

The MiT-B2 candidate is the first Transformer/SegFormer-like architecture
screen. Pure HuggingFace SegFormer integration remains optional after the first
wave signal.

## Balanced Monitor Split

Added:

- `configs/task1/splits/released_eval_balanced_1024.csv`

Composition:

- 512 `animal`
- 512 `phantom`

This avoids the old `max_eval_samples: 512` behavior where the ordered released
eval CSV sampled only `animal_test` rows.

## Verification

Completed checks:

- `python3 -m py_compile src/cathaction/training/task1_baseline.py scripts/task1/train_baseline.py scripts/task1/evaluate_baseline.py scripts/task1/predict_baseline.py`
- YAML parsing for all `*shootout.yaml` configs
- balanced monitor count check: 1024 rows, 512 animal, 512 phantom
- `cathaction-task1` unit test: 8 passed, 1 skipped
- `cardiac-diffusion` direct SMP forward check: output shape `(1, 3, 64, 64)`
- `git diff --check`

`cardiac-diffusion` does not currently have `pytest`, so the SMP branch was
verified there with a direct Python model construction and forward pass.

## First Active Run

Started:

- config: `configs/task1/smp_unet_mit_b2_512_shootout.yaml`
- log: `quality_reports/logs/task1_smp_unet_mit_b2_512_shootout.log`
- output: `outputs/task1/smp_unet_mit_b2_512_shootout`
- launcher PID: `664492`

Epoch 1 balanced monitor metrics:

- mean Dice: `0.45025541594366203`
- `label_1` Dice: `0.5783878776195535`
- `label_2` Dice: `0.3221229542677707`
- animal Dice: `0.4800553241738727`
- phantom Dice: `0.4204555077134514`

Interpretation: epoch 1 is not comparable to the 100-epoch MONAI UNet anchor,
but the run is healthy and learning non-trivial foreground structure.

## Completed Result

Training completed normally:

- end time: `2026-05-21T21:43:19+02:00`
- exit status: `0`
- epochs: `50`

Best balanced-monitor checkpoint:

- epoch: `36`
- balanced monitor Dice: `0.5664497634128358`
- `label_1` Dice: `0.6201485319013156`
- `label_2` Dice: `0.5127509949243563`
- animal Dice: `0.6140205709691877`
- phantom Dice: `0.518878955856484`

Full released-eval result for the best checkpoint:

- output: `outputs/task1/smp_unet_mit_b2_512_shootout/eval_best_full_released.json`
- samples: `4691`
- mean Dice: `0.5431901033460851`
- `label_1` Dice: `0.5958993409185247`
- `label_2` Dice: `0.4904808657736453`
- animal Dice: `0.6213138562177019`
- phantom Dice: `0.5218624248145121`

Compared with the previous MONAI UNet full-eval anchor:

- mean Dice: `+0.012829777509168789`
- `label_1` Dice: `+0.004129755559615452`
- `label_2` Dice: `+0.021529799458721897`
- animal Dice: `-0.002756786542088195`
- phantom Dice: `+0.01708488834649967`

Conclusion: MiT-B2 is a valid Stage 2 winner candidate. The gain is modest but
real on full released eval, mainly through `label_2` and phantom improvement.

## Second Active Run

Started the next architecture screen:

- config: `configs/task1/smp_unet_efficientnet_b3_512_shootout.yaml`
- log: `quality_reports/logs/task1_smp_unet_efficientnet_b3_512_shootout.log`
- output: `outputs/task1/smp_unet_efficientnet_b3_512_shootout`
- launcher PID: `714431`
- main training PID: `714465`

Smoke test completed first:

- log: `quality_reports/logs/task1_smp_unet_efficientnet_b3_512_smoke.log`
- output: `outputs/task1/smp_unet_efficientnet_b3_512_smoke`
- status: passed; checkpoint and metrics were written

Initial full-run status:

- start time: `2026-05-22T10:27:46+02:00`
- GPU memory for main process: about `7.8GB`
- early throughput: about `8 it/s`
- epoch size: `1564` train batches

## EfficientNet-B3 Completed Result

Training completed normally:

- end time: `2026-05-22T13:21:46+02:00`
- exit status: `0`
- epochs: `50`

Best balanced-monitor checkpoint:

- epoch: `41`
- balanced monitor Dice: `0.6550091608871695`
- `label_1` Dice: `0.6398593384420751`
- `label_2` Dice: `0.6701589833322641`
- animal Dice: `0.7071171786546679`
- phantom Dice: `0.6029011431196711`

Full released-eval result for the best checkpoint:

- output: `outputs/task1/smp_unet_efficientnet_b3_512_shootout/eval_best_full_released.json`
- samples: `4691`
- mean Dice: `0.627019953686694`
- `label_1` Dice: `0.6001191579699536`
- `label_2` Dice: `0.6539207494034345`
- animal Dice: `0.7070860300984764`
- phantom Dice: `0.6051620234641016`

Compared with the MONAI UNet full-eval anchor:

- mean Dice: `+0.09665962784977777`
- `label_1` Dice: `+0.008349572611044298`
- `label_2` Dice: `+0.18496968308851108`
- animal Dice: `+0.0830153873386863`
- phantom Dice: `+0.10038448699608926`

Compared with the MiT-B2 full-eval result:

- mean Dice: `+0.08382985034060897`
- `label_1` Dice: `+0.004219817051428931`
- `label_2` Dice: `+0.16343988362978917`
- animal Dice: `+0.08577217388077452`
- phantom Dice: `+0.08329959864958958`

Conclusion: EfficientNet-B3 UNet is the current clear Stage 2 winner. The main
gain is `label_2`, but both animal and phantom domains improve strongly.

## ConvNeXt-Tiny FPN Completed Result

Initial ConvNeXt-Tiny FPN screen:

- config: `configs/task1/smp_fpn_convnext_tiny_512_shootout.yaml`
- implementation: SMP `FPN` decoder with `tu-convnext_tiny` ImageNet encoder
- log: `quality_reports/logs/task1_smp_fpn_convnext_tiny_512_shootout.log`
- epochs: `50`
- exit status: `0`

The run diverged numerically:

- best balanced-monitor checkpoint: epoch `7`
- best balanced-monitor Dice: `0.5100636033468431`
- `label_1` Dice: `0.6192568789448915`
- `label_2` Dice: `0.4008703277487946`
- loss became `NaN` from epoch `8`

Full released-eval result for the best checkpoint:

- output: `outputs/task1/smp_fpn_convnext_tiny_512_shootout/eval_best_full_released.json`
- samples: `4691`
- mean Dice: `0.4801980713555774`
- `label_1` Dice: `0.5782261531691921`
- `label_2` Dice: `0.38216998954196274`
- animal Dice: `0.5689914828436068`
- phantom Dice: `0.45595759049887247`

Conclusion: the initial ConvNeXt screen is not competitive and is numerically
unstable under the generic Stage 2 recipe.

## ConvNeXt Rescue Check

The official ConvNeXt repository was cloned for recipe comparison:

- path: `external_repos/ConvNeXt`
- git policy: ignored by `.gitignore`

Useful differences from the official downstream segmentation recipe:

- official semantic segmentation uses UPerNet rather than FPN;
- ImageNet mean/std normalization is expected;
- AdamW uses `lr=0.0001` and `weight_decay=0.05`;
- schedule uses warmup + poly decay;
- configs include layer-wise LR decay;
- official segmentation uses AMP hooks, but the classification script also
  supports disabling AMP and explicit gradient clipping.

Added rescue config:

- `configs/task1/smp_fpn_convnext_tiny_512_rescue_stable.yaml`
- ImageNet normalization enabled
- learning rate lowered from `0.0003` to `0.0001`
- weight decay raised to ConvNeXt-style `0.05`
- AMP disabled
- gradient clipping set to `1.0`

Verification before launch:

- `py_compile` passed for the training/eval scripts
- direct normalization smoke passed
- `pytest` component tests could not run because neither base nor
  `cardiac-diffusion` currently has `pytest` installed
- small CUDA smoke passed with finite loss and exit status `0`

Full rescue run started:

- log: `quality_reports/logs/task1_smp_fpn_convnext_tiny_512_rescue_stable.log`
- output: `outputs/task1/smp_fpn_convnext_tiny_512_rescue_stable`
- launcher PID: `745823`
- main training PID: `745858`
- start time: `2026-05-22T16:48:50+02:00`
- early loss values were finite through startup; no immediate NaN observed

Full rescue run completed:

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

Full released-eval result for the final checkpoint:

- output: `outputs/task1/smp_fpn_convnext_tiny_512_rescue_stable/eval_final_full_released.json`
- mean Dice: `0.6305445227190412`
- `label_1` Dice: `0.6085550893978662`
- `label_2` Dice: `0.6525339560402161`
- animal Dice: `0.7208472468119479`
- phantom Dice: `0.6058920015691187`

Compared with the EfficientNet-B3 full-eval result:

- mean Dice: `+0.004105814700915866`
- `label_1` Dice: `+0.008666135198649116`
- `label_2` Dice: `-0.0004545057968174948`
- animal Dice: `+0.02101523584358267`
- phantom Dice: `-0.0005104343274485723`

Conclusion: ConvNeXt-Tiny FPN rescue is now the top Stage 2 single-model by
mean full released Dice, but the margin over EfficientNet-B3 is narrow. The win
comes mostly from animal-domain and `label_1` improvements; `label_2` and
phantom are effectively tied or slightly worse.

## EfficientNet-B3 ImageNet-Normalization Check

Rationale: ConvNeXt rescue used ImageNet mean/std normalization while the
original EfficientNet-B3 UNet screen used ImageNet weights without explicit
input normalization. A controlled follow-up isolated this one variable while
keeping the original EfficientNet recipe otherwise unchanged.

Configuration:

- config: `configs/task1/smp_unet_efficientnet_b3_512_imagenet_norm.yaml`
- architecture: SMP `Unet`
- encoder: `efficientnet-b3`
- encoder weights: `imagenet`
- learning rate: `0.0003`
- weight decay: `0.0001`
- AMP: enabled
- added ImageNet normalization

Training completed normally:

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

Conclusion: ImageNet normalization does not explain ConvNeXt rescue's narrow
lead. Under the original EfficientNet-B3 recipe it hurts released full eval,
mostly by reducing `label_2`. Keep original EfficientNet-B3 as the EfficientNet
reference and keep ConvNeXt rescue as the current top single-model checkpoint.
