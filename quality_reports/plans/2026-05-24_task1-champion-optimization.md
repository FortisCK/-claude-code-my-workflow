# Plan: Task 1 Champion Optimization

**Date:** 2026-05-24
**Status:** STAGE 3A COMPLETED; STAGE 3B COMPLETED
**Scope:** Improve the current best Task 1 segmentation route before adding
more unrelated architectures.

## Context

Current best single-model full released-eval results:

- ConvNeXt-Tiny FPN rescue: `0.6311257683876099`
- EfficientNet-B3 UNet original: `0.627019953686694`

The two checkpoints are close but not identical in error profile:

- ConvNeXt rescue is stronger on mean Dice, `label_1`, and animal domain.
- EfficientNet-B3 original is competitive on `label_2` and phantom domain.

This makes eval-time improvements and soft ensembling the lowest-cost next
step before launching new overnight training.

## Stage 3A: Eval-Only Optimization

Implement a dedicated evaluator that can run:

- single-checkpoint no-TTA sanity checks;
- horizontal-flip TTA;
- optional vertical-flip TTA only if explicitly enabled;
- softmax-probability ensemble across checkpoints;
- simple weight sweep for ConvNeXt/EfficientNet combinations;
- full released-eval metrics with per-class and per-domain reporting.

Initial experiments:

1. ConvNeXt-Tiny FPN rescue best checkpoint, no TTA sanity check.
2. ConvNeXt-Tiny FPN rescue best checkpoint, horizontal-flip TTA.
3. ConvNeXt-Tiny FPN rescue + EfficientNet-B3 original soft ensemble:
   - `0.5 / 0.5`
   - `0.6 / 0.4`
   - `0.7 / 0.3`

Primary comparison target:

- ConvNeXt-Tiny FPN rescue best full released Dice: `0.6311257683876099`

Secondary comparison target:

- EfficientNet-B3 original best full released Dice: `0.627019953686694`

## Stage 3B: Training-Side Optimization

Only after Stage 3A:

1. ConvNeXt rescue at `640 x 640`.
2. ConvNeXt rescue at `640 x 640` plus class-wise soft-clDice.
3. Domain-balanced training or sampling changes if domain metrics indicate a
   clear imbalance.

Started first Stage 3B run:

- config: `configs/task1/smp_fpn_convnext_tiny_640_rescue_stable.yaml`
- output: `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable`
- log: `quality_reports/logs/task1_smp_fpn_convnext_tiny_640_rescue_stable.log`
- systemd user unit: `cathaction-task1-convnext640-rescue.service`
- recipe:
  - `640 x 640`
  - `aspect_pad`
  - ImageNet normalization
  - MONAI DiceCE
  - `lr=0.0001`
  - `weight_decay=0.05`
  - AMP disabled
  - gradient clipping `1.0`
  - 80 epochs
  - batch size 8

Pre-launch checks:

- YAML/config load passed.
- 1-epoch CUDA smoke on 16 train / 8 eval samples passed:
  - output: `outputs/task1/smp_fpn_convnext_tiny_640_rescue_smoke`
  - finite train loss: `2.541735887527466`
  - checkpoint written.
- Full training service is active.
- Initial GPU process observed:
  - process: Python PID `844782`
  - GPU memory: about `9898 MB`
- Initial log showed epoch 1 progressing with finite loss around `1.0024`.

Restarted on 2026-05-25:

- Issue found: the training loop only wrote `best_checkpoint.pt`,
  `checkpoint.pt`, and `metrics.json` after all epochs completed. That is too
  fragile for long 80-epoch runs.
- Fix: `src/cathaction/training/task1_baseline.py` now writes:
  - `checkpoint.pt` and `metrics.json` after every epoch;
  - `best_checkpoint.pt` immediately when the monitored metric improves.
- The partial pre-fix log was preserved as:
  - `quality_reports/logs/task1_smp_fpn_convnext_tiny_640_rescue_stable_pre_checkpoint_restart.log`
- The `640 x 640` service was restarted with the fixed loop:
  - systemd user unit: `cathaction-task1-convnext640-rescue.service`
  - new GPU process: Python PID `850238`
  - GPU memory: about `9898 MB`
  - initial epoch 1 loss values were finite.
- Per-epoch persistence verified after epoch 1:
  - `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/best_checkpoint.pt`
  - `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/checkpoint.pt`
  - `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/metrics.json`
- Epoch 1 balanced-monitor metrics:
  - mean Dice: `0.5449038706854183`
  - `label_1`: `0.6173195653414567`
  - `label_2`: `0.47248817602938`
  - animal: `0.5837925472211452`
  - phantom: `0.5060151941496914`
- Training continued into epoch 2 after the checkpoint write.

## Prediction Path

Added an ensemble/TTA prediction writer:

- `scripts/task1/predict_tta_ensemble.py`

Added an original-mask-space evaluator:

- `scripts/task1/evaluate_tta_ensemble_original_space.py`

Rationale:

- The current active champion uses `direct` resize at `512 x 512`.
- The Stage 3B ConvNeXt run uses `aspect_pad` at `640 x 640`.
- These preprocessing geometries should not be ensembled directly in resized
  tensor space. The original-space evaluator maps every model's probability
  output back to the original mask shape before ensembling and computing
  metrics.

Added a post-training evaluation runner:

- `scripts/task1/run_stage3_convnext640_eval.sh`

It is intended to run after the `640 x 640` training service completes and will
write:

- `convnext640_best_resized_full.json`
- `incumbent_512_ensemble_w050_050_hflip_original_full.json`
- `convnext640_best_hflip_original_full.json`
- `ensemble_convnext640_efficientnet512_w050_050_hflip_original_full.json`

It reuses the Stage 3A evaluator's model loading, TTA, and softmax-probability
averaging logic, then writes:

- PNG masks;
- `predictions.csv`;
- `summary.json`.

Smoke verification:

- command used current champion:
  - ConvNeXt rescue `0.5`
  - EfficientNet-B3 original `0.5`
  - hflip TTA
- 16 released-eval samples generated:
  - `outputs/task1/champion_eval/predict_smoke_ensemble_w050_050_hflip_16`
- PNG count: `16`
- `predictions.csv` rows: `16` plus header
- The existing evaluator read the generated manifest successfully:
  - matched predictions: `16 / 16`
  - no missing sample IDs
  - smoke Dice: `0.7471756569997943`
- Original-space evaluator smoke verification:
  - output: `outputs/task1/champion_eval/original_space_smoke_ensemble_w050_050_hflip_2.json`
  - samples: `2`
  - device: CPU, to avoid interfering with the active GPU training service
  - smoke Dice: `0.6975087523267691`

## Verification

- Python compile checks for new scripts and touched modules.
- Small max-sample CUDA smoke for the new evaluator.
- Full released eval for any result considered comparable.
- Record output JSON paths and metric deltas in this plan and the Stage 2/3
  decision log.

Completed:

- `py_compile` passed for:
  - `scripts/task1/evaluate_tta_ensemble.py`
  - `scripts/task1/evaluate_tta_ensemble_original_space.py`
  - `scripts/task1/predict_tta_ensemble.py`
  - `src/cathaction/training/task1_baseline.py`
- `bash -n` passed for:
  - `scripts/task1/run_stage3_convnext640_eval.sh`
- `git diff --check` passed before the full Stage 3A sweep.
- Small CUDA smoke passed on 32 released-eval samples:
  - `outputs/task1/champion_eval/convnext_best_notta_smoke32.json`
- ConvNeXt no-TTA sanity matched the existing evaluator exactly on full
  released eval:
  - old reference Dice: `0.6311257683876099`
  - new evaluator Dice: `0.6311257683876099`

## Stage 3A Results

Full released-eval set: `configs/task1/splits/released_eval.csv`

| Candidate | TTA | Mean Dice | Label 1 | Label 2 | Animal | Phantom | Output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt rescue | none | `0.631125768388` | `0.608785293169` | `0.653466243607` | `0.728101265942` | `0.604651589137` | `outputs/task1/champion_eval/convnext_best_notta_full_released.json` |
| ConvNeXt rescue | hflip | `0.631236797431` | `0.610529352228` | `0.651944242633` | `0.721221271419` | `0.606671158128` | `outputs/task1/champion_eval/convnext_best_hflip_tta_full_released.json` |
| ConvNeXt/EfficientNet `0.4/0.6` | none | `0.630756374976` | `0.609082140838` | `0.652430609113` | `0.711162924950` | `0.608805495932` | `outputs/task1/champion_eval/ensemble_convnext_efficientnet_w040_060_notta_full_released.json` |
| ConvNeXt/EfficientNet `0.5/0.5` | none | `0.635726895204` | `0.614962793296` | `0.656490997113` | `0.720209478370` | `0.612663264630` | `outputs/task1/champion_eval/ensemble_convnext_efficientnet_w050_050_notta_full_released.json` |
| ConvNeXt/EfficientNet `0.6/0.4` | none | `0.632226610623` | `0.613305942099` | `0.651147279147` | `0.721078716922` | `0.607970106163` | `outputs/task1/champion_eval/ensemble_convnext_efficientnet_w060_040_notta_full_released.json` |
| ConvNeXt/EfficientNet `0.7/0.3` | none | `0.635676526936` | `0.613949916396` | `0.657403137475` | `0.731633747146` | `0.609480336018` | `outputs/task1/champion_eval/ensemble_convnext_efficientnet_w070_030_notta_full_released.json` |
| ConvNeXt/EfficientNet `0.5/0.5` | hflip | `0.641864190668` | `0.618312392822` | `0.665415988514` | `0.730384483323` | `0.617698270882` | `outputs/task1/champion_eval/ensemble_convnext_efficientnet_w050_050_hflip_tta_full_released.json` |
| ConvNeXt/EfficientNet `0.7/0.3` | hflip | `0.638736651746` | `0.616439192337` | `0.661034111154` | `0.729869548422` | `0.613857494607` | `outputs/task1/champion_eval/ensemble_convnext_efficientnet_w070_030_hflip_tta_full_released.json` |

Stage 3A active champion:

- ConvNeXt rescue + EfficientNet-B3 original soft ensemble, `0.5/0.5` weights,
  with horizontal-flip TTA.
- Full released mean Dice: `0.6418641906679287`.
- Improvement over ConvNeXt rescue single-checkpoint no-TTA:
  - mean Dice: `+0.0107384222803188`
  - `label_1`: `+0.0095270996532133`
  - `label_2`: `+0.0119497449074243`
  - animal: `+0.0022832173811237`
  - phantom: `+0.0130466817453364`

Interpretation:

- Single-checkpoint hflip TTA is effectively neutral.
- Soft ensembling is useful; the benefit is not just noise because the
  no-TTA `0.5/0.5` ensemble already improves mean Dice by about `+0.0046`.
- Hflip TTA becomes useful when applied to the complementary ensemble,
  improving the `0.5/0.5` ensemble by another about `+0.0061`.
- The `0.7/0.3` ensemble slightly improves animal in no-TTA mode, but it is
  weaker than `0.5/0.5` with hflip on mean Dice and phantom.

## Stage 3B Results

The `640 x 640` ConvNeXt rescue training service completed all `80` epochs.

Training monitor:

- output: `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable`
- log: `quality_reports/logs/task1_smp_fpn_convnext_tiny_640_rescue_stable.log`
- best epoch: `74`
- best balanced-monitor Dice: `0.6617878685971026`
- best balanced-monitor `label_1`: `0.6572927231708008`
- best balanced-monitor `label_2`: `0.6662830140234042`
- best balanced-monitor animal: `0.7293322993478828`
- best balanced-monitor phantom: `0.5942434378463224`

Post-training full released evaluation:

- output directory: `outputs/task1/stage3_convnext640_eval`
- eval manifest: `configs/task1/splits/released_eval.csv`
- samples: `4691`

| Candidate | Metric Space | TTA | Mean Dice | Label 1 | Label 2 | Animal | Phantom | Output |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ConvNeXt640 | resized | hflip | `0.629154587839` | `0.606109108731` | `0.652200066947` | `0.723500446454` | `0.603398296450` | `outputs/task1/stage3_convnext640_eval/convnext640_best_resized_full.json` |
| ConvNeXt640 | original mask | hflip | `0.631223867357` | `0.609780229598` | `0.652667505116` | `0.725717874803` | `0.605427131539` | `outputs/task1/stage3_convnext640_eval/convnext640_best_hflip_original_full.json` |
| ConvNeXt512/EfficientNet512 `0.5/0.5` | original mask | hflip | `0.642831838445` | `0.619251062719` | `0.666412614171` | `0.731303937752` | `0.618679075378` | `outputs/task1/stage3_convnext640_eval/incumbent_512_ensemble_w050_050_hflip_original_full.json` |
| ConvNeXt640/EfficientNet512 `0.5/0.5` | original mask | hflip | `0.643867793255` | `0.623889804455` | `0.663845782054` | `0.727624263860` | `0.621002390424` | `outputs/task1/stage3_convnext640_eval/ensemble_convnext640_efficientnet512_w050_050_hflip_original_full.json` |

Updated active champion:

- ConvNeXt640 rescue + EfficientNet-B3 original soft ensemble, `0.5/0.5`
  weights, with horizontal-flip TTA, evaluated in original-mask space.
- Full released mean Dice: `0.6438677932546133`.
- Delta versus the original-space incumbent ConvNeXt512/EfficientNet512
  ensemble:
  - mean Dice: `+0.0010359548097881`
  - `label_1`: `+0.0046387417360744`
  - `label_2`: `-0.0025668321164984`
  - animal: `-0.0036796738915814`
  - phantom: `+0.0023233150468513`

Interpretation:

- ConvNeXt640 alone does not beat the prior ConvNeXt512 single-checkpoint
  reference.
- The mixed-geometry original-space ensemble gives the best mean Dice so far,
  but the gain over the previous ensemble is small and comes mainly from
  `label_1` and phantom.
- Further work should not keep pushing plain `aspect_pad 640` alone; the next
  training-side attempt should add class-wise soft-clDice or use a same-geometry
  ensemble candidate.

## Decision Rule

- If TTA or ensemble improves full released Dice by at least `0.002`, keep it
  as the active submission-style evaluator.
- If improvements are within noise, keep the single ConvNeXt rescue checkpoint
  as the champion and move to `640 x 640` training.
- If ensemble improves one domain/class while hurting mean Dice, preserve it as
  a diagnostic but do not make it the active champion.
