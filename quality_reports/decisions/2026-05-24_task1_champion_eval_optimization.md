# Decision: Task 1 Stage 3A Champion Eval Optimization

**Date:** 2026-05-24
**Status:** Completed; Stage 3B follow-up evaluated

## Purpose

Before launching more expensive training runs, test whether the two strongest
Stage 2 checkpoints have complementary errors that can be exploited at eval
time:

- ConvNeXt-Tiny FPN rescue single-model champion.
- EfficientNet-B3 UNet original close second.

The decision rule from the Stage 3 plan was to keep TTA or ensembling only if
it improves full released Dice by at least `0.002`.

## Implementation

Added:

- `scripts/task1/evaluate_tta_ensemble.py`

The evaluator supports:

- one or more checkpoint/config pairs;
- per-model preprocessing from each model's own YAML config;
- softmax probability averaging across checkpoints;
- horizontal-flip TTA;
- full released metrics with per-class and per-domain reporting.

## Verification

Completed checks:

- `py_compile` passed for:
  - `scripts/task1/evaluate_tta_ensemble.py`
  - `src/cathaction/training/task1_baseline.py`
- `git diff --check` passed before the full Stage 3A sweep.
- 32-sample CUDA smoke passed:
  - `outputs/task1/champion_eval/convnext_best_notta_smoke32.json`
- Full released ConvNeXt no-TTA sanity exactly matched the previous evaluator:
  - old Dice: `0.6311257683876099`
  - new Dice: `0.6311257683876099`

## Results

Full released-eval set: `configs/task1/splits/released_eval.csv`

| Candidate | TTA | Mean Dice | Label 1 | Label 2 | Animal | Phantom |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt rescue | none | `0.631125768388` | `0.608785293169` | `0.653466243607` | `0.728101265942` | `0.604651589137` |
| ConvNeXt rescue | hflip | `0.631236797431` | `0.610529352228` | `0.651944242633` | `0.721221271419` | `0.606671158128` |
| ConvNeXt/EfficientNet `0.4/0.6` | none | `0.630756374976` | `0.609082140838` | `0.652430609113` | `0.711162924950` | `0.608805495932` |
| ConvNeXt/EfficientNet `0.5/0.5` | none | `0.635726895204` | `0.614962793296` | `0.656490997113` | `0.720209478370` | `0.612663264630` |
| ConvNeXt/EfficientNet `0.6/0.4` | none | `0.632226610623` | `0.613305942099` | `0.651147279147` | `0.721078716922` | `0.607970106163` |
| ConvNeXt/EfficientNet `0.7/0.3` | none | `0.635676526936` | `0.613949916396` | `0.657403137475` | `0.731633747146` | `0.609480336018` |
| ConvNeXt/EfficientNet `0.5/0.5` | hflip | `0.641864190668` | `0.618312392822` | `0.665415988514` | `0.730384483323` | `0.617698270882` |
| ConvNeXt/EfficientNet `0.7/0.3` | hflip | `0.638736651746` | `0.616439192337` | `0.661034111154` | `0.729869548422` | `0.613857494607` |

Output JSON files live under:

- `outputs/task1/champion_eval/`

## Decision

Promote the `0.5/0.5` ConvNeXt/EfficientNet ensemble with horizontal-flip TTA as
the active eval champion.

Key deltas versus the ConvNeXt rescue single-checkpoint no-TTA champion:

- mean Dice: `+0.0107384222803188`
- `label_1`: `+0.0095270996532133`
- `label_2`: `+0.0119497449074243`
- animal: `+0.0022832173811237`
- phantom: `+0.0130466817453364`

Interpretation:

- Hflip TTA on ConvNeXt alone is not meaningful.
- The ensemble is meaningful even without TTA, improving mean Dice by about
  `+0.0046`.
- The best result comes from combining both complementary checkpoints and
  hflip TTA.
- The `0.7/0.3` ensemble is not the active choice because its mean Dice and
  phantom Dice are below the `0.5/0.5` hflip result.

## Stage 3B Launch

Stage 3B is optimizing the ConvNeXt/EfficientNet family instead of adding
unrelated architectures.

Started higher-resolution ConvNeXt rescue training:

- config: `configs/task1/smp_fpn_convnext_tiny_640_rescue_stable.yaml`
- output: `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable`
- log: `quality_reports/logs/task1_smp_fpn_convnext_tiny_640_rescue_stable.log`
- systemd user unit: `cathaction-task1-convnext640-rescue.service`
- epochs: `80`
- batch size: `8`
- resize: `aspect_pad`

Pre-launch checks:

- YAML/config load passed.
- 1-epoch CUDA smoke passed on 16 train / 8 eval samples.
- Full service started and was active.
- GPU process observed using about `9898 MB`.

2026-05-25 restart:

- The initial launch exposed a durability gap: checkpoints were written only
  after all epochs completed.
- `src/cathaction/training/task1_baseline.py` was updated to write:
  - latest `checkpoint.pt` after every epoch;
  - `metrics.json` after every epoch;
  - `best_checkpoint.pt` as soon as the monitor metric improves.
- The partial pre-fix log was retained:
  - `quality_reports/logs/task1_smp_fpn_convnext_tiny_640_rescue_stable_pre_checkpoint_restart.log`
- The fixed service was restarted:
  - unit: `cathaction-task1-convnext640-rescue.service`
  - Python PID: `850238`
  - GPU memory: about `9898 MB`
- Per-epoch persistence verified after epoch 1:
  - `best_checkpoint.pt` exists
  - `checkpoint.pt` exists
  - `metrics.json` exists
- Epoch 1 balanced-monitor Dice: `0.5449038706854183`
- Training continued into epoch 2 after checkpoint write.

## Prediction Writer

Added:

- `scripts/task1/predict_tta_ensemble.py`
- `scripts/task1/evaluate_tta_ensemble_original_space.py`
- `scripts/task1/run_stage3_convnext640_eval.sh`

This makes the active `0.5/0.5 + hflip` ensemble usable as a prediction writer,
not only as an evaluator.

The original-space evaluator maps each model's probability output back to the
original mask shape before ensembling. This is required for fair comparison
between the original `direct 512` checkpoints and the Stage 3B `aspect_pad 640`
checkpoint.

Smoke test:

- output: `outputs/task1/champion_eval/predict_smoke_ensemble_w050_050_hflip_16`
- generated PNG masks: `16`
- `predictions.csv`: `16` prediction rows
- read-back eval matched: `16 / 16`
- no missing sample IDs
- original-space evaluator CPU smoke:
  - output: `outputs/task1/champion_eval/original_space_smoke_ensemble_w050_050_hflip_2.json`
  - samples: `2`
  - Dice: `0.6975087523267691`

## Stage 3B Follow-Up

The `640 x 640` ConvNeXt rescue run completed all `80` epochs.

Training monitor:

- output: `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable`
- best epoch: `74`
- best balanced-monitor Dice: `0.6617878685971026`
- best balanced-monitor `label_1`: `0.6572927231708008`
- best balanced-monitor `label_2`: `0.6662830140234042`
- best balanced-monitor animal: `0.7293322993478828`
- best balanced-monitor phantom: `0.5942434378463224`

Full released post-training eval:

| Candidate | Metric Space | Mean Dice | Label 1 | Label 2 | Animal | Phantom |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt640 | resized | `0.629154587839` | `0.606109108731` | `0.652200066947` | `0.723500446454` | `0.603398296450` |
| ConvNeXt640 | original mask | `0.631223867357` | `0.609780229598` | `0.652667505116` | `0.725717874803` | `0.605427131539` |
| ConvNeXt512/EfficientNet512 `0.5/0.5` | original mask | `0.642831838445` | `0.619251062719` | `0.666412614171` | `0.731303937752` | `0.618679075378` |
| ConvNeXt640/EfficientNet512 `0.5/0.5` | original mask | `0.643867793255` | `0.623889804455` | `0.663845782054` | `0.727624263860` | `0.621002390424` |

Output JSON files:

- `outputs/task1/stage3_convnext640_eval/convnext640_best_resized_full.json`
- `outputs/task1/stage3_convnext640_eval/convnext640_best_hflip_original_full.json`
- `outputs/task1/stage3_convnext640_eval/incumbent_512_ensemble_w050_050_hflip_original_full.json`
- `outputs/task1/stage3_convnext640_eval/ensemble_convnext640_efficientnet512_w050_050_hflip_original_full.json`

Decision update:

- Promote the original-space ConvNeXt640/EfficientNet512 `0.5/0.5` hflip
  ensemble as the new active champion.
- Its mean Dice is `0.6438677932546133`, which is `+0.0010359548097881` over
  the original-space ConvNeXt512/EfficientNet512 incumbent.
- The improvement is below the earlier `0.002` threshold, so treat this as a
  provisional champion rather than a robust breakthrough.
- ConvNeXt640 alone is not stronger than the existing ConvNeXt512 family; the
  next meaningful training change should be class-wise soft-clDice or another
  same-geometry complementary checkpoint, not more plain `aspect_pad 640`.
