# Plan: Task 1 Best-Model Push

**Date:** 2026-05-26
**Status:** Active; official repository setup completed
**Scope:** Move from conservative champion hardening to an aggressive top-model
push for CATHACTION Task 1 segmentation.

## Why This Stage Exists

Stage 2 and Stage 3 established a fair evaluation and prediction pipeline:

- released-eval full-set metrics;
- per-class and per-domain reporting;
- original-mask-space evaluation for mixed preprocessing geometries;
- hflip TTA and softmax ensemble support;
- a current champion around `0.6439` mean Dice.

That work reduced uncertainty, but it is not sufficient if the goal is the
strongest possible model. Stage 4 should widen the search and then invest
training budget into the most promising family.

## Current Champion

Active provisional champion:

- ConvNeXt640 rescue + EfficientNet-B3 512 soft ensemble
- weights: `0.5 / 0.5`
- TTA: hflip
- metric space: original mask
- released-eval mean Dice: `0.6438677932546133`

The gain over the previous original-space incumbent is only about `+0.0010`,
so it should be treated as a provisional champion rather than a robust
breakthrough.

## Strategy Change

Stop treating `+0.001` improvements as the main objective.

Use a two-lane strategy:

1. **Architecture shootout lane:** run stronger candidate families under a
   consistent short-budget protocol to identify which family has real upside.
2. **Champion maximization lane:** once a family wins, spend budget on
   high-resolution training, class-wise thin-structure losses, checkpoint
   averaging, and ensemble/TTA.

## Stage 4A: Aggressive Architecture Shootout

Run comparable short-to-medium training jobs, not one-off micro tweaks.

Candidates:

1. SegFormer / MiT-B2 or MiT-B3 style model.
2. Swin or Swin-like U-Net/FPN model if available in the current environment.
3. Strong ConvNeXt variant beyond Tiny if supported by installed libraries.
4. EfficientNet-B4/B5 or another high-capacity encoder if GPU memory allows.

Initial protocol:

- same released train/eval split discipline;
- same metric reporting;
- original-mask-space eval when preprocessing differs;
- 512 or 640 input, depending on model stability;
- 30-50 epochs for ranking, not final convergence;
- keep loss simple for the first pass: DiceCE, same augmentations where
  possible;
- promote only candidates that beat current single-model references or show a
  clearly complementary class/domain profile.

Goal:

- identify whether a Transformer/hybrid/high-capacity encoder actually
  dominates ConvNeXt/EfficientNet on this dataset.

## Stage 4B: Thin-Structure Loss Push

Apply only to the strongest candidate family after Stage 4A.

Primary recipe:

- DiceCE base loss;
- class-wise soft-clDice for `label_1` and `label_2`;
- optional higher weight for the thinner or lower-performing class based on
  measured mask statistics, not assumed semantics;
- delayed clDice schedule if early training becomes unstable.

Decision target:

- look for improvements in `label_2`, phantom, and skeleton-like continuity,
  not just mean Dice.

## Stage 4C: Spatial Strategy

If full-image 640 is not enough:

- foreground-centered crops;
- patch/ROI training around sparse tool pixels;
- full-image validation and inference;
- optionally high-resolution final inference if memory allows.

This is likely more important than another small architecture tweak because the
foreground structures are thin and sparse.

## Stage 4D: Final Champion Construction

Build the final model as a system, not as one checkpoint:

- best single model from Stage 4A/4B;
- best ConvNeXt/EfficientNet existing checkpoints if complementary;
- checkpoint ensemble from top validation epochs;
- hflip TTA by default;
- test multi-scale TTA only if it improves full released eval enough to justify
  extra inference cost;
- original-mask-space ensembling for mixed preprocessing geometries.

## Decision Rules

- A single new architecture must beat the current best single-model family by
  at least `0.005` or show a strong complementary profile to continue.
- A final champion update should improve released-eval mean Dice by at least
  `0.002`, unless it materially improves the weaker class/domain.
- Any run used for champion comparison must report:
  - mean Dice;
  - `label_1`;
  - `label_2`;
  - animal;
  - phantom;
  - output JSON path.

## Immediate Next Actions After Approval

1. Inspect installed model libraries and available encoders in the current
   training environment.
2. Create configs for a fast Stage 4A shootout.
3. Start the first aggressive candidate run, preferably SegFormer/MiT if
   supported cleanly; otherwise start the strongest installed ConvNeXt or Swin
   candidate.
4. Record every run in the decision log and only keep models that improve the
   champion system.

## Official Repository Setup

Completed initial clone/inventory:

- `external_repos/ConvNeXt`
- `external_repos/ConvNeXt-V2`
- `external_repos/SegFormer`
- `external_repos/Swin-Unet`
- `external_repos/TransUNet`
- `external_repos/FGA-Net-Guidewire-Segmentation`
- `external_repos/MSLNet`
- `external_repos/WT-CMUNeXt`
- `external_repos/nnUNet`

Inventory and integration decision:

- `quality_reports/decisions/2026-05-26_task1_official_repo_inventory.md`

Important finding:

- Current `cardiac-diffusion` has `segmentation_models_pytorch`, `timm`,
  `monai`, and `nnunetv2`, but not `mmcv/mmseg`.
- Therefore official repositories should first be used as reference/model
  sources while keeping the existing CATHACTION data/eval/checkpoint pipeline.
  A separate MMseg environment remains an option if SegFormer official configs
  are necessary.

## Stage 4A Launch And Results

Launched on 2026-05-26 as a user systemd service:

- unit: `cathaction-task1-stage4a-shootout.service`
- runner: `scripts/task1/run_stage4a_shootout.sh`
- master log: `quality_reports/logs/task1_stage4a_shootout_master_systemd.log`
- full-eval output directory: `outputs/task1/stage4a_eval`

Candidate order:

1. `configs/task1/smp_segformer_mit_b3_512_stage4a.yaml`
2. `configs/task1/smp_segformer_mit_b4_512_stage4a.yaml`
3. `configs/task1/smp_fpn_convnext_small_512_stage4a.yaml`
4. `configs/task1/smp_fpn_convnextv2_base_512_stage4a.yaml`
5. `configs/task1/smp_unet_efficientnet_b5_512_stage4a.yaml`

Startup checks:

- GPU was free before launch except desktop processes.
- `Segformer mit_b3` 1-epoch CUDA smoke passed:
  - log: `quality_reports/logs/task1_smp_segformer_mit_b3_512_stage4a_smoke.log`
  - output: `outputs/task1/smp_segformer_mit_b3_512_stage4a_smoke`
- A direct debug smoke with the same config/log naming also passed:
  - log: `quality_reports/logs/smp_segformer_mit_b3_512_stage4a_debug.log`
  - output: `outputs/task1/smp_segformer_mit_b3_512_stage4a_debug`
- Initial `nohup` launch exited early before creating child logs, so the long
  run was relaunched through `systemd-run --user`.
- Confirmed active service after launch:
  - service state: `active`
  - active training config: `smp_segformer_mit_b3_512_stage4a`
  - GPU memory: about `11272 MiB`
  - GPU utilization: about `97%`
  - first observed train loss values were finite, around `1.18`, `1.06`,
    `1.09`, and `0.99`.

Monitoring commands:

```bash
systemctl --user status cathaction-task1-stage4a-shootout.service --no-pager -l
tail -f quality_reports/logs/task1_stage4a_shootout_master_systemd.log
tail -f quality_reports/logs/smp_segformer_mit_b3_512_stage4a.log
find outputs/task1/stage4a_eval -maxdepth 1 -type f -name '*.json' -print
```

Completed on 2026-05-28:

- service state after completion: `inactive`
- all five full released evaluation JSON files were written under
  `outputs/task1/stage4a_eval/`

Full released evaluation results:

| Candidate | Mean Dice | Label 1 | Label 2 | Animal | Phantom | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `FPN + ConvNeXtV2-Base` | `0.634086861627` | `0.609813554681` | `0.658360168573` | `0.717981853848` | `0.611183642584` | `outputs/task1/stage4a_eval/smp_fpn_convnextv2_base_512_stage4a_full_released.json` |
| `FPN + ConvNeXt-Small` | `0.633066549973` | `0.609565498660` | `0.656567601285` | `0.722420953553` | `0.608672919036` | `outputs/task1/stage4a_eval/smp_fpn_convnext_small_512_stage4a_full_released.json` |
| `Segformer + MiT-B3` | `0.629093403226` | `0.605300341391` | `0.652886465061` | `0.711088410427` | `0.606708877515` | `outputs/task1/stage4a_eval/smp_segformer_mit_b3_512_stage4a_full_released.json` |
| `UNet + EfficientNet-B5` | `0.628982855360` | `0.614394692940` | `0.643571017781` | `0.710592786740` | `0.606703454826` | `outputs/task1/stage4a_eval/smp_unet_efficientnet_b5_512_stage4a_full_released.json` |
| `Segformer + MiT-B4` | `0.627912671935` | `0.599148031785` | `0.656677312084` | `0.708191497738` | `0.605996661417` | `outputs/task1/stage4a_eval/smp_segformer_mit_b4_512_stage4a_full_released.json` |

Interpretation:

- None of the Stage 4A single models beat the provisional champion
  `ConvNeXt640/EfficientNet512 0.5/0.5 + hflip` ensemble
  (`0.6438677932546133`).
- The best Stage 4A single model is `FPN + ConvNeXtV2-Base`, but it is still
  about `0.00978` mean Dice below the active ensemble champion.
- SegFormer/MiT did not produce the hoped-for architecture jump under this
  training recipe; MiT-B3/B4 are both below ConvNeXtV2-Base and ConvNeXt-Small.
- EfficientNet-B5 looked strong on the balanced 1024-sample monitor
  (`best balanced Dice 0.6592878756872932` at epoch 47), but its full released
  eval was only `0.6289828553604564`, so the balanced monitor overestimated its
  full-set generalization.
- Next step should be eval-side complementarity testing: ensemble
  ConvNeXtV2-Base and ConvNeXt-Small with the current champion family before
  investing in another long single-model architecture run.
