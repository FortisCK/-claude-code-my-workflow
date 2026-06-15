# Decision: Start Stage 8 Warm-Started ROI/Patch Refinement

Date: 2026-06-01

## Context

The current Task 1 local champion remains the seven-model ensemble
`add_both_010_010` with full released-eval original-space mean Dice
`0.6536723455148790`.

The next intended jump is not another full-frame backbone swap, but a
coarse-to-fine refinement path for thin foreground structures.

## Implementation

Stage 8 adds:

- train-time foreground-centered patch sampling in
  `src/cathaction/training/task1_baseline.py`;
- optional training warm-start from an existing checkpoint via
  `training.initial_checkpoint`;
- a patch-refiner config at
  `configs/task1/smp_fpn_convnext_small_640_toolness_aux_stage8_patch_refiner.yaml`;
- an original-space ROI fusion predictor at
  `scripts/task1/predict_roi_patch_refine_original_space.py`.

Patch training uses ConvNeXt-Small FPN with the existing toolness auxiliary
head and initializes from the Stage 7 best checkpoint:

`outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/best_checkpoint.pt`

The ROI predictor forms ROIs from coarse model predictions only. It must not use
released-eval or hidden-test masks for ROI creation.

## Verification

Completed checks:

- `py_compile` on the modified training module and ROI refine script;
- `bash -n` on the Stage 8 run script;
- patch dataset smoke sample: image `(3, 640, 640)`, mask `(640, 640)`, labels
  `[0, 1, 2]`;
- cold-start 1-epoch smoke: completed and wrote checkpoint;
- warm-start 1-epoch smoke: loaded Stage 7 checkpoint with no missing or
  unexpected keys;
- ROI refine smoke on 2 released-eval images: wrote PNG predictions and an
  evaluatable `predictions.csv`.

Warm-start smoke metric on 4 animal eval samples was mean Dice `0.7893102599286348`.
This is only a pipeline check, not a leaderboard estimate.

## Run Started

Started systemd service:

`cathaction-task1-stage8-patch-refiner.service`

Logs:

- master: `quality_reports/logs/task1_stage8_convnext_small_640_patch_refiner_master.log`
- train: `quality_reports/logs/task1_smp_fpn_convnext_small_640_toolness_aux_stage8_patch_refiner.log`

The service is training the warm-started patch refiner. The Stage 8 checkpoint
will be under:

`outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage8_patch_refiner/`

## Next Step

After the patch-refiner has a good checkpoint, run ROI fusion probes over a
subset first to tune `fuse_alpha`, `roi_size`, `max_rois`, and the coarse model
source. Then run the full released-eval original-space comparison against the
`0.6536723455148790` seven-model fallback.
