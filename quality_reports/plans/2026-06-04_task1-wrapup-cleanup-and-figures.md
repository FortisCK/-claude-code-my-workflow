# Task 1 Wrap-Up Cleanup and Result Figures

Date: 2026-06-04

Status: completed; conservative cleanup executed

## Objective

Freeze Stage9A as the current Task 1 candidate, clean unneeded experiment
outputs, and generate qualitative figures for inspection and method reporting.

## Preserve

Keep the complete Stage9A reproducibility chain:

- `outputs/task1/stage9a_seven_model_add_both_010_010/`
- `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/best_checkpoint.pt`
- `outputs/task1/smp_unet_efficientnet_b3_512_shootout/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnextv2_base_512_stage4a/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_512_stage4a/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_stage5/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_cldice_stage6/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/best_checkpoint.pt`

Also keep:

- configs under `configs/task1/`
- source scripts under `scripts/task1/` and `src/`
- decision notes, plans, logs, and paper PDFs under `quality_reports/` and
  `master_supporting_docs/`
- Task 1 data: `datasets/segmentation/`
- Task 2 data: `datasets/collision_detection/`

## Output Cleanup Candidates

Safe candidates to remove after confirmation:

- smoke/debug/checkpoint-only failed model directories
- old baseline directories that are not part of Stage9A
- failed Stage8/Stage9B/Stage9C/Stage9D output directories
- temporary coarse prediction directories
- old full prediction PNG folders that are documented in decision notes

Do not remove Stage9A predictions or the seven Stage9A best checkpoints.

Optional extra cleanup:

- remove `checkpoint.pt` from retained model directories while keeping
  `best_checkpoint.pt`, `metrics.json`, and `resolved_config.json`.

## Dataset Cleanup Candidates

Do not remove by default:

- `datasets/segmentation/`
- `datasets/collision_detection/`

Optional large deletion only if explicitly confirmed:

- `datasets/video_action_understanding/` (~54G), because the current workflow is
  Task 1 segmentation and Task 2 collision detection.

## Result Figures

Generate Stage9A qualitative outputs under `outputs/task1/stage9a_final_figures/`:

1. A small representative overlay set:
   - source: `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv`
   - manifest: `configs/task1/splits/released_eval.csv`
   - output: `outputs/task1/stage9a_final_figures/representative_overlays/`

2. Worst-case diagnostic panels:
   - output: `outputs/task1/stage9a_final_figures/error_analysis/`
   - buckets: worst overall, label 1, label 2, animal, phantom
   - includes per-sample metrics and contact sheets

## Verification

- Confirm Stage9A prediction manifest still exists after cleanup.
- Confirm all seven referenced `best_checkpoint.pt` files still exist.
- Confirm generated figure summaries exist.
- Re-read Stage9A `eval.json` and report official Dice.

## Outcome

Completed on 2026-06-04.

Generated final Stage9A figures:

- `outputs/task1/stage9a_final_figures/representative_overlays/`
- `outputs/task1/stage9a_final_figures/error_analysis/`

Executed conservative output cleanup only. Excluded from deletion:

- Stage9A predictions and final figures
- diagnostics and morphology-grid records
- the seven Stage9A `best_checkpoint.pt` model directories
- all datasets

Disk usage:

- Before cleanup: `outputs/task1` was `9.9G`.
- After cleanup: `outputs/task1` is `2.7G`.

Verification passed:

- Stage9A `predictions.csv` exists.
- Stage9A `eval.json` exists.
- final figure summaries exist.
- all seven Stage9A `best_checkpoint.pt` files exist.

