# Task 1 Cleanup Candidates

Date: 2026-06-04

Status: conservative output cleanup executed

## Generated Figures

Created Stage9A final qualitative figures:

- Representative overlays:
  `outputs/task1/stage9a_final_figures/representative_overlays/`
- Worst-case error analysis:
  `outputs/task1/stage9a_final_figures/error_analysis/`
- Error-analysis summary:
  `outputs/task1/stage9a_final_figures/error_analysis/summary.json`

Verified aggregate Stage9A metrics from the generated error analysis:

- Dice: `0.6551964758686204`
- Label 1 Dice: `0.6339706899994402`
- Label 2 Dice: `0.6764222617378007`
- Animal Dice: `0.742322590929714`
- Phantom Dice: `0.6314111646741944`

## Keep

Keep Stage9A final predictions:

- `outputs/task1/stage9a_seven_model_add_both_010_010/`
- `outputs/task1/stage9a_final_figures/`

Keep Stage9A model checkpoints:

- `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/best_checkpoint.pt`
- `outputs/task1/smp_unet_efficientnet_b3_512_shootout/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnextv2_base_512_stage4a/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_512_stage4a/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_stage5/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_cldice_stage6/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/best_checkpoint.pt`

Keep supporting outputs:

- `outputs/task1/diagnostics/`
- `outputs/task1/stage9a_morphology_grid/`

Keep datasets required for current challenge work:

- `datasets/segmentation/`
- `datasets/collision_detection/`

## Conservative Output Delete Candidates

These are not part of the Stage9A champion chain:

- `outputs/task1/baseline_unet_multiclass_mild_weighted_full_512/`
- `outputs/task1/champion_eval/`
- `outputs/task1/monai_unet_cldice_domain_balanced_640/`
- `outputs/task1/monai_unet_dicece_full_512_overnight/`
- `outputs/task1/smp_fpn_convnext_small_512_binary_hardneg_stage9d/`
- `outputs/task1/smp_fpn_convnext_small_512_binary_hardneg_stage9d_smoke/`
- `outputs/task1/smp_fpn_convnext_small_640_cldice_stage6_smoke/`
- `outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7_smoke/`
- `outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage8_patch_refiner/`
- `outputs/task1/smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b/`
- `outputs/task1/smp_fpn_convnext_small_640_toolness_cbdice_hd_stage9b_smoke/`
- `outputs/task1/smp_fpn_convnext_tiny_512_rescue_stable/`
- `outputs/task1/smp_fpn_convnext_tiny_512_rescue_stable_smoke/`
- `outputs/task1/smp_fpn_convnext_tiny_512_shootout/`
- `outputs/task1/smp_fpn_convnext_tiny_512_smoke/`
- `outputs/task1/smp_fpn_convnext_tiny_640_rescue_smoke/`
- `outputs/task1/smp_segformer_mit_b3_512_stage4a/`
- `outputs/task1/smp_segformer_mit_b3_512_stage4a_debug/`
- `outputs/task1/smp_segformer_mit_b3_512_stage4a_smoke/`
- `outputs/task1/smp_segformer_mit_b4_512_stage4a/`
- `outputs/task1/smp_unet_efficientnet_b3_512_imagenet_norm/`
- `outputs/task1/smp_unet_efficientnet_b3_512_imagenet_norm_smoke/`
- `outputs/task1/smp_unet_efficientnet_b3_512_smoke/`
- `outputs/task1/smp_unet_efficientnet_b5_512_stage4a/`
- `outputs/task1/smp_unet_mit_b2_512_shootout/`
- `outputs/task1/smp_unet_mit_b2_512_smoke/`
- `outputs/task1/stage3_convnext640_eval/`
- `outputs/task1/stage4a_eval/`
- `outputs/task1/stage5_convnext_ensemble_search/`
- `outputs/task1/stage5_convnext_ensemble_search_full/`
- `outputs/task1/stage5_convnext_small640_ensemble_search/`
- `outputs/task1/stage5_convnext_small_640_eval/`
- `outputs/task1/stage5_error_analysis/`
- `outputs/task1/stage5_submission_smoke/`
- `outputs/task1/stage5_weight_refine/`
- `outputs/task1/stage6_convnext_small_640_cldice_eval/`
- `outputs/task1/stage7_convnext_small_640_toolness_aux_eval/`
- `outputs/task1/stage8_convnext_small_640_patch_refiner_eval/`
- `outputs/task1/stage8_patch_refiner_smoke/`
- `outputs/task1/stage8_patch_refiner_warmstart_smoke/`
- `outputs/task1/stage8_roi_probe/`
- `outputs/task1/stage8_roi_refine_smoke/`
- `outputs/task1/stage9b_convnext_small_640_toolness_cbdice_hd_eval/`
- `outputs/task1/stage9c_mslnet_hard_postprocess/`
- `outputs/task1/stage9c_mslnet_probability_gate/`
- `outputs/task1/stage9d_binary_gate_full_t065/`
- `outputs/task1/stage9d_binary_gate_smoke32/`
- `outputs/task1/stage9d_train_coarse_stage7_hflip/`

Approximate expected savings: about `6-7G`, depending on current small generated
files.

Execution result:

- Conservative output cleanup was executed on 2026-06-04.
- `outputs/task1` decreased from `9.9G` to `2.7G`.
- Stage9A predictions, final figures, diagnostics, morphology-grid records, and
  the seven champion `best_checkpoint.pt` files were preserved.

## Optional Safe Checkpoint Trimming

The Stage9A summary references only `best_checkpoint.pt`. The following retained
directories also contain `checkpoint.pt` files that can be removed while keeping
the champion reproducible:

- `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/checkpoint.pt` (~114M)
- `outputs/task1/smp_unet_efficientnet_b3_512_shootout/checkpoint.pt` (~51M)
- `outputs/task1/smp_fpn_convnextv2_base_512_stage4a/checkpoint.pt` (~343M)
- `outputs/task1/smp_fpn_convnext_small_512_stage4a/checkpoint.pt` (~197M)
- `outputs/task1/smp_fpn_convnext_small_640_stage5/checkpoint.pt` (~197M)
- `outputs/task1/smp_fpn_convnext_small_640_cldice_stage6/checkpoint.pt` (~197M)
- `outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/checkpoint.pt` (~197M)

Expected savings: about `1.3G`.

Status: not executed.

## Optional Dataset Delete Candidate

Only delete with explicit confirmation:

- `datasets/video_action_understanding/` (~54G)

Rationale: current repository plan focuses on Task 1 segmentation and Task 2
collision detection. This directory is not required for either immediate task,
but it is challenge data, so deletion should be deliberate.

Status: not executed.
