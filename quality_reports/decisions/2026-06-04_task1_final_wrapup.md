# Task 1 Final Wrap-Up

Date: 2026-06-04

## Status

Task 1 is frozen for now with Stage9A as the current submission candidate.

The project should move to Task 2 after this point. Task 1 should only be
reopened for low-risk packaging, Docker inference, or final submission-format
work unless a clearly stronger idea is available.

## Champion

Stage9A seven-model ensemble with flip TTA and light component post-processing.

Final prediction manifest:

- `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/predictions.csv`

Final evaluation JSON:

- `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/eval.json`

Official-style released-eval metrics:

- Mean Dice: `0.6551964758686204`
- Label 1 Dice: `0.6339706899994402`
- Label 2 Dice: `0.6764222617378007`
- Animal Dice: `0.742322590929714`
- Phantom Dice: `0.6314111646741944`
- Mean IoU: `0.5331671049089795`
- Pixel accuracy: `0.9948812664975007`

## Model Chain

The Stage9A raw ensemble uses these seven `best_checkpoint.pt` files:

- `outputs/task1/smp_fpn_convnext_tiny_640_rescue_stable/best_checkpoint.pt`
- `outputs/task1/smp_unet_efficientnet_b3_512_shootout/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnextv2_base_512_stage4a/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_512_stage4a/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_stage5/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_cldice_stage6/best_checkpoint.pt`
- `outputs/task1/smp_fpn_convnext_small_640_toolness_aux_stage7/best_checkpoint.pt`

Stage9A raw ensemble summary:

- `outputs/task1/stage9a_seven_model_add_both_010_010/raw_predictions/summary.json`

Stage9A post-processing summary:

- `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/summary.json`

## Result Figures

Final qualitative figures:

- Representative overlays:
  `outputs/task1/stage9a_final_figures/representative_overlays/`
- Worst-case error analysis:
  `outputs/task1/stage9a_final_figures/error_analysis/`

Contact sheets:

- `outputs/task1/stage9a_final_figures/error_analysis/worst_overall/contact_sheet.png`
- `outputs/task1/stage9a_final_figures/error_analysis/worst_label_1/contact_sheet.png`
- `outputs/task1/stage9a_final_figures/error_analysis/worst_label_2/contact_sheet.png`
- `outputs/task1/stage9a_final_figures/error_analysis/worst_animal/contact_sheet.png`
- `outputs/task1/stage9a_final_figures/error_analysis/worst_phantom/contact_sheet.png`

The error-analysis aggregate re-computed the same Stage9A metrics, confirming
that the figures are tied to the frozen champion predictions.

## Cleanup

Conservative output cleanup was executed on 2026-06-04.

- Before cleanup: `outputs/task1` was `9.9G`.
- After cleanup: `outputs/task1` is `2.7G`.

Preserved:

- Stage9A predictions and final figures
- diagnostics and morphology-grid records
- seven Stage9A champion model directories
- all datasets

Not executed:

- checkpoint trimming inside retained champion directories
- deletion of `datasets/video_action_understanding/`

Cleanup details:

- `quality_reports/decisions/2026-06-04_task1_cleanup_candidates.md`

## Negative Routes Closed

The following directions were tested and are not current champion routes:

- MONAI/UNet baseline
- SegFormer/MiT and other architecture shootouts
- standalone clDice/cbDice/Hausdorff-style alignment losses
- ROI patch refinement
- MSLNet-style hard postprocess
- Stage9D binary hard-negative gate/refiner

Important diagnostic conclusion:

Stage9D improved binary foreground precision but hurt recall and official
multiclass Dice. Future Task 1 work should avoid hard foreground deletion unless
it is demonstrably recall-preserving.

## Next Work

Immediate next project phase:

- start Task 2 collision detection setup and baseline.

Task 1 remaining submission work:

- package Stage9A inference into a stable entrypoint
- create Docker-compatible prediction flow
- write the Task 1 method/report section using the frozen metrics and figures

