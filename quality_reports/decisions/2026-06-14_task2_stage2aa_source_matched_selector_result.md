# Task2 Stage2AA Source-Matched Selector Result

Date: 2026-06-14

## Question

Can a source-matched listwise/tabular selector use the Stage2X candidate gains and beat the current Stage2V champion?

## Setup

Current champion:

`outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`

Champion metric:

- `valid_combined` mAP50: 0.20096803092334323
- `valid_combined` mAP50-95: 0.04924044637514719

Stage2AA candidate sources:

- `yolo_stage2l`
- `task1_geometry_rect`
- `stage2x_class1`

Train candidate rows:

`outputs/task2/stage2u_quality_ranker/stage2aa_three_source_train_eval_v2/valid_combined_candidates_used.csv`

Train prediction rows:

`outputs/task2/stage2u_quality_ranker/stage2aa_three_source_train_eval_v2/valid_combined_eval_prediction_rows.csv`

Validation candidate rows:

`outputs/task2/stage2aa_source_matched/valid_combined_yologeom_stage2x_candidates.csv`

Validation prediction rows:

`outputs/task2/stage2aa_source_matched/valid_combined_yologeom_stage2x_prediction_rows.csv`

Source-match audit:

`quality_reports/decisions/2026-06-14_task2_stage2aa_three_source_full_match_audit.json`

The audit passed:

- train candidate rows: 278,006
- train prediction rows: 278,006
- valid candidate rows: 113,535
- valid prediction rows: 113,535
- train/valid source set match: true

## Result

Output:

`outputs/task2/stage2t_tabular_selector/histgb_stage2aa_three_source_matched`

Best policy:

- quality mode: `pred_iou`
- score mode: `rank_decay_roi`
- k: 50

Best full `valid_combined` metrics:

- mAP50: 0.1537369641375607
- mAP50-95: 0.03899863697727917

This does **not** beat Stage2V:

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2V champion | 0.20096803092334323 | 0.04924044637514719 |
| Stage2AA source-matched selector | 0.1537369641375607 | 0.03899863697727917 |

## Interpretation

Stage2AA fixed the unfair source mismatch in Stage2Z, so this is a cleaner test than the previous five-source transfer. The result is still lower than Stage2V.

The key lesson is that adding `stage2x_class1` candidates improves candidate-pool localization on train-side diagnostics, but the current tabular selector still fails to rank high-IoU validation candidates strongly enough to improve AP.

## Decision

Do not promote Stage2AA.

Keep Stage2V as the Task2 champion.

Future work should not spend more time tuning this HistGradientBoosting selector unless we change the learning target/objective. The next plausible accuracy route is a true AP-aware/listwise neural ranker or a simpler class-aware calibration on the existing Stage2V outputs.

