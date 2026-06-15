# Task2 Stage2AA Source-Matched Listwise Selector Plan

Date: 2026-06-14

## Goal

Move beyond Stage2V by fixing the main weakness of Stage2T/Stage2Z: train and validation candidate sources must match before a per-frame/listwise selector can learn useful ranking.

## Current Evidence

Current champion:

`outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`

Metric:

- `valid_combined` mAP50: 0.20096803092334323
- `valid_combined` mAP50-95: 0.04924044637514719

Stage2T source-matched yolo+geometry result:

- train sources: `yolo_stage2l`, `task1_geometry_rect`
- valid sources: `yolo_stage2l`, `task1_geometry_rect`
- best `valid_combined` mAP50-95: about 0.0403
- conclusion: source-matched but candidate pool is weaker than Stage2V.

Stage2Z five-source transfer result:

- train sources: `yolo_stage2l`, `task1_geometry_rect`
- valid sources: `yolo_stage2l`, `task1_geometry_rect`, `sequence_tip`, `stage2w_dense`, `stage2x_class1`
- best `valid_combined` mAP50-95: about 0.0371
- conclusion: not a fair or useful selector test because validation contains sources unseen by the selector.

Available candidate pools:

- train-side three-source candidates:
  `outputs/task2/stage2o_candidate_pool/stage2x_train_subset_yolo_geometry_stage2x_top50/valid_combined_candidates.csv`
- valid-side five-source candidates:
  `outputs/task2/stage2o_candidate_pool/stage2x_five_source_valid_top50/valid_combined_candidates.csv`
- valid-side five-source prediction rows:
  `outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval/valid_combined_eval_prediction_rows.csv`

Missing artifact:

- train-side prediction rows for the train-side three-source pool.

## Stage2AA Steps

1. Add a source-match audit script.
   - Input train/valid candidate CSVs and optional prediction-row CSVs.
   - Check row counts, source sets, per-source rows, sample counts, and alignment feasibility.
   - Fail loudly when train and valid source sets differ.

2. Run the audit on existing Stage2T and Stage2Z artifacts.
   - Confirm yolo+geometry is source-matched but weak.
   - Confirm five-source transfer is source-mismatched.

3. Export train-side prediction rows for:

   `outputs/task2/stage2o_candidate_pool/stage2x_train_subset_yolo_geometry_stage2x_top50/valid_combined_candidates.csv`

   using the existing Stage2O/Stage2U ranker checkpoint.

4. Filter valid-side five-source rows down to the same three sources:

   `yolo_stage2l`, `task1_geometry_rect`, `stage2x_class1`

5. Train/evaluate a source-matched three-source Stage2AA selector.
   - Decision metric: full `valid_combined` mAP50-95.
   - Promotion threshold: must beat Stage2V mAP50-95 0.04924044637514719.

## Stop Conditions

Stop the Stage2AA branch if:

- the source-matched candidate pool oracle is not better than Stage2V's current candidate source;
- the full `valid_combined` mAP50-95 remains below Stage2V;
- train-side prediction-row export is too slow or brittle relative to expected gains.

