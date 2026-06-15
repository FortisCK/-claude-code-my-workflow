# Task2 Stage2Q Plan: Refined-Candidate mAP Check

Date: 2026-06-11
Status: completed
Parent plan: `quality_reports/plans/2026-06-11_task2-stage2p-box-refinement.md`

## Motivation

Stage2P showed that refined boxes are useful only as candidate augmentation:

- combined R@0.75: 0.2213 -> 0.2513 with original+refined candidates;
- phantom R@0.75: 0.2500 -> 0.2840;
- animal R@0.75 remains 0.0000.

This is an oracle candidate-pool gain. It does not prove the Stage2O ranker can select those refined candidates.

## Objective

Convert Stage2P refined boxes into Stage2O candidate CSV format and run the existing ranker checkpoint in eval-only mode on an expanded candidate pool.

## Inputs

Original candidate pool:

- `outputs/task2/stage2o_candidate_pool/stage2o_valid_yolo_geometry_top50/*_candidates.csv`

Refined candidate rows:

- `outputs/task2/stage2p_box_refiner/convnext_tiny_yolo_geometry_iou20_full_eval_augmented/*_eval_refined_candidates.csv`

Ranker checkpoint:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_valid256_e4/checkpoints/best.pt`

## Evaluation

Compare existing ranker eval metrics on:

- original YOLO+geometry pool;
- expanded YOLO+geometry+Stage2P-refined pool.

Primary metric:

- valid_combined mAP50-95.

Secondary:

- valid_combined mAP50;
- phantom/animal breakdown;
- class-wise AP.

## Gates

Keep refined candidates in the Task 2 pipeline if either:

- valid_combined mAP50-95 improves by >= +0.005 absolute; or
- valid_phantom mAP50-95 improves without hurting combined mAP50-95.

If mAP does not improve while oracle R@0.75 does improve, retrain the ranker with refined candidates included in the train candidate pool.

## Verification

- focused pytest for exporter mechanics;
- candidate CSV audit by row count and Stage2P source counts;
- eval-only ranker run writes `eval_metrics.json`.

## Result

Decision report:

- `quality_reports/decisions/2026-06-11_task2_stage2q_refined_candidate_map_check.md`

Outcome:

- Expanded candidate pool improved valid_combined loc R@0.75 from 0.2213 to 0.2513.
- Existing ranker mAP50-95 dropped from 0.0388 to 0.0280 on valid_combined.
- Directly adding refined candidates to the old ranker is rejected.

Next:

- generate train-subset Stage2P refined candidates;
- retrain Stage2O ranker with source-consistent original+refined candidates;
- re-check whether the localization gain can be converted into mAP50-95.
