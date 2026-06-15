# Task2 Stage2R Plan: Source-Consistent Refined Ranker

Date: 2026-06-11
Status: completed
Parent plan: `quality_reports/plans/2026-06-11_task2-stage2q-refined-candidate-map-check.md`

## Motivation

Stage2Q showed a split result:

- adding Stage2P refined candidates improves localization upper bound;
- the old ranker loses mAP because it was never trained on the `stage2p_refined` source.

Therefore the next test must keep source distribution consistent between train and validation.

## Objective

Generate Stage2P refined candidates for the ranker train subset, then retrain the Stage2O ranker on:

- `yolo_stage2l`;
- `task1_geometry_rect`;
- `stage2p_refined`.

Evaluate the retrained ranker on the expanded validation pool.

## Inputs

Stage2P refiner checkpoint:

- `outputs/task2/stage2p_box_refiner/convnext_tiny_yolo_geometry_iou20_valid256_e4/checkpoints/best.pt`

Train candidate pool:

- `outputs/task2/stage2o_candidate_pool/stage2o_train_subset_yolo_geometry_top50/valid_combined_candidates.csv`

Train split:

- `configs/task2/splits_stage2o_ranker/train_v0_v1_val_v2_animal_all_phantom1000pc_labels.txt`

Expanded validation pool:

- `outputs/task2/stage2q_refined_candidate_pool/yolo_geometry_stage2p_refined/*_candidates.csv`

## Steps

1. Add an explicit overlap-check bypass for Stage2P eval-only train-candidate generation.
2. Generate refined candidates for the train subset.
3. Export an expanded train candidate CSV.
4. Retrain the ranker for a short pilot.
5. Full-eval the best checkpoint against the expanded validation pool.

## Gates

Continue with a longer ranker train if the short pilot achieves either:

- valid_combined mAP50-95 >= 0.0388, matching or beating the old YOLO+geometry ranker;
- valid_combined mAP50 improves without reducing mAP50-95 below 0.0388 by more than 0.002.

Stop if source-consistent retraining still stays below the old ranker.

## Verification

- `py_compile` and pytest for touched scripts.
- train refined candidate CSV exists and matches the base rows.
- ranker eval writes `eval_metrics.json`.

## Result

Decision report:

- `quality_reports/decisions/2026-06-11_task2_stage2r_source_consistent_refined_ranker_result.md`

Outcome:

- valid256 pilot reached rank_decay mAP50-95 0.0665 at epoch 1.
- full-valid combined mAP50-95 did not generalize:
  - Stage2R ranker + expanded pool: 0.0203;
  - Stage2R ranker + original pool control: 0.0288.
- Current best remains old Stage2O ranker + original YOLO+geometry pool:
  - valid_combined mAP50: 0.1887;
  - valid_combined mAP50-95: 0.0388.

Decision:

- reject Stage2R as champion;
- keep Stage2P/Stage2Q outputs as diagnostics only;
- next useful direction would need a different ranking objective or candidate pruning, not more unfiltered refined candidates.
