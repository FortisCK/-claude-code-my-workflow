# Task2 Stage2P Plan: Local Box Refinement

Date: 2026-06-11
Status: completed
Parent plan: `quality_reports/plans/2026-06-11_task2-stage2o-aprime-candidate-ranker.md`

## Motivation

Stage2O ranker improved candidate ranking and background rejection, but full-valid mAP50-95 remained low:

- valid_combined rank_decay mAP50: 0.1887
- valid_combined rank_decay mAP50-95: 0.0388
- valid_animal R@0.50: 0.9533
- valid_animal R@0.75: 0.0000

This pattern indicates box quality rather than ROI classification is now the main bottleneck.

## Objective

Train a local box refiner on non-leakage YOLO+geometry candidates to improve high-IoU localization, especially R@0.75 and final mAP50-95.

## Inputs

Train candidates:

- `outputs/task2/stage2o_candidate_pool/stage2o_train_subset_yolo_geometry_top50/valid_combined_candidates.csv`

Validation candidates:

- `outputs/task2/stage2o_candidate_pool/stage2o_valid_yolo_geometry_top50/valid_combined_candidates.csv`
- `outputs/task2/stage2o_candidate_pool/stage2o_valid_yolo_geometry_top50/valid_phantom_candidates.csv`
- `outputs/task2/stage2o_candidate_pool/stage2o_valid_yolo_geometry_top50/valid_animal_candidates.csv`

Optional ranker checkpoint for final mAP evaluation:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_valid256_e4/checkpoints/best.pt`

## Method

Train a crop-based refiner:

- input: ROI crop around the candidate box;
- target: box delta from candidate box to GT box;
- model: lightweight ConvNeXt-Tiny regression head;
- train rows: candidates with IoU >= 0.20, with optional GT boxes as exact positives;
- loss: SmoothL1 over normalized `(dx, dy, log(dw), log(dh))`.

## Evaluation

Primary refinement metrics:

- mean best IoU before/after;
- R@0.50 before/after;
- R@0.75 before/after;
- per-domain and per-class breakdown.

Secondary detection metrics:

- if localization improves, apply refined boxes to ranker-scored candidates;
- compare mAP50 and mAP50-95 before/after using the same ranker scores.

## Gates

Proceed to a longer train/refine run if a short pilot achieves at least one:

- valid_combined R@0.75 improves by >= +0.03 absolute;
- valid_animal R@0.75 rises above 0.05;
- valid_combined mAP50-95 improves by >= +0.01 absolute after ranker-score reuse.

Stop if:

- R@0.75 does not improve or gets worse;
- refined boxes improve train but not validation;
- gains only come from shrinking/expanding boxes in a way that hurts mAP50.

## Verification

- `py_compile` for new scripts.
- focused pytest for box-delta encode/decode and candidate filtering.
- smoke train/eval on a small subset.
- full-panel eval-only for the best pilot checkpoint.

## Result

Decision report:

- `quality_reports/decisions/2026-06-11_task2_stage2p_box_refiner_result.md`

Conclusion:

- Direct replacement refinement is not safe: refined-only boxes reduce full-valid R@0.75.
- Candidate augmentation is useful: original+refined candidates improve full-valid combined R@0.75 from 0.2213 to 0.2513.
- Phantom gains are meaningful: R@0.75 from 0.2500 to 0.2840.
- Animal high-IoU localization remains unresolved: R@0.75 stays 0.0000.

Next:

- export Stage2P refined boxes back into Stage2O candidate format;
- evaluate a three-source candidate pool with the existing ranker;
- retrain ranker with the refined source if score calibration cannot use the new high-IoU boxes.
