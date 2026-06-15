# Task2 Stage2AB Domain-Aware Policy Result

Date: 2026-06-14

## Question

Can we improve Stage2V without adding new candidate sources, by selecting score modes separately for class and domain?

## Setup

Base run:

`outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`

Sweep script:

`scripts/task2/sweep_stage2v_domain_policy.py`

Output:

`outputs/task2/stage2v_domain_policy/yolo_only_domain_policy_sweep.json`

Prediction CSVs:

`outputs/task2/stage2v_domain_policy/yolo_only_domain_policy_predictions`

The exported CSVs contain only inference fields:

`sample_id, video_id, frame_index, domain, class_id, score, x1, y1, x2, y2, source, source_rank, score_mode, policy_name`

No `gt_*`, `candidate_iou`, or `verifier_label` fields are present.

## Selected Policy

Class 0 / normal:

- phantom: `prob_iou75_source_rank_decay_roi`
- animal: `prob_iou75_source_rank_decay_roi`

Class 1 / collision:

- phantom: `rank_decay_roi`
- animal: `prob_iou75_roi`

## Result

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2V champion | 0.20096803092334323 | 0.04924044637514719 |
| Stage2AB domain-aware policy | 0.21128164745000094 | 0.050628135044670744 |

Split results for Stage2AB:

| Split | mAP50 | mAP50-95 | Rows |
| --- | ---: | ---: | ---: |
| valid_combined | 0.21128164745000094 | 0.050628135044670744 | 40,970 |
| valid_phantom | 0.10462265654206386 | 0.04301139057740514 | 39,338 |
| valid_animal | 0.4997355309073923 | 0.050008473801353336 | 1,632 |

## Interpretation

Stage2AB is the current best public-validation metric, improving Stage2V by:

- +0.01031361652665771 mAP50;
- +0.00138768866952355 mAP50-95.

This is a small but real public-valid improvement without adding candidate sources. It is more reliable than Stage2AA in the sense that it keeps the simpler yolo-only candidate source. It is less reliable than Stage2V in the sense that it adds domain-aware public-validation calibration.

## Decision

Treat Stage2AB as the current public-validation best and tentative Task2 champion.

Keep Stage2V as the conservative fallback because Stage2AB is explicitly tuned on public validation domain behavior. For hidden-test preparation, package both policies unless the official validation confirms the domain-aware policy generalizes.

