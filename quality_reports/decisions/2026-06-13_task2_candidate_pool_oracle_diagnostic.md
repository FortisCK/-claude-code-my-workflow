# Task2 Candidate-Pool Oracle Diagnostic

Date: 2026-06-13

## Context

The current Task2 pipeline is limited by a coarse candidate pool followed by a Stage2U/Stage2V quality ranker. Before investing more time in ranker tuning, I evaluated whether the candidate pools contain boxes that overlap the ground truth well enough.

Script added:

- `scripts/task2/evaluate_candidate_oracle.py`

The script reads a saved `*_candidates_used.csv`, groups candidates by sample, and reports the best available ground-truth IoU per sample. This measures the localization ceiling available to any downstream ranker using that candidate pool.

## Candidate Pools Compared

| pool | candidate CSV |
| --- | --- |
| yolo-only | `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval/valid_combined_candidates_used.csv` |
| mixed yolo + task1 geometry | `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_yologeom_eval/valid_combined_candidates_used.csv` |
| augmented yolo + task1 geometry + Stage2P refined | `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_stage2p_augmented_eval/valid_combined_candidates_used.csv` |

Outputs:

- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/yolo_only_candidate_oracle.json`
- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/mixed_candidate_oracle.json`
- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/augmented_candidate_oracle.json`

## Key Results

| pool | group | mean best IoU | recall@0.50 | recall@0.75 | candidates/sample |
| --- | --- | ---: | ---: | ---: | ---: |
| yolo-only | all | 0.5070 | 0.5456 | 0.2170 | 22.0 |
| yolo-only | class 0 | 0.5790 | 0.7096 | 0.3302 | 23.9 |
| yolo-only | class 1 | 0.4460 | 0.4067 | 0.1210 | 20.4 |
| mixed | all | 0.5216 | 0.5585 | 0.2213 | 71.9 |
| mixed | class 0 | 0.6046 | 0.7377 | 0.3396 | 73.8 |
| mixed | class 1 | 0.4513 | 0.4067 | 0.1210 | 70.4 |
| augmented | all | 0.5310 | 0.5618 | 0.2513 | 143.9 |
| augmented | class 0 | 0.6171 | 0.7400 | 0.3934 | 147.6 |
| augmented | class 1 | 0.4581 | 0.4107 | 0.1310 | 140.8 |

Important subgroup observations:

- `domain=phantom|class=1` remains weak even in the augmented pool: recall@0.50 `0.2791`, recall@0.75 `0.1602`.
- `domain=animal|class=0` is fixed by geometry/refined sources at IoU@0.50 but still has recall@0.75 `0.0`.
- Adding Stage2P refined candidates increases the raw localization ceiling slightly, but it also increases candidate count massively and previously hurt ranker AP when used naively.

## Interpretation

The main bottleneck is candidate localization, especially for class 1 / collision. The ranker is often choosing among boxes whose best possible IoU is below high AP thresholds. This explains the current pattern:

- mAP50 can move upward with better ranking.
- mAP50-95 remains low because IoU@0.75+ candidates are rare.
- Adding many refined candidates helps oracle recall slightly but hurts practical AP unless the ranker can reliably isolate the few useful refined boxes.

## Decision

Do not spend the next major effort on small Stage2V score tweaks. The next meaningful direction should improve the coarse candidate generator or generate collision-centered high-IoU candidates, especially for class 1.

The current best export remains:

- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`
- `valid_combined` mAP50 `0.2009680309`
- `valid_combined` mAP50-95 `0.0492404464`

## Next Candidate Direction

Stage2W should target candidate localization rather than candidate scoring:

1. Build a collision-centered localizer that predicts a small ROI around the interaction point rather than broad object boxes.
2. Train/evaluate it on a balanced split with phantom and animal visible in training, while keeping a held-out validation split.
3. Add its candidates to the pool only if oracle recall@0.75 improves substantially without exploding candidates/sample.
