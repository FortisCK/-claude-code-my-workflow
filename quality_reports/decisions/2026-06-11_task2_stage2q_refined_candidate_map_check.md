# Task2 Stage2Q Result: Refined-Candidate mAP Check

Date: 2026-06-11
Status: completed

## Question

Stage2P improved candidate-pool high-IoU recall when refined boxes were appended to the original pool. Does this translate to better mAP with the existing Stage2O ranker checkpoint?

## Implementation

Added:

- `scripts/task2/export_stage2p_refined_candidates.py`
- `tests/test_task2_stage2p_export_refined_candidates.py`

The exporter reads:

- original YOLO+geometry candidate CSVs;
- Stage2P refined box CSVs;

and writes expanded candidate CSVs with a new source:

- `stage2p_refined`

Important leakage control:

- `source_conf` is inherited from the original candidate.
- `after_iou` is written only to `candidate_iou` for local evaluation/labels.
- `after_iou` is not used as a ranking confidence.

Expanded candidate pool:

- `outputs/task2/stage2q_refined_candidate_pool/yolo_geometry_stage2p_refined`

## Candidate-Pool Audit

`valid_combined`:

- original rows: 66,985
- refined rows: 66,985
- expanded rows: 133,970
- refined rows with IoU >= 0.75: 224

`valid_phantom`:

- original rows: 60,869
- refined rows: 60,869
- expanded rows: 121,738
- refined rows with IoU >= 0.75: 224

`valid_animal`:

- original rows: 6,116
- refined rows: 6,116
- expanded rows: 12,232
- refined rows with IoU >= 0.75: 0

## Ranker Eval-Only

Checkpoint:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_valid256_e4/checkpoints/best.pt`

Original YOLO+geometry full eval:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/eval_metrics.json`

Expanded YOLO+geometry+Stage2P refined full eval:

- `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_stage2p_refined_full_eval/eval_metrics.json`

Rank-decay score mode comparison:

| Split | Pool | mAP50 | mAP50-95 | loc R@0.50 | loc R@0.75 |
| --- | --- | ---: | ---: | ---: | ---: |
| valid_combined | original | 0.1887 | 0.0388 | 0.5585 | 0.2213 |
| valid_combined | expanded | 0.1519 | 0.0280 | 0.5618 | 0.2513 |
| valid_phantom | original | 0.1022 | 0.0340 | 0.5073 | 0.2500 |
| valid_phantom | expanded | 0.0663 | 0.0216 | 0.5109 | 0.2840 |
| valid_animal | original | 0.4609 | 0.0467 | 0.9533 | 0.0000 |
| valid_animal | expanded | 0.4595 | 0.0464 | 0.9533 | 0.0000 |

## Decision

Do not use the expanded refined candidate pool with the old ranker checkpoint as-is.

The refined candidates improve localization upper bound, especially R@0.75, but the existing ranker was trained without `stage2p_refined` examples and fails to rank the new high-IoU candidates ahead of the added false positives. As a result, mAP50 and mAP50-95 drop on combined and phantom validation.

The correct next experiment is refined-source ranker retraining:

- generate Stage2P refined candidates for the train subset;
- build a train/valid candidate pool with source consistency;
- retrain the Stage2O ranker with `stage2p_refined` included;
- re-evaluate mAP50-95.

## Interpretation

This resolves the apparent contradiction:

- Stage2P is useful for candidate generation: loc R@0.75 improves from 0.2213 to 0.2513 on valid_combined.
- Stage2P is not yet useful for final detection: old ranker mAP50-95 drops from 0.0388 to 0.0280.

So the bottleneck after Stage2P is ranking/calibration, not whether refined candidates contain some better boxes.

## Verification

Passed:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile scripts/task2/export_stage2p_refined_candidates.py scripts/task2/train_stage2p_box_refiner.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2p_box_refiner.py tests/test_task2_stage2p_export_refined_candidates.py tests/test_task2_stage2o_ranker.py tests/test_task2_stage2o_candidate_pool.py
```

Result:

- 13 tests passed.
