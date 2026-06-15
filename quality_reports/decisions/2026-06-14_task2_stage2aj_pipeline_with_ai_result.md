# Task2 Stage2AJ Four-Variant Champion Pipeline Result

Date: 2026-06-14

## Decision

Extend the standard Task2 champion pipeline to export Stage2AI in addition to AB/AD/AE.

This is necessary because the official PDF says Task2 primary ranking is `mAP`, while the current public documents do not explicitly specify the IoU-threshold convention. Stage2AE and Stage2AI optimize different metric tradeoffs:

- Stage2AE preserves AP50/mAP50.
- Stage2AI maximizes our COCO-style mAP50-95 implementation.

## Command

```bash
python3 scripts/task2/run_stage2_champion_pipeline.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2aj_champion_pipeline_with_ai/public_valid_ab_ad_ae_ai
```

Output:

`outputs/task2/stage2aj_champion_pipeline_with_ai/public_valid_ab_ad_ae_ai`

## Public-Validation Results

| Variant | valid_combined mAP50 | valid_combined mAP50-95 | Role |
| --- | ---: | ---: | --- |
| stage2ab | 0.21128164745000094 | 0.050628135044670744 | no-calibration fallback |
| stage2ad | 0.21128164745000094 | 0.06251332119288261 | conservative calibration fallback |
| stage2ae | 0.21128164745000094 | 0.06327005557430047 | balanced/default champion |
| stage2ai | 0.18439741617609534 | 0.08394956471533466 | high mAP50-95 candidate |

Split metrics:

| Variant | Split | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: |
| stage2ab | valid_phantom | 0.10462265654206386 | 0.04301139057740514 |
| stage2ab | valid_animal | 0.4997355309073923 | 0.050008473801353336 |
| stage2ad | valid_phantom | 0.10462265654206386 | 0.04301139057740514 |
| stage2ad | valid_animal | 0.4997355309073923 | 0.1269937435713481 |
| stage2ae | valid_phantom | 0.10462265654206386 | 0.04301139057740514 |
| stage2ae | valid_animal | 0.4997355309073923 | 0.1431702793939404 |
| stage2ai | valid_phantom | 0.09742357618031064 | 0.040369930409670254 |
| stage2ai | valid_animal | 0.48785015325856906 | 0.23876800871618054 |

## Official Metric Interpretation

The 2026 CATHACTION PDF states that Task2 uses AP and mAP, with mAP as the primary private-test ranking metric. It does not clearly state whether mAP means:

- AP averaged over cases/classes/confidence thresholds only; or
- COCO-style AP averaged over IoU thresholds, which corresponds to our `mAP50-95`.

Therefore:

- Use Stage2AE as the default/balanced submission candidate until the official evaluation code clarifies the metric.
- Use Stage2AI as the high-IoU/high-mAP candidate if official validation rewards COCO-style mAP.
- Run the four-variant pipeline on official validation as soon as labels/evaluation are available.

## Verification

The four-variant pipeline completed on current public validation and reproduced known metrics for AB/AD/AE/AI. The focused tests passed after adding Stage2AI support:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_champion_pipeline.py \
  tests/test_task2_stage2ab_gt_free_export.py
```

Result: 13 passed.

Full Task2 export/metric regression after the documentation update:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/sweep_stage2ae_calibrated_policy.py \
  scripts/task2/export_stage2ab_domain_policy_predictions.py \
  scripts/task2/run_stage2_champion_pipeline.py \
  scripts/task2/evaluate_clean_prediction_topk.py \
  scripts/task2/evaluate_stage2ac_box_calibration.py

PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_stage2ae_policy_sweep.py \
  tests/test_task2_stage2ab_gt_free_export.py \
  tests/test_task2_clean_prediction_topk.py \
  tests/test_task2_champion_pipeline.py \
  tests/test_task2_stage2ab_verify.py \
  tests/test_task2_stage2v_domain_policy.py \
  tests/test_task2_stage2v_champion_verify.py \
  tests/test_task2_export_clean_predictions.py \
  tests/test_task2_detection_metrics.py
```

Result: 29 passed.
