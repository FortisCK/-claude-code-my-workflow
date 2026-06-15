# Task2 Stage2AR Stage2AQ Export Hardening Result

Date: 2026-06-14

## Question

Can Stage2AQ be exported by the standard champion pipeline and reproduce the
public-validation metrics from the Stage2AQ sweep?

## Implementation

Updated:

- `scripts/task2/run_stage2_champion_pipeline.py`
- `tests/test_task2_champion_pipeline.py`

The champion pipeline now supports six variants:

- `stage2ab`
- `stage2ad`
- `stage2ae`
- `stage2ai`
- `stage2an`
- `stage2aq`

Stage2AQ is implemented inside the pipeline as:

1. export Stage2AE-style baseline predictions using the Stage2AE box transform;
2. remove only rows where `domain == phantom` and `class_id == 1`;
3. load multi-source rows from
   `outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval`;
4. add only phantom class1 rows from source policy `yolo_stage2x`, i.e.
   `yolo_stage2l + stage2x_class1`;
5. use `rank_decay_roi_score_collision`, top-k 5 per sample/class, score scale
   0.75;
6. validate clean schema and recompute metrics.

## Commands

Stage2AQ-only reproduction:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/run_stage2_champion_pipeline.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2ar_champion_pipeline_with_aq/public_valid_aq \
  --variant stage2aq
```

All-variant export:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/run_stage2_champion_pipeline.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --output-dir outputs/task2/stage2ar_champion_pipeline_with_aq/public_valid_ab_ad_ae_ai_an_aq
```

## Metrics

Stage2AQ-only pipeline output:

| Split | rows | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: |
| `valid_combined` | 25,421 | 0.23661313056264216 | 0.07023557986011882 |
| `valid_phantom` | 23,789 | 0.12983063980792314 | 0.04994025156087294 |
| `valid_animal` | 1,632 | 0.4997355309073923 | 0.1431702793939404 |

This exactly reproduces the Stage2AQ sweep metrics.

All-variant public-validation summary:

| Variant | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2AB | 0.21128164745000094 | 0.050628135044670744 |
| Stage2AD | 0.21128164745000094 | 0.06251332119288261 |
| Stage2AE | 0.21128164745000094 | 0.06327005557430047 |
| Stage2AI | 0.18439741617609534 | 0.08394956471533466 |
| Stage2AN | 0.20037493968130657 | 0.06478975652205522 |
| Stage2AQ | 0.23661313056264216 | 0.07023557986011882 |

## Decision

Stage2AQ is now a reproducible pipeline candidate and should be treated as the
default balanced Task2 candidate on current public validation.

Keep Stage2AI as the high-IoU candidate until the official evaluator clarifies
whether primary `mAP` is COCO-style `mAP50-95`.

## Remaining Work

The pipeline can now reproduce Stage2AQ from saved Stage2AE and Stage2X
candidate/prediction rows. For hidden-test use, we still need to ensure the
upstream scripts generate the required Stage2X multi-source candidate rows for
the official split before running `stage2aq`.

## Verification

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/run_stage2_champion_pipeline.py \
  scripts/task2/sweep_stage2aq_phantom_class1_multisource.py

/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_champion_pipeline.py \
  tests/test_task2_stage2ab_gt_free_export.py \
  tests/test_task2_stage2ae_policy_sweep.py
```

Focused result: 18 passed.

Full Task2 regression also passed:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 85 passed.
