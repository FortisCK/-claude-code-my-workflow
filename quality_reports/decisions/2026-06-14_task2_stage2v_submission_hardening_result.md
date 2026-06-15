# Task2 Stage2V Submission Hardening Result

Date: 2026-06-14

## Decision

Keep Stage2V as the current Task2 internal champion and freeze it with a verifier plus manifest rather than opening another model branch immediately.

## Added Artifacts

- Plan: `quality_reports/plans/2026-06-14_task2-stage2v-submission-hardening.md`
- Verifier: `scripts/task2/verify_stage2v_champion_export.py`
- Tests: `tests/test_task2_stage2v_champion_verify.py`
- Hidden-test checklist: `quality_reports/reports/2026-06-14_task2_stage2v_hidden_test_checklist.md`
- Generated manifest:
  `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/champion_manifest.json`

## Frozen Champion

Path:

`outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`

Policy:

- class 0: `prob_iou75_source_rank_decay_roi`
- class 1: `roi`

Metrics:

| Split | mAP50 | mAP50-95 | Rows |
| --- | ---: | ---: | ---: |
| valid_combined | 0.20096803092334323 | 0.04924044637514719 | 40,970 |
| valid_phantom | 0.09603994929405452 | 0.04086057058652194 | 39,338 |
| valid_animal | 0.49984030661130624 | 0.05000256627474636 | 1,632 |

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/verify_stage2v_champion_export.py \
  scripts/task2/export_clean_predictions.py \
  scripts/task2/export_stage2v_class_aware_predictions.py

PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest tests/test_task2_stage2v_champion_verify.py \
  tests/test_task2_export_clean_predictions.py

python3 scripts/task2/verify_stage2v_champion_export.py \
  --prediction-dir outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export
```

Result:

- `5 passed`
- verifier status: `ok`
- manifest written beside the champion prediction CSVs.

## Interpretation

This does not improve model accuracy. It improves reproducibility and reduces submission risk. The next accuracy-improving step should be a true source-matched train/validation listwise selector or a revised Task2 detector once official 2026 validation data is released.

