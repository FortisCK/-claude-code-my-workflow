# Task2 Stage2V Hidden-Test Preparation Checklist

Date: 2026-06-14

## Frozen Internal Policies

Conservative internal Task2 champion:

`outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`

Current public-validation best:

`outputs/task2/stage2v_domain_policy`

Frozen policy:

- class 0 / normal:
  - source run: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`
  - score mode: `prob_iou75_source_rank_decay_roi`
- class 1 / collision:
  - source run: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`
  - score mode: `roi`

Frozen validation metrics:

- `valid_combined` mAP50: 0.20096803092334323
- `valid_combined` mAP50-95: 0.04924044637514719
- `valid_phantom` mAP50: 0.09603994929405452
- `valid_phantom` mAP50-95: 0.04086057058652194
- `valid_animal` mAP50: 0.49984030661130624
- `valid_animal` mAP50-95: 0.05000256627474636

Stage2AB public-validation best:

- `valid_combined` mAP50: 0.21128164745000094
- `valid_combined` mAP50-95: 0.050628135044670744
- `valid_phantom` mAP50: 0.10462265654206386
- `valid_phantom` mAP50-95: 0.04301139057740514
- `valid_animal` mAP50: 0.4997355309073923
- `valid_animal` mAP50-95: 0.050008473801353336

Stage2AB policy:

- class 0 / normal:
  - phantom: `prob_iou75_source_rank_decay_roi`
  - animal: `prob_iou75_source_rank_decay_roi`
- class 1 / collision:
  - phantom: `rank_decay_roi`
  - animal: `prob_iou75_roi`

## Verification Command

Run this before using the artifact in any report or submission adapter:

```bash
python3 scripts/task2/verify_stage2v_champion_export.py \
  --prediction-dir outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export
```

Expected output:

- writes `champion_manifest.json` beside the prediction CSVs;
- rejects any prediction CSV containing validation-only fields;
- rejects row-count or frozen-metric drift.

## Current Internal CSV Schema

The clean internal prediction files use:

`sample_id, video_id, frame_index, domain, class_id, score, x1, y1, x2, y2, source, source_rank, score_mode, policy_name`

This is **not yet guaranteed** to be the official hidden-test schema. It is a clean internal artifact that can be adapted once official instructions specify the result file format.

## Hidden-Test Adaptation Steps

1. Generate the same YOLO-only candidate/eval rows for the official validation or hidden-test split.
2. Apply both frozen policies:
   - conservative Stage2V;
   - public-valid-best Stage2AB domain-aware calibration.
3. If the official validation data includes reliable domain metadata, compare Stage2AB against Stage2V on official validation before choosing one for hidden test.
4. If domain metadata is missing or ambiguous, default to Stage2V unless the official instructions provide an equivalent domain indicator.
5. Convert the selected clean internal CSV to the official challenge result schema.
6. Build and smoke-test the Docker submission.

Conservative Stage2V policy:

   - class 0 uses `prob_iou75_source_rank_decay_roi`;
   - class 1 uses `roi`.

For a non-public-valid export whose metric should not be compared with the frozen champion, use:

```bash
python3 scripts/task2/verify_stage2v_champion_export.py \
  --prediction-dir PATH_TO_EXPORT \
  --skip-frozen-metric-check
```

## Risks To Report Honestly

- The champion is still modest in absolute mAP50-95.
- Animal class 0 is weak in current validation.
- Expanded candidate pools improved oracle recall but hurt AP because scoring/ranking was not reliable.
- The official 2026 split may differ from the current public data split; procedure-level split discipline must be preserved.
