# Task2 Stage2U Plan: Image-aware Candidate Quality Ranker

Date: 2026-06-12
Status: proposed, awaiting approval
Task: CATHACTION Task2 collision detection

## Goal

Improve the current three-stage Task2 pipeline by replacing the weak tabular-only candidate selector with an image-aware ROI quality ranker.

Current pipeline:

1. YOLO + Task1-geometry candidate pool.
2. Stage2O ConvNeXt ROI ranker trained as background/normal/collision classifier.
3. Stage2T tabular selector trained from candidate metadata and Stage2O probabilities.

The next experiment keeps this structure, but changes Stage 2 from ordinary ROI classification to direct candidate-quality learning.

## Evidence Behind This Direction

Current Stage2O champion:

- Run: `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval`
- Score mode: `rank_decay_roi`
- valid_combined mAP50: `0.1887`
- valid_combined mAP50-95: `0.0388`
- candidate-pool valid_combined R@0.50: `0.5585`
- candidate-pool valid_combined R@0.75: `0.2213`

Stage2T result:

- Run: `outputs/task2/stage2t_tabular_selector/histgb_stage2o_trainpred_fullvalid`
- best policy: `prob_iou75 * rank_decay_roi`, `k=50`
- valid_combined mAP50: `0.1551`
- valid_combined mAP50-95: `0.0403`

Interpretation:

- Stage2T gives only `+0.0015` mAP50-95 over Stage2O, while reducing mAP50 by `0.0336`.
- The best Stage2T policy keeps `k=50`, close to no pruning, so simple candidate count control is not the fix.
- Stage2S showed oracle candidate selection has much higher headroom, meaning the pool contains useful candidates that the current scorer does not rank correctly.
- Stage2P box refinement improves augmented R@0.75, but the standalone refined box is not reliably better than the original box; refinement needs a better selector.

Therefore the next bottleneck is not another detector recipe or another tabular score rule. It is image-aware candidate quality estimation.

## Proposed Method

Implement a new CSV-driven ROI model:

`candidate crop + candidate metadata -> class logits + IoU quality heads`

The model should predict:

- `class_logits`: background / normal / collision;
- `iou_regression`: continuous candidate IoU with GT;
- `iou50_logit`: whether candidate IoU >= 0.50;
- `iou75_logit`: whether candidate IoU >= 0.75.

The final detection score should be swept over policies such as:

- `class_prob * prob_iou50`;
- `class_prob * prob_iou75`;
- `class_prob * pred_iou`;
- `rank_decay(class_prob * prob_iou75)`;
- optional blend with source confidence only after image-only quality is evaluated.

This is intentionally different from Stage2O:

- Stage2O asks, "What class is this crop?"
- Stage2U asks, "Is this crop both the correct class and the correct box?"

## Implementation Steps

1. Add `scripts/task2/train_stage2u_quality_ranker.py`.
   - Reuse `CandidateExample`, `load_candidate_csv`, `apply_candidate_budget`, ROI crop helpers, metric conversion, and split loading from Stage2O where practical.
   - Keep leakage checks enabled by default.
   - Support eval-only checkpoint loading.

2. Define a multi-task ConvNeXt model.
   - Backbone: existing `convnext_tiny` creation path from Stage2O.
   - Heads:
     - 3-way class head.
     - 1-way IoU regression head with sigmoid output.
     - 2-way quality logits or two binary heads for IoU50/IoU75.
   - First implementation can use a shared backbone and separate linear heads.

3. Define training targets.
   - Classification:
     - IoU >= 0.50: positive normal/collision according to GT class.
     - IoU <= 0.20: background.
     - 0.20 < IoU < 0.50: keep for quality heads, but downweight or ignore for class CE.
   - Quality:
     - continuous target `gt_iou`, clipped to [0, 1].
     - binary targets `gt_iou >= 0.50` and `gt_iou >= 0.75`.

4. Sampling.
   - Balance positives/background.
   - Balance normal/collision.
   - Balance domain when possible.
   - Keep source diversity so the model does not become only a YOLO-source prior learner.

5. Evaluation.
   - Produce prediction CSVs for `valid_combined`, `valid_phantom`, and `valid_animal`.
   - Report mAP50, mAP50-95, loc R@0.50, loc R@0.75, and class-wise AP.
   - Compare against:
     - Stage2O `rank_decay_roi`: `0.1887 / 0.0388`;
     - Stage2T best: `0.1551 / 0.0403`.

6. Add focused tests.
   - Dataset target construction for background, positive, and intermediate-IoU candidates.
   - Multi-task loss masks do not leak GT columns into inference features.
   - Prediction rows include both class detections and quality scores.
   - Per-sample top-k selection preserves candidate order and does not drop all candidates.

## Success Gates

Treat Stage2U as successful only if it clears at least one of these:

- valid_combined mAP50-95 >= `0.0500`;
- valid_combined mAP50 >= `0.1887` while mAP50-95 >= `0.0403`;
- valid_phantom mAP50-95 improves by at least `0.005` without animal collapse;
- quality ranking clearly beats Stage2T tabular ranking at the same candidate pool and k.

If it only improves mAP50-95 by another `0.001` while reducing mAP50, record it as a weak calibration result, not a solved model.

## Stop Conditions

Stop this line and revisit candidate generation if:

- candidate-pool R@0.75 remains the dominant ceiling after Stage2U;
- Stage2U cannot beat Stage2O or Stage2T on either mAP50 or mAP50-95;
- gains come only from valid-policy selection and do not generalize across phantom/animal split reports.

## Verification

Before full training:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile scripts/task2/train_stage2u_quality_ranker.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2u_quality_ranker.py tests/test_task2_stage2o_ranker.py
```

Smoke run:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python scripts/task2/train_stage2u_quality_ranker.py \
  --name smoke_convnext_tiny_stage2u_quality_limit512 \
  --train-limit 512 \
  --valid-limit 128 \
  --epochs 1 \
  --batch-size 64 \
  --workers 2 \
  --device auto
```

Full run after smoke pass:

```bash
nohup /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python scripts/task2/train_stage2u_quality_ranker.py \
  --name convnext_tiny_stage2u_quality_yolo_geometry_e8 \
  --epochs 8 \
  --batch-size 256 \
  --workers 8 \
  --device auto \
  > quality_reports/logs/task2_stage2u_convnext_tiny_quality_yolo_geometry_e8.log 2>&1 &
```

## Approval Request

If approved, implement Stage2U as the next main Task2 experiment. Do not start another YOLO/YOLOV or pure temporal-detector run before this quality-ranker test, because the current evidence points to candidate selection quality as the stronger bottleneck.
