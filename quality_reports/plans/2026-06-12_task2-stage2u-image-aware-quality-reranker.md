# Task2 Stage2U: Image-Aware ROI Quality Reranker

Date: 2026-06-12
Status: proposed

## Motivation

The current Task2 pipeline is functional but not yet strong enough:

- Stage2O ConvNeXt ROI ranker is the current AP50/robustness baseline:
  - valid_combined mAP50: 0.1887
  - valid_combined mAP50-95: 0.0388
- Stage2T tabular selector is the current mAP50-95 champion but only by a very small margin:
  - valid_combined mAP50: 0.1551
  - valid_combined mAP50-95: 0.0403
- Candidate-pool localization recall remains limited:
  - valid_combined R@0.50: 0.5585
  - valid_combined R@0.75: 0.2213
- Stage2S oracle diagnostics showed that the candidate pool has headroom, but the learned/non-oracle score often selects the wrong candidate.
- Stage2P box refinement can improve augmented R@0.75, but the current scoring stack does not reliably exploit those improved candidates.

The next experiment should directly train the ROI image model to predict candidate localization quality, not only class identity.

## Hypothesis

A ConvNeXt ROI model with multi-task outputs can improve final detection ranking by combining:

1. class logits for background / normal / collision;
2. IoU regression for candidate-vs-GT quality;
3. IoU >= 0.50 binary quality logit;
4. IoU >= 0.75 binary quality logit.

This should be more aligned with mAP than the current Stage2O classifier and more informative than Stage2T's tabular-only selector.

## Proposed Implementation

Add a new script:

- `scripts/task2/train_stage2u_quality_ranker.py`

The script should reuse Stage2O/Stage2P components where possible:

- candidate CSV loading from `train_stage2o_candidate_ranker.py`;
- ROI cropping from `cathaction.data.task2_roi`;
- model creation pattern from Stage2O/Stage2P;
- detection mAP evaluation from `cathaction.metrics.detection`;
- candidate-budget and split-overlap guards from Stage2O.

Training samples:

- input: candidate ROI crop;
- targets:
  - verifier label: background / normal / collision;
  - `gt_iou` regression target;
  - `gt_iou >= 0.50`;
  - `gt_iou >= 0.75`.

Loss:

- CE(class logits, verifier_label);
- SmoothL1(pred_iou, gt_iou);
- BCE(iou50_logit, gt_iou >= 0.50);
- BCE(iou75_logit, gt_iou >= 0.75).

Initial default weights:

- class CE: 1.0
- IoU regression: 1.0
- IoU50 BCE: 0.5
- IoU75 BCE: 1.0

Scoring modes to evaluate:

- class-only baseline equivalent to Stage2O;
- `quality_iou75 * class_prob`;
- `quality_iou50 * class_prob`;
- `pred_iou * class_prob`;
- `rank_decay_quality_iou75 * class_prob`.

## Validation

Required metrics:

- valid_combined mAP50 and mAP50-95;
- valid_phantom mAP50 and mAP50-95;
- valid_animal mAP50 and mAP50-95;
- localization recall retained after top-k selection;
- class-wise AP if available.

Success criteria for smoke test:

- script compiles;
- tests pass;
- `--train-limit` / `--valid-limit` run completes;
- output contains all score modes and detection metrics.

Success criteria for real run:

- Stage2U beats Stage2O on mAP50-95 by at least +0.005 absolute without catastrophic AP50 loss, or
- Stage2U beats Stage2O on mAP50 while preserving mAP50-95, or
- Stage2U clearly identifies that the remaining bottleneck is candidate-pool recall rather than scoring.

## Risks

- Positive IoU75 examples are sparse, especially on animal, so IoU75 classifier may be unstable.
- If candidate-pool R@0.75 remains too low, quality ranking alone cannot produce high COCO-style mAP.
- Training on train-side Stage2O predictions must keep normal split-overlap guards enabled except for intentional train-side prediction export.
- Overweighting quality may reduce AP50 by suppressing low-IoU but valid AP50 matches.

## Decision Gate

If Stage2U fails to improve beyond Stage2T/Stage2O, the next direction should be candidate-pool redesign or integrating Stage2P refined boxes into Stage2U training/evaluation, not more tabular-only tuning.
