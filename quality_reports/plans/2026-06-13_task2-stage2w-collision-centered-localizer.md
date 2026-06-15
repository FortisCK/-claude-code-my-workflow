# Task2 Stage2W Collision-Centered Localizer Plan

Date: 2026-06-13

## Motivation

Stage2V class-aware fusion is now the current best public validation export:

- `valid_combined` mAP50: `0.2009680309`
- `valid_combined` mAP50-95: `0.0492404464`

The candidate-pool oracle diagnostic shows that the main limitation is localization ceiling, not just ranking:

- yolo-only class 1 recall@0.75: `0.1210`
- mixed class 1 recall@0.75: `0.1210`
- augmented class 1 recall@0.75: `0.1310`

So the next useful step is to create higher-IoU candidates around the actual collision/contact point.

## Objective

Build and evaluate a Stage2W candidate generator that predicts collision-centered boxes, then integrate it into the existing Stage2U/Stage2V ranker pipeline only if it improves candidate-pool oracle recall@0.75.

## Proposed Design

1. **Target definition**
   - Use existing Task2 GT boxes and class labels.
   - Train a localizer to predict a compact box centered on the GT collision/contact region.
   - Keep both class 0 / normal and class 1 / collision targets available, but optimize primarily for class 1 high-IoU candidates.

2. **Model**
   - Start with a simple high-resolution detector/localizer already supported by the project stack.
   - Avoid another large architecture detour unless oracle diagnostics show this target formulation is promising.

3. **Data split**
   - Use a balanced public-data split for development:
     - training includes both phantom and animal where labels are available;
     - validation remains held out at case/video level.
   - Keep the current official-like split metrics as a separate reporting baseline.

4. **Evaluation gates**
   - Primary gate before ranker training: candidate-pool oracle.
   - Required signal to continue:
     - class 1 recall@0.75 should improve materially over `0.1310`;
     - candidates/sample should stay controlled enough for Stage2U inference.
   - Secondary gate: Stage2V full valid mAP after integration.

## Work Items

1. Audit current Task2 labels and split metadata to confirm target boxes and case/video grouping.
2. Generate a Stage2W training manifest with balanced phantom/animal case-level split.
3. Train a compact collision-centered localizer or adapt the existing detector target formulation.
4. Export Stage2W candidates to the existing candidate CSV schema.
5. Run `evaluate_candidate_oracle.py` against Stage2W candidates.
6. If oracle improves, integrate Stage2W candidates into Stage2U/Stage2V and run full validation.
7. Record a decision comparing Stage2V current best vs Stage2W-integrated result.

## Stop Conditions

Stop Stage2W if either is true:

- candidate-pool class 1 recall@0.75 does not exceed the augmented pool by a meaningful margin;
- candidate count explodes without improving AP after ranker integration.

## Current Best to Beat

- Current export: `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`
- `valid_combined` mAP50: `0.2009680309`
- `valid_combined` mAP50-95: `0.0492404464`
