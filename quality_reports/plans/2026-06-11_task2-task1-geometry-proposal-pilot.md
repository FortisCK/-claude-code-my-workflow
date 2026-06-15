# Task 2 Pilot: Task1-Segmentation Geometry Proposals

Date: 2026-06-11

Status: completed; result recorded in `quality_reports/decisions/2026-06-11_task2_task1_geometry_proposal_result.md`

## Objective

Test whether Task1 catheter/guidewire segmentation can generate better Task2
collision-detection candidates than YOLO coarse detection, especially for:

- animal normal, currently top50 R@0.50 = 0.0000;
- phantom collision, currently top50 R@0.50 around 0.27.

This is a proposal-recall pilot, not a final detector.

## Motivation

Stage2L showed that one-class YOLO proposal detection is structurally weak:

- it covers animal collision well;
- it misses animal normal completely;
- it has weak phantom collision coverage.

The GT-ROI classifier result suggests local classification is feasible once the
right region is supplied. The bottleneck is candidate generation. Task1
segmentation can provide a tool-geometry prior, so we should test whether
segmentation-derived candidates improve high-recall coverage.

## Experiment

Run on the Stage2L held-out panels first:

- `train_v0_v1_val_v2_valid_animal` (`video_2_animal`, 107 frames);
- `valid_phantom_balanced_small` (824 frames);
- combined panel for aggregate comparison.

Use a Task1 segmentation checkpoint to predict masks on these Task2 frames.
Then generate candidate boxes from:

1. foreground-union mask regions;
2. class-wise 3px dilation/proximity regions;
3. skeleton/endpoint/branch-like points where available;
4. fallback fixed-scale boxes around foreground centroids.

Evaluate these candidates using the same top-k proposal metric used for YOLO:

- top1/top5/top10/top20/top50 recall@IoU 0.50 and 0.75;
- class-wise recall for original Task2 class0/class1;
- split-wise recall for animal and phantom.

## Success Criteria

Continue this route if geometry proposals improve either of the current hard
failures:

- animal class0 top50 R@0.50 rises from 0.0000 to a nontrivial value;
- phantom class1 top50 R@0.50 rises above the current Stage2L value, 0.2743.

If promising, the next step is to union geometry proposals with YOLO proposals
and retrain/evaluate the ROI verifier.

## Verification

- Added synthetic tests in `tests/test_task2_geometry_proposals.py`.
- Full metrics were saved under `outputs/task2/geometry_proposals/`.
- Visual overlays were saved under the run-specific `visuals/` directories.
- Union-oracle metrics were saved under
  `outputs/task2/geometry_proposals/union_oracle_yolo_stage2l_task1_geometry_stage2l_panels/summary_metrics.json`.

## Result

Task1 geometry proposals alone are not competitive with the current YOLO
proposal model. They are useful as a complementary candidate source: rectangular
geometry boxes recover a meaningful fraction of animal class0 cases that YOLO
misses completely, but they hurt phantom and class1 coverage when used as the
only ranked proposal set.

The viable next route is multi-source proposal union plus a learned ROI
verifier/ranker, not replacing YOLO with raw Task1-derived geometry.
