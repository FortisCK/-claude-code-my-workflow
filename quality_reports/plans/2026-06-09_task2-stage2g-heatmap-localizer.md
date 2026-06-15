# Task 2 Stage2G: Single-Object Heatmap Localizer

Date: 2026-06-09

Status: approved by user request ("先把现在跑的停了？然后开始新的")

## Objective

Replace the failing YOLO-only first stage with a detector formulation better
matched to Task 2: each frame has one annotated ROI, so predict class-specific
center heatmaps plus bbox geometry directly.

The immediate target is the known failure mode from Stage2F:

- valid_animal class1 has 0.0 top50 IoU>=0.25 oracle recall from the current
  YOLO candidate pool.

## Design

Use a MONAI 2D UNet as a compact heatmap localizer:

- input: resized/padded fluoroscopy frame;
- outputs:
  - 2 class center heatmaps;
  - width/height map;
  - optional center offset map;
- losses:
  - BCE or focal-style loss on class heatmaps;
  - L1/Huber size loss around GT center;
  - CE/BCE class supervision through the heatmap channel.

Inference:

1. Take the highest response over both class heatmaps.
2. Decode class, center, width, and height.
3. Convert back to original image coordinates.
4. Evaluate with Task 2 mAP and class/domain-specific IoU recall.

## Scope

1. Add target generation utilities/tests for heatmaps and box decoding.
2. Add `scripts/task2/train_heatmap_localizer.py`.
3. Run a smoke test on a tiny subset.
4. If smoke passes, start a short diagnostic training run.

## Non-Goals

- Do not remove previous YOLO experiments.
- Do not tune on hidden test.
- Do not assume this replaces the reranker path; reranker remains useful for
  phantom/class0 where candidate oracle recall is non-trivial.

## Success Criteria

This route is worth scaling if validation animal class1 improves from the
current YOLO-candidate oracle floor:

- valid_animal class1 center/box recall at IoU>=0.25 becomes non-zero; and/or
- valid_animal class1 mean IoU becomes materially above 0.0.

The first diagnostic run does not need to beat YOLO mAP; it must show that
heatmap localization can at least produce candidates near animal collision GT.
