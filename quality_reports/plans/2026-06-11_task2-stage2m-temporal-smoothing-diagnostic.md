# Task 2 Stage2M-A: Temporal Smoothing Diagnostic

Date: 2026-06-11

Status: completed; result recorded in `quality_reports/decisions/2026-06-11_task2_stage2m_temporal_smoothing_result.md`

## Objective

Test whether the current YOLO proposal detector already contains the true tip
location in its top-K candidates, but fails because frame-wise ranking is noisy.

## Experiment

Use existing Stage2L YOLO proposal outputs and run per-video temporal decoding:

- group candidate boxes by video and frame;
- keep top-K candidates per frame;
- find a smooth trajectory using dynamic programming;
- score candidates by detection confidence minus transition penalties for
  center jump, box-size change, and optional class switching;
- evaluate the decoded trajectory against GT boxes.

## Metrics

Compare with raw YOLO proposal metrics:

- IoU recall at 0.25, 0.50, 0.75;
- center-distance recall at 5, 10, 20 px;
- mean IoU;
- class-wise and domain-wise metrics.

## Success Criteria

Continue this route if temporal smoothing improves animal class0 recall or
phantom class1 localization without damaging animal class1.

## Verification

- Add synthetic unit tests for dynamic programming path selection.
- Save metrics and per-frame decoded predictions under
  `outputs/task2/temporal_smoothing/`.

## Result

Implemented `scripts/task2/evaluate_temporal_smoothing.py` and
`tests/test_task2_temporal_smoothing.py`.

The diagnostic did not improve IoU-based detection enough to continue simple
YOLO-topK temporal smoothing as the main route. Existing YOLO proposals have
some center proximity on phantom, but the selected boxes and ranking remain too
weak; animal class0 is not covered by YOLO top50 in a way smoothing can rescue.
