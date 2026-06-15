# Task 2 Stage2M-B: Sequence-Aware Tip Localizer

Date: 2026-06-11

Status: completed; result recorded in `quality_reports/decisions/2026-06-11_task2_stage2m_sequence_tip_localizer_result.md`

## Objective

Implement and run a first sequence-aware, class-agnostic tip localizer for Task2.
This replaces class-specific YOLO coarse detection as the main proposal route.

## Design

- Input: 5 grayscale frames centered at target frame `t`.
- Target: class-agnostic tip center heatmap for frame `t`.
- Heads:
  - heatmap objectness;
  - width/height regression;
  - center offset regression.
- Loss:
  - weighted BCE/focal-style heatmap loss;
  - L1 size loss at GT center;
  - L1 offset loss at GT center.

## First Evaluation

Report proposal localization metrics, not final mAP:

- center recall at 5/10/20 px;
- IoU recall at 0.25/0.50/0.75;
- mean center distance;
- mean IoU;
- class-wise and split-wise metrics for animal and phantom panels.

## Data Splits

Use the existing public-development Stage2L split discipline:

- train: phantom train plus animal `video_0_animal` and `video_1_animal`;
- validation: held-out `video_2_animal` plus balanced phantom panel.

## Verification

- Add/compile script.
- Add synthetic tests for clip indexing and class-agnostic target generation.
- Run a short smoke training before any longer run.

## Result

Implemented `scripts/task2/train_sequence_tip_localizer.py` and
`tests/test_task2_sequence_tip_localizer.py`.

The first from-scratch MONAI UNet localizer was weak. A pretrained
SMP/ConvNeXt-FPN localizer produced real center-localization signal on phantom
and animal class1, but did not solve animal class0. Training also required fp32
or stabilization; AMP runs can become NaN after several epochs.
