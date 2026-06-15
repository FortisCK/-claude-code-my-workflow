# Task 2 Stage2C: Class-Agnostic Proposal Detector

Date: 2026-06-08

Status: approved by user request ("可以")

## Objective

Improve Task 2 coarse localization by training YOLO as a class-agnostic proposal
detector. The detector should learn only where the local tool/contact ROI is,
while the Stage2A ConvNeXt ROI classifier remains responsible for normal vs
collision classification.

## Motivation

Stage2B showed that GT-ROI collision classification is strong, but YOLO-derived
ROIs are often mislocalized:

- top1 candidate recall at IoU 0.5 on `valid_combined`: 0.3810
- top5 best-candidate recall at IoU 0.5 on `valid_combined`: 0.4931
- best top5 fusion with `YOLO_conf * ROI_prob`: 0.0406 mAP50-95, close to but
  not above the YOLO11s two-class baseline best of 0.0421

The next detector should maximize proposal recall rather than solve the subtle
normal/collision distinction in the full frame.

## Scope

1. Create a class-agnostic Task 2 YOLO data mirror:
   - keep images via symlink to the original Task 2 image directory;
   - rewrite every label line to class id `0`, preserving bbox coordinates;
   - preserve the leakage-free train/validation split manifests;
   - write YOLO YAMLs with one class: `tool_roi`.
2. Verify:
   - rewritten labels are all class `0`;
   - split counts remain unchanged;
   - smoke YOLO training can start and validate.
3. Start the first real class-agnostic YOLO proposal run:
   - model: YOLO11s pretrained checkpoint;
   - image size: 1024 initially, so it is comparable with the two-class baseline;
   - validation: `valid_combined`;
   - save under `outputs/task2/yolo_proposal/`.

## Non-Goals

- Do not use hidden-test feedback.
- Do not change Stage2A ROI classifier in this stage.
- Do not claim class-agnostic YOLO mAP as final Task 2 mAP without ROI
  reclassification.

## Success Criteria

The first target is proposal coverage, not final two-class mAP:

- top1 IoU>=0.5 should exceed the current 0.3810.
- top5 IoU>=0.5 should exceed the current 0.4931.
- If proposal recall improves materially, run Stage2D fusion:
  class-agnostic YOLO topK proposals + ConvNeXt ROI classifier +
  `proposal_conf * ROI_prob`.
