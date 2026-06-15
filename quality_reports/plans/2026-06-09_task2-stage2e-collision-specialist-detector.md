# Task 2 Stage2E: Collision-Specialist Detector

Date: 2026-06-09

Status: approved by user request ("你来吧，直接写完脚本并开始训练")

## Objective

Train a detector dedicated to collision localization. Class-1 collision frames
are positive images with one `collision_roi` bbox; class-0 normal frames are
negative images with empty YOLO label files.

This directly tests whether the current Task 2 bottleneck is generic YOLO
localization or the normal/collision class conflict in the two-class setup.

## Scope

1. Add a data-preparation script that creates a collision-only YOLO data mirror:
   - images symlink to `datasets/collision_detection/images`;
   - class-1 labels are rewritten to class id `0`;
   - class-0 labels are rewritten to empty files;
   - leakage-free train/validation split manifests are preserved;
   - YAML files expose one class: `collision_roi`.
2. Add focused tests for the collision-only label rewriting.
3. Generate the collision-only data mirror and YAMLs.
4. Smoke-validate syntax/tests.
5. Start the first real collision-specialist YOLO11s run:
   - image size: 1024;
   - epochs: 80 initially;
   - batch: 32 if memory allows;
   - validation: combined split.

## Evaluation Plan

After training starts or finishes, evaluate proposal recall with the existing
proposal evaluator:

- combined class-1 top5 IoU>=0.5
- animal class-1 top5 IoU>=0.5
- class-1 AP/mAP in a merged detector setup, if proposal recall improves

## Non-Goals

- Do not change hidden-test assumptions.
- Do not alter the original Task 2 dataset.
- Do not replace the current best detector until collision-specialist results
  are measured.

## Success Criteria

This stage is useful if it lifts collision proposal recall materially above the
class-agnostic detector:

- combined class-1 top5 IoU>=0.5 should exceed 0.0365;
- animal class-1 top5 IoU>=0.5 should no longer be 0.0;
- if localization improves, merge class-1 predictions with the existing
  two-class YOLO/ROI pipeline in the next stage.
