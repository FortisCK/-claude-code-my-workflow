# Task 2 Stage2A: GT-ROI Collision Classifier Diagnostic

Date: 2026-06-05

Status: approved by user request ("那试试")

## Objective

Test whether Task 2 collision vs normal can be learned when the tip location is
already known.

The first YOLO11s full-frame detector baseline reached only about `0.042`
combined `mAP50-95`, with collision AP nearly zero. This suggests that the
single-stage detector may be mixing two hard problems:

1. tiny tip localization;
2. subtle collision-state classification.

Stage2A isolates the second problem by cropping a region around the ground-truth
bbox and training a binary classifier on the ROI.

## Scope

1. Reuse the clean Task 2 split files already generated under
   `configs/task2/splits/`.
2. Implement a GT-bbox-centered ROI dataset:
   - crop around the bbox center;
   - include context via fixed-size square crop or bbox expansion;
   - resize to classifier input size.
3. Train a lightweight image classifier:
   - first preference: `timm` ConvNeXt-Tiny if available;
   - fallback: torchvision ConvNeXt/ResNet if available;
   - final fallback: small local CNN for smoke only.
4. Report validation metrics:
   - accuracy;
   - balanced accuracy;
   - collision precision/recall/F1;
   - AP and AUROC when available;
   - combined, phantom, and animal validation separately.
5. Save output artifacts under `outputs/task2/roi_classifier/` and logs under
   `quality_reports/logs/`.

## Non-Goals

- Do not replace the final Task 2 detector yet.
- Do not train on hidden-test data or hidden-test feedback.
- Do not use predicted YOLO boxes in Stage2A; this stage uses GT boxes to
  measure the classification upper bound.
- Do not add temporal modeling yet.

## Verification

- Python files must pass `py_compile`.
- A small smoke run must complete before any longer run starts.
- The training log must identify the exact split files, crop settings, model,
  metrics, and checkpoint path.
- The decision record must state whether GT-ROI classification looks promising.

