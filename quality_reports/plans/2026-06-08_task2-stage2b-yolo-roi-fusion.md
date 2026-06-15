# Task 2 Stage2B: YOLO + ROI Classifier Fusion

Date: 2026-06-08

Status: approved by user request ("可以")

## Objective

Build and evaluate the first deployable two-stage Task 2 prototype:

1. use the existing full-frame YOLO detector to predict a tip bbox;
2. crop an ROI around the predicted bbox;
3. use the Stage2A ConvNeXt GT-ROI classifier to classify `normal` vs
   `collision`;
4. compute detector-style AP/mAP on the released validation splits.

Stage2A showed that collision/normal is learnable when tip localization is
known. Stage2B tests how much of that signal survives when the ROI is cropped
from predicted YOLO boxes instead of GT boxes.

## Inputs

Detector checkpoint:

```text
outputs/task2/yolo/yolo11s_1024_clean_combined_e100/weights/best.pt
```

ROI classifier checkpoint:

```text
outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/checkpoints/best.pt
```

Validation splits:

```text
configs/task2/splits/valid_combined_labels.txt
configs/task2/splits/valid_phantom_labels.txt
configs/task2/splits/valid_animal_labels.txt
```

## Scope

1. Implement a fusion evaluator:
   - run YOLO on validation images;
   - select the top predicted box per image;
   - crop ROI using the predicted box;
   - classify with the ConvNeXt ROI classifier;
   - output one detection per image with fused class and score.
2. Compute detection AP:
   - class-wise AP at IoU thresholds `0.50:0.05:0.95`;
   - `mAP50`;
   - `mAP50-95`;
   - split-specific combined / phantom / animal results.
3. Save JSON/CSV outputs under `outputs/task2/stage2b_yolo_roi_fusion/`.

## Non-Goals

- Do not retrain YOLO yet.
- Do not use GT bboxes during fusion, except as validation labels.
- Do not tune on hidden-test feedback.
- Do not claim Stage2A classification AP as detector mAP.

## Verification

- Python files must pass `py_compile`.
- A small smoke evaluation on a limited validation subset must run before full
  validation.
- The output must include enough diagnostics to decide whether failure is due
  to bbox localization or ROI classification.

