# Task 2 Stage2G Heatmap Localizer Result

Date: 2026-06-09

## Run

```text
outputs/task2/heatmap_localizer/monai_unet_heatmap256_stage2g_e12/
```

Log:

```text
quality_reports/logs/task2_heatmap_monai_unet256_stage2g_e12.log
```

Model:

- MONAI UNet
- input size 256
- class-specific center heatmaps + size/offset heads
- 12 epochs
- primary metric: `valid_animal/class1_recall_iou_0.25`

## Result

The diagnostic objective failed:

```text
best valid_animal/class1_recall_iou_0.25 = 0.0000
valid_animal/class1_mean_iou = 0.0000 throughout
```

The model learned a weak phantom/class0 signal:

```text
epoch 12 valid_combined/mean_iou = 0.0633
epoch 12 valid_combined/recall_iou_0.25 = 0.1309
epoch 12 valid_combined/class0_recall_iou_0.25 = 0.2617
epoch 12 valid_combined/class1_recall_iou_0.25 = 0.0000
```

## Important Split Fact

The Task 2 released training split currently contains no animal-named samples:

```text
train_clean total 35084, animal_name 0
valid_phantom total 11024, animal_name 0
valid_animal total 398, animal_name 398
```

Thus animal validation is a true domain shift from phantom-only supervised
training in the current split setup.

## Interpretation

Naive single-frame heatmap localization trained on phantom data does not solve
animal collision localization. This supports the diagnosis that animal class1
is not just a YOLO ranking failure; it is a domain/generalization problem and
possibly a weak-visual-signal problem.

## Decision

Do not scale this exact heatmap setup as-is.

Next steps should focus on diagnosing/fixing the first-stage formulation:

1. Visualize heatmap predictions to see whether the model follows phantom
   spatial priors or collapses to broad false peaks.
2. Change the localizer target from class-specific heatmaps to:
   - class-agnostic ROI/objectness heatmap;
   - separate class head;
   - optional bbox refinement.
3. Preserve aspect ratio with letterbox resizing rather than square warping.
4. Add stronger domain augmentation and/or use task/video geometry, because
   supervised animal labels are not in the training split.
5. Keep two-stage reranking for phantom/class0 where candidate oracle recall is
   non-trivial, but treat animal class1 as a separate domain-generalization
   problem.
