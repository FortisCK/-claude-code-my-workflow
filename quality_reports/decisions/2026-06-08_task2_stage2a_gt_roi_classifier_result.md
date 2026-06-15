# Task 2 Stage2A GT-ROI Classifier Result

Date: 2026-06-08

## Status

Completed.

The Stage2A GT-ROI classifier finished all 30 epochs. The best checkpoint is
epoch 27 by the primary metric `valid_combined/average_precision`.

This experiment uses ground-truth bboxes to crop tip-centered ROIs. Therefore,
the result measures whether `normal` vs `collision` is learnable once tip
localization is known. It is not a final object-detection mAP result and should
not be directly compared as a detector score.

## Run

Output directory:

```text
outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/
```

Log:

```text
quality_reports/logs/task2_roi_convnext_tiny_gtroi224_scale8_e30.log
```

Best checkpoint:

```text
outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/checkpoints/best.pt
```

Metrics:

```text
outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/metrics.csv
```

## Configuration

| Item | Value |
| --- | --- |
| Backbone | `convnext_tiny` |
| Pretraining | public ImageNet weights via `timm` |
| Input | GT bbox-centered ROI |
| Input size | `224` |
| Crop scale | `8x` max bbox side |
| Crop clamp | min `224`, max `512` pixels before resize |
| Loss | class-weighted cross entropy |
| Sampler | balanced sampler |
| Epochs | `30` |
| Batch size | `128` |
| Device | `cuda:0` |

## Best Epoch

Best epoch:

```text
27
```

Primary metric:

```text
valid_combined/average_precision = 0.9067337231137343
```

## Best Metrics

| Split | Accuracy | Balanced Acc | Collision Precision | Collision Recall | Collision F1 | AP | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combined | `0.9642` | `0.9460` | `0.6387` | `0.9254` | `0.7558` | `0.9067` | `0.9907` |
| phantom | `0.9654` | `0.9226` | `0.5224` | `0.8762` | `0.6546` | `0.8838` | `0.9885` |
| animal | `0.9296` | `0.8889` | `0.9067` | `1.0000` | `0.9510` | `0.9725` | `0.9664` |

## Interpretation

This is a strong positive diagnostic for the two-stage direction.

The previous full-frame YOLO11s detector had a very low collision detection
score, with collision `mAP50-95` nearly zero. In contrast, when the true tip
location is given and the model only has to classify a local ROI, the collision
state is clearly learnable on the released validation splits.

The important conclusion is not that the final Task 2 problem is solved. The
important conclusion is:

```text
Task 2 failure is not simply "collision is visually impossible";
it is likely the coupling of tiny localization and subtle state classification.
```

Therefore, the next serious route should be a two-stage pipeline:

1. detector localizes the tip bbox;
2. ROI classifier assigns `normal` vs `collision`;
3. final result fuses detector localization with classifier state probability.

## Caveats

- This uses GT bboxes at validation time, so it is an upper-bound diagnostic for
  classification, not a deployable hidden-test pipeline.
- The released validation splits may not match hidden-test distributions.
- `0 = normal`, `1 = collision` remains a working class-name assumption pending
  official class-name confirmation.
- Classification AP/AUROC are not object-detection mAP.

## Next Step

Build Stage2B:

1. run the trained YOLO detector on validation images;
2. crop ROIs from predicted boxes;
3. classify those predicted-box ROIs with the Stage2A classifier;
4. report detector-style AP/mAP and compare with YOLO-only.

If predicted-box ROI classification drops sharply, improve localization first.
If it remains strong, tune score fusion and thresholds.

