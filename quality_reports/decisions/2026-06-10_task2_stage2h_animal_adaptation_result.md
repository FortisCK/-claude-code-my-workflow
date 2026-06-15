# Task 2 Stage2H Animal Adaptation Diagnostic Result

Date: 2026-06-10

## Run

Output:

```text
outputs/task2/yolo_adapt_animal/yolo11s_1024_train_v1_val_v2_e40
```

Log:

```text
quality_reports/logs/task2_yolo11s_1024_adapt_animal_train_v1_val_v2_e40.log
```

Split:

```text
configs/task2/collision_detection_adapt_animal_train_v1_val_v2.local.yaml
```

Training configuration:

- Model: YOLO11s pretrained checkpoint
- Input size: 1024
- Batch size: 32
- Planned epochs: 40
- Early stopping patience: 10
- Train: `train_clean + video_1_animal`
- Validation: held-out `video_2_animal`

## Completion

The run stopped early at epoch 26:

```text
EarlyStopping: Training stopped early as no improvement observed in last 10 epochs.
Best results observed at epoch 16, best model saved as best.pt.
26 epochs completed in 2.555 hours.
```

The GPU was idle after completion.

## Aggregate Validation Metrics

From `results.csv`:

| Metric | Value | Epoch |
| --- | ---: | ---: |
| Best mAP50 | 0.49750 | 12 |
| Best mAP50-95 | 0.14601 | 16 |
| Final epoch mAP50 | 0.47558 | 26 |
| Final epoch mAP50-95 | 0.14070 | 26 |

Final validation of `best.pt`:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 107 | 107 | 0.981 | 0.500 | 0.487 | 0.146 |
| normal | 15 | 15 | 1.000 | 0.000 | 0.000 | 0.000 |
| collision | 92 | 92 | 0.962 | 1.000 | 0.973 | 0.292 |

## Interpretation

The animal-adaptation diagnostic strongly supports the domain-shift hypothesis. The previous phantom-only models had near-zero animal collision behavior. After adding `video_1_animal` to training, held-out `video_2_animal` collision detection becomes strong:

- collision mAP50: 0.973
- collision mAP50-95: 0.292
- collision recall: 1.000

The aggregate `all` mAP50 remains around 0.49 because class 0 (`normal`) is not recovered on this held-out animal fold. This implies the current adaptation model behaves like a collision specialist on animal video rather than a balanced two-class detector.

This result should not be reported as clean public validation performance, because one animal video from the public validation split was moved into training. It is a diagnostic showing that animal-domain supervision substantially changes Task 2 performance.

## Next Steps

1. Run the reverse animal fold: train on `video_2_animal`, validate on `video_1_animal`.
2. Evaluate whether a collision-specialist formulation is closer to the challenge metric than two-class normal/collision detection.
3. Visualize `best.pt` predictions on `video_2_animal` to verify whether high collision AP corresponds to clinically meaningful localization.
4. Keep the original clean split as the no-leakage reference; treat Stage2H as an adaptation diagnostic.

