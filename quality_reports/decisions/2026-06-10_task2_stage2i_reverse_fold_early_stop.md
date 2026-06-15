# Task2 Stage2I Reverse Animal Fold Early Stop

Date: 2026-06-10

## Context

Stage2H showed that adding one animal procedure to the YOLO training set can rescue held-out animal collision detection on the opposite animal fold:

- Train: `train_clean + video_1_animal`
- Val: `video_2_animal`
- Best validation: mean mAP50 `0.4867`, collision mAP50 `0.973`, normal mAP50 `0.000`

However, the same Stage2H best checkpoint performed poorly on the original phantom validation set:

- Valid phantom mean mAP50 `0.0722`
- Normal mAP50 `0.143`
- Collision mAP50 `0.00149`

This suggested animal specialization rather than a generally robust detector.

Stage2I tested the reverse animal fold:

- Train: `train_clean + video_2_animal`
- Val: `video_1_animal`
- Config: `configs/task2/collision_detection_adapt_animal_train_v2_val_v1.local.yaml`
- Log: `quality_reports/logs/task2_yolo11s_1024_adapt_animal_train_v2_val_v1_e40.log`
- Output: `outputs/task2/yolo_adapt_animal/yolo11s_1024_train_v2_val_v1_e40`

## Early Validation Curve

The run was stopped manually after epoch 5 because the validation curve did not show the early lift observed in Stage2H.

| Epoch | Precision | Recall | mAP50 | mAP50-95 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | no valid detections |
| 2 | 0.00496 | 0.48333 | 0.00733 | 0.00095 | weak recall, almost no AP |
| 3 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | collapsed again |
| 4 | 0.00280 | 0.46944 | 0.03432 | 0.01325 | small transient lift |
| 5 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | collapsed again |

For comparison, the forward fold already reached mean mAP50 `0.35149` at epoch 4 and `0.49698` at epoch 5. The reverse fold therefore did not justify continuing to the planned 40 epochs.

## Decision

Stop this reverse-fold training run early.

The result does not refute the main Stage2H finding that animal-domain supervision matters, but it shows that the one-animal-procedure adaptation is not stable across animal procedures. In particular:

1. `video_1_animal` is not recovered well by training on `video_2_animal` alone.
2. Animal procedures are not interchangeable enough for a one-procedure adaptation result to be treated as robust.
3. Naive addition of a small animal subset can specialize or destabilize the detector instead of producing a general domain-robust model.

## Implication For Next Stage

The next useful direction is not another plain YOLO run with a tiny animal add-on. The next Task2 stage should use a training design that explicitly handles:

- domain balance: phantom and animal should be sampled deliberately rather than letting phantom dominate;
- class balance: collision examples are rare and should be over-sampled or weighted;
- validation discipline: keep at least one animal procedure and a phantom validation set untouched for reporting;
- detector objective: consider class-agnostic localization plus a stronger local collision/normal classifier, or YOLO with custom sampling/copy-paste rather than default full-dataset sampling.

The current internal conclusion is:

> Phantom-only training fails on animal collision; one-procedure animal adaptation can rescue a matched held-out animal fold, but the reverse fold is unstable. Final Task2 modeling needs domain-balanced and class-balanced training, not naive fine-tuning or naive data mixing.
