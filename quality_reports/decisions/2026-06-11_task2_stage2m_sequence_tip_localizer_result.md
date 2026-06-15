# Task 2 Stage2M-B Result: Sequence-Aware Tip Localizer

Date: 2026-06-11

## Objective

Implement a two-stage-compatible first-stage localizer that reformulates Task2
coarse detection as:

> class-agnostic catheter/guidewire tip localization from a short frame sequence.

This differs from YOLO-style detection: the localizer does not predict
`normal`/`collision`; it only predicts the tip center plus box size/offset.

## Implemented Artifacts

Code:

- `scripts/task2/train_sequence_tip_localizer.py`

Tests:

- `tests/test_task2_sequence_tip_localizer.py`

Main outputs:

- `outputs/task2/sequence_tip_localizer/pilot_sequence_tip384_train4096_e8/`
- `outputs/task2/sequence_tip_localizer/pilot_sequence_tip384_bce_coord_train4096_e8/`
- `outputs/task2/sequence_tip_localizer/pilot_sequence_tip384_smp_convnext_tiny_train4096_e8/`
- `outputs/task2/sequence_tip_localizer/pilot_sequence_tip384_smp_convnext_tiny_lr1e4_clip_train4096_e8/`
- `outputs/task2/sequence_tip_localizer/pilot_sequence_tip384_smp_convnext_tiny_lr1e4_noamp_train4096_e6/`

Verification:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python -m py_compile scripts/task2/train_sequence_tip_localizer.py tests/test_task2_sequence_tip_localizer.py
PYTHONPATH=src /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest -q tests/test_task2_sequence_tip_localizer.py
```

Pytest result: `2 passed`.

## Design

The implemented localizer supports:

- 5-frame grayscale clip input by default;
- letterbox resizing to preserve aspect ratio;
- class-agnostic Gaussian tip heatmap target;
- width/height and center-offset regression;
- MONAI UNet from scratch;
- SMP FPN with pretrained timm encoders, including `tu-convnext_tiny`;
- optional weighted BCE or CenterNet-style heatmap loss;
- optional soft-argmax center loss;
- validation outputs with center-distance recall and IoU recall.

The most useful configuration so far:

```text
model_type=smp_fpn
encoder=tu-convnext_tiny
encoder_weights=imagenet
input_size=384
frame_radius=2
train_limit=4096
lr=1e-4
amp=false
grad_clip_norm=1.0
heatmap_loss=bce
coord_loss_weight=5
```

## Results

Best key pilot results:

| Run | Best Epoch | Combined Center@20 | Combined IoU@0.50 | Mean IoU | Animal C0 Center@20 | Animal C1 Center@20 | Phantom C0 Center@20 | Phantom C1 Center@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MONAI UNet, BCE | 7 | 0.1579 | 0.0086 | 0.0476 | 0.0000 | 1.0000 | 0.1335 | 0.0000 |
| MONAI UNet, BCE + coord | 8 | 0.0376 | 0.0011 | 0.0405 | 0.0667 | 0.0000 | 0.0825 | 0.0000 |
| SMP ConvNeXt, lr=3e-4 AMP | 5 | 0.6509 | 0.0193 | 0.1955 | 0.0667 | 1.0000 | 0.7063 | 0.5388 |
| SMP ConvNeXt, lr=1e-4 AMP+clip | 2 | 0.7261 | 0.2105 | 0.2531 | 0.0000 | 0.0000 | 0.8932 | 0.7476 |
| SMP ConvNeXt, lr=1e-4 fp32+clip | 3 | 0.7798 | 0.0816 | 0.2447 | 0.0000 | 1.0000 | 0.8641 | 0.6748 |

The fp32 ConvNeXt run was the most stable completed run. The AMP+clip run had
the best combined IoU@0.50 but later hit NaN. The fp32 run avoided NaN over the
6-epoch pilot.

## Interpretation

This experiment validates part of the redesign:

- from-scratch UNet is too weak for full-image tip localization;
- pretrained ConvNeXt/FPN provides a real localization signal;
- phantom class0/class1 center recall improves substantially compared with the
  from-scratch version;
- animal class1 can be localized well in some epochs.

It does not yet solve the main hard case:

- animal class0 remains essentially unrecovered;
- center localization is better than the failed UNet, but still not enough as a
  final proposal generator;
- IoU@0.50 is limited, so size/box refinement is still weak;
- training can be unstable with AMP and lr=3e-4 or 1e-4, so fp32 or stronger
  stabilization is preferred for now.

## Comparison To Current YOLO Diagnosis

The localizer is not yet a drop-in replacement for YOLO. YOLO Stage2L still has
strong center coverage on some panels and excellent animal class1 localization.

The important difference is qualitative: ConvNeXt/FPN tip localization is now
learning a task-aligned signal, whereas from-scratch heatmap and simple YOLO
temporal smoothing did not. The remaining bottlenecks are:

1. animal class0 domain/generalization;
2. box size/shape refinement;
3. stable training;
4. converting center candidates into ranked detection proposals for the ROI
   verifier.

## Decision

Continue the sequence-aware tip-localizer route, but only with pretrained
encoder-decoder models.

Do not scale the MONAI from-scratch UNet variant.

Next steps:

1. Add explicit top-K heatmap peak export so the localizer can act as a
   high-recall proposal source, not only a top1 predictor.
2. Train a stable fp32 ConvNeXt/FPN run on more data with lower lr or scheduler.
3. Add animal/domain-balanced sampling that oversamples animal class0.
4. Add a box-refinement or template-size calibration stage because center recall
   is much better than IoU@0.50.
5. Feed localizer proposals into the existing ROI verifier together with YOLO
   and Task1 geometry proposals.
