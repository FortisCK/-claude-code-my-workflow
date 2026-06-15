# Task2 Stage2W Sequence-Tip Localizer Result

Date: 2026-06-13

## Objective

Stage2W tested whether a stronger sequence-aware ConvNeXt/FPN localizer could improve the coarse candidate pool for Task2, especially class 1 / collision high-IoU candidates.

The gate from the Stage2W plan was:

- class 1 recall@0.75 should improve materially over the current augmented candidate-pool ceiling of about `0.1310`;
- candidate count should remain controlled enough for downstream Stage2U/Stage2V ranking.

## Training Run

Run:

- `outputs/task2/sequence_tip_localizer/stage2w_sequence_tip384_convnext_tiny_train20k_e12_screen`

Log:

- `quality_reports/logs/task2_stage2w_sequence_tip384_convnext_tiny_train20k_e12_screen.log`

Configuration:

- model: `smp_fpn`
- encoder: `tu-convnext_tiny`
- encoder weights: ImageNet
- input: 5-frame grayscale sequence, `384`
- train samples: `20000`
- epochs: `12`
- batch size: `16`
- AMP: off / fp32
- primary metric: `valid_combined/center_recall_20px`

Best checkpoint:

- `outputs/task2/sequence_tip_localizer/stage2w_sequence_tip384_convnext_tiny_train20k_e12_screen/checkpoints/best.pt`
- best epoch: `2`
- best `valid_combined/center_recall_20px`: `0.7883995704`

Training finished normally. Later epochs overfit/oscillated and did not beat epoch 2.

## Direct Localizer Metrics At Best Epoch

Selected metrics from `best_metrics.json`:

| split/group | center@20 | recall@0.50 | recall@0.75 | mean IoU |
| --- | ---: | ---: | ---: | ---: |
| valid_combined all | 0.7884 | 0.1633 | 0.0150 | 0.2913 |
| valid_combined class0 | 0.8689 | 0.3443 | 0.0328 | 0.3844 |
| valid_combined class1 | 0.7202 | 0.0099 | 0.0000 | 0.2125 |
| valid_phantom class1 | 0.6578 | 0.0121 | 0.0000 | 0.1771 |
| valid_animal class1 | 1.0000 | 0.0000 | 0.0000 | 0.3710 |

Interpretation: the localizer learned centers, but boxes were too weak for high-IoU AP thresholds.

## Proposal Export

Command pattern:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_sequence_tip_proposals.py \
  --checkpoint outputs/task2/sequence_tip_localizer/stage2w_sequence_tip384_convnext_tiny_train20k_e12_screen/checkpoints/best.pt \
  --name stage2w_tip384_convnext_tiny_train20k_e12_best_top20_templates \
  --batch-size 16 \
  --workers 8 \
  --top-peaks 20 \
  --max-proposals 50 \
  --top-k 1,5,10,20,50 \
  --device auto
```

Output:

- `outputs/task2/sequence_tip_proposals/stage2w_tip384_convnext_tiny_train20k_e12_best_top20_templates/summary_metrics.json`

Standalone top50 proposal metrics:

| split/group | recall@0.50 | recall@0.75 | mean best IoU | center@20 |
| --- | ---: | ---: | ---: | ---: |
| valid_combined all | 0.2632 | 0.0569 | 0.3807 | 0.8153 |
| valid_combined class0 | 0.5597 | 0.1194 | 0.4973 | 0.9110 |
| valid_combined class1 | 0.0119 | 0.0040 | 0.2819 | 0.7341 |
| valid_phantom all | 0.2961 | 0.0643 | 0.3705 | 0.7913 |
| valid_phantom class1 | 0.0146 | 0.0049 | 0.2371 | 0.6748 |
| valid_animal all | 0.0093 | 0.0000 | 0.4587 | 1.0000 |
| valid_animal class0 | 0.0667 | 0.0000 | 0.3141 | 1.0000 |
| valid_animal class1 | 0.0000 | 0.0000 | 0.4823 | 1.0000 |

Compared with the older Stage2M sequence-tip pilot, Stage2W improved standalone sequence-tip proposals overall:

| model | valid_combined top50 R@0.50 | valid_combined top50 R@0.75 | valid_combined mean best IoU |
| --- | ---: | ---: | ---: |
| old sequence-tip pilot | 0.2352 | 0.0397 | 0.3490 |
| Stage2W 20k sequence-tip | 0.2632 | 0.0569 | 0.3807 |

However, the improvement is not on class 1 high-IoU localization:

| model | valid_combined class1 top50 R@0.50 | valid_combined class1 top50 R@0.75 |
| --- | ---: | ---: |
| old sequence-tip pilot | 0.0198 | 0.0000 |
| Stage2W 20k sequence-tip | 0.0119 | 0.0040 |

## Union Oracle

I evaluated Stage2W as a supplemental proposal source.

Three-source union, replacing old sequence-tip with Stage2W:

- `outputs/task2/proposal_union_oracle/yolo_geometry_stage2w_tip20k_top50/summary_metrics.json`

Four-source union, adding Stage2W to the prior YOLO + geometry + old sequence-tip pool:

- `outputs/task2/proposal_union_oracle/yolo_geometry_oldseq_stage2w_top50/summary_metrics.json`

Key comparison:

| proposal set | valid_combined R@0.50 | valid_combined R@0.75 | valid_combined mean IoU | valid_combined class1 R@0.75 |
| --- | ---: | ---: | ---: | ---: |
| YOLO + geometry + old sequence-tip | 0.5832 | 0.2567 | 0.5459 | 0.1230 |
| YOLO + geometry + Stage2W | 0.5768 | 0.2524 | 0.5400 | 0.1250 |
| YOLO + geometry + old sequence-tip + Stage2W | 0.5832 | 0.2739 | 0.5502 | 0.1250 |

Stage2W adds some useful high-IoU class0 candidates:

| proposal set | valid_combined class0 R@0.75 | valid_phantom class0 R@0.75 | valid_animal class0 R@0.50 |
| --- | ---: | ---: | ---: |
| YOLO + geometry + old sequence-tip | 0.4145 | 0.4296 | 0.6667 |
| YOLO + geometry + old sequence-tip + Stage2W | 0.4496 | 0.4660 | 0.7333 |

But class1 remains essentially unchanged:

| proposal set | valid_combined class1 R@0.50 | valid_combined class1 R@0.75 |
| --- | ---: | ---: |
| YOLO + geometry + old sequence-tip | 0.4187 | 0.1230 |
| YOLO + geometry + old sequence-tip + Stage2W | 0.4107 | 0.1250 |

## Decision

Stage2W did **not** pass the original collision-centered gate. It improves sequence-tip center/box proposals versus the older pilot and can slightly improve class0 oracle recall when included as a fourth proposal source, but it does not materially improve class1 / collision high-IoU recall.

Do not immediately retrain Stage2U/Stage2V with Stage2W candidates as a full additional source. The candidate count would increase, and the class1 bottleneck would remain.

Keep Stage2W artifacts as a possible class0/normal supplemental source, but the next major Task2 effort should target class1/collision box generation directly.

## Current Best Still Active

The active metric best remains Stage2V class-aware export:

- `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export`
- `valid_combined` mAP50: `0.2009680309`
- `valid_combined` mAP50-95: `0.0492404464`

## Next Direction

The next candidate generator should not only find the tool tip center. It needs a class1-specific box target or a geometric construction around the actual collision/contact region. Good next experiments:

1. class1-only localizer trained/evaluated only on collision frames;
2. localizer with fixed class1 box templates calibrated from GT class1 box statistics;
3. using the sequence-tip center as a seed but sweeping class1-specific box scales/aspect ratios more aggressively;
4. adding a learned box-size calibration model conditioned on local crop appearance and class.

## Dense Template Follow-Up

After the main Stage2W evaluation, I ran a denser template sweep around the same Stage2W centers:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_sequence_tip_proposals.py \
  --checkpoint outputs/task2/sequence_tip_localizer/stage2w_sequence_tip384_convnext_tiny_train20k_e12_screen/checkpoints/best.pt \
  --name stage2w_tip384_dense_templates_top30 \
  --batch-size 16 \
  --workers 8 \
  --top-peaks 30 \
  --max-proposals 100 \
  --top-k 1,5,10,20,50,100 \
  --templates 16x16,18x18,20x20,22x22,24x24,26x26,28x28,30x30,34x34,38x38,44x44,52x52,24x58,28x58,30x64,30x70,34x70,38x70,24x80,30x80,40x80,58x24,70x30,80x40 \
  --device auto
```

Output:

- `outputs/task2/sequence_tip_proposals/stage2w_tip384_dense_templates_top30/summary_metrics.json`

Standalone dense-template metrics:

| split/group | top50 recall@0.50 | top50 recall@0.75 | top50 mean best IoU |
| --- | ---: | ---: | ---: |
| valid_combined all | 0.3759 | 0.0580 | 0.3873 |
| valid_combined class0 | 0.5972 | 0.1218 | 0.5040 |
| valid_combined class1 | 0.1885 | 0.0040 | 0.2884 |
| valid_phantom all | 0.3167 | 0.0655 | 0.3765 |
| valid_phantom class1 | 0.0170 | 0.0049 | 0.2444 |
| valid_animal all | 0.8318 | 0.0000 | 0.4705 |
| valid_animal class1 | 0.9565 | 0.0000 | 0.4960 |

This confirms that template geometry can recover many IoU@0.50 cases, especially animal class1, but it still does not solve high-IoU localization.

Dense-template four-source union:

- `outputs/task2/proposal_union_oracle/yolo_geometry_oldseq_stage2w_dense_top50/summary_metrics.json`

| proposal set | valid_combined R@0.50 | valid_combined R@0.75 | valid_combined mean IoU | valid_combined class1 R@0.75 |
| --- | ---: | ---: | ---: | ---: |
| YOLO + geometry + old sequence-tip + Stage2W default | 0.5832 | 0.2739 | 0.5502 | 0.1250 |
| YOLO + geometry + old sequence-tip + Stage2W dense | 0.5854 | 0.2750 | 0.5513 | 0.1250 |

Dense templates slightly improve the overall union oracle, but the gain is still class0/shape coverage rather than class1 high-IoU collision localization. The decision remains unchanged: do not retrain the full Stage2U/Stage2V ranker on this source yet; the next large change should improve center precision for class1.
