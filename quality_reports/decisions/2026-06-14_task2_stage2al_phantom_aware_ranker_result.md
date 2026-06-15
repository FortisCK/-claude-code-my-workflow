# Task2 Stage2AL Phantom-Aware Ranker Result

Date: 2026-06-14

## Question

Can a yolo-only Stage2U fine-tune improve the Stage2AK ranking bottleneck,
especially `valid_phantom` class1 AP50?

## Code Changes

`scripts/task2/train_stage2u_quality_ranker.py` now supports:

- `--train-source-keep`: restrict training candidates to specified sources;
- `--primary-metric-key`: select checkpoints by an exact flattened metric key,
  e.g. `valid_phantom/prob_iou75_rank_decay_roi/class1_ap50`.

These changes are reusable for later ranker experiments.

## Sanity Check

A small GPU sanity run with `train-limit=512` confirmed:

- CUDA path works;
- `--train-source-keep yolo_stage2l` works;
- exact metric-key checkpoint selection works when full validation contains the target class;
- `--backend auto` is required to warm-start the old checkpoint correctly.

Important correction:

- `--backend torchvision` loaded 0 warm-start tensors because its checkpoint key names differ.
- `--backend auto` loaded 182 warm-start tensors from the Stage2U checkpoint.

## Main Run

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/train_stage2u_quality_ranker.py \
  --train-csv outputs/task2/stage2o_candidate_pool/stage2o_train_subset_yolo_geometry_top50/valid_combined_candidates.csv \
  --valid-csv valid_combined=outputs/task2/stage2q_refined_candidate_pool/yolo_geometry_stage2p_refined/valid_combined_candidates.csv \
  --valid-csv valid_phantom=outputs/task2/stage2q_refined_candidate_pool/yolo_geometry_stage2p_refined/valid_phantom_candidates.csv \
  --valid-csv valid_animal=outputs/task2/stage2q_refined_candidate_pool/yolo_geometry_stage2p_refined/valid_animal_candidates.csv \
  --output-dir outputs/task2/stage2u_quality_ranker \
  --name stage2al_yolo_only_phantom_class1_finetune_e4 \
  --backend auto \
  --model convnext_tiny \
  --no-pretrained \
  --train-source-keep yolo_stage2l \
  --valid-source-keep yolo_stage2l \
  --init-checkpoint outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt \
  --primary-metric-key valid_phantom/prob_iou75_rank_decay_roi/class1_ap50 \
  --epochs 4 \
  --batch-size 256 \
  --workers 8 \
  --lr 0.00003 \
  --class-weight \
  --device cuda
```

Output:

`outputs/task2/stage2u_quality_ranker/stage2al_yolo_only_phantom_class1_finetune_e4`

## Result

Best checkpoint:

- best epoch: 2;
- primary key: `valid_phantom/prob_iou75_rank_decay_roi/class1_ap50`;
- best value: 0.008476280150359738.

Selected metrics at best epoch:

| Metric | Value |
| --- | ---: |
| `valid_phantom/prob_iou75_rank_decay_roi/class1_ap50` | 0.008476280150359738 |
| `valid_phantom/prob_iou75_rank_decay_roi/mAP50` | 0.05952999505867054 |
| `valid_phantom/prob_iou75_rank_decay_roi/mAP50-95` | 0.024682251301697482 |
| `valid_combined/prob_iou75_rank_decay_roi/mAP50` | 0.15423433650893215 |
| `valid_combined/prob_iou75_rank_decay_roi/mAP50-95` | 0.0333372607179991 |

Best phantom class1 AP50 across all score modes in the run:

| Epoch | Best score mode | `valid_phantom` class1 AP50 |
| ---: | --- | ---: |
| 1 | `rank_decay_roi` | 0.042455 |
| 2 | `rank_decay_roi` | 0.043082 |
| 3 | `rank_decay_roi` | 0.043619 |
| 4 | `rank_decay_roi` | 0.041043 |

Stage2AE baseline for `valid_phantom` class1 AP50 is 0.045713631775941704.

## Decision

Do not promote Stage2AL.

The experiment shows that full-model yolo-only fine-tuning does not recover the
Stage2AK oracle ranking gap. It slightly underperforms the existing Stage2AE
rank-decay behavior on phantom class1, and quality-head-adjusted scores are
worse.

The current default remains Stage2AE, with Stage2AI retained as the high
`mAP50-95` alternative.

## Next Direction

Avoid further full-backbone fine-tuning for this specific yolo-only ranker path.

If continuing the ranker route, use a more conservative method:

- freeze the image backbone and train only a small calibration/head layer; or
- train a lightweight tabular/listwise score combiner using existing prediction
  columns, with strict validation against Stage2AE.

The next experiment should be designed so that the old Stage2AE ranking is the
starting point/fallback rather than overwritten by a full CNN fine-tune.

