# Task2 Result: Stage2U Source-Consistent Refined-Source Fine-Tune

Date: 2026-06-13
Status: completed

## Question

Can Stage2U recover the AP loss caused by appending Stage2P refined candidates if it is fine-tuned on the same refined candidate source during training?

Previous evidence:

- Original YOLO+geometry Stage2U full-valid result:
  - valid_combined `blend_rank_decay_roi`: mAP50 `0.1875`, mAP50-95 `0.0414`
- Direct Stage2P augmented pool evaluation with the original Stage2U checkpoint:
  - valid_combined `blend_rank_decay_roi`: mAP50 `0.1493`, mAP50-95 `0.0308`
- The augmented pool improves localization recall:
  - valid_combined R@0.75: `0.2213 -> 0.2513`

The hypothesis was that Stage2P refined boxes contain useful high-IoU candidates, but the Stage2U scorer was not calibrated for the `stage2p_refined` source.

## Training

Run:

- `outputs/task2/stage2u_quality_ranker/stage2u_refined_source_finetune_train100k_valid512_e3_rerun`

Checkpoint initialization:

- `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt`

Training candidate pool:

- `outputs/task2/stage2q_refined_candidate_pool/train_yolo_geometry_stage2p_refined/valid_combined_candidates.csv`

Training setup:

- epochs: `3`
- batch size: `256`
- learning rate: `1e-4`
- train limit: `100000`, stratified down to `67394`
- validation limit during training: `512`
- candidate sources in train:
  - `stage2p_refined`: `33056`
  - `task1_geometry_rect`: `23785`
  - `yolo_stage2l`: `10553`

GPU verification:

- external GPU execution was required because the sandbox does not expose `/dev/nvidia*`;
- host GPU was visible as NVIDIA RTX 6000 Ada Generation;
- training ran with about `16.8 GB` VRAM and high GPU utilization.

Best training-validation checkpoint:

- best epoch: `2`
- primary metric: `valid_combined/blend_rank_decay_roi/mAP50_95 = 0.0790` on valid512

## Full-Validation Evaluation

Run:

- `outputs/task2/stage2u_quality_ranker/stage2u_refined_source_finetune_train100k_fullvalid_augmented_eval`

Checkpoint:

- `outputs/task2/stage2u_quality_ranker/stage2u_refined_source_finetune_train100k_valid512_e3_rerun/checkpoints/best.pt`

Candidate pool:

- `outputs/task2/stage2q_refined_candidate_pool/yolo_geometry_stage2p_refined/*_candidates.csv`

Full-valid results:

| Split | Best mode | mAP50 | mAP50-95 | R@0.50 | R@0.75 |
| --- | --- | ---: | ---: | ---: | ---: |
| valid_combined | `rank_decay_roi` | `0.1501` | `0.0318` | `0.5618` | `0.2513` |
| valid_phantom | `rank_decay_roi` | `0.0599` | `0.0242` | `0.5109` | `0.2840` |
| valid_animal | `pred_iou_rank_decay_roi` | `0.5006` | `0.0556` | `0.9533` | `0.0000` |

## Source Diagnostics

A fast AP-equivalent diagnostic using stored `gt_iou` showed:

| Policy | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| all augmented sources | `0.1493` | `0.0308` |
| original sources only | `0.1875` | `0.0414` |
| YOLO source only | `0.1883` | `0.0416` |
| refined source only | `0.1269` | `0.0341` |
| all sources, refined x0.2 | `0.1875` | `0.0414` |
| all sources, refined x0.05 | `0.1876` | `0.0414` |

Interpretation:

- `stage2p_refined` is not useless, but it is not strong enough as a full-volume source.
- It adds a small number of high-IoU boxes, but doubles the candidate count and hurts ranking precision.
- Simple refined-source downweighting can recover the original score, but does not produce a meaningful gain.

## Decision

Do not adopt Stage2P refined candidates as a main inference source for the current Task2 pipeline.

The source-consistent fine-tune improved the augmented-pool full-valid mAP50-95 from `0.0308` to `0.0318`, but it remains far below the original-pool Stage2U result of `0.0414`.

This closes the current refined-box line as a primary optimization path.

## Next Direction

Keep the current Stage2U original-pool checkpoint as the scoring baseline. The next improvement should focus on candidate source selection and candidate generation:

1. Treat `yolo_stage2l`-only or source-filtered inference as a lightweight candidate selection baseline, because it slightly outperformed the original mixed source in the fast diagnostic.
2. Build a proper source-filtered evaluation/submission path rather than appending all candidate sources blindly.
3. Move the main research effort from refined-box calibration to candidate generation redesign, especially for phantom and high-IoU localization.
4. Candidate generation directions worth prioritizing:
   - Task1 mask / centerline structured proposals;
   - keypoint or heatmap-style interaction localizer;
   - temporal consistency proposals;
   - phantom-specific candidate generator.
