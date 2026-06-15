# Task2 Result: Stage2P Refined Candidates with Stage2U Scoring

Date: 2026-06-12
Status: completed

## Question

Can Stage2P refined boxes improve the current Stage2U scoring baseline?

Current fair Stage2U baseline on the YOLO+geometry candidate pool:

| Split | Best mode | mAP50 | mAP50-95 | R@0.75 | candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| valid_combined | `blend_rank_decay_roi` | 0.1875 | 0.0414 | 0.2213 | 66,985 |
| valid_phantom | `blend_rank_decay_roi` | 0.0891 | 0.0342 | 0.2500 | 60,869 |
| valid_animal | `roi` | 0.5009 | 0.0502 | 0.0000 | 6,116 |

## Candidate Pool

Used the existing Stage2Q augmented candidate pool:

- `outputs/task2/stage2q_refined_candidate_pool/yolo_geometry_stage2p_refined/valid_combined_candidates.csv`
- `outputs/task2/stage2q_refined_candidate_pool/yolo_geometry_stage2p_refined/valid_phantom_candidates.csv`
- `outputs/task2/stage2q_refined_candidate_pool/yolo_geometry_stage2p_refined/valid_animal_candidates.csv`

This pool appends Stage2P refined candidates to the original YOLO+geometry candidates.

Stage2P augmented localization upper bound:

- valid_combined R@0.75: 0.2213 -> 0.2513
- valid_phantom R@0.75: 0.2500 -> 0.2840
- valid_animal R@0.75: unchanged at 0.0000

## Evaluation

Checkpoint:

- `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt`

Full augmented pool evaluation:

- `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_stage2p_augmented_eval`
- used `--max-candidates-per-sample 0`

Budget-100 diagnostic evaluation:

- `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_stage2p_augmented_budget100_eval`
- used `--max-candidates-per-sample 100`

## Results

| Pool | Split | Best mode | mAP50 | mAP50-95 | R@0.75 | candidates |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| original YOLO+geometry | valid_combined | `blend_rank_decay_roi` | 0.1875 | 0.0414 | 0.2213 | 66,985 |
| original YOLO+geometry | valid_phantom | `blend_rank_decay_roi` | 0.0891 | 0.0342 | 0.2500 | 60,869 |
| original YOLO+geometry | valid_animal | `roi` | 0.5009 | 0.0502 | 0.0000 | 6,116 |
| Stage2P augmented full | valid_combined | `blend_rank_decay_roi` | 0.1493 | 0.0308 | 0.2513 | 133,970 |
| Stage2P augmented full | valid_phantom | `prob_iou50_rank_decay_roi` | 0.0547 | 0.0223 | 0.2840 | 121,738 |
| Stage2P augmented full | valid_animal | `blend_rank_decay_roi` | 0.4962 | 0.0537 | 0.0000 | 12,232 |
| Stage2P augmented budget100 | valid_combined | `blend_rank_decay_roi` | 0.1496 | 0.0309 | 0.2427 | 93,100 |
| Stage2P augmented budget100 | valid_phantom | `prob_iou50_rank_decay_roi` | 0.0551 | 0.0225 | 0.2743 | 82,400 |
| Stage2P augmented budget100 | valid_animal | `prob_iou75_rank_decay_roi` | 0.4967 | 0.0537 | 0.0000 | 10,700 |

## Decision

Do not adopt Stage2P augmented candidates in the current Stage2U submission path.

The refined boxes improve localization upper bound but hurt AP:

- valid_combined mAP50-95 drops from 0.0414 to about 0.0308-0.0309;
- valid_phantom mAP50-95 drops from 0.0342 to about 0.0223-0.0225;
- valid_animal mAP50-95 improves from 0.0502 to 0.0537, but animal R@0.75 remains 0.0000 and the combined/phantom losses dominate.

The budget100 diagnostic did not recover the loss, so the failure is not just caused by doubling the number of predictions. The current Stage2U scorer was trained on original YOLO+geometry candidates and does not reliably rank the unseen `stage2p_refined` source.

## Interpretation

Stage2P remains useful as an oracle/upper-bound diagnostic because it proves that high-IoU candidate recall can be increased. However, refined candidates must be introduced during training or with a dedicated refined-source calibration/ranker. Appending refined boxes only at validation/inference time is harmful.

## Next Direction

The next refined-candidate experiment should be source-consistent:

1. build a train-side Stage2Q augmented pool by appending Stage2P refined candidates to train candidates;
2. train or fine-tune Stage2U on original + refined candidates so it sees `stage2p_refined`;
3. evaluate again on the Stage2Q augmented validation pool;
4. compare against the current Stage2U original-pool baseline.

If that also fails, the project should move from box-refinement augmentation to candidate generation redesign for phantom/normal cases.
