# Task2 Stage2U Result: Image-Aware ROI Quality Reranker

Date: 2026-06-12
Status: completed

## Question

Can an image-aware candidate quality ranker improve over the previous Task2 scoring stack?

Previous baselines on the YOLO+geometry validation candidate pool:

| Method | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | ---: |
| Stage2O ConvNeXt ROI ranker, `rank_decay_roi` | 0.1887 | 0.0388 |
| Stage2T tabular selector, `prob_iou75 * rank_decay_roi` | 0.1551 | 0.0403 |

## Implementation

Added:

- `scripts/task2/train_stage2u_quality_ranker.py`
- `tests/test_task2_stage2u_quality_ranker.py`
- `quality_reports/plans/2026-06-12_task2-stage2u-image-aware-quality-reranker.md`

Stage2U is a ConvNeXt ROI model with 6 outputs:

- 3 class logits: background / normal / collision;
- 1 IoU-quality regression output;
- 1 IoU >= 0.50 logit;
- 1 IoU >= 0.75 logit.

The final detection score is evaluated both as plain class score and as image-predicted quality multiplied by the existing Stage2O base scores.

Important implementation detail:

- from-scratch Stage2U was weak;
- warm-starting from the existing Stage2O checkpoint was necessary;
- warm-start loaded 180 tensors and skipped only `head.fc.weight` / `head.fc.bias`, because Stage2O has 3 outputs and Stage2U has 6.

## Verification

Static and unit checks:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile scripts/task2/train_stage2u_quality_ranker.py
PYTHONDONTWRITEBYTECODE=1 /home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2u_quality_ranker.py tests/test_task2_stage2o_ranker.py tests/test_task2_stage2p_box_refiner.py tests/test_task2_stage2t_tabular_selector.py
```

Result:

- 14 tests passed.

GPU environment:

- machine GPU is visible outside the Codex sandbox: NVIDIA RTX 6000 Ada, 49 GB VRAM;
- `cathaction-task1` torch is CPU-only;
- `cardiac-diffusion` has CUDA-enabled torch 2.11.0+cu128 and was used for GPU training/evaluation.

## Runs

Smoke/pilot runs:

- CPU smoke: `outputs/task2/stage2u_quality_ranker/smoke_stage2u_cpu_limit4_v2`
- from-scratch GPU pilot: `outputs/task2/stage2u_quality_ranker/pilot_stage2u_convnext_tiny_quality_train4096_valid128_e2`
- warm-start GPU pilot: `outputs/task2/stage2u_quality_ranker/pilot_stage2u_warm_stage2o_train20k_valid256_e3`

Full train:

- train run: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6`
- best epoch: 3
- selected primary during training: `valid_combined/pred_iou_rank_decay_roi/mAP50_95 = 0.0801` on valid512

Full-valid fair evaluation on the same YOLO+geometry candidate pool as Stage2O/Stage2T:

- `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_yologeom_eval`

## Results

Best valid_combined modes on YOLO+geometry full validation:

| Method / mode | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| Stage2O `rank_decay_roi` | 0.1887 | 0.0388 |
| Stage2T `prob_iou75 * rank_decay_roi`, k=50 | 0.1551 | 0.0403 |
| Stage2U `blend_rank_decay_roi` | 0.1875 | 0.0414 |
| Stage2U `pred_iou_rank_decay_roi` | 0.1844 | 0.0398 |
| Stage2U `prob_iou75_rank_decay_roi` | 0.1722 | 0.0382 |

Split-level Stage2U best modes:

| Split | Best mode | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: |
| valid_combined | `blend_rank_decay_roi` | 0.1875 | 0.0414 |
| valid_phantom | `blend_rank_decay_roi` | 0.0891 | 0.0342 |
| valid_animal | `roi` / similar modes | 0.5009 | 0.0502 |

Candidate-pool localization on the fair YOLO+geometry evaluation remains unchanged:

- valid_combined R@0.50: 0.5585
- valid_combined R@0.75: 0.2213

## Decision

Accept Stage2U `blend_rank_decay_roi` as the current best single-pipeline mAP50-95 result on the YOLO+geometry candidate pool.

This is a real but small improvement:

- vs Stage2O mAP50-95: 0.0388 -> 0.0414, +0.0026 absolute;
- vs Stage2T mAP50-95: 0.0403 -> 0.0414, +0.0011 absolute;
- vs Stage2O mAP50: 0.1887 -> 0.1875, only -0.0012 absolute.

The result validates the image-aware quality-reranking idea, but it does not solve Task2 at a large scale. The remaining bottleneck is still candidate localization and phantom-domain performance.

## Important Caveat

An intermediate full-valid Stage2U evaluation accidentally used `stage2o_aprime_default_sources_top50`, which includes `sequence_tip` proposals:

- `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_eval`

That run is not directly comparable with the Stage2O/Stage2T champion because the candidate pool differs. The fair comparison is:

- `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_fullvalid_yologeom_eval`

## Next Direction

Stage2U should be kept as the current scoring baseline, but further gains likely require improving candidate generation/localization rather than another tabular reranker:

1. integrate Stage2P refined boxes into Stage2U scoring carefully;
2. build a better candidate pool for phantom and normal cases;
3. evaluate a submission-time ensemble of Stage2O and Stage2U modes if the challenge rules permit model ensembling;
4. preserve candidate-pool identity in every report, because different proposal pools change the interpretation of mAP.
