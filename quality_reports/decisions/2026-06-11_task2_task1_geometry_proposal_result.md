# Task 2 Decision: Task1 Geometry Proposals

Date: 2026-06-11

## Question

Can Task1 catheter/guidewire segmentation predictions provide better Task2
collision-detection candidates than the current YOLO coarse detector, especially
for the animal class0 failure case?

## Implementation

Added `scripts/task2/evaluate_task1_geometry_proposals.py`.

The script:

- loads the current Task1 segmentation checkpoint
  `outputs/task1/smp_fpn_convnext_small_640_stage5/best_checkpoint.pt`;
- predicts masks on Task2 validation frames;
- restores masks to original image geometry;
- generates proposal boxes from class proximity, foreground components,
  skeleton endpoints, branch-like pixels, foreground grids, and fixed box
  templates;
- evaluates proposals with the same proposal-recall metric used for YOLO.

Added `tests/test_task2_geometry_proposals.py` for synthetic geometry behavior
and preprocessing/postprocessing restoration checks.

## Verification

Commands completed:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python -m py_compile scripts/task2/evaluate_task1_geometry_proposals.py tests/test_task2_geometry_proposals.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest -q tests/test_task2_geometry_proposals.py
```

Pytest result: `3 passed`.

Full evaluation outputs:

- square geometry:
  `outputs/task2/geometry_proposals/task1_stage5_geometry_d3_stage2l_panels/summary_metrics.json`
- rectangular geometry:
  `outputs/task2/geometry_proposals/task1_stage5_geometry_d3_rect_stage2l_panels/summary_metrics.json`
- YOLO + geometry union oracle:
  `outputs/task2/geometry_proposals/union_oracle_yolo_stage2l_task1_geometry_stage2l_panels/summary_metrics.json`

## Main Results

Current Stage2L YOLO proposal baseline:

| Split | Top50 R@0.50 | Class0 R@0.50 | Class1 R@0.50 |
| --- | ---: | ---: | ---: |
| combined | 0.5456 | 0.7096 | 0.4067 |
| phantom | 0.5049 | 0.7354 | 0.2743 |
| animal | 0.8598 | 0.0000 | 1.0000 |

Task1 geometry, square templates only:

| Split | Top50 R@0.50 | Top50 R@0.75 | Mean Best IoU |
| --- | ---: | ---: | ---: |
| combined | 0.1676 | 0.0236 | 0.3099 |
| phantom | 0.1893 | 0.0267 | 0.3025 |
| animal | 0.0000 | 0.0000 | 0.3669 |

Task1 geometry, rectangular templates:

| Split | Top50 R@0.50 | Class0 R@0.50 | Class1 R@0.50 | Mean Best IoU |
| --- | ---: | ---: | ---: | ---: |
| combined | 0.0977 | 0.1920 | 0.0179 | 0.2291 |
| phantom | 0.0874 | 0.1748 | 0.0000 | 0.1963 |
| animal | 0.1776 | 0.6667 | 0.0978 | 0.4822 |

Oracle union over YOLO + Task1 geometry top50 candidates:

| Proposal Set | Combined R@0.50 | Phantom R@0.50 | Animal R@0.50 | Animal Class0 R@0.50 |
| --- | ---: | ---: | ---: | ---: |
| YOLO Stage2L | 0.5456 | 0.5049 | 0.8598 | 0.0000 |
| YOLO + geometry rect | 0.5585 | 0.5073 | 0.9533 | 0.6667 |
| YOLO + geometry rect + square | 0.5661 | 0.5158 | 0.9533 | 0.6667 |

The union numbers are an oracle over candidate sources, not a deployable ranked
top50 detector. They show whether a downstream ranker/verifier could benefit
from seeing both sources.

## Interpretation

Task1 geometry does not solve Task2 by itself. It produces many badly ranked
boxes, and the raw top50 proposal recall is below YOLO on the combined and
phantom panels.

The experiment is still useful because it identifies a complementary signal:
rectangular skeleton-endpoint geometry recovers animal class0 cases where YOLO
has zero recall. In other words, the geometry prior is not a replacement coarse
detector, but it can supply missed candidates to a two-stage system.

The 3px proximity idea was not the main source of gains. The useful candidates
mostly came from skeleton/endpoint-derived regions plus more realistic
rectangular box templates. Simple overlap/proximity boxes are too noisy and too
poorly ranked.

## Decision

Do not continue optimizing Task1 geometry as a standalone detector.

Continue with a multi-source candidate strategy:

1. Keep YOLO Stage2L as the main proposal source.
2. Add Task1-derived geometry candidates, especially endpoint/rectangular
   candidates, as supplemental high-recall proposals.
3. Train or adapt the ROI verifier/ranker to choose among the unioned
   candidates.
4. Evaluate final output as ranked detections, not as source-wise oracle recall.

This is the most defensible next step because it directly targets the current
failure mode without discarding the strongest existing proposal model.
