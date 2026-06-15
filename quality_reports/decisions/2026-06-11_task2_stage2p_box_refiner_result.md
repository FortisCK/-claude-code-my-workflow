# Task2 Stage2P Result: Local Box Refiner

Date: 2026-06-11
Status: completed

## Question

Can a local ROI model refine YOLO+geometry candidate boxes enough to improve high-IoU localization for Task 2?

## Implementation

Added:

- `scripts/task2/train_stage2p_box_refiner.py`
- `tests/test_task2_stage2p_box_refiner.py`

The refiner trains a ConvNeXt-Tiny regression head on ROI crops around candidate boxes. The regression target is the normalized delta from the source candidate box to the GT box:

- `dx, dy` normalized by source box width/height;
- `dw, dh` as log scale ratios;
- SmoothL1 loss.

The evaluation reports three views:

- `before`: original YOLO+geometry candidate pool;
- `after`: refined boxes only, replacing originals;
- `augmented`: original boxes plus refined boxes, which is the correct downstream use when refinement is uncertain.

## Runs

Smoke:

- `/tmp/stage2p_box_refiner_smoke/smoke_stage2p_refiner_limit64_valid16_e1`

Short pilot:

- `outputs/task2/stage2p_box_refiner/convnext_tiny_yolo_geometry_iou20_valid256_e4`
- best checkpoint: epoch 4

Full eval:

- `outputs/task2/stage2p_box_refiner/convnext_tiny_yolo_geometry_iou20_full_eval_augmented/eval_metrics.json`

## Full-Validation Results

`valid_combined`:

- before R@0.50: 0.5585
- after R@0.50: 0.5542
- augmented R@0.50: 0.5618
- before R@0.75: 0.2213
- after R@0.75: 0.2062
- augmented R@0.75: 0.2513
- augmented mean best IoU gain: +0.0094

`valid_phantom`:

- before R@0.50: 0.5073
- after R@0.50: 0.5049
- augmented R@0.50: 0.5109
- before R@0.75: 0.2500
- after R@0.75: 0.2330
- augmented R@0.75: 0.2840
- augmented mean best IoU gain: +0.0095

`valid_animal`:

- before R@0.50: 0.9533
- after R@0.50: 0.9346
- augmented R@0.50: 0.9533
- before R@0.75: 0.0000
- after R@0.75: 0.0000
- augmented R@0.75: 0.0000
- augmented mean best IoU gain: +0.0086

## Decision

Keep Stage2P as a candidate-pool augmentation source, not as a replacement/refinement stage.

The direct replacement view is negative: `after` reduces R@0.75 on combined and phantom validation. The augmented view is positive: retaining original boxes and appending refined boxes improves combined R@0.75 by about +0.030 and phantom R@0.75 by about +0.034. This passes the Stage2P gate for candidate-pool improvement.

However, Stage2P does not solve animal high-IoU localization. Animal R@0.75 remains 0.0000, meaning the current local refiner can nudge boxes but cannot create precise animal boxes from the present candidate distribution.

## Next Step

The next actionable step is to export refined boxes back into Stage2O candidate format and re-run the ranker/mAP evaluation on an expanded candidate pool:

- original YOLO Stage2L candidates;
- task1-geometry candidates;
- Stage2P refined candidates.

If mAP50-95 improves, keep the three-source pool. If ranking cannot use the new high-IoU refined boxes, train/re-evaluate the ranker with the refined source included in both train and validation pools.

## Verification

Passed:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile scripts/task2/train_stage2p_box_refiner.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2p_box_refiner.py tests/test_task2_stage2o_ranker.py tests/test_task2_stage2o_candidate_pool.py
```

Result:

- 12 tests passed.
