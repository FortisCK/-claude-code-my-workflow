# Task 2 Stage2M-A Result: YOLO Top-K Temporal Smoothing

Date: 2026-06-11

## Objective

Test whether the current YOLO Stage2L proposal detector already contains the
correct catheter/guidewire tip location in top-K candidates, and whether
temporal smoothing can select a better candidate trajectory than frame-wise
confidence ranking.

## Artifacts

Code:

- `scripts/task2/evaluate_temporal_smoothing.py`
- `tests/test_task2_temporal_smoothing.py`

Outputs:

- `outputs/task2/temporal_smoothing/stage2m_yolo_stage2l_viterbi_top50/`
- `outputs/task2/temporal_smoothing/stage2m_yolo_stage2l_viterbi_top50_highsmooth/`

Verification:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python -m py_compile scripts/task2/evaluate_temporal_smoothing.py tests/test_task2_temporal_smoothing.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest -q tests/test_task2_temporal_smoothing.py
```

Pytest result: `2 passed`.

## Method

For each video:

1. load the existing YOLO proposal CSVs;
2. keep top50 proposals per frame;
3. run dynamic programming/Viterbi over candidates;
4. score each path by detection confidence minus transition penalties for
   center jump and box-size change;
5. evaluate the selected path against GT.

The script reports:

- raw top1: frame-wise highest-confidence proposal;
- raw top50 oracle: best proposal by GT IoU within top50, an upper bound;
- temporal decoded: best smoothed trajectory without GT.

## Main Results

Default sweep best:

| Split | Method | IoU@0.50 Recall | Center@20px Recall | Mean IoU |
| --- | --- | ---: | ---: | ---: |
| animal | raw top1 | 0.8318 | 0.8598 | 0.4598 |
| animal | top50 oracle | 0.8598 | 0.8692 | 0.4692 |
| animal | temporal | 0.8318 | 0.8598 | 0.4601 |
| phantom | raw top1 | 0.1869 | 0.9005 | 0.3058 |
| phantom | top50 oracle | 0.5049 | 0.9951 | 0.5119 |
| phantom | temporal | 0.1857 | 0.9078 | 0.3064 |
| combined | raw top1 | 0.2610 | 0.8958 | 0.3235 |
| combined | top50 oracle | 0.5456 | 0.9807 | 0.5070 |
| combined | temporal | 0.2599 | 0.9023 | 0.3241 |

High-smoothing sweep:

| Split | Method | IoU@0.50 Recall | Center@20px Recall | Mean IoU |
| --- | --- | ---: | ---: | ---: |
| animal | temporal high-smooth | 0.8318 | 0.8598 | 0.4622 |
| phantom | temporal high-smooth | 0.1820 | 0.9126 | 0.3056 |
| combined | temporal high-smooth | 0.2567 | 0.9066 | 0.3233 |

Class-wise default result:

| Split | Method | Class0 IoU@0.50 | Class1 IoU@0.50 | Class0 Center@20 | Class1 Center@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| animal | raw top1 | 0.0000 | 0.9674 | 0.0000 | 1.0000 |
| animal | top50 oracle | 0.0000 | 1.0000 | 0.0667 | 1.0000 |
| animal | temporal | 0.0000 | 0.9674 | 0.0000 | 1.0000 |
| phantom | raw top1 | 0.3544 | 0.0194 | 0.9684 | 0.8325 |
| phantom | top50 oracle | 0.7354 | 0.2743 | 1.0000 | 0.9903 |
| phantom | temporal | 0.3592 | 0.0121 | 0.9709 | 0.8447 |

## Interpretation

Simple temporal smoothing does not solve the coarse-detection problem.

The failure modes differ by domain:

1. Animal class0:
   - YOLO top50 oracle has IoU@0.50 = 0.0000 and Center@20px = 0.0667.
   - This means the correct animal class0 region is mostly absent from the
     candidate set.
   - Temporal smoothing cannot recover boxes that are not proposed.

2. Animal class1:
   - Raw YOLO top1 is already strong.
   - Temporal smoothing has little room to improve.

3. Phantom class1:
   - Top50 oracle has Center@20px = 0.9903, so many candidates are near the
     correct center.
   - But top50 oracle IoU@0.50 is only 0.2743 and temporal IoU@0.50 is around
     0.0121 for class1.
   - This suggests the bottleneck is not only trajectory selection; box shape,
     scale, and local classification/ranking are still wrong.

## Decision

Do not continue with simple YOLO-topK Viterbi smoothing as the main Task2 route.

The diagnostic is still useful because it separates two problems:

- animal class0 needs a new proposal generator/localizer, not just smoothing;
- phantom class1 needs better box refinement and candidate ranking around
  already-near centers.

Next direction:

1. Train a sequence-aware, class-agnostic tip heatmap localizer rather than
   relying on YOLO proposals.
2. Decode top-K heatmap peaks with temporal smoothing.
3. Add a box refinement head or template fitting to convert center candidates
   into better IoU boxes.
4. Use an ROI state classifier for normal/collision after localization.
