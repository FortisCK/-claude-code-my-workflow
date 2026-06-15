# Task 2 Stage2F Candidate-Union Oracle Result

Date: 2026-06-09

## What Ran

Script:

```bash
scripts/task2/evaluate_candidate_union_oracle.py
```

Run:

```bash
env YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/matplotlib-codex \
  /home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/evaluate_candidate_union_oracle.py \
  --device cpu \
  --samples-per-class 128 \
  --max-det-per-source 50 \
  --batch-size 8 \
  --source-chunk-size 32 \
  --name stratified128_per_class_cpu \
  --output-dir outputs/task2/candidate_union_oracle
```

Candidate sources:

- two-class YOLO11s
- class-agnostic YOLO11s
- collision-only YOLO11s

Output:

```text
outputs/task2/candidate_union_oracle/stratified128_per_class_cpu/
```

## Key Results

Oracle recall means: if a perfect reranker could choose the best candidate
inside the union top-k pool, what fraction of samples would have a candidate
with the requested IoU to GT?

### Valid Combined, Stratified 128/Class

| GT class | top50 IoU>=0.5 | top50 IoU>=0.25 | mean best IoU |
| --- | ---: | ---: | ---: |
| class0 | 0.6562 | 0.8516 | 0.5923 |
| class1 | 0.2031 | 0.4141 | 0.2495 |

### Valid Phantom, Stratified 128/Class

| GT class | top50 IoU>=0.5 | top50 IoU>=0.25 | mean best IoU |
| --- | ---: | ---: | ---: |
| class0 | 0.6562 | 0.8516 | 0.5923 |
| class1 | 0.3750 | 0.8594 | 0.4738 |

### Valid Animal, Stratified 128/Class

| GT class | top50 IoU>=0.5 | top50 IoU>=0.25 | mean best IoU |
| --- | ---: | ---: | ---: |
| class0 | 0.8571 | 0.8571 | 0.5027 |
| class1 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

The candidate-union pool is useful for normal/class0 and phantom collision.
For phantom class1, top50 IoU>=0.25 reaches 0.8594, so many candidates are near
the right area but poorly ranked or not tight enough. This supports a
proposal-aware reranker/refiner.

Animal class1 is different. The union top50 and top100 pools still have zero
IoU>=0.25 recall in the stratified diagnostic sample. A reranker cannot recover
animal collision if the correct candidate never appears. This requires a new
first-stage localizer, likely heatmap/CenterNet-style or another domain-aware
localization route.

## Decision

Do not treat two-stage reranking as the only solution.

Recommended next structure:

1. Train proposal-aware hard-negative reranker for domains/classes where the
   candidate pool has recall, especially phantom and class0.
2. In parallel, implement a class1/animal-focused first-stage localizer, not
   another YOLO-only reranker.
3. Stop or deprioritize collision-only YOLO if later epochs remain near current
   mAP; by epoch 27 it still has mAP50 about 0.00116 and mAP50-95 about
   0.00038.
