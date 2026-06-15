# Task 2 Startup Inventory and Baseline Decision

Date: 2026-06-04

## Status

Accepted as the Task 2 starting point.

Task 2 should begin as a tiny-object detection problem, not as a plain image
classification problem.

## Official Task Definition

CATHACTION Task 2 is collision detection in fluoroscopic endovascular
intervention video frames.

The CathAction benchmark paper defines the task as object detection:

- annotate the catheter or guidewire tip with a bounding box in each frame;
- classify each box as either collision or normal;
- evaluate with AP and mAP.

The MICCAI 2026 challenge guide states that:

- AP and mAP are used for Task 2;
- mAP on the private test set is the primary ranking metric;
- AP is a complementary metric;
- final submissions are Docker inference pipelines on hidden test data.

## Local Data Structure

Local folder:

- `datasets/collision_detection/images/`
- `datasets/collision_detection/labels/`
- `datasets/collision_detection/train_phantom.txt`
- `datasets/collision_detection/valid_phantom.txt`
- `datasets/collision_detection/valid_animal.txt`

Counts from local files:

| Item | Count |
| --- | ---: |
| Images | `46506` |
| Label files | `46506` |
| Missing labels | `0` |
| Missing images | `0` |
| Empty label files | `0` |
| Bad label lines | `0` |
| Boxes per image | exactly `1` |
| Video IDs | `36` |

Label format is YOLO-style:

```text
class x_center y_center width height
```

All coordinates are normalized to `[0, 1]`.

Example:

```text
0 0.5972850678733032 0.4218009478672986 0.014660633484162732 0.019194312796208583
```

## Class Distribution

Overall local class counts:

| Class ID | Count |
| --- | ---: |
| `0` | `28369` |
| `1` | `18137` |

Working assumption:

- `0` = normal
- `1` = collision

Reason: collision is described in the paper as the rarer event, and local
`valid_phantom` has class `1` as the rare class (`412 / 11024`). This mapping
must be confirmed against any official class-name file or evaluator once
released.

## Bounding Box Scale

The boxes are small. Normalized width/height medians:

| Class | Median width | Median height | Median area |
| --- | ---: | ---: | ---: |
| `0` | `0.03255` | `0.04626` | `0.00155` |
| `1` | `0.03322` | `0.05428` | `0.00182` |

This supports the paper's conclusion that the task is a tiny-object detection
problem. High-resolution input and small-object-sensitive detection heads are
likely important.

## Split Files

Local split-list counts:

| File | Lines | Notes |
| --- | ---: | --- |
| `train_phantom.txt` | `35788` | label paths, not image paths |
| `valid_phantom.txt` | `11024` | label paths, all `video_0` |
| `valid_animal.txt` | `398` | label paths, `video_0_animal`, `video_1_animal`, `video_2_animal` |

Important issue:

- `train_phantom.txt` and `valid_phantom.txt` overlap by `704` label files.
- Therefore, using these files directly for local model selection causes
  leakage on `valid_phantom`.

Conservative local-validation policy:

1. For local validation, remove all `valid_phantom` and `valid_animal` labels
   from the training manifest.
2. Report validation separately for:
   - `valid_phantom`;
   - `valid_animal`;
   - combined validation.
3. Prefer case/video-level validation reporting, because `video_0` dominates
   the phantom validation set.

For final challenge training, reassess whether the official rules allow using
all public labeled data before hidden-test submission.

## Published CathAction Baselines

From the CathAction benchmark paper, Task 2 baselines:

| Method | Type | Mean AP | Mean mAP |
| --- | --- | ---: | ---: |
| STEP | spatiotemporal action detection | `10.98` | `9.08` |
| YOWO | spatiotemporal action localization | `11.73` | `9.92` |
| YOWO-Plus | improved YOWO | `11.77` | `10.28` |
| HIT | holistic interaction transformer | `12.14` | `10.81` |
| Yolov | video/tiny object detector | `15.89` | `14.11` |
| EFF | tiny object detector / feature fusion | `16.91` | `14.88` |

The paper's interpretation:

- tiny-object detectors outperform generic spatiotemporal action detectors;
- all scores are low;
- main bottlenecks are small tip boxes and collision/normal imbalance.

## Literature Search Result

As of this startup pass, no MSLNet-like independent paper was found that
directly studies the released CathAction Task 2 collision-detection split and
proposes a clearly stronger method.

The direct public reference remains:

- Huang et al., "CathAction: A Benchmark for Endovascular Intervention
  Understanding", arXiv:2408.13126.

## First Baseline Decision

Start with a modern YOLO-family tiny-object detector.

Rationale:

- local labels are already YOLO-style;
- each frame has exactly one small bbox;
- paper baselines show tiny-object detectors beat temporal action detectors;
- a YOLO baseline is fastest to make reproducible and easy to Dockerize.

Recommended first baseline:

1. Build clean YOLO manifests:
   - clean phantom train = `train_phantom - valid_phantom - valid_animal`;
   - validation = `valid_phantom` and `valid_animal` separately.
2. Train a small/medium detector at high resolution.
   - start with `imgsz=1024` if GPU memory allows;
   - keep `640` as a fast smoke setting;
   - use pretrained public weights.
3. Report:
   - mAP50-95;
   - AP/mAP per class;
   - phantom validation and animal validation separately;
   - confusion/failure overlays.
4. After the image-detector baseline:
   - add high-res / small-object tuning;
   - test temporal smoothing or short-window fusion;
   - consider Task1-guided ROI/tip prior if detection localization is weak.

## Open Questions

1. Confirm class ID mapping (`0/1`) from official evaluator or class-name file.
2. Confirm official AP/mAP threshold definitions when submission instructions are
   released.
3. Decide whether public validation labels can be used for final model training
   before hidden-test submission.
4. Determine Docker runtime and image-size limits before promoting a large model.

## Verification

Local checks completed:

- `46506` images and `46506` labels;
- no missing image/label pairs;
- no bad label lines;
- every image has exactly one bbox;
- class counts and split overlap computed from local files;
- benchmark baselines verified from the CathAction paper and existing local
  literature scan.
