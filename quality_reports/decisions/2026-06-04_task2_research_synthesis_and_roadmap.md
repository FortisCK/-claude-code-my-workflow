# Task 2 Research Synthesis and Roadmap

Date: 2026-06-04

## Status

Accepted as the working Task 2 research roadmap.

This document incorporates the user-provided Task 2 literature review and the
local startup inventory.

## Main Conclusion

Task 2 should be treated as a tip-centric tiny-object detection and local state
classification problem.

The first strong baseline should be a high-resolution YOLO/EFF-style detector,
not a heavyweight action-localization transformer. Later improvements should
add structure and temporal priors in a controlled order.

## Why This Is the Right Formulation

Public CathAction materials define collision detection as:

- catheter/guidewire tip bounding-box localization;
- two classes: collision and normal;
- AP/mAP evaluation;
- mAP as the private-test primary metric in the MICCAI 2026 guide.

Local data agrees with this formulation:

- `46506` images;
- `46506` label files;
- exactly one bbox per image;
- no empty labels or bad label lines;
- small normalized boxes, with median width around `0.033`.

Therefore, once class mapping is confirmed, inference should likely enforce a
top-1 tip output per frame rather than allowing arbitrary multi-instance
detections.

## Baseline Evidence

CathAction paper Task 2 baselines show that tiny-object detectors outperform
generic spatiotemporal action detectors.

| Method | Mean AP | Mean mAP |
| --- | ---: | ---: |
| STEP | `10.98` | `9.08` |
| YOWO | `11.73` | `9.92` |
| YOWO-Plus | `11.77` | `10.28` |
| HIT | `12.14` | `10.81` |
| Yolov | `15.89` | `14.11` |
| EFF | `16.91` | `14.88` |

The paper also reports very low phantom-to-animal domain-adaptation results,
with EFF mean mAP only `7.88`. This makes domain shift a first-order issue.

## Literature Directions Verified

The user-provided research directions are mostly credible and relevant.

Verified directions:

1. **AttWire-style wire attention**
   - Relevant because collision depends on thin wire/catheter structures.
   - Useful later as a structural prior or feature module, not first baseline.

2. **ConTrack-style contextual tip tracking**
   - Relevant because it uses spatial context, segmentation context, template
     frames, and flow information to track catheter tips in X-ray.
   - Strong inspiration for a Stage 3 temporal/segmentation-aware refiner.

3. **Lightweight multi-frame YOLO**
   - Relevant because it stacks consecutive frames and supervises one target
     frame, adding temporal context without a large video model.
   - Good Stage 2 candidate after single-frame YOLO is stable.

4. **Task 1 segmentation prior**
   - Relevant because Task 1 provides tool/shaft information, and ConTrack-like
     methods use full-device segmentation context for tip prediction.
   - Practical options:
     - use Task 1 mask to restrict ROI;
     - add a binary toolness/shaft auxiliary head;
     - use detected shaft endpoint as a prior for the tip.

5. **Phantom-to-animal adaptation**
   - Important because local splits and the CathAction paper both expose a
     large domain shift.
   - Should be pursued only after a single-frame detector baseline is stable.

## Immediate Roadmap

### Stage T2-0: Data Protocol and Evaluator

Goals:

- confirm class mapping;
- generate leakage-free manifests;
- implement local AP/mAP evaluator;
- generate overlay visualizations.

Requirements:

- remove `valid_phantom` and `valid_animal` from local training for model
  selection;
- report `valid_phantom`, `valid_animal`, and combined metrics separately;
- never use frame-random splits for reported results.

### Stage T2-1: High-Resolution Single-Frame YOLO Baseline

Goals:

- train a reliable modern YOLO-family detector;
- use public pretrained weights;
- test `640` smoke and `1024`/higher full run;
- evaluate mAP50-95, AP50, AP75, per-class AP, per-domain AP.

Expected role:

- establish whether a modern detector already beats CathAction paper EFF
  baselines under local validation.

### Stage T2-2: Tiny-Object Optimization

Candidate changes:

- P2/small-object head if available;
- high-resolution training;
- SAHI/tiling only if full-frame high resolution is insufficient;
- class imbalance handling through focal-style losses or sampling;
- top-1 per-frame postprocess and WBF/TTA where runtime permits.

### Stage T2-3: Lightweight Temporal Refinement

Candidate changes:

- multi-frame stacked input;
- frame-neighbor feature aggregation;
- top-1 tracklet smoothing;
- hysteresis / median filtering for collision state;
- short-window rescoring rather than a heavy action detector.

### Stage T2-4: Task1-Guided Tip Prior

Candidate changes:

- Task1 Stage9A segmentation mask as ROI prior;
- endpoint extraction from predicted shaft mask;
- detector run on full frame plus ROI crop;
- auxiliary binary shaft/toolness head if implementing custom model.

### Stage T2-5: Domain Adaptation

Candidate changes:

- phantom-to-animal calibration;
- target-style augmentation;
- pseudo-label/self-training on validation-like or allowed unlabeled frames;
- domain-specific thresholds only if official test-domain information supports
  it.

## Current Decisions

1. Start with single-frame YOLO, not temporal transformer.
2. Keep temporal and Task1 priors as second-stage improvements, not baseline
   dependencies.
3. Treat validation leakage in the provided split files as a real issue and
   build clean manifests.
4. Treat class ID mapping as an unresolved setup question.
5. Use domain-separated evaluation from the first baseline onward.

## Sources

- User-provided Task 2 literature review:
  `/home/mingzhang/.codex/attachments/bcd06a1b-4305-411d-b488-fee7b026d29c/pasted-text.txt`
- Task 2 startup inventory:
  `quality_reports/decisions/2026-06-04_task2_startup_inventory_and_baseline.md`
- CathAction paper:
  https://arxiv.org/abs/2408.13126
- CathAction dataset card:
  https://huggingface.co/datasets/airvlab/CathAction
- AttWire:
  https://arxiv.org/abs/2503.06190
- ConTrack:
  https://arxiv.org/abs/2307.07541
- Lightweight multi-frame YOLO:
  https://arxiv.org/abs/2506.20550

## Verification

- Local data facts were verified from `datasets/collision_detection/`.
- Published CathAction Task 2 baselines were verified from the CathAction paper.
- AttWire, ConTrack, and lightweight multi-frame YOLO directions were checked
  against public arXiv pages.
