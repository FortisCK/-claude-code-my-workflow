# Task2 Metric Alignment and High-IoU Route

Date: 2026-06-14

## Question

How does the current Task2 system compare with the CathAction paper's YOLOV and
EFF baselines, and what should we optimize next?

## Paper Reference

The CathAction paper defines collision detection as object detection of the
catheter/guidewire tip box with two labels:

- collision;
- normal.

It evaluates AP and mAP. Table V reports percentages:

| Paper method | AP Collision | AP Normal | AP Mean | mAP Collision | mAP Normal | mAP Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STEP | 7.79 | 11.21 | 10.98 | 6.92 | 11.29 | 9.08 |
| YOWO | 8.32 | 12.18 | 11.73 | 7.46 | 12.28 | 9.92 |
| YOWO-Plus | 8.92 | 12.23 | 11.77 | 7.86 | 12.48 | 10.28 |
| HIT | 9.37 | 12.74 | 12.14 | 8.18 | 12.72 | 10.81 |
| YOLOV | 12.30 | 21.08 | 15.89 | 11.88 | 20.04 | 14.11 |
| EFF | 13.70 | 22.10 | 16.91 | 12.14 | 20.78 | 14.88 |

The paper says YOLOV and EFF are tiny-object detectors and outperform the
normal object-detection baselines, but all collision-detection results remain
low.

## Current Local Results

Current public-validation metrics:

| Local method | valid_combined mAP50 | valid_combined mAP50-95 | Role |
| --- | ---: | ---: | --- |
| Stage2AE | 21.13% | 6.33% | default before Stage2AQ |
| Stage2AQ | 23.66% | 7.02% | current balanced candidate |
| Stage2AI | 18.44% | 8.39% | high-IoU / mAP50-95 candidate |

Domain split for the current default:

| Method | valid_phantom mAP50 | valid_phantom mAP50-95 | valid_animal mAP50 | valid_animal mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Stage2AQ | 12.98% | 4.99% | 49.97% | 14.32% |

## Comparison Interpretation

This is not a strict leaderboard comparison because the paper split/evaluator
and the current local public-validation split/evaluator are not guaranteed to
match. The numbers are still useful as a reference.

If paper `AP Mean` is treated as AP50-like / loose localization:

| Method | AP50-like score |
| --- | ---: |
| YOLOV paper AP Mean | 15.89% |
| EFF paper AP Mean | 16.91% |
| Stage2AQ local mAP50 | 23.66% |

Under this loose-localization interpretation, Stage2AQ looks stronger than the
paper's YOLOV/EFF references.

If paper `mAP Mean` is treated as COCO-style / stricter multi-IoU localization:

| Method | high-IoU/mAP-like score |
| --- | ---: |
| YOLOV paper mAP Mean | 14.11% |
| EFF paper mAP Mean | 14.88% |
| Stage2AQ local mAP50-95 | 7.02% |
| Stage2AI local mAP50-95 | 8.39% |

Under this stricter-localization interpretation, our current system is still
below the paper's YOLOV/EFF references.

## Practical Conclusion

Current Stage2AQ is probably not failing at coarse detection. It finds enough
boxes to make AP50 competitive. The main weakness is high-IoU localization and
ranking:

- Stage2AQ improves valid_combined mAP50 from Stage2AE by about +2.53 points.
- Stage2AQ improves mAP50-95 only by about +0.70 points.
- Stage2AI improves mAP50-95 to 8.39%, but loses AP50.
- The current domain split shows animal AP50 is much stronger than phantom AP50,
  while phantom dominates many public-validation rows.

The next performance work should target:

```text
better high-IoU box localization + better selector/ranker for near-correct boxes
```

not another generic single-frame detector run.

## Recommended Next Experiments

### Priority 1: Confirm official metric mapping

Before running expensive training, align the local evaluator with the official
metric once validation code is released. Until then, keep two candidates:

- Stage2AQ if official ranking is AP50-like;
- Stage2AI if official ranking is COCO-style mAP50-95.

### Priority 2: High-IoU box refinement

Train or evaluate a box refinement head on existing candidates:

- input: candidate crop plus original candidate geometry;
- target: GT center offset and width/height correction;
- train only on candidates with non-trivial overlap, for example IoU >= 0.20;
- optimize SmoothL1/GIoU-style box regression;
- evaluate by:
  - candidate R@0.75;
  - final mAP50-95;
  - AP75;
  - whether Stage2U can rank refined candidates without adding false positives.

Prior Stage2P/Stage2Q attempts showed refined candidates can improve the
localization upper bound, but the old ranker did not rank them well. Therefore,
the refinement route must include either:

- source-consistent ranker training with refined candidates in both train and
  validation pools; or
- a dedicated refined-candidate score calibration policy.

### Priority 3: Proper YOLOV/EFF two-class baseline on our split

The previous YOLOV work was a reduced/pilot proposal-style run and should not
be treated as a full paper-baseline reproduction. If we need a stronger
external baseline, run:

- YOLOV two-class on current public split;
- EFF or another tiny-object detector if implementation cost is acceptable;
- same local mAP50/mAP50-95 evaluator as Stage2AQ.

This will tell us whether paper-style tiny detectors really dominate our
current Stage2AQ pipeline under the same split.

### Priority 4: Domain-aware hidden policy

Hidden filenames may not encode domain. Current submission entrypoint defaults
to `--fallback-domain phantom`, which is safe for public phantom-heavy behavior
but may not be optimal if hidden includes animal/human. Add one of:

- explicit domain metadata parsing from official validation package, if
  provided;
- a lightweight image/video domain classifier;
- a domain-agnostic fallback policy that avoids over-specializing to phantom.

## Current Decision

For submission hardening, keep Stage2AQ as the default internal output because
it has the strongest local AP50-like score and best balanced combined result.

For the next model-improvement stage, do not start with a new generic detector.
Start with metric alignment and high-IoU box refinement/ranking, while keeping
YOLOV/EFF two-class reproduction as the main sanity baseline if we suspect the
paper methods transfer better to our split.

