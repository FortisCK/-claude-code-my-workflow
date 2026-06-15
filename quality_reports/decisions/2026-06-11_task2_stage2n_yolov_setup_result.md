# Task 2 Stage2N YOLOV Official Setup Result

Date: 2026-06-11

## Question

Can we replace the single-frame YOLO first-stage detector with a real temporal
video object detector, instead of continuing to patch the custom sequence-tip
localizer?

## Short Answer

Yes, technically. The official YOLOV repository has been cloned, adapted to
CATHACTION Task 2, and verified through data conversion, loader/model smoke
checks, GPU loss computation, and a 1-epoch smoke train/eval.

This does **not** yet prove that YOLOV improves Task 2 performance. It proves
that the official temporal detector route is now runnable on our data.

## Why YOLOV

The local CathAction paper describes Task 2 as frame-wise bbox detection of the
catheter/guidewire tip in fluoroscopy video, with normal/collision classes. It
also discusses video/tiny-object detection baselines, including YOLOV and EFF.
This makes YOLOV a more principled first-stage detector candidate than the
previous homemade heatmap/sequence localizer.

## Implementation

Official repo:

- `external_repos/YOLOV`

New conversion script:

- `scripts/task2/prepare_yolov_cathaction.py`

New CATHACTION YOLOV experiment:

- `external_repos/YOLOV/exps/cathaction/cathaction_yolov_s_agnostic.py`

Local patches to official YOLOV:

- `external_repos/YOLOV/yolox/data/datasets/vid.py`
  - added missing `json`, `COCO`, and `remove_useless_info` imports;
  - patched `TrainSampler.__init__` for current PyTorch compatibility.
- `external_repos/YOLOV/yolox/evaluators/vid_evaluator_v2.py`
  - forced standard `pycocotools.cocoeval.COCOeval` to avoid the optional C++
    fast evaluator requiring `ninja`;
  - replaced hard-coded ImageNet VID category names with dynamic CathAction
    category metadata (`tip` for agnostic mode).
- `external_repos/YOLOV/yolox/evaluators/coco_evaluator.py`
  - same standard COCOeval fallback to avoid optional C++ evaluator failures.

Pretrained checkpoint:

- `weights/yolox_s.pth`

## Converted Data

Primary agnostic dataset:

- `datasets/collision_detection_yolov_agnostic`

Summary:

- train: 35,375 samples, 35 videos
  - original class counts: 17,742 normal / 17,633 collision
  - domain counts: 291 animal / 35,084 phantom
- valid_combined: 931 samples, 2 videos
  - original class counts: 427 normal / 504 collision
  - domain counts: 107 animal / 824 phantom
- valid_phantom: 824 samples, 1 video
- valid_animal: 107 samples, 1 video

Smoke dataset:

- `datasets/collision_detection_yolov_agnostic_smoke`
- generated with `--train-limit 64 --valid-limit 32`
- used only to validate training/evaluation plumbing.

## Verification

Passed:

- `python -m py_compile scripts/task2/prepare_yolov_cathaction.py`
- JSON parse check on:
  - `datasets/collision_detection_yolov_agnostic/cathaction_valid_combined.json`
- YOLOV dataset loader smoke:
  - full `valid_combined` loads as temporal samples;
  - full train loader builds batches.
- YOLOV model build smoke:
  - model parameters: 10,251,732
  - trainable parameters: 10,251,732
- GPU forward/loss smoke:
  - one batch computes YOLOV training loss successfully.
- 1-epoch smoke train/eval:
  - output: `outputs/task2/yolov_smoke/cathaction_yolov_s_agnostic`
  - final train loss line: total loss about 18.7
  - evaluation runs with standard COCOeval
  - category table correctly reports `tip`

Smoke AP is 0.0, as expected for one training batch on a tiny non-representative
smoke split. It is not a performance result.

## Important Caveats

1. Current YOLOV agnostic mode is a first-stage proposal detector. It predicts
   one `tip` class and does not directly solve normal-vs-collision mAP.
2. The default official YOLOV data path uses video context, but the current
   experiment is global-frame mode (`gframe=8`) rather than a strictly adjacent
   local-frame model.
3. Full-data training is expensive at the current setting. The smoke run measured
   roughly 32 seconds for one 8-sample train iteration, so launching full data
   blindly would be risky.
4. The first smoke conversion used COCO-style category id `1`, while YOLOX/YOLOV
   maps categories internally to contiguous class id `0`. This has been cleaned
   up for the current full and pilot conversions; they now export `0`-based
   category ids.

## Decision

Run a representative reduced YOLOV pilot before any full-data overnight job.

Next controlled step:

1. Create a representative reduced train split rather than a naive first-N smoke
   split:
   - include animal and phantom;
   - include normal and collision;
   - preserve video/case boundaries;
   - sample temporally coherent windows.
2. Run a short YOLOV pilot on that subset.
3. Evaluate against Stage2L with the same proposal-recall panels:
   - top1/top5/top10/top50 R@0.50/R@0.75;
   - animal/phantom breakdown;
   - class0/class1 breakdown;
   - mAP/AP if the detector is trained in two-class mode.

The route should be kept only if it improves proposal recall over Stage2L or
solves the animal class0 failure mode.

## Representative Pilot Result

After the smoke train, a representative pilot dataset was generated:

- `datasets/collision_detection_yolov_agnostic_pilot`
- train: 1,024 samples, 35 videos
  - original class counts: 483 normal / 541 collision
  - domain counts: 291 animal / 733 phantom
- valid_combined: 256 samples
- valid_phantom: 256 samples
- valid_animal: 107 samples

Command class:

- official YOLOV-S, agnostic `tip` class
- input/test size: 576
- checkpoint init: `weights/yolox_s.pth`
- pilot training: 3 epochs

Outputs:

- training run: `outputs/task2/yolov_pilot/cathaction_yolov_s_agnostic`
- proposal evaluation:
  - `outputs/task2/yolov_proposal_eval/yolov_s_576_agnostic_pilot_e3_fullval`
- union oracle:
  - `outputs/task2/proposal_union_oracle/yolo_stage2l_plus_yolov_pilot_top50`

Official YOLOV AP on the pilot validation subset:

| epoch | AP50:95 | AP50 | AR@100 |
| --- | ---: | ---: | ---: |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.056 | 0.373 | 0.102 |
| 3 | 0.082 | 0.401 | 0.125 |

Full-panel proposal recall for the epoch-3 best checkpoint:

| source | combined top50 R@0.50 | phantom top50 R@0.50 | animal top50 R@0.50 | animal class0 top50 R@0.50 |
| --- | ---: | ---: | ---: | ---: |
| Stage2L YOLO | 0.5456 | 0.5049 | 0.8598 | 0.0000 |
| YOLOV pilot | 0.3629 | 0.3027 | 0.8542 | 0.0000 |
| Stage2L + YOLOV oracle union | 0.5542 | 0.5158 | 0.8598 | 0.0000 |

Class-wise YOLOV finding:

- combined class0 top50 R@0.50: 0.5786
- combined class1 top50 R@0.50: 0.1789
- phantom class0 top50 R@0.50: 0.5907
- phantom class1 top50 R@0.50: 0.0147
- animal class0 top50 R@0.50: 0.0000
- animal class1 top50 R@0.50: 1.0000

## Updated Decision

YOLOV should **not** replace Stage2L as the main first-stage proposal detector.
It is worse on combined and phantom proposal recall, and it does not recover the
animal class0 failure mode.

However, it is not useless. As a supplemental proposal source, it increases the
oracle union from 0.5456 to 0.5542 combined top50 R@0.50 and from 0.5049 to
0.5158 phantom top50 R@0.50. That gain is small but real. If we build a final
candidate-pool ensemble, YOLOV can be kept as a secondary source behind:

1. Stage2L YOLO as the main detector.
2. Task1 geometry proposals for animal class0 recovery.
3. Sequence-tip proposals as a weak supplemental source.
4. YOLOV as an additional video-context source only if candidate budget allows.
