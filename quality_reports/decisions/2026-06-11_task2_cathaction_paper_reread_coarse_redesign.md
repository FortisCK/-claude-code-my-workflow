# Task 2 Decision: CathAction Paper Reread and Coarse Detector Redesign

Date: 2026-06-11

## Source

Primary local source:

- `master_supporting_docs/supporting_papers/Cathaction.pdf`

Supporting local stats:

- `quality_reports/decisions/2026-06-11_task2_dataset_reinspect_for_coarse_redesign.json`
- `quality_reports/decisions/2026-06-11_task2_temporal_box_stats_for_coarse_redesign.json`

## What the Paper Actually Says

The Task 2 formulation is not a normal multi-object detection problem. The
paper defines collision detection as detecting the tip of the catheter or
guidewire in each frame with one bounding box. The class attached to that box is
the state of the tip:

- `collision`: the tip collides with the blood vessel wall;
- `normal`: the tip does not collide with the blood vessel wall.

This matters because `normal` is not a separate visible object category. It is
the same kind of tip region under a different contact state. Therefore, a model
that treats `normal` and `collision` as visually separate object classes can
fail structurally.

The paper reports very low Task 2 results even for specialized baselines. The
best reported overall collision-detection result in Table V is EFF, with mean
AP 16.91 and mean mAP 14.88. Under phantom-to-animal domain adaptation, the best
mean mAP in Table VIII is only 7.88. The paper explicitly attributes the
difficulty to:

- very small catheter/guidewire tips in X-ray images;
- imbalance between collision and normal classes;
- domain shift between phantom and animal data.

The baselines in the paper are also mostly video/action or tiny-object detection
methods, including YOWO, YOWO-Plus, STEP, HIT, YOLOV, and EFF. Our current
YOLO11 single-frame detector is therefore not fully aligned with the paper's
stronger hint: Task 2 should use video/temporal context and tiny-object design.

## Local Dataset Facts

The local released public data has:

- 46,506 labeled frames;
- exactly one box per label file;
- 36 videos;
- class counts: class0 = 28,369, class1 = 18,137;
- `train_phantom.txt`: 35,788 samples, no animal-named samples;
- `valid_animal.txt`: 398 samples from 3 animal videos;
- `valid_phantom.txt`: one phantom video with 11,024 samples.

Bbox sizes in pixels are small:

- overall median width/height: about 28 px / 28 px;
- animal median width/height: about 30 px / 50 px;
- animal p75 width/height: about 37 px / 58 px.

Consecutive-frame tip motion is extremely smooth:

- all videos, median center movement: about 1.3 px;
- all videos, p75 center movement: about 2.2 px;
- animal videos, median center movement: 0 px;
- animal videos, p95 center movement: about 2.9 px.

This is the strongest local evidence that frame-independent detection is using
the wrong inductive bias.

## Diagnosis of Current Direction

The previous coarse-detection path is not robust enough because it treats Task 2
as ordinary single-frame object detection:

1. YOLO has to discover a tiny tip box from a full X-ray image, despite the fact
   that the tip is visually weak and many vessel/tool-like structures exist.
2. The class label is a contact state, not an object type, so class-specific
   detection is unstable.
3. The strongest available signal, temporal continuity of the tip location, is
   almost unused.
4. Task1 segmentation geometry is not sufficient by itself. It can supplement
   candidate proposals, especially for animal class0, but it cannot replace a
   learned localizer.

Therefore, the current question should not be "how do we make YOLO produce
better boxes?" The better question is:

> How do we localize the catheter/guidewire tip as a temporally smooth point or
> small region, and then classify the localized tip as collision or normal?

## Redesign

Replace the generic coarse detector with a sequence-aware tip localizer.

The recommended pipeline is:

1. Class-agnostic tip localization:
   - input: a short clip around frame `t`, e.g. 5 or 9 frames;
   - target: one class-agnostic center heatmap plus size/offset regression;
   - output: top-K tip candidates, not class-specific object detections.

2. Temporal decoding:
   - enforce smooth center trajectories per video;
   - use a simple Viterbi/Kalman-style smoother or dynamic programming over
     candidate peaks;
   - penalize large frame-to-frame jumps unless image evidence is strong.

3. ROI state classifier:
   - crop around localized candidate;
   - classify `normal` vs `collision`;
   - optionally include neighboring frames or frame differences in the crop.

4. Supplemental proposal sources:
   - keep YOLO Stage2L as an additional proposal source;
   - keep Task1 geometry endpoints as an additional source for animal class0;
   - let the ROI verifier/ranker select among localizer, YOLO, and geometry
     candidates.

## Why This Is Better Than Continuing YOLO Tuning

This formulation matches the paper and the data:

- The paper says the annotated object is the tip in every frame.
- Local data confirms there is exactly one box per frame.
- Consecutive-frame tip motion is tiny, so temporal smoothing should be a strong
  prior.
- The class label is contact state, so localization and classification should be
  separated.
- Tiny object baselines outperform generic video/action detectors in the paper,
  so a heatmap/point-localization head is more appropriate than large-anchor
  object detection.

## Next Experiment

Run a Stage2M sequence-aware class-agnostic tip localizer:

- start with a 5-frame input stack;
- preserve aspect ratio with letterbox or fixed 512 animal / 768 phantom mode;
- train on the current mixed/balanced split that includes animal videos where
  allowed for local public-data development;
- validate on held-out `video_2_animal` plus balanced phantom panel;
- evaluate proposal recall at IoU 0.25/0.50 and center-distance recall at
  5/10/20 px;
- compare against Stage2L YOLO proposal recall.

Success criteria:

- animal class0 candidate recall improves over YOLO's 0.0000 without destroying
  animal class1;
- phantom class1 center-distance recall improves, even if IoU@0.50 is still
  hard;
- temporal smoothing improves localization consistency over single-frame
  prediction.

If this works, train the ROI state classifier on localizer candidates and report
final mAP.
