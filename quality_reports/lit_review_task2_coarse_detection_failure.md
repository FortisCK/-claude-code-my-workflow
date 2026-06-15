# Literature Review: Task 2 Coarse Detection Failure

**Date:** 2026-06-09  
**Query:** Whether noisy, poorly aligned coarse detection boxes in CATHACTION Task 2 have appeared in prior fluoroscopy / guidewire / small-object detection literature.

## Summary

Yes. The failure mode we are seeing is not surprising and is directly aligned
with the CathAction benchmark paper itself. CathAction defines collision
detection as object detection of the catheter/guidewire tip bbox, with class
normal or collision. Its reported baselines are all low: the best listed mean
AP is 16.91 and best listed mean mAP is 14.88, despite using video action
detectors and two tiny-object methods. The paper attributes this to small tip
boxes and normal/collision class imbalance.

Adjacent fluoroscopy work often avoids relying on a single raw detector. Common
patterns are: detector as a coarse proposal stage, post-processing or temporal
constraints to remove false detections, a second-stage keypoint/heatmap/local
network for endpoint localization, or segmentation/response-map methods for
thin structures. This supports our current pivot from "keep tuning YOLO" toward
"measure proposal oracle recall, then use hard-negative reranking or a
heatmap/CenterNet-style localizer."

The broader small-object and domain-shift literature also matches our
diagnostics. Small objects have weak features, limited context, severe
foreground/background imbalance, and too few useful positive examples. Domain
shift between catheterization labs or imaging domains can damage YOLO feature
maps and the downstream regression head. Our observed train/valid shift between
phantom and animal collision boxes is therefore a plausible explanation for why
top-1 boxes look visually unrelated to GT.

## Key Papers

### Huang et al. — CathAction Benchmark

- **Main contribution:** Introduces CathAction and includes collision detection
  baselines.
- **Relevant finding:** Collision detection performance is low across baselines.
  Tiny-object methods, Yolov and EFF, outperform general baselines but still
  reach only mean AP 16.91 / mean mAP 14.88 at best in the table.
- **Stated difficulty:** The paper explicitly points to the small catheter or
  guidewire tip and normal/collision imbalance.
- **Relevance:** This is the closest source to our current task; our low YOLO
  behavior is consistent with the official benchmark being intrinsically hard.

### Li et al. 2021 — Real-Time Multi-Guidewire Endpoint Localization

- **Main contribution:** Two-stage guidewire endpoint localization in
  fluoroscopy.
- **Method:** YOLOv3 detects guidewire instances; post-processing refines
  detection; SA-hourglass predicts endpoint locations in the crop.
- **Key result:** PubMed reports mean pixel error of 2.20 pixels on the
  guidewire dataset.
- **Relevance:** This is a direct precedent for not trusting raw detector boxes
  as the final endpoint localization output.

### Zhang et al. 2024 — Real-Time Guidewire Tracking and Segmentation

- **Main contribution:** Two-stage guidewire tracking/segmentation in
  intraoperative X-ray.
- **Method:** YOLOv5s outputs possible guidewire boxes, then a spatiotemporal
  refinement module robustly localizes guidewires and removes false detections;
  segmentation is performed inside detected boxes.
- **Relevance:** Very close to our situation: detector proposals are useful, but
  need temporal constraints/refinement because fluoroscopy guidewires are
  elongated, low contrast, and noisy.

### Wen et al. 2024 — Sim-to-Real Guidewire Segmentation

- **Main contribution:** Adapts segmentation foundation models under sim-to-real
  domain shift for X-ray fluoroscopy guidewires.
- **Method:** Coarse-to-fine SAM adaptation with style transfer, pseudo-labels,
  self-training, and consistency.
- **Relevance:** Confirms that domain shift is a first-class issue in
  guidewire fluoroscopy. While it is segmentation rather than detection, the
  lesson maps to Task 2: phantom/animal/human shifts are not cosmetic.

### Ambrosini et al. 2017 — Catheter Segmentation in X-Ray Fluoroscopy

- **Main contribution:** Real-time catheter/guidewire segmentation using the
  current frame plus three previous frames.
- **Key result:** Reports small tip/centerline errors using sequence context.
- **Relevance:** Supports using temporal information instead of single-frame
  boxes only.

### Li and Barbu 2020 — Steerable CNN for Guidewire Detection

- **Main contribution:** Pixelwise guidewire response-map detection with
  orientation-sensitive filters/CNNs.
- **Motivation:** Fluoroscopy images are low-dose, guidewires are thin and
  poorly visible, and orientation matters.
- **Relevance:** Supports a heatmap/response-map formulation for small/thin
  targets.

### Liu et al. 2021 — Small Object Detection Survey

- **Main contribution:** Survey and performance evaluation of deep learning
  methods for small object detection.
- **Relevant finding:** Identifies feature weakness, missing context,
  foreground/background imbalance, and insufficient positive examples as
  central challenges.
- **Relevance:** These are exactly the failure mechanisms we see in Task 2.

### Towards Robust Object Detection in Unseen Catheterization Laboratories

- **Main contribution:** Studies YOLO object detection robustness under unseen
  cath-lab domain shift.
- **Relevant point:** Discusses one-stage YOLO localization limits for small
  objects and frames domain shift as a practical cause of degraded detection.
- **Relevance:** Matches our phantom/animal domain split and supports feature
  distribution diagnostics.

## Thematic Organization

### 1. CathAction Collision Detection Is Known to Be Low-AP

CathAction's own baselines are much lower than ordinary COCO-style object
detection. The benchmark explicitly says the top mean AP is only 16.91 and
attributes the difficulty to small tip boxes and class imbalance. This means our
current "messy boxes" are not an isolated engineering bug by default.

### 2. Raw YOLO Is Usually a Proposal Generator, Not the Final Answer

Fluoroscopy guidewire endpoint papers often use YOLO to crop or propose a
region, then add a specialized endpoint/keypoint/localization network. That
maps well to our proposed hard-negative reranker and heatmap localizer.

### 3. Temporal and Geometric Constraints Matter

Several fluoroscopy methods use sequence context or spatiotemporal refinement.
This is reasonable for Task 2 because tip position and collision status should
not jump arbitrarily between adjacent frames.

### 4. Domain Shift Is a Real Problem

The sim-to-real and unseen-catheterization-lab papers support our observation
that phantom/animal/human imaging shifts can break a detector. Our bbox stats
show a large class-1 distribution shift between train, valid_phantom, and
valid_animal, which likely worsens localization.

## Gaps and Opportunities

1. **Direct collision detection work is sparse.** CathAction itself is the main
   direct reference; most adjacent papers are guidewire endpoint or segmentation.
2. **The literature supports two-stage/local refinement.** Our next experiment
   should test whether the correct box exists in the union of top-k detector
   proposals. If yes, train a hard-negative reranker; if no, move to heatmap
   localization.
3. **Temporal consistency is underused in our current pipeline.** Once we have
   a reasonable per-frame localizer, video-level smoothing should be added.
4. **Domain-aware validation is essential.** We should report phantom and
   animal separately and avoid assuming one global detector will solve both.

## Suggested Next Steps

1. Run candidate-union oracle recall over two-class YOLO, class-agnostic YOLO,
   and collision-only YOLO at top10/top20/top50.
2. If oracle recall is high, train a proposal-aware hard-negative reranker.
3. If oracle recall is low, implement a CenterNet-style single-object heatmap
   localizer with class-specific center heatmaps and size/offset heads.
4. Add temporal smoothing only after per-frame candidate quality improves.
5. Consider a feature/domain diagnostic similar to the unseen cath-lab paper:
   compare train/valid phantom/animal feature clusters and bbox priors.

## BibTeX Entries

```bibtex
@article{huang2024cathaction,
  title={CathAction: A Benchmark for Endovascular Intervention Understanding},
  author={Huang, B. and Vo, T. and Kongtongvattana, C. and others},
  journal={arXiv preprint arXiv:2408.13126},
  year={2024}
}

@article{li2021realtime,
  title={Real-Time Multi-Guidewire Endpoint Localization in Fluoroscopy Images},
  author={Li, Rui-Qi and Xie, Xiao-Liang and Zhou, Xiao-Hu and Liu, Shi-Qi and Ni, Zhen-Liang and Zhou, Yan-Jie and Bian, Gui-Bin and Hou, Zeng-Guang},
  journal={IEEE Transactions on Medical Imaging},
  volume={40},
  number={8},
  pages={2002--2014},
  year={2021},
  doi={10.1109/TMI.2021.3069998}
}

@article{zhang2024realtime,
  title={Real-time guidewire tracking and segmentation in intraoperative x-ray},
  author={Zhang, Baochang and Bui, Mai and Wang, Cheng and Bourier, Felix and Schunkert, Heribert and Navab, Nassir},
  journal={arXiv preprint arXiv:2404.08805},
  year={2024}
}

@article{wen2024generalizing,
  title={Generalizing Segmentation Foundation Model Under Sim-to-real Domain-shift for Guidewire Segmentation in X-ray Fluoroscopy},
  author={Wen, Yuxuan and Roussinova, Evgenia and Brina, Olivier and Machi, Paolo and Bouri, Mohamed},
  journal={arXiv preprint arXiv:2410.07460},
  year={2024}
}

@article{ambrosini2017fully,
  title={Fully Automatic and Real-Time Catheter Segmentation in X-Ray Fluoroscopy},
  author={Ambrosini, Pierre and Ruijters, Daniel and Niessen, Wiro J. and Moelker, Adriaan and van Walsum, Theo},
  journal={arXiv preprint arXiv:1707.05137},
  year={2017}
}

@inproceedings{li2020steerable,
  title={Training a Steerable CNN for Guidewire Detection},
  author={Li, Donghang and Barbu, Adrian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

@article{liu2021smallobject,
  title={A survey and performance evaluation of deep learning methods for small object detection},
  author={Liu, Yang and Sun, Peng and Wergeles, Nickolas and Shang, Yi},
  journal={Expert Systems with Applications},
  volume={172},
  pages={114602},
  year={2021},
  doi={10.1016/j.eswa.2021.114602}
}

@inproceedings{gong2021eff,
  title={Effective Fusion Factor in FPN for Tiny Object Detection},
  author={Gong, Yuqi and Yu, Xuehui and Ding, Yao and Peng, Xiaoke and Zhao, Jian and Han, Zhenjun},
  booktitle={WACV},
  year={2021}
}

@inproceedings{shi2023yolov,
  title={Yolov: Making Still Image Object Detectors Great at Video Object Detection},
  author={Shi, Yuheng and Wang, Naiyan and Guo, Xiaojie},
  booktitle={AAAI},
  year={2023}
}
```

## Verification Note

Claims above were checked against source pages/PDFs located during the mini
search. This is a compact, source-linked mini-review rather than a full
systematic review; exact leaderboard comparability still depends on our split,
class mapping, and evaluation protocol.
