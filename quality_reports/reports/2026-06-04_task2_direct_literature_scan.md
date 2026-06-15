# Task 2 Direct Literature Scan: CathAction Collision Detection

Date: 2026-06-04

## Question

Is there an MSLNet-like paper that directly uses the CathAction Task 2
collision-detection data and proposes a stronger method?

## Short Answer

I did not find a post-CathAction independent paper that directly uses the
CathAction Task 2 collision-detection split in the way MSLNet directly studies
CathAction Task 1 segmentation.

The directly relevant public source is currently the CathAction benchmark paper
itself:

- Huang et al. (2024), **CathAction: A Benchmark for Endovascular Intervention
  Understanding**, arXiv:2408.13126.

This paper introduces the dataset and reports Task 2 collision-detection
baselines. It defines collision detection as an object-detection problem: the
catheter/guidewire tip is annotated with a bounding box, and each box is labeled
as either collision or normal.

## Direct CathAction Task 2 Baselines

CathAction benchmarks these methods on the collision-detection task:

| Method | Type | Collision AP | Normal AP | Mean AP | Collision mAP | Normal mAP | Mean mAP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STEP | spatiotemporal action detection | 7.79 | 11.21 | 10.98 | 6.92 | 11.29 | 9.08 |
| YOWO | real-time spatiotemporal action localization | 8.32 | 12.18 | 11.73 | 7.46 | 12.28 | 9.92 |
| YOWO-Plus | improved YOWO | 8.92 | 12.23 | 11.77 | 7.86 | 12.48 | 10.28 |
| HIT | holistic interaction transformer | 9.37 | 12.74 | 12.14 | 8.18 | 12.72 | 10.81 |
| Yolov | video object detection / tiny object detector in this context | 12.30 | 21.08 | 15.89 | 11.88 | 20.04 | 14.11 |
| EFF | tiny object detection / feature-fusion FPN method | 13.70 | 22.10 | 16.91 | 12.14 | 20.78 | 14.88 |

The important reading is that the best reported mean AP is only `16.91` and
the best reported mean mAP is `14.88`, both from EFF. The paper explicitly
frames this as evidence that CathAction collision detection is difficult.

## Domain-Adaptation Baselines

CathAction also reports a phantom-to-animal domain-adaptation setting:

| Method | Collision AP | Normal AP | Mean AP | Collision mAP | Normal mAP | Mean mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STEP | 1.53 | 2.12 | 1.87 | 1.09 | 1.98 | 1.62 |
| YOWO | 2.12 | 4.11 | 3.09 | 1.97 | 3.68 | 2.92 |
| YOWO-Plus | 1.18 | 1.43 | 1.21 | 1.07 | 1.26 | 1.09 |
| HIT | 1.31 | 1.19 | 1.24 | 1.06 | 1.18 | 1.11 |
| Yolov | 7.31 | 8.92 | 8.09 | 6.28 | 7.49 | 7.21 |
| EFF | 8.27 | 9.16 | 8.19 | 7.61 | 8.29 | 7.88 |

This reinforces two likely bottlenecks for us:

1. tiny target detection at the catheter/guidewire tip;
2. strong phantom-to-animal domain shift.

## What the Existing Baselines Imply

The benchmark paper itself points to two main reasons the scores are low:

- the catheter/guidewire tip is very small in X-ray images;
- collision and normal classes are imbalanced.

It also suggests that future methods may need attention mechanisms,
transformers, foundation models, temporal information, graph models,
multi-modal learning, or transfer/domain adaptation.

## Methods Worth Considering First

Based on the CathAction benchmark results, the most relevant baselines are not
generic image classifiers. The task should start as YOLO-style object detection
or video object detection.

Priority candidates:

1. **EFF-style tiny object detector**
   - Best reported CathAction Task 2 result in the benchmark table.
   - Directly relevant because collision boxes are tiny tip boxes.

2. **Modern YOLO baseline**
   - Our local labels are already YOLO-style:
     `class x_center y_center width height`.
   - Even if the paper used Yolov, a modern YOLO implementation is likely the
     fastest reproducible first baseline.

3. **Temporal/video detector**
   - YOWO, YOWO-Plus, STEP, HIT are included because collision can depend on
     temporal context.
   - But their benchmark scores are below tiny object detectors, so temporal
     modeling alone is not enough.

4. **Task1-guided detector**
   - Not in the benchmark table, but plausible for our project.
   - Use Task1 segmentation/toolness to focus candidate regions around the
     guidewire/catheter tip.

5. **Domain-balanced training**
   - Needed if animal/phantom domain labels can be recovered.
   - The benchmark domain-adaptation table shows severe degradation.

## Negative Finding

Searches for terms such as:

- `CathAction collision detection paper`
- `CathAction collision_detection`
- `CathAction YOWO collision`
- `CathAction EFF collision detection`
- `CathAction object detection collision`
- `CathAction normal collision mAP`

did not reveal a later independent method paper directly using CathAction Task
2 collision detection. Most results point back to the original CathAction
benchmark paper, the dataset card, or summaries of the benchmark.

This means Task 2 is more open than Task 1: there is no obvious MSLNet-like
published method to directly follow.

## Immediate Recommendation

Do not start with a complex temporal transformer.

Start with a reproducible object-detection baseline:

1. inventory the local labels;
2. confirm class-id mapping for `collision` vs `normal`;
3. generate case/video-level splits;
4. train a modern YOLO-family detector;
5. evaluate AP/mAP with the same class split;
6. visualize failures;
7. only then add temporal context or Task1 segmentation priors.

The benchmark numbers are low enough that a strong, well-tuned tiny-object
detector with high-resolution input, appropriate anchors/assigner settings, and
domain-balanced training may already be competitive.

## Sources

- CathAction arXiv page:
  https://arxiv.org/abs/2408.13126
- CathAction HTML full text:
  https://ar5iv.org/html/2408.13126v2
- CathAction dataset card:
  https://huggingface.co/datasets/airvlab/CathAction
- CATHACTION 2026 challenge PDF:
  `datasets/343-CATHACTION_Endovascular_Intervention_Tool_Segmentation_and_Collision_Detection_2026-04-22T16-37-17.pdf`

## BibTeX

```bibtex
@misc{huang2024cathaction,
  title = {CathAction: A Benchmark for Endovascular Intervention Understanding},
  author = {Huang, Baoru and Vo, Tuan and Kongtongvattana, Chayun and Dagnino, Giulio and Kundrat, Dennis and Chi, Wenqiang and Abdelaziz, Mohamed and Kwok, Trevor and Jianu, Tudor and Do, Tuong and Le, Hieu and Nguyen, Minh and Nguyen, Hoan and Tjiputra, Erman and Tran, Quang and Xie, Jianyang and Meng, Yanda and Bhattarai, Binod and Tan, Zhaorui and Liu, Hongbin and Gan, Hong Seng and Wang, Wei and Yang, Xi and Wang, Qiufeng and Su, Jionglong and Huang, Kaizhu and Stefanidis, Angelos and Guo, Min and Du, Bo and Tao, Rong and Vu, Minh and Zheng, Guoyan and Zheng, Yalin and Vasconcelos, Francisco and Stoyanov, Danail and Elson, Daniel and Rodriguez y Baena, Ferdinando and Nguyen, Anh},
  year = {2024},
  eprint = {2408.13126},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2408.13126}
}

@misc{kopuklu2019yowo,
  title = {You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization},
  author = {Kopuklu, Okan and Wei, Xi and Rigoll, Gerhard},
  year = {2019},
  eprint = {1911.06644},
  archivePrefix = {arXiv}
}

@misc{yang2022yowoplus,
  title = {YOWO-Plus: An Incremental Improvement},
  author = {Yang, Junwei},
  year = {2022},
  eprint = {2210.11219},
  archivePrefix = {arXiv}
}

@inproceedings{yang2019step,
  title = {STEP: Spatio-Temporal Progressive Learning for Video Action Detection},
  author = {Yang, Xitong and Yang, Xiaodong and Liu, Ming-Yu and Xiao, Fanyi and Davis, Larry S. and Kautz, Jan},
  booktitle = {CVPR},
  year = {2019}
}

@inproceedings{faure2023hit,
  title = {Holistic Interaction Transformer Network for Action Detection},
  author = {Faure, Guillaume J. and Chen, Min-Hung and Lai, Shang-Hong},
  booktitle = {WACV},
  year = {2023}
}

@inproceedings{shi2023yolov,
  title = {Yolov: Making Still Image Object Detectors Great at Video Object Detection},
  author = {Shi, Yuheng and Wang, Naiyan and Guo, Xiaojie},
  booktitle = {AAAI},
  year = {2023}
}

@inproceedings{gong2021eff,
  title = {Effective Fusion Factor in FPN for Tiny Object Detection},
  author = {Gong, Yu and Yu, Xiaodong and Ding, Yuhang and Peng, Xue and Zhao, Jian and Han, Zhen},
  booktitle = {WACV},
  year = {2021}
}
```

## Verification Note

The direct CathAction Task 2 baseline table and task definition were verified
against the ar5iv HTML rendering of arXiv:2408.13126. The negative claim
(`no later independent direct CathAction Task 2 method paper found`) is a search
result, not a proof of nonexistence; it should be treated as current working
evidence as of 2026-06-04.

