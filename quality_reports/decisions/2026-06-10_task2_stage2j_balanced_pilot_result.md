# Task 2 Stage2J Balanced Pilot Result

Date: 2026-06-10

## Experiment

Evaluated the Stage2J balanced pilot model:

- Model: `outputs/task2/yolo_stage2j_balanced/yolo11s_1024_pilot_v1_val_v2_e15/weights/best.pt`
- Training split: `pilot_train_v1_val_v2`
- Training composition:
  - 10,000 phantom samples
  - `video_1_animal` repeated 50 times
- Animal validation: `video_2_animal`
- Additional phantom checks:
  - balanced-small phantom panel
  - full `valid_phantom`

## Animal Validation Result

Validation target: `configs/task2/collision_detection_stage2j_pilot_train_v1_val_v2_animal.local.yaml`

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 107 | 107 | 0.104 | 0.484 | 0.134 | 0.026 |
| normal | 15 | 15 | 0.000 | 0.000 | 0.000 | 0.000 |
| collision | 92 | 92 | 0.209 | 0.967 | 0.267 | 0.052 |

## Phantom Balanced-Small Result

Validation target: `configs/task2/collision_detection_stage2j_pilot_train_v1_val_v2_valid_phantom_balanced_small.local.yaml`

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 824 | 824 | 0.118 | 0.0947 | 0.0735 | 0.0224 |
| normal | 412 | 412 | 0.236 | 0.189 | 0.145 | 0.0443 |
| collision | 412 | 412 | 0.000 | 0.000 | 0.00192 | 0.000561 |

## Phantom Full Result

Validation target: `configs/task2/collision_detection_stage2j_pilot_train_v1_val_v2_valid_phantom_full.local.yaml`

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 11,024 | 11,024 | 0.159 | 0.0912 | 0.0747 | 0.0242 |
| normal | 10,612 | 10,612 | 0.317 | 0.182 | 0.149 | 0.0483 |
| collision | 412 | 412 | 0.000 | 0.000 | 0.000102 | 0.0000305 |

## Interpretation

The simple balanced-list strategy did not solve the Task 2 detector problem.

Compared with Stage2H naive animal adaptation, the balanced pilot is worse on
the held-out animal collision class:

- Stage2H collision mAP50 on `video_2_animal`: 0.973
- Stage2J balanced collision mAP50 on `video_2_animal`: 0.267

It also fails to preserve phantom-domain collision performance:

- phantom full collision mAP50: 0.000102
- phantom balanced-small collision mAP50: 0.00192

Normal detection remains effectively absent on animal validation and weak on
phantom validation.

## Decision

Do not continue this exact Stage2J recipe as the main Task 2 route.

The next useful direction should be a structured two-stage design:

1. make the detector/proposal stage class-agnostic or collision-focused and
   optimize for recall rather than direct class AP;
2. use a local ROI verifier/classifier for normal-vs-collision judgment;
3. keep domain balancing as a sampler or model-selection tool, not as a simple
   repeated-list substitute for a robust proposal design.
