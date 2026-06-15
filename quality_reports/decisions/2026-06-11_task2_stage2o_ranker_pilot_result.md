# Task2 Stage2O Ranker Pilot Result

Date: 2026-06-11
Plan: `quality_reports/plans/2026-06-11_task2-stage2o-aprime-candidate-ranker.md`
Status: Phase 3 pilot completed; ranking helps mAP50, but mAP50-95 remains box-quality limited.

## Objective

Train a CSV-driven background-aware ROI ranker without validation leakage. The first attempt trained on old Stage2K YOLO-only candidates and evaluated on all-source Stage2O candidates. It failed because the train candidate distribution did not match validation sources. We then generated a matched YOLO+geometry train candidate subset and repeated the pilot.

## New Artifacts

Code:

- `scripts/task2/prepare_stage2o_ranker_train_subset.py`
- `scripts/task2/train_stage2o_candidate_ranker.py`

Train subset:

- `configs/task2/splits_stage2o_ranker/train_v0_v1_val_v2_animal_all_phantom1000pc_labels.txt`
- `configs/task2/splits_stage2o_ranker/train_v0_v1_val_v2_animal_all_phantom1000pc_summary.json`

Train candidate sources:

- YOLO: `outputs/task2/yolo_proposal_eval/stage2o_train_subset_yolo_stage2l_top50/valid_combined_proposals.csv`
- Geometry: `outputs/task2/geometry_proposals/stage2o_train_subset_task1_geometry_d3_rect/valid_combined_proposals.csv`

Candidate pools:

- Train YOLO+geometry: `outputs/task2/stage2o_candidate_pool/stage2o_train_subset_yolo_geometry_top50/valid_combined_candidates.csv`
- Valid YOLO+geometry: `outputs/task2/stage2o_candidate_pool/stage2o_valid_yolo_geometry_top50/summary_audit.json`

Ranker:

- Training run: `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_valid256_e4`
- Full eval: `outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/eval_metrics.json`

## Train Subset

The non-leakage train subset was sampled from `train_v0_v1_val_v2_train`.

| Group | Samples |
| --- | ---: |
| animal:0 | 111 |
| animal:1 | 180 |
| phantom:0 | 1000 |
| phantom:1 | 1000 |
| total | 2291 |

This fixes the previous pilot's biggest defect: the old train candidate CSV had only 26 animal:0 candidates and no geometry/sequence-like candidate distribution.

## Candidate Pool Checks

### Train YOLO+Geometry Candidate Pool

`stage2o_train_subset_yolo_geometry_top50`:

- samples: 2291
- rows: 163,456
- union R@0.50: 0.9838
- union R@0.75: 0.8018
- source-order top50 R@0.50: 0.9830

### Valid YOLO+Geometry Candidate Pool

`stage2o_valid_yolo_geometry_top50`:

| Split | Samples | Rows | Union R@0.50 | Union R@0.75 | Stage2L baseline R@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| valid_combined | 931 | 66,985 | 0.5585 | 0.2213 | 0.5456 |
| valid_phantom | 824 | 60,869 | 0.5073 | 0.2500 | 0.5049 |
| valid_animal | 107 | 6,116 | 0.9533 | 0.0000 | 0.8598 |

Interpretation: YOLO+geometry is useful mainly because it recovers animal class0 candidates. It does not provide the full all-source union gain, and it still has weak high-IoU box quality.

## Failed Pilot: Old YOLO-only Train CSV

Training on `outputs/task2/roi_verifier/convnext_tiny_stage2h_proposals_v1_val_v2_spdc1000_e8/train_candidates.csv` and validating on all-source candidates produced poor results:

- best valid256 combined `source_roi` mAP50-95: 0.0011
- the model almost never predicted background;
- training had only YOLO/GT candidates, while validation was dominated by geometry and sequence-tip candidates.

Decision: do not use that old train CSV for Stage2O ranker training.

## Matched YOLO+Geometry Ranker Pilot

Training:

- model: ConvNeXt-Tiny ROI ranker
- classes: background / normal / collision
- train candidates: YOLO+geometry train subset
- valid candidates: YOLO+geometry valid pool
- best checkpoint: epoch 3
- primary metric for checkpoint selection: valid256 combined `rank_decay_roi` mAP50-95
- score mode that worked best: `rank_decay_roi`

### Valid256 Best Checkpoint

| Split | rank_decay mAP50 | rank_decay mAP50-95 |
| --- | ---: | ---: |
| valid_combined | 0.5356 | 0.0778 |
| valid_phantom | 0.2786 | 0.0925 |
| valid_animal | 0.4609 | 0.0467 |

This showed the ranker can learn useful background rejection and ranking when train/valid source distributions match.

### Full Valid Evaluation

Same checkpoint evaluated on full YOLO+geometry valid panels:

| Split | rank_decay mAP50 | rank_decay mAP50-95 | loc R@0.50 | loc R@0.75 |
| --- | ---: | ---: | ---: | ---: |
| valid_combined | 0.1887 | 0.0388 | 0.5585 | 0.2213 |
| valid_phantom | 0.1022 | 0.0340 | 0.5073 | 0.2500 |
| valid_animal | 0.4609 | 0.0467 | 0.9533 | 0.0000 |

Class-wise full valid combined:

| Class | AP50 | AP50-95 |
| --- | ---: | ---: |
| class0 | 0.1548 | 0.0507 |
| class1 | 0.2226 | 0.0269 |

Candidate classification on full combined:

- accuracy: 0.7785
- background recall: 0.7758
- normal recall: 0.8068
- collision recall: 0.9640

## Gate Check

Original Stage2O Phase3 gates:

- Combined mAP50 >= 0.13: PASS (`0.1887`)
- Combined mAP50-95 >= 0.055: FAIL (`0.0388`)
- Class1/collision AP50-95 >= 0.010: PASS (`0.0269`)
- Animal class0 candidate-stage R@0.50 >= 0.50: PASS from candidate audit (`0.6667`)

Decision: Phase3 ranker is useful but not sufficient. The main remaining bottleneck is box quality, not ROI classification.

## Interpretation

This is the first Stage2O result that looks structurally sane:

- The ranker now predicts background and rejects many bad proposals.
- Full combined mAP50 improves to 0.1887, clearly above the YOLO checkpoint's own mAP50 of about 0.112.
- Full combined mAP50-95 remains 0.0388, roughly limited by candidate IoU and close to the Stage2L YOLO mAP50-95 region.

The mAP50 vs mAP50-95 gap is diagnostic:

- At IoU 0.50, candidate ranking/classification can help.
- At stricter IoU thresholds, the boxes are not tight enough.
- Animal remains especially box-limited: R@0.50 = 0.9533, R@0.75 = 0.0000.

## Decision

Do not spend the next iteration mainly on a bigger ranker. Move to Phase4 box refinement.

Recommended next step:

1. Train a local box refiner on the YOLO+geometry candidate pool.
2. Use the ranker as a scorer after refinement.
3. Optimize/refine for mAP50-95, not just mAP50.

The box refiner should target candidate boxes with IoU >= 0.20 and learn center offset plus width/height correction. It should be evaluated by matched-candidate IoU improvement, R@0.75 gain, and final detector-style mAP50-95.

## Verification

Commands run:

```bash
python3 -m py_compile scripts/task2/prepare_stage2o_ranker_train_subset.py
python3 scripts/task2/prepare_stage2o_ranker_train_subset.py
python3 -m py_compile scripts/task2/build_stage2o_candidate_pool.py scripts/task2/train_stage2o_candidate_ranker.py scripts/task2/prepare_stage2o_ranker_train_subset.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2o_candidate_pool.py tests/test_task2_stage2o_ranker.py
```

GPU commands were run outside the sandbox because the sandbox cannot access the NVIDIA driver, while non-sandbox `nvidia-smi` and PyTorch CUDA work.

Result:

- compile checks: PASS
- Stage2O unit tests: PASS, 8 tests
- train subset generation: PASS
- YOLO train proposal export: PASS
- geometry train proposal export: PASS for `valid_combined`; duplicate second split was stopped intentionally after the needed CSV was written
- YOLO+geometry train candidate pool build: PASS
- YOLO+geometry valid candidate pool build: PASS
- ranker short pilot: PASS
- full eval-only run: PASS
