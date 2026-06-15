# Task2 Stage2O Candidate Pool Audit Result

Date: 2026-06-11
Plan: `quality_reports/plans/2026-06-11_task2-stage2o-aprime-candidate-ranker.md`
Status: Phase 1/2 completed; Phase 3 ranker is justified by candidate-recall gates.

## Objective

Stage2O tests whether the existing proposal sources have enough complementary localization recall to justify a learned background-aware ROI ranker. This step does not train a new model. It only builds a unified candidate CSV and audits oracle candidate coverage.

## Implemented Artifact

Added:

- `scripts/task2/build_stage2o_candidate_pool.py`
- `tests/test_task2_stage2o_candidate_pool.py`

Main output:

- `outputs/task2/stage2o_candidate_pool/stage2o_aprime_default_sources_top50/valid_combined_candidates.csv`
- `outputs/task2/stage2o_candidate_pool/stage2o_aprime_default_sources_top50/valid_phantom_candidates.csv`
- `outputs/task2/stage2o_candidate_pool/stage2o_aprime_default_sources_top50/valid_animal_candidates.csv`
- `outputs/task2/stage2o_candidate_pool/stage2o_aprime_default_sources_top50/summary_audit.json`
- `outputs/task2/stage2o_candidate_pool/stage2o_aprime_default_sources_top50/candidate_pool_audit.md`

The unified CSV keeps these fields: sample/video/frame/domain identifiers, image/label paths, image size, GT class and box, source name/priority/rank/confidence, candidate box, candidate IoU, matched flag, and verifier label (`background`, `normal`, `collision`, `ignore`).

## Candidate Sources

The default frozen sources were used:

1. `yolo_stage2l`: `outputs/task2/yolo_proposal_eval/yolo11s_1024_stage2l_combined_bal_e20_stage2l_panels`
2. `task1_geometry_rect`: `outputs/task2/geometry_proposals/task1_stage5_geometry_d3_rect_stage2l_panels`
3. `sequence_tip`: `outputs/task2/sequence_tip_proposals/convnext_tip384_lr1e4_noamp_e3_top20_templates`
4. `yolov_pilot`: `outputs/task2/yolov_proposal_eval/yolov_s_576_agnostic_pilot_e3_fullval`

Each source contributes up to top 50 candidates per sample. The audit reports both all-source oracle coverage and global top-K coverage by fixed source order.

## Key Results

| Split | Samples | Rows | Union R@0.50 | Union R@0.75 | Stage2L baseline R@0.50 | Source-order top50 R@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid_combined | 931 | 119,990 | 0.5789 | 0.2771 | 0.5456 | 0.5553 |
| valid_phantom | 824 | 107,354 | 0.5316 | 0.3131 | 0.5049 | 0.5049 |
| valid_animal | 107 | 12,653 | 0.9533 | 0.0000 | 0.8598 | 0.9439 |

Domain/class detail:

| Group | Samples | Union R@0.50 | Union R@0.75 | Stage2L baseline R@0.50 | Source-order top50 R@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| animal:0 | 15 | 0.6667 | 0.0000 | 0.0000 | 0.6000 |
| animal:1 | 92 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| phantom:0 | 412 | 0.7816-0.7840 | 0.4782 | 0.7354 | 0.7354 |
| phantom:1 | 412 | 0.2791 | 0.1481 | 0.2743 | 0.2743 |

Best-source counts on `valid_combined`:

- `yolo_stage2l`: 554
- `yolov_pilot`: 184
- `sequence_tip`: 105
- `task1_geometry_rect`: 88

The best-source counts show that the added sources are not redundant: geometry, sequence-tip, and YOLOV each supply the best-overlap candidate for a non-trivial number of frames.

## Gate Check

Predefined Phase3 go gates:

- Combined R@0.50 gain vs Stage2L baseline >= +0.02: PASS (`+0.0333`)
- Animal class0 R@0.50 >= 0.50: PASS (`0.6667`)
- Phantom R@0.50 no collapse vs baseline: PASS (`0.5316` vs `0.5049`)

Decision: GO for Phase 3 CSV-driven ROI verifier/ranker.

## Interpretation

The audit validates the Stage2O direction: multi-source proposals recover missing candidate coverage, especially animal class0, where Stage2L alone had R@0.50 = 0.0000 and the union reaches 0.6667.

However, this is still only candidate-pool upper bound. It does not prove that a learned ranker can select the correct candidate. It also exposes a clear remaining localization-quality issue:

- Animal R@0.75 is 0.0000 even though animal R@0.50 is 0.9533.
- Phantom class1 remains weak: R@0.50 = 0.2791, R@0.75 = 0.1481.

This means Phase3 should train the ranker, but Phase4 box refinement is likely needed if mAP50-95 remains low after ranking.

## Verification

Commands run:

```bash
python3 -m py_compile scripts/task2/build_stage2o_candidate_pool.py
python3 scripts/task2/build_stage2o_candidate_pool.py --name smoke_stage2o_default_sources_top5 --source-top-k 5 --global-top-k 1,5,10 --output-dir /tmp/stage2o_candidate_pool_smoke
python3 scripts/task2/build_stage2o_candidate_pool.py --name stage2o_aprime_default_sources_top50
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile scripts/task2/build_stage2o_candidate_pool.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_stage2o_candidate_pool.py
```

Result:

- `py_compile`: PASS
- smoke candidate audit: PASS
- full Stage2O candidate audit: PASS
- `tests/test_task2_stage2o_candidate_pool.py`: PASS, 5 tests

Note: base Python did not have `pytest`; tests were run in the existing `cathaction-task1` conda environment.

## Next Step

Implement Phase 3: CSV-driven background-aware ROI verifier/ranker using the Stage2O candidate CSVs. The first ranker should be judged by detector-style mAP, not ROI classification accuracy.
