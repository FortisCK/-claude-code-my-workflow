# Task 2 Decision: Sequence Tip Localizer Proposals

Date: 2026-06-11

## Question

Can the pretrained ConvNeXt/FPN sequence-aware tip localizer replace or
complement the current YOLO Stage2L coarse proposal detector?

## Implementation

Added `scripts/task2/evaluate_sequence_tip_proposals.py`.

The script:

- loads a `train_sequence_tip_localizer.py` checkpoint;
- reconstructs the model without downloading encoder weights;
- exports top-K heatmap peaks;
- converts each peak into proposal boxes using both predicted width/height and
  fixed original-pixel box templates;
- evaluates proposal recall with the same top-k IoU metrics used for YOLO;
- additionally records center recall at 5/10/20 px.

Added `scripts/task2/evaluate_proposal_csv_union.py`.

The script:

- reads saved `{split}_proposals.csv` files from multiple proposal sources;
- keeps each source's top-K candidates;
- reports source-wise oracle union recall, class-wise recall, and best-source
  counts.

## Verification

Commands completed:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python -m py_compile scripts/task2/evaluate_sequence_tip_proposals.py
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python -m py_compile scripts/task2/evaluate_proposal_csv_union.py
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest -q tests/test_task2_sequence_tip_localizer.py
```

Pytest result: `2 passed`.

Full proposal export:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python scripts/task2/evaluate_sequence_tip_proposals.py \
  --checkpoint outputs/task2/sequence_tip_localizer/pilot_sequence_tip384_smp_convnext_tiny_lr1e4_noamp_train4096_e6/checkpoints/best.pt \
  --name convnext_tip384_lr1e4_noamp_e3_top20_templates \
  --batch-size 16 \
  --workers 8 \
  --top-peaks 20 \
  --max-proposals 50 \
  --top-k 1,5,10,20,50 \
  --device auto
```

Outputs:

- `outputs/task2/sequence_tip_proposals/convnext_tip384_lr1e4_noamp_e3_top20_templates/summary_metrics.json`
- `outputs/task2/proposal_union_oracle/yolo_stage2l_plus_sequence_tip_top50/summary_metrics.json`
- `outputs/task2/proposal_union_oracle/yolo_stage2l_plus_geometry_top50/summary_metrics.json`
- `outputs/task2/proposal_union_oracle/yolo_stage2l_plus_geometry_plus_sequence_tip_top50/summary_metrics.json`

## Main Results

Standalone ConvNeXt/FPN sequence-tip proposals:

| Split | Top50 R@0.50 | Top50 R@0.75 | Mean Best IoU | Center@20 |
| --- | ---: | ---: | ---: | ---: |
| combined | 0.2352 | 0.0397 | 0.3490 | 0.7809 |
| phantom | 0.2658 | 0.0449 | 0.3499 | 0.7706 |
| animal | 0.0000 | 0.0000 | 0.3420 | 0.8598 |

The center localization signal is real, but the box proposals are too weak to
replace YOLO. The model finds approximate centers but does not produce reliable
IoU-level boxes.

Oracle union over source top50 candidates:

| Proposal Set | Combined R@0.50 | Combined R@0.75 | Phantom R@0.50 | Animal R@0.50 | Animal Class0 R@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO Stage2L | 0.5456 | 0.2170 | 0.5049 | 0.8598 | 0.0000 |
| YOLO + sequence-tip | 0.5661 | 0.2449 | 0.5279 | 0.8598 | 0.0000 |
| YOLO + geometry | 0.5661 | 0.2309 | 0.5158 | 0.9533 | 0.6667 |
| YOLO + geometry + sequence-tip | 0.5832 | 0.2567 | 0.5352 | 0.9533 | 0.6667 |

Class-wise R@0.50 for the final three-source oracle:

| Split | Class0 R@0.50 | Class1 R@0.50 |
| --- | ---: | ---: |
| combined | 0.7775 | 0.4187 |
| phantom | 0.7816 | 0.2888 |
| animal | 0.6667 | 1.0000 |

## Interpretation

The sequence-tip localizer should not be promoted to the primary coarse
detector. Its top50 IoU recall is far below YOLO because the output is closer to
a center heatmap than a box detector.

It is still useful as a supplemental proposal source. It improves YOLO on
phantom and improves the full multi-source oracle:

- YOLO -> YOLO + sequence-tip: combined R@0.50 +0.0205.
- YOLO + geometry -> YOLO + geometry + sequence-tip: combined R@0.50 +0.0171.
- Phantom R@0.50 rises from 0.5158 to 0.5352 in the three-source union.

It does not fix the animal class0 failure. That remains supplied by Task1
geometry, not by the sequence-tip localizer.

## Decision

Use the sequence-tip localizer as an optional third proposal source in the next
ROI verifier/ranker experiment.

Do not continue training or tuning this localizer as a standalone detector now.
The next highest-leverage step is a learned ROI verifier/ranker over a
multi-source candidate set:

1. YOLO Stage2L proposals as the main detector source.
2. Task1 geometry proposals as the animal class0 recovery source.
3. Sequence-tip localizer template proposals as a phantom/box-quality
   supplemental source.

The final deployable system still needs a ranking/classification stage; these
union numbers are oracle upper bounds, not final mAP.
