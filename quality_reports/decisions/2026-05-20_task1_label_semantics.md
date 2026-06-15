# Decision: Task 1 Label Semantics

**Date:** 2026-05-20
**Scope:** CATHACTION Task 1 segmentation masks
**Status:** Active with provisional class-name mapping

---

## Evidence Checked

- Local MICCAI 2026 PDF:
  `datasets/343-CATHACTION_Endovascular_Intervention_Tool_Segmentation_and_Collision_Detection_2026-04-22T16-37-17.pdf`
- CATHACTION public segmentation page:
  <https://airvlab.github.io/cathaction/docs/segmentation/>
- Hugging Face dataset README:
  <https://huggingface.co/datasets/airvlab/CathAction/blob/main/README.md>
- Local extracted masks under `datasets/segmentation/`.
- Local diagnostic overlays generated from released eval samples:
  - `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/diagnostics/label_semantics_contact_sheet.png`
  - `outputs/task1/baseline_unet_multiclass_mild_weighted_gpu_pilot/diagnostics/label_semantics_zoom_sheet.png`
- CathAction paper figure caption, arXiv v2:
  <https://arxiv.org/abs/2408.13126>

The MICCAI PDF explicitly states that Task 1 masks distinguish catheter and
guidewire as separate classes. The public website and Hugging Face README
confirm the segmentation task but do not provide a numeric label map for mask
values.

The CathAction paper caption for Figure 1 states that its visualization uses
blue for catheter and dark yellow for guidewire. That color map is a paper
visualization, not a numeric `.npy` label map.

---

## Local Label Inventory

Animal and phantom masks are `.npy` arrays with values `0,1,2`.

Full pixel-count scan:

| Collection | Files | Label 1 pixels | Label 2 pixels | Label 1 foreground fraction |
| --- | ---: | ---: | ---: | ---: |
| `animal_train` | 4,021 | 4,926,859 | 1,492,425 | 0.7675 |
| `animal_test` | 1,006 | 1,217,178 | 362,067 | 0.7707 |
| `phantom_train` | 14,737 | 52,465,596 | 6,854,851 | 0.8844 |
| `phantom_test` | 3,685 | 12,907,528 | 1,718,795 | 0.8825 |

Human masks are binary PNGs with observed values `0,255`.

---

## Decision

For animal/phantom segmentation, support `multiclass_012` as the challenge-aligned
label mode:

- `0`: background
- `1`: provisional catheter class
- `2`: provisional guidewire class

Basis for the provisional mapping:

- In local GT overlays, label `1` is usually the larger/thicker proximal or
  main tool segment.
- Label `2` is usually the thinner distal continuation, and some frames contain
  label `2` without label `1`.
- This morphology is consistent with catheter vs. guidewire behavior and with
  the CathAction paper's statement that catheter and guidewire are separately
  visualized.

Do not treat this as an official numeric mapping until the organizers,
official platform, or a reliable dataset label map confirms it.

Keep `binary_foreground` available for:

- smoke tests;
- compatibility with human binary masks;
- quick foreground-only ablations.

Do not mix human binary masks into `multiclass_012` training without a documented
conversion strategy.

---

## Consequence

The next baseline step should train and evaluate three-channel outputs on
released animal/phantom manifests using numeric classes `0/1/2`. Reports can
state metrics for `label_1` and `label_2`. Internal notes may describe the
working interpretation as `label_1 ~= catheter` and `label_2 ~= guidewire`, but
external reports should mark that mapping as provisional unless confirmed by an
official label map.
