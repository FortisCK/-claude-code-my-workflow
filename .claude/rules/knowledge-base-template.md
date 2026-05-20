---
paths:
  - "Slides/**/*.tex"
  - "Quarto/**/*.qmd"
  - "scripts/**/*.R"
  - "scripts/**/*.py"
  - "src/**/*.py"
  - "*.tex"
  - "*.md"
---

# CATHACTION Knowledge Base

This file gives agents project facts before they create, review, or modify CATHACTION artifacts.

## Project Identity

| Field | Value |
|-------|-------|
| Project | CATHACTION: Endovascular Intervention Tool Segmentation and Collision Detection |
| Institution | Université de Rennes |
| Role | MICCAI 2026 participant submission repository |
| Primary outputs | Challenge code, Docker submission, final method report/paper, publication-ready visuals |

## Challenge Source Registry

| Source | Status | Notes |
|--------|--------|-------|
| 2026-04-22 CATHACTION MICCAI PDF | Authoritative | Use for task definitions, dataset counts, schedule, policy, and metrics |
| https://airvlab.github.io/cathaction/ | Background / older public site | Currently shows older counts; do not use as the final 2026 spec unless updated |

## Task Registry

| Task | Target | Labels | Primary Metric | Secondary Metrics |
|------|--------|--------|----------------|-------------------|
| Task 1 | Catheter and guidewire segmentation in X-ray fluoroscopy | Pixel-level masks; catheter and guidewire distinguished | DSC | IoU/Jaccard, mIoU, pixel-wise accuracy |
| Task 2 | Collision detection in endovascular intervention | Frame-level collision labels / event confidence outputs per official schema | mAP | AP |

## Dataset Registry

| Quantity | Working Fact |
|----------|--------------|
| Cases | 650 videos/procedures |
| Domains | Silicon vascular phantom, preclinical animal X-ray, real human X-ray |
| Task 1 size | Approximately 40,000 annotated frames |
| Task 2 size | Approximately 600,000 annotated frames |
| Split | 70% train, 15% validation, 15% hidden test |
| Split unit | Procedure/case level |

## Notation Registry

| Symbol / Term | Meaning | Notes |
|---------------|---------|-------|
| DSC | Dice Similarity Coefficient | Primary Task 1 ranking metric |
| IoU / Jaccard | Intersection over Union | Secondary Task 1 metric |
| mIoU | mean Intersection over Union | Secondary Task 1 metric |
| AP | Average Precision | Complementary Task 2 metric |
| mAP | mean Average Precision | Primary Task 2 ranking metric |
| Domain | Phantom, animal, or human acquisition environment | Report stratified performance when possible |
| Case | One fluoroscopy-guided catheterization procedure/video | Split and evaluate without cross-case leakage |

## Data And Submission Policy

| Policy | Rule |
|--------|------|
| External data | Allowed only if publicly available at challenge launch |
| Private clinical data | Prohibited |
| Additional annotations | Allowed on public data only, not derived from private sources |
| Assessment interaction | No user interaction during algorithm assessment |
| Submission | Docker container, predefined result file, short method description/report |
| Hidden test | Labels unavailable; no manual tuning or reverse engineering |

## Review Priorities

| Priority | What To Check |
|----------|---------------|
| Leakage | Case-level split, no temporal/frame bleed across train/val/test |
| Metric fidelity | DSC for segmentation ranking; mAP for collision ranking |
| Domain shift | Phantom-to-animal-to-human generalization and failure modes |
| Clinical realism | Avoid overstating clinical deployment from benchmark evidence |
| Reproducibility | Configs, seeds, checkpoints, Docker path, output provenance |
| Visual polish | Paper-quality figures with explicit metric/domain/sample context |

## Anti-Patterns

| Anti-Pattern | Why It Is Bad | Correction |
|--------------|---------------|------------|
| Frame-random train/validation split | Leaks procedure context and inflates validation metrics | Split by video/procedure/case |
| Reporting AP as the final Task 2 ranking | Challenge primary metric is mAP | Report mAP as primary, AP as secondary |
| Mixing old website counts with 2026 PDF counts | Creates inconsistent method/report facts | Cite the PDF or document a newer official source |
| Manual hidden-test prediction fixes | Violates no-interaction evaluation | Keep inference fully automated |
| Figure without domain/sample label | Makes robustness claims impossible to audit | Add caption/legend with domain and split |
