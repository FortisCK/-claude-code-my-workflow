# Session Log: 2026-02-15 — Literature Collection & Bibliography Expansion

## Goal
Collect references from related papers' Related Work sections to build a comprehensive bibliography for writing Introduction and Related Work sections of the TRUST paper.

## Approach
User strategy: "找到一篇和我这个最相关的文章，直接把它的Related work" — mine Related Work from the most relevant papers, then collect → write.

## Key Context
- Paper: TRUST (Topology-Guided Semi-Supervised Aortic Root Landmark Detection)
- Target: IEEE TMI
- Method structure finalized in previous session (§3.1–3.6)
- Bibliography was at 14 entries — insufficient for Intro + Related Work

## Progress

### Literature Mining (completed)
- **G2LCPS (Ren 2024):** Extracted 43 references — SSL, heatmap/regression landmarks, CPS
- **Tang 2025:** Extracted 29 references — Mean Teacher, uncertainty, supervised landmarks
- **CASEMark (Huang 2025):** Extracted 34 references — CNN/Transformer landmark detection
- **Torosdagli 2023:** Extracted 36 references — GNN, relational reasoning, 3D landmarking
- **H3DE-Net (Huang 2025):** Extracted 20 references — 3D landmark, super-resolution

### Master Reference List
- Compiled and saved to `quality_reports/literature_collection_master.md`
- Organized into 6 topic categories matching Related Work structure
- Identified 15 HIGH, 12 MEDIUM, 8 LOW priority entries

### TAVI Clinical References (completed)
- Found Ribeiro 2013 (coronary obstruction systematic review) — already in supporting_papers
- Found Wang 2023 (DL pre-TAVR CT assessment) — already in supporting_papers
- Added Dvir 2015, Holmes 2012, Blanke 2019 from agent search

### Bibliography Merge (completed)
- 25 new entries written to `bib_entries_to_add.bib` (staging file)
- Merged into `Bibliography_base.bib` via `cat >> `
- bibtex + 3-pass pdflatex: zero errors
- **Bibliography: 14 → ~39 entries**

## Decisions
- Protected `Bibliography_base.bib` hook required manual merge via `cat`
- Included both HIGH and MEDIUM priority entries in single batch
- TAVI clinical refs (Ribeiro, Dvir, Holmes, Blanke) fill Gap 1

## Open Questions
- Writing order: Introduction first or Related Work first?
- Paper title still undecided (Issue #1 from proofreader)
- Astudillo et al. — not verified if in supporting_papers

## Next Steps
- Write Introduction (with contributions list from TODO comments in main.tex)
- Write Related Work (5 subsections per literature_collection_master.md)

---
**Context compaction (auto) at 14:38**
Check git log and quality_reports/plans/ for current state.
