---
paths:
  - "paper/**/*.tex"
  - "scripts/python/**/*.py"
---

# Project Knowledge Base: TRUST

## Notation Registry

| Rule | Convention | Example | Anti-Pattern |
|------|-----------|---------|-------------|
| Vectors | Bold lowercase | $\mathbf{x}$ | $x$ (scalar notation for vectors) |
| Matrices | Bold uppercase | $\mathbf{W}$ | $W$ (no bold) |
| Sets | Calligraphic | $\mathcal{D}$ | $D$ (plain) |
| Losses | Calligraphic L + subscript | $\mathcal{L}_{sup}$ | $L_{sup}$ (plain L) |
| Parameters | Greek letters | $\theta$, $\theta'$ | $w$, $w'$ |
| Landmarks | Indexed set | $\{l_i\}_{i=1}^{K}$ | $L$ (ambiguous) |

## Symbol Reference

| Symbol | Meaning | Introduced |
|--------|---------|------------|
| $\mathbf{x}$ | Input 3D CT volume | Method |
| $K$ | Number of landmarks (=10) | Method |
| $f_\theta$ | Student network | Method |
| $f_{\theta'}$ | Teacher network (EMA) | Method |
| $\alpha$ | EMA decay rate | Method |
| $\mathcal{L}_{sup}$ | Supervised landmark loss | Method |
| $\mathcal{L}_{con}$ | Uncertainty-aware consistency loss | Method |
| $\mathcal{L}_{topo}$ | Topology-guided graph loss | Method |
| $u_i$ | Per-landmark uncertainty | Method |
| MRE | Mean Radial Error (mm) | Experiments |
| SDR | Success Detection Rate (%) | Experiments |

## Paper Section Progression

| # | Section | Core Question | Key Notation | Key Method |
|---|---------|--------------|-------------|------------|
| 1 | Introduction | Why automate aortic root landmarks? | MRE, SDR | -- |
| 2 | Related Work | What exists? What's missing? | -- | -- |
| 3 | Method | How does TRUST work? | All symbols | Mean-Teacher + UG-HCO + GCN Refinement |
| 4 | Experiments | Does it work? How much does each part help? | MRE, SDR | Ablation, comparison |
| 5 | Discussion | Limitations? Clinical impact? | -- | -- |

## Terminology Consistency

| Preferred Term | Avoid | Context |
|----------------|-------|---------|
| aortic root | aortic valve root | Anatomical structure |
| landmark | keypoint (except when citing others) | Detection target |
| TAVI | TAVR (unless US context) | Procedure name |
| Mean Teacher | mean teacher, mean-teacher | Architecture name |
| semi-supervised | semisupervised, semi supervised | Learning paradigm |
| pseudo-label | pseudo label, pseudolabel | Generated label |
| uncertainty-aware | uncertainty aware | Modifier |

## Anti-Patterns (Don't Do This)

| Anti-Pattern | What Happened | Correction |
|-------------|---------------|-----------|
| Mixed notation for same concept | $L$ and $\mathcal{L}$ both used for loss | Always use $\mathcal{L}$ |
| Undefined abbreviation | "UG-HCO" used without expansion | Define on first use |
| Inconsistent metric reporting | MRE in some tables, MAE in others | Always use MRE |

## Python Script Pitfalls

| Bug | Impact | Fix |
|-----|--------|-----|
| matplotlib PDF backend missing fonts | Figures render with wrong fonts | Use `plt.rcParams['pdf.fonttype'] = 42` |
| seaborn style reset | Figures lose consistent styling | Set style at script top, not per-figure |
| DPI not set on save | Low-res figures in paper | Always pass `dpi=300` to `savefig()` |
