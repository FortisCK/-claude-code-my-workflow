---
name: figure-generation
description: Python figure generation workflow for publication-ready plots and diagrams
disable-model-invocation: true
argument-hint: "[description of figure to generate or script path]"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Edit", "Bash", "Task"]
---

# Figure Generation Workflow

Generate publication-ready figures using Python (matplotlib/seaborn).

**Input:** `$ARGUMENTS` -- a description of the figure to create, or a path to an existing script to run.

---

## Constraints

- **Follow Python conventions** in `.claude/rules/python-conventions.md`
- **Save all scripts** to `scripts/python/` with descriptive names
- **Save all figures** to `figs/`
- **Use project figure standards** from `.claude/rules/figure-quality.md`
- **Run python-reviewer** on the generated script before presenting results

---

## Workflow Phases

### Phase 1: Setup

1. Read `.claude/rules/python-conventions.md` and `.claude/rules/figure-quality.md`
2. Create Python script with proper header
3. Set up imports and random seeds
4. Set up output directory: `os.makedirs('figs', exist_ok=True)`

### Phase 2: Script Structure

```python
#!/usr/bin/env python3
"""
[Descriptive title]
Purpose: [What this script generates]
Outputs: figs/[figure_name].pdf, figs/[figure_name].png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # Editable text in PDFs

# Configuration
SEED = 42
DPI = 300
SINGLE_COL_WIDTH = 3.5  # inches (IEEE)
DOUBLE_COL_WIDTH = 7.16  # inches (IEEE)

np.random.seed(SEED)
os.makedirs('figs', exist_ok=True)

# [Figure generation code]

# Save
plt.savefig('figs/figure_name.pdf', dpi=DPI, bbox_inches='tight', format='pdf')
plt.savefig('figs/figure_name.png', dpi=DPI, bbox_inches='tight', format='png')
plt.close()
```

### Phase 3: Generate and Verify

1. Run the script: `python3 scripts/python/[script_name].py`
2. Verify output files exist and are non-zero size
3. Check image dimensions match IEEE column widths

### Phase 4: Review

Run the python-reviewer agent on the script:
```
Delegate to the python-reviewer agent:
"Review the script at scripts/python/[script_name].py"
```

Address any Critical or High issues.

---

## Important

- **Use colorblind-safe palettes** (seaborn 'colorblind' or Okabe-Ito)
- **Font sizes:** minimum 7pt at final print size
- **Vector format** (PDF) for all line art, diagrams, and plots
- **PNG** only for CT images, heatmaps, or other raster content
- **No hardcoded paths.** Use relative paths from repo root.
