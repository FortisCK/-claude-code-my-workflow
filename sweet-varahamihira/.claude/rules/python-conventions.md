---
paths:
  - "scripts/**/*.py"
---

# Python Code Conventions

## Reproducibility

- Set random seeds explicitly at the top of every script:
  ```python
  import random, numpy as np
  random.seed(42)
  np.random.seed(42)
  ```
- All imports at the top of the file (stdlib, third-party, local)
- Use relative paths from repo root only; never hardcode absolute paths
- Use `os.makedirs(..., exist_ok=True)` for output directories

## Function Design

- `snake_case` for functions and variables
- Type hints for function signatures
- Google-style docstrings for non-trivial functions
- Default parameters for configurable values (DPI, colors, sizes)
- No magic numbers -- define constants at module level

## Figure Output Standards

- Always save with explicit parameters:
  ```python
  plt.savefig(path, dpi=300, bbox_inches='tight', format='pdf')
  ```
- Use `plt.rcParams['pdf.fonttype'] = 42` for editable text in PDFs
- Set seaborn/matplotlib style once at script top
- Use colorblind-safe palette consistently
- Single-column figures: 3.5 in wide; double-column: 7.16 in wide

## Quality Checklist

```
[ ] Imports at top (stdlib / third-party / local)
[ ] Seeds set if stochastic
[ ] No hardcoded absolute paths
[ ] Functions have type hints and docstrings
[ ] Figures saved with dpi=300, bbox_inches='tight'
[ ] PDF fonttype set to 42
[ ] Output directory created with exist_ok=True
```
