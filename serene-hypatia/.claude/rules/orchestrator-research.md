---
paths:
  - "scripts/**/*.py"
  - "explorations/**"
---

# Research Project Orchestrator (Simplified)

**For Python scripts, figure generation, and data analysis** -- use this simplified loop instead of the full multi-agent orchestrator.

## The Simple Loop

```
Plan approved -> orchestrator activates
  |
  Step 1: IMPLEMENT -- Execute plan steps
  |
  Step 2: VERIFY -- Run code, check outputs
  |         Python scripts: python3 runs without error
  |         Seeds: reproducibility verified
  |         Plots: PDF/PNG created, correct format
  |         If verification fails -> fix -> re-verify
  |
  Step 3: SCORE -- Apply quality-gates rubric
  |
  +-- Score >= 80?
        YES -> Done (commit when user signals)
        NO  -> Fix blocking issues, re-verify, re-score
```

**No 5-round loops. No multi-agent reviews. Just: write, test, done.**

## Verification Checklist

- [ ] Script runs without errors
- [ ] All imports at top
- [ ] No hardcoded absolute paths
- [ ] Random seeds set explicitly if stochastic
- [ ] Output files created at expected paths
- [ ] Tolerance checks pass (if applicable)
- [ ] Quality score >= 80
