---
name: review-paper
description: Comprehensive manuscript review covering method novelty, experimental design, clinical relevance, reproducibility, and potential referee objections
disable-model-invocation: true
argument-hint: "[paper filename in master_supporting_docs/ or path to .tex/.pdf]"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Task"]
---

# Manuscript Review

Produce a thorough, constructive review of an academic manuscript -- the kind of report a top IEEE TMI referee would write.

**Input:** `$ARGUMENTS` -- path to a paper (.tex, .pdf), or a filename in `master_supporting_docs/`.

---

## Steps

1. **Locate and read the manuscript.** Check:
   - Direct path from `$ARGUMENTS`
   - `master_supporting_docs/supporting_papers/$ARGUMENTS`
   - `paper/$ARGUMENTS`
   - Glob for partial matches

2. **Read the full paper** end-to-end. For long PDFs, read in chunks (5 pages at a time).

3. **Evaluate across 6 dimensions** (see below).

4. **Generate 3-5 "referee objections"** -- the tough questions a top reviewer would ask.

5. **Produce the review report.**

6. **Save to** `quality_reports/paper_review_[sanitized_name].md`

---

## Review Dimensions

### 1. Argument Structure
- Is the research question clearly stated?
- Does the introduction motivate the question effectively?
- Is the logical flow sound (clinical need -> method -> results -> impact)?
- Are the conclusions supported by the evidence?
- Are limitations acknowledged?

### 2. Method Novelty
- Is the contribution clearly novel vs. existing methods?
- What specific gap does this method fill?
- Is the novelty overstated or understated?
- Are the key components (UG-HCO, GCN Refinement, or similar) well-motivated?

### 3. Experimental Design
- Are baselines fair and recent?
- Is the evaluation protocol standard for the field?
- Are ablation studies comprehensive (isolate each component)?
- Are error bars / statistical significance reported?
- Is cross-validation or train/test split appropriate?

### 4. Clinical Relevance
- Does the paper connect to actual clinical workflow?
- Are the metrics clinically meaningful (e.g., SDR at clinically relevant thresholds)?
- Would a clinician understand the implications?
- Is the dataset representative of clinical practice?

### 5. Reproducibility
- Is enough detail provided to reproduce the method?
- Are hyperparameters fully specified?
- Is the dataset described adequately (or publicly available)?
- Is the code available (or planned for release)?

### 6. Writing & Presentation
- Clarity and concision
- Academic tone
- Consistent notation throughout
- Figures and tables are self-contained (clear labels, captions)
- Paper is the right length for the contribution

---

## Output Format

```markdown
# Manuscript Review: [Paper Title]

**Date:** [YYYY-MM-DD]
**Reviewer:** review-paper skill
**File:** [path to manuscript]

## Summary Assessment

**Overall recommendation:** [Strong Accept / Accept / Revise & Resubmit / Reject]

[2-3 paragraph summary: main contribution, strengths, and key concerns]

## Strengths

1. [Strength 1]
2. [Strength 2]
3. [Strength 3]

## Major Concerns

### MC1: [Title]
- **Dimension:** [Novelty / Experiments / Clinical / Reproducibility / Writing]
- **Issue:** [Specific description]
- **Suggestion:** [How to address it]
- **Location:** [Section/page/table if applicable]

[Repeat for each major concern]

## Minor Concerns

### mc1: [Title]
- **Issue:** [Description]
- **Suggestion:** [Fix]

[Repeat]

## Referee Objections

These are the tough questions a top referee would likely raise:

### RO1: [Question]
**Why it matters:** [Why this could be fatal]
**How to address it:** [Suggested response or additional analysis]

[Repeat for 3-5 objections]

## Summary Statistics

| Dimension | Rating (1-5) |
|-----------|-------------|
| Argument Structure | [N] |
| Method Novelty | [N] |
| Experimental Design | [N] |
| Clinical Relevance | [N] |
| Reproducibility | [N] |
| Writing & Presentation | [N] |
| **Overall** | **[N]** |
```

---

## Principles

- **Be constructive.** Every criticism should come with a suggestion.
- **Be specific.** Reference exact sections, equations, tables.
- **Think like Reviewer 2 at IEEE TMI.** What would make them reject?
- **Distinguish fatal flaws from minor issues.** Not everything is equally important.
- **Acknowledge what's done well.** Good research deserves recognition.
- **Do NOT fabricate details.** If you can't read a section clearly, say so.
