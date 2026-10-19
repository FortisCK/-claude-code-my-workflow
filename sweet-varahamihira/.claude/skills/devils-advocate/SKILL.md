---
name: devils-advocate
description: Challenge paper design with 5-7 specific questions. Checks argument structure, missing comparisons, experimental gaps, notation conflicts, and reviewer objections.
disable-model-invocation: true
argument-hint: "[paper section filename or topic description]"
allowed-tools: ["Read", "Grep", "Glob"]
---

# Devil's Advocate Review

Critically examine the paper and challenge its design with 5-7 specific questions.

**Philosophy:** "We arrive at the best possible paper through active dialogue."

---

## Setup

1. **Read the target file** (the paper section or full paper being challenged)
2. **Read the knowledge base** in `.claude/rules/` for notation conventions and method context
3. If applicable, **read related sections** for consistency

---

## Challenge Categories

Generate 5-7 challenges from these categories:

### 1. Argument Structure Challenges
> "Is the motivation for component X convincing enough for a skeptical reviewer?"

### 2. Missing Comparison Challenges
> "Should we compare against method Y? A reviewer will ask why it's absent."

### 3. Experimental Gap Challenges
> "What ablation is missing that a reviewer will request?"

### 4. Notation Conflict Challenges
> "This symbol conflicts with standard usage in the field."

### 5. Assumption Challenges
> "Is this assumption too strong? Would a weaker assumption suffice?"

### 6. Clinical Relevance Challenges
> "How does this translate to actual TAVI planning? A clinical reviewer will ask."

### 7. Reproducibility Challenges
> "Is there enough detail to reproduce this? What's missing?"

---

## Output Format

```markdown
# Devil's Advocate: [Section/Paper Title]

## Challenges

### Challenge 1: [Category] -- [Short title]
**Question:** [The specific question]
**Why it matters:** [What could go wrong / what a reviewer would say]
**Suggested resolution:** [Specific action]
**Section affected:** [Section name or number]
**Severity:** [High / Medium / Low]

[Repeat for 5-7 challenges]

## Summary Verdict
**Strengths:** [2-3 things done well]
**Critical changes:** [0-2 changes before submission]
**Suggested improvements:** [2-3 nice-to-have changes]
```

---

## Principles

- **Be specific:** Reference exact sections, equations, and claims
- **Be constructive:** Every challenge has a suggested resolution
- **Be honest:** If the paper is strong, say so
- **Think like Reviewer 2:** Where will they object?
- **Prioritize:** Missing experiments > notation issues
