# Meta-Governance: CATHACTION Repository

This repository is now a working CATHACTION participant submission repo. It still contains reusable workflow infrastructure, but project-specific correctness takes precedence over preserving template genericity.

## Repository Identity

- **Working project:** CATHACTION code, experiments, Docker packaging, method report, paper figures, and optional slides.
- **Workflow infrastructure:** rules, agents, skills, hooks, templates, and memory that help the working project stay rigorous.

When these conflict, prefer the CATHACTION project need and document reusable lessons separately.

## What Belongs In Git

Commit:

- Project rules, plans, specs, session logs, and decision records.
- Code, tests, configs, Docker files, and reproducible scripts.
- Publicly shareable method reports, figures, and slides.
- Generic or project-relevant `[LEARN]` entries in `MEMORY.md`.

Do not commit:

- Private data, patient identifiers, hospital identifiers, credentials, tokens, or private download links.
- Machine-specific dataset paths or local workarounds.
- Hidden-test labels, manual hidden-test edits, or platform secrets.
- Large generated artifacts unless they are intentionally part of a release or docs output.

## Memory Management

Use `MEMORY.md` for corrections and decisions that should persist across sessions:

```markdown
[LEARN:category] wrong -> right
```

Use session logs for narrative context and `quality_reports/decisions/` for explicit source-of-truth or design choices.

## Dogfooding Rules

- Non-trivial work starts with a saved plan.
- Complex or ambiguous work gets a requirements spec first.
- Every completed task reports verification.
- README, guide, and docs should move together when public-facing workflow changes.
- Challenge facts must trace to the 2026 PDF or a documented newer official source.

## Source-Of-Truth Changes

If the official challenge platform supersedes the 2026 PDF:

1. Create a decision record in `quality_reports/decisions/`.
2. Update `CLAUDE.md`, `AGENTS.md`, README, guide/docs, and `.claude/rules/knowledge-base-template.md`.
3. Scan for old counts or metric definitions.
4. Record the change in the session log.

## Amendment Process

For governance changes:

1. Propose the change in a plan or session log.
2. Apply the change to the relevant config files.
3. Verify with drift scans.
4. Add a `[LEARN]` entry when the lesson should persist.
