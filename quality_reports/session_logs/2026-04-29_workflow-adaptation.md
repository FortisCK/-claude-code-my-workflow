# Session log — 2026-04-29 — Workflow adaptation for the cardiac-CT diffusion project

**Date:** 2026-04-29
**Author:** CMZ + Claude (Opus 4.7)
**Plan:** [`../plans/wondrous-honking-gray.md`](../plans/wondrous-honking-gray.md) (APPROVED)
**Status:** COMPLETED (pending claim-verifier outcome on bibliography)

---

## 高层目标

把这个 fork 自 `pedrohcgs/claude-code-my-workflow` 的模板仓库,从一个为
**Beamer / Quarto 讲义 + R 分析** 优化的形态,适配到一个为
**3D conditional latent-diffusion 心脏 CT 运动伪影校正研究项目** 优化的形态。
不动任何模型代码或数据 —— 这只是工作流配置层的改造。

## Approach

User answered four scoping questions:
1. **Moderate prune** — archive lecture/R surface to `.claude/archive/`
2. **Python primary** — add `/review-python` + `python-reviewer` + `python-code-conventions.md`
3. **Bilingual internal docs** — English in `.claude/` and shipped artefacts; 中文+English-terms in user-authored docs
4. **Paper + code primary** — create `manuscript/`, `code/`, `experiments/`, `data/` skeletons

The plan ([`wondrous-honking-gray.md`](../plans/wondrous-honking-gray.md)) decomposed this into 9 implementation steps.

## What changed

### File operations

- **Archived to `.claude/archive/`:** 14 skills, 8 agents, 10 rules, the 6-script R analysis pipeline, both HelloWorld samples, and the upstream template's MEMORY.md.
- **Created skeletons:** `manuscript/{miccai2027,cvpr2027,tmi-extension}/`, `code/{data,models,training,evaluation,inference,notebooks}/`, `experiments/runs/`, `data/`, `scripts/python/` — all with convention READMEs (where applicable) or `.gitkeep`.
- **Relocated:** `simple_plan.html` → `quality_reports/decisions/2026-04-29_research-direction-v2.html` + a markdown wrapper that summarises V1→V2 diffs and links the HTML.

### Newly authored

- [`.claude/rules/python-code-conventions.md`](../../.claude/rules/python-code-conventions.md) — PyTorch + MONAI standards (reproducibility, GPU-memory hygiene, MONAI idioms, numerical discipline, run-card linkage).
- [`.claude/rules/experiments-protocol.md`](../../.claude/rules/experiments-protocol.md) — run-card discipline (intent before, outcome after; required fields).
- [`.claude/skills/review-python/SKILL.md`](../../.claude/skills/review-python/SKILL.md) — Python analogue of the archived `/review-r`.
- [`.claude/agents/python-reviewer.md`](../../.claude/agents/python-reviewer.md) — agent backing the skill; 12-category review protocol.
- [`.claude/archive/README.md`](../../.claude/archive/README.md) — index of dormant template surface + how to promote items back.
- New `CLAUDE.md` (rewritten from placeholders → project-specific).
- New `MEMORY.md` (5 seed `[LEARN]` entries on project framing / scope / bilingual policy / reproducibility).
- Seeded `Bibliography_base.bib` with 9 anchor entries (TT U-Net, HM-EDM ×2, ProDM, C2F-MC, DPS, ATOM, TW-MoCoNet, TAVI clinical anchor — all marked `% TODO: verify`).
- This session log + the markdown wrapper for the V2 strategic brief.

### Edited

- `.claude/WORKFLOW_QUICK_REF.md` — placeholders filled in (paths, seeds, run cards, figure standards-TBD, tolerance-TBD, language policy, reporting preference, "first ~5 sessions: check in more").

## Key decisions (locked in)

- **Archive over delete.** Lecture / R surface stays recoverable via `git mv`.
- **`code/` over `src/`.** Matches the simple_plan's references and the R-pipeline analogue.
- **`manuscript/<target>/` per submission venue.** Cleaner than a single `paper/` when MICCAI + CVPR + TMI may all coexist.
- **Bibliography keyed by `Author<Year>_keyword`.** Direct continuation of the template's convention; matches the cite-keys mentioned in `simple_plan.html` §07.
- **`Slides/` and `Quarto/` kept as DORMANT directories**, not deleted, in case a thesis defense / lab talk needs slides later. `theme-template.scss` retained for that contingency.

## Verification (Step 9)

| Check                                                | Outcome     |
| ---------------------------------------------------- | ----------- |
| `bash scripts/validate-setup.sh`                     | 7 passed, 2 warnings, 1 failure (expected: missing LaTeX / Quarto toolchain — to install when manuscript work starts) |
| `python3 scripts/check-skill-integrity.py`           | PASS        |
| Active skill list shows `/review-python`             | PASS (visible in skill loader) |
| Active skill list does NOT show `/qa-quarto`         | PASS (archived) |
| New CLAUDE.md has zero `[YOUR …]` placeholders       | PASS (rewritten from scratch) |
| Bibliography `claim-verifier` round-trip             | DONE — 1/9 VERIFIED (DPS), 8/9 PARTIAL with concrete corrections; corrections applied to `Bibliography_base.bib`. |
| `simple_plan.html` accessible at new path            | PASS (`quality_reports/decisions/2026-04-29_research-direction-v2.html`) |

## Open questions / deferred items

These are explicitly out of scope for the current adaptation but parked for the next plans:

1. **README.md rewrite.** Currently the template's marketing README; needs full project-specific rewrite. Defer until the project has at least a first checkpoint or arXiv preprint to point at.
2. **`pyproject.toml` + lock file.** Wait until first dependency install.
3. **Figure colour palette.** Lock in alongside the first paper-figure draft.
4. **Tolerance thresholds (`replication-protocol.md`).** Set after the first VAE HU-fidelity sanity check produces baseline numbers.
5. **MICCAI / CVPR / TMI calibration of `editor` / `domain-referee` / `methods-referee`.** Add to `.claude/references/journal-profiles.md` before the first `/review-paper --peer` invocation.
6. **`venue-templates`-style integration.** Consider once the MICCAI 2027 deadline gets close — the upstream `venue-templates` skill (in the available-skills list) could be useful.
7. **Bibliography verification — partially resolved.** The `claim-verifier` agent (CoVe / fresh fork) returned with 8 PARTIAL findings; corrections applied. Cite-key changes worth flagging:
   - `Chen2025_HMEDM_CMIG` → `Chen2024_HMEDM_CMIG` (online publication was Dec 2024).
   - `Gong2024_ATOM` → `Lu2024_ATOM` (lead author surname needs manual verification — Y. Lu per dblp/IOP).
   - `TAVI2023_SciRep_motion` → `Matsumoto2023_SSF2_TAVI` (first author confirmed Matsumoto).
   `% NEEDS-MANUAL-CHECK` markers remain on three entries where an authoritative author list could not be pulled (Springer / Nature pages 303-redirected the WebFetch). Re-run `/verify-claims` before the first manuscript draft.
8. **PAD pipeline reproducibility.** Listed as a Medium-severity risk in the V2 brief; some of PAD is MATLAB. May need an Octave fallback or a contact-Deng-lab ask. Not blocking until the actual data pipeline is being built.

## Session learnings ([LEARN] candidates)

None proposed for `MEMORY.md` from this session — all the entries in the new `MEMORY.md` were seeded from the V2 strategic brief, not discovered during this session. Discovery-flavoured `[LEARN]` entries will accumulate as actual model / data / writing work begins.

## Handoff state

- Branch: `cardiac-artifacts` (no commit yet — that's the user's call).
- 58 file-tree changes are uncommitted (as of `git status` at the verification step).
- The plan file [`wondrous-honking-gray.md`](../plans/wondrous-honking-gray.md) is APPROVED → COMPLETED conceptually, but the plan file's frontmatter still says DRAFT. Updating the status string is a minor follow-up.
- The user expects more frequent check-ins for the first ~5 sessions per their initial brief; for this kickoff session the check-in came at the AskUserQuestion → ExitPlanMode boundary.

## Next steps (suggested)

1. User reviews the file-tree diff (`git diff`, `git status`).
2. If happy → `/commit "feat: adapt workflow for cardiac-CT diffusion project"`.
3. Read the `claim-verifier` output once it lands; correct or augment the bibliography accordingly.
4. Decide whether to download ImageCAS now (kicks off the May 2026 timeline) or to defer until after the first commit settles.


---
**Context compaction (auto) at 15:55**
Check git log and quality_reports/plans/ for current state.


---
**Context compaction (auto) at 15:51**
Check git log and quality_reports/plans/ for current state.
