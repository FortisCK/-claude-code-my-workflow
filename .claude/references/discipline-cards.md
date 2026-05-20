# Discipline Cards

Short reference cards for research-ideation, preregistration, and peer-review routing in this CATHACTION repo.

## Medical Imaging / Surgical AI (`medical-imaging`)

**Paper-type frequencies.**

| Type | Share | Notes |
|------|-------|-------|
| Segmentation-only | Common | Tool, anatomy, or lesion segmentation with overlap metrics. |
| Detection-only | Common | Object/event detection with AP/mAP or sensitivity/specificity tradeoffs. |
| Multi-task | Common | Shared representation across segmentation, detection, tracking, or phase tasks. |
| Domain-generalization | Common | Cross-site, cross-device, synthetic-to-real, phantom-to-human, or animal-to-human transfer. |
| Challenge report | Common | System description, ablations, leaderboard metrics, reproducibility details. |

**Dominant venues shipped in `journal-profiles.md`.** MICCAI, Medical Image Analysis (MedIA), IEEE Transactions on Medical Imaging (TMI), International Journal of Computer Assisted Radiology and Surgery (IJCARS), Nature Biomedical Engineering (NBE).

**Preregistration norms.**

- Challenge submissions usually follow challenge rules rather than formal preregistration.
- External validation or clinical studies may require protocol registration, ethics approval, and data-governance review.
- Any public external data or pretrained model used for CATHACTION must be disclosed and policy-compliant.

**Method conventions.**

- Report primary challenge metrics first, then secondary metrics.
- Split by patient/procedure/case, not frame, slice, or patch.
- Provide domain-stratified metrics when domain metadata exists.
- Include strong baselines, ablations, and failure cases.
- Keep inference automated and Docker-friendly for challenge submission.

**Cross-references.** `methods-referee.md` paper types: segmentation-only, collision-only, multi-task, domain-generalization, challenge report. `journal-profiles.md`: MICCAI, MedIA, TMI, IJCARS, NBE.
