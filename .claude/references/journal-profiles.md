# Journal Profiles

Calibration data for `/review-paper --peer [journal]` in this CATHACTION repo. Profiles are tuned for medical imaging, surgical AI, and MICCAI-style method papers.

## Schema

Each profile includes:

- **Short name**: string passed to `--peer`.
- **Focus**: what the venue values.
- **Bar**: what clears review.
- **Typical concerns**: questions reviewers will ask.
- **Referee-pool weights**: weights over CLINICAL, GENERALIZATION, SEGMENTATION, DETECTION, REPRODUCIBILITY, SKEPTIC.
- **Formatting / reporting notes**: venue-specific expectations.

---

## MICCAI

**Short name:** `MICCAI`

**Focus.** Novel, rigorous medical image computing methods with clear clinical motivation and careful experiments.

**Bar.** The method must be technically credible, clinically motivated, and evaluated against appropriate baselines. Claims must be supported by ablations and failure analysis.

**Typical concerns.**
- "Is the contribution more than an engineering combination of known parts?"
- "Are comparisons fair and sufficiently strong?"
- "Are the challenge metrics reported correctly?"
- "Does the method generalize across phantom, animal, and human domains?"
- "Is the clinical framing careful and not overclaimed?"

**Referee-pool weights.**
- CLINICAL: 0.15
- GENERALIZATION: 0.25
- SEGMENTATION: 0.20
- DETECTION: 0.15
- REPRODUCIBILITY: 0.15
- SKEPTIC: 0.10

**Formatting / reporting notes.** Concise method description, strong visual explanation, clear ablations, metric table, limitations.

---

## Medical Image Analysis

**Short name:** `MedIA`

**Focus.** Deep, mature medical image analysis contributions with thorough validation and careful methodological explanation.

**Bar.** Stronger validation and analysis than a conference paper. Reviewer expects robust baselines, ablations, error analysis, and clear clinical relevance.

**Typical concerns.**
- "Is the method sufficiently novel and broadly useful?"
- "Are experiments deep enough to support the claims?"
- "Are domain-shift limitations handled honestly?"
- "Can another lab reproduce the training and evaluation?"

**Referee-pool weights.**
- CLINICAL: 0.15
- GENERALIZATION: 0.25
- SEGMENTATION: 0.15
- DETECTION: 0.15
- REPRODUCIBILITY: 0.20
- SKEPTIC: 0.10

**Formatting / reporting notes.** Full limitations, detailed ablations, reproducibility details, high-quality figures.

---

## IEEE Transactions on Medical Imaging

**Short name:** `TMI`

**Focus.** Technically rigorous medical imaging methods with strong quantitative validation and careful engineering detail.

**Bar.** Methodological clarity, robust quantitative evidence, and reproducible implementation matter heavily.

**Typical concerns.**
- "Is the technical method described precisely enough to reproduce?"
- "Are metric implementations and statistical comparisons correct?"
- "Are baselines strong and implementation details fair?"
- "Is the submission/inference pipeline deterministic?"

**Referee-pool weights.**
- CLINICAL: 0.10
- GENERALIZATION: 0.20
- SEGMENTATION: 0.20
- DETECTION: 0.15
- REPRODUCIBILITY: 0.25
- SKEPTIC: 0.10

**Formatting / reporting notes.** Precise algorithms, equations where helpful, complete experimental setup.

---

## International Journal of Computer Assisted Radiology and Surgery

**Short name:** `IJCARS`

**Focus.** Computer-assisted intervention, surgical workflow, clinical translation, and system-level validation.

**Bar.** The method must connect clearly to interventional workflow and procedural safety while keeping claims evidence-based.

**Typical concerns.**
- "How does this support an endovascular workflow?"
- "Are runtime, integration, and failure modes discussed?"
- "Are clinical claims grounded in the benchmark evidence?"
- "Does the method handle realistic fluoroscopy artifacts?"

**Referee-pool weights.**
- CLINICAL: 0.30
- GENERALIZATION: 0.20
- SEGMENTATION: 0.10
- DETECTION: 0.20
- REPRODUCIBILITY: 0.10
- SKEPTIC: 0.10

**Formatting / reporting notes.** Clinical workflow diagrams, runtime notes, limitations, deployment caution.

---

## Nature Biomedical Engineering

**Short name:** `NBE`

**Focus.** Broad biomedical engineering impact, strong validation, and translational significance.

**Bar.** A challenge method alone is unlikely to clear the bar unless it demonstrates broad conceptual or translational significance.

**Typical concerns.**
- "Why does this matter beyond the challenge leaderboard?"
- "Is the validation compelling enough for translational claims?"
- "Are limitations and risks disclosed clearly?"
- "Does the method change what is possible in endovascular AI?"

**Referee-pool weights.**
- CLINICAL: 0.25
- GENERALIZATION: 0.25
- SEGMENTATION: 0.10
- DETECTION: 0.10
- REPRODUCIBILITY: 0.15
- SKEPTIC: 0.15

**Formatting / reporting notes.** Strong narrative, polished figures, transparent limitations, careful claims.
