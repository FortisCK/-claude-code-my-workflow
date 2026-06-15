# Task2 Stage2AF Calibration Fine Sweep Result

Date: 2026-06-14

## Question

Is Stage2AE's animal-only strength `1.25` actually near the best point, or can a nearby strength/class-specific transform improve it further?

## Setup

Base run:

`outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval`

Output directory:

`outputs/task2/stage2af_calibration_fine_sweep`

Summary JSON:

`outputs/task2/stage2af_calibration_fine_sweep/summary.json`

## Result

| Variant | Strength | Class-specific | valid_combined mAP50 | valid_combined mAP50-95 |
| --- | ---: | --- | ---: | ---: |
| Stage2AE | 1.25 | no | 0.21128164745000094 | 0.06327005557430047 |
| domain-shared | 1.20 | no | 0.21128164745000094 | 0.06321969776599151 |
| domain-shared | 1.15 | no | 0.21128164745000094 | 0.06320773879399151 |
| domain-shared | 1.10 | no | 0.21128164745000094 | 0.06320767500203121 |
| domain-shared | 1.05 | no | 0.21128164745000094 | 0.06297167200145336 |
| domain-shared | 1.30 | no | 0.21128164745000094 | 0.052743035383064886 |
| domain-shared | 1.35 | no | 0.21128164745000094 | 0.05242042495126746 |
| domain-shared | 1.40 | no | 0.21128164745000094 | 0.05219348612367537 |
| class-specific | 1.00 | yes | 0.21128164745000094 | 0.051751206499361136 |
| class-specific | 1.25 | yes | 0.09718886742331332 | 0.04030938820715173 |

## Decision

Do not create a new champion from Stage2AF. Keep Stage2AE as the current public-validation champion.

Rationale:

- Strength `1.25` remains the best tested point.
- Nearby strengths `1.10` to `1.20` are close but lower.
- Strengths above `1.25` degrade sharply.
- Class-specific animal transforms are worse and can hurt mAP50 substantially.

Interpretation:

The animal calibration gain has a narrow useful range. Stage2AE is not just a random coarse-grid pick, but it is still public-validation tuned. Stage2AD remains the more conservative train-fitted fallback.

