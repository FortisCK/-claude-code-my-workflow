# Task2 Stage2AF Calibration Fine Sweep Plan

Date: 2026-06-14

## Goal

Check whether Stage2AE's animal-only calibration strength `1.25` is a local optimum, and whether class-specific animal transforms can beat the domain-shared transform.

## Steps

1. Sweep domain-shared animal-only strengths around Stage2AE.
2. Evaluate class-specific animal-only transforms at representative strengths.
3. Summarize the results in a machine-readable JSON file.
4. Keep Stage2AE if no setting improves public-validation mAP50-95 without hurting mAP50.

