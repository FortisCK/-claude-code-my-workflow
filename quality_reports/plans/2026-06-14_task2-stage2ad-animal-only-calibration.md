# Task2 Stage2AD Animal-Only Box Calibration Plan

Date: 2026-06-14

## Goal

Improve Stage2AC by applying train-fitted box calibration only to animal rows while keeping phantom boxes unchanged.

## Rationale

Stage2AC improved valid_combined mAP50-95 but slightly reduced mAP50 because the phantom transform introduced a small degradation. The split results showed all meaningful gain came from animal rows. A domain-selective transform should preserve Stage2AB's phantom behavior and keep the animal high-IoU gain.

## Steps

1. Add a domain-selection option to the Stage2AC calibration diagnostic.
2. Fit animal-only, domain-shared transforms from train candidates.
3. Evaluate on valid_combined, valid_phantom, and valid_animal.
4. Export clean Stage2AD predictions through the GT-free Stage2AB exporter with the animal-only transform JSON.
5. Verify schema/policy and record whether Stage2AD should replace Stage2AC.

## Acceptance Criteria

- valid_combined mAP50 is not lower than Stage2AB.
- valid_combined mAP50-95 is higher than Stage2AB and Stage2AC.
- valid_phantom metrics match Stage2AB.
- Clean CSV export passes schema/policy verification.

