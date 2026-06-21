# Blind DPS on NATIVE real motion — Leo058 (honest probe of the operator wall)

Date: 2026-06-19 CEST
Author: MZ
Status: DONE — EXPLORATORY, NEGATIVE (no genuine de-artifacting; metric trap exposed)
Git SHA: be17e6b (working tree: blind_dps_native.py uncommitted)
Config: diffusion_v2_residual.yaml + unet_v1; assumed pop-mean motion (s=0.9,T=800), no-motion HU calib
Seed: DPS 0
Dataset: Leo058 REAL native-motion clinical CCTA, NO GT
Planned GPU-hours: <0.2

## Intent (EXPLORATORY)

User: real-data de-artifacting is the goal, not just synthetic. Probe whether our
tools touch NATIVE real motion (058, dirtiest candidate). DPS needs a known
operator we don't have on real scans -> attempt blind DPS (assumed pop-mean motion
+ no-motion HU calibration) alongside U-Net and feed-forward diffusion. No GT ->
no-reference in-heart sharpness + visual only. Expectation: weak (operator unknown,
forward model != real scanner).

## Result — NEGATIVE, and instructive

in-heart sharpness (grad-mag mean): corrupted 51.8 -> U-Net 71.1 -> ff_diff 79.9
-> blind_dps 77.6. Panel: experiments/runs/blind_dps_native/case_058__panel.png

Reading the IMAGE (not the number):
1. **The sharpness rise is a METRIC TRAP.** It comes from (a) HALLUCINATED CT-texture
   grain (the diffusion prior paints realistic clean-CCTA texture on OOD input) and
   (b) a bright rectangular FRAME artifact at the scan-FOV/crop boundary — NOT from
   genuine coronary deblurring. No-reference gradient sharpness is gamed by
   hallucination. (Direct cautionary evidence for the eval-protocol leg.)
2. **blind DPS (77.6) ~= feed-forward (79.9), below it.** The assumed operator added
   NOTHING over the feed-forward prior -> the measurement-consistency mechanism is
   inert without the true operator. The unknown-operator wall, made concrete.
3. The frame artifact is partly a preprocessing/FOV-edge issue (fixable); the
   texture-hallucination + inert-DPS findings are fundamental.

Contrast with same-day external-validity run (real anatomy + KNOWN synthetic
motion): there DPS gave +0.13 Dice, 5/5. The ONLY difference is operator knowledge.

## Decision

DISCARD as a result; KEEP as decisive evidence that native real-motion de-artifacting
needs (a) per-case motion-OPERATOR ESTIMATION (to make DPS valid) and (b) a
resolution-matched, realistic-motion prior (to stop hallucination). Current
feed-forward/blind tools HALLUCINATE rather than de-artifact on native motion.
This sets the agenda: build the motion-operator estimator next.

## Cross-references

- External validity (KNOWN operator, PASS): 2026-06-19_dps-real-leo-external-validity.md
- Feed-forward 058 (smooth/hallucinate): 2026-06-19_real-leo058-feedforward-demo.md
- Synthetic DPS win: 2026-06-17_dps-mc-stage1-pilot.md
- Code: scripts/python/blind_dps_native.py
