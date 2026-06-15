# Decision: Stop Stage 8 Patch Training and Continue as Conservative ROI Refiner

Date: 2026-06-02

## Status

The Stage 8 warm-started patch-refiner training service was still active after
about 20 hours. It reached epoch 77 in progress, but the best checkpoint had not
improved since epoch 12.

Service stopped:

`cathaction-task1-stage8-patch-refiner.service`

GPU was released after stopping.

## Training Metrics

Stage 8 patch-refiner, balanced 1024 eval during training:

- best epoch: `12`
- best mean Dice: `0.5653137703974893`
- label_1 Dice: `0.5314602411315835`
- label_2 Dice: `0.5991672996633951`
- animal Dice: `0.6823995934740131`
- phantom Dice: `0.4482279473209655`

Stage 7 toolness auxiliary, same training-eval口径:

- best epoch: `46`
- best mean Dice: `0.6682097615492055`
- label_1 Dice: `0.6638849936363767`
- label_2 Dice: `0.6725345294620345`
- animal Dice: `0.7314241126141201`
- phantom Dice: `0.6049954104842911`

Conclusion: the Stage 8 patch-refiner must not be promoted as a standalone
full-frame model.

## ROI Probe

A small ROI-fusion probe used Stage 7 as the coarse model and Stage 8 best epoch
as a local patch refiner. ROI boxes were generated from coarse predictions only,
not from evaluation masks.

64 animal samples from released eval:

- Stage 7 coarse: `0.73587140623559`
- Stage 7 + Stage 8 ROI alpha=0.15: `0.7380333688997551`
- delta: `+0.0021619626641651`

Mixed 128 probe, 64 animal + 64 phantom:

- Stage 7 coarse mean Dice: `0.6666689791444314`
- Stage 7 + Stage 8 ROI alpha=0.15 mean Dice: `0.6691574299615879`
- delta: `+0.0024884508171565`
- animal delta: `+0.0021619626641651`
- phantom delta: `+0.0028149389701480`

## Decision

Continue Stage 8 only as a conservative ROI refinement component. Do not train
this patch model longer and do not evaluate it as a standalone full-frame model.

Next useful step is a parameter probe on mixed subsets and then a full released-
eval run against the current seven-model champion `0.6536723455148790`.
