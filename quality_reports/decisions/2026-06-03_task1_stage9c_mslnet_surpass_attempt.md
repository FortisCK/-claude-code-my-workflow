# Decision: Stage 9C Threshold/Gate Route Does Not Beat MSLNet

Date: 2026-06-03

## Goal

Use the MSLNet paper's CathAction-style binary foreground metrics as a local
diagnostic target and push the current Stage 9A champion beyond the reported
MSLNet numbers.

## Reference

MSLNet CathAction segmentation numbers from the paper:

- Dice: `0.6251`
- IoU: `0.4658`
- AHD: `1.5`
- F1 r3: `0.9305`
- precision r3: `0.9152`
- recall r3: `0.9462`

Current Stage 9A seven-model champion with remove-small-min32 postprocessing:

- MSLNet-style Dice: `0.6215624175077626`
- IoU: `0.4605049121513648`
- AHD: `1.3713229393268282`
- F1 r3: `0.911577665104627`
- precision r3: `0.8663946640069181`
- recall r3: `0.9677293936026353`

Domain split:

- animal F1 r3: `0.9827937447392838`
- phantom F1 r3: `0.892135771994053`

To match MSLNet's overall F1 r3 while keeping the current animal score, phantom
F1 r3 would need to rise to roughly `0.9162`.

## Experiments Run

### Hard-mask foreground calibration

Full released-eval targeted hard-mask search:

- Output: `outputs/task1/stage9c_mslnet_hard_postprocess/full_post_targeted3/summary.json`
- Best candidate: `all_erode1`
- Exact output: `outputs/task1/diagnostics/stage9c_all_erode1_mslnet_style/eval_mslnet_style.json`

`all_erode1` exact metrics:

- Dice: `0.6166986524865422`
- IoU: `0.4573580007316683`
- AHD: `1.404407495224451`
- F1 r3: `0.9175400141830308`
- precision r3: `0.8951362530072895`
- recall r3: `0.9450977034033231`
- animal F1 r3: `0.985275865313633`
- phantom F1 r3: `0.8990482187319084`

This improved F1 r3 by about `+0.006` but reduced Dice and still missed MSLNet.

### Soft probability gate

Phantom 512-sample probability-gate probe:

- Output: `outputs/task1/stage9c_mslnet_probability_gate/phantom512_quick/summary.json`
- Best candidate: `all_foreground_prob_t0p7`

Best phantom-probe metrics:

- Dice: `0.591107345612506`
- IoU: `0.428690762751617`
- precision r3: `0.8724868783828252`
- recall r3: `0.9343578207712437`
- F1 r3: `0.9002408161032309`

The gain was again about `+0.006` on the probe and followed the same
precision/recall tradeoff as hard erosion.

### Stage 7 toolness auxiliary gate

Phantom 512-sample toolness probe:

- Output: `outputs/task1/stage9c_mslnet_probability_gate/phantom512_toolness/summary.json`

Best candidate was the baseline. Toolness thresholding raised precision but
deleted too many true foreground pixels, so F1 r3 did not improve.

## Interpretation

The current champion already has high localization recall. The main MSLNet gap
is phantom precision: the model predicts too much foreground on line-like
background structures. Uniform operations, including erosion and global
probability thresholds, can raise precision by about `0.03`, but they also lose
about `0.02` to `0.03` recall. That caps full released-eval F1 r3 around
`0.917`, below MSLNet's `0.9305`.

This is a useful failure: the missing component is selective false-positive
rejection, not a better global threshold.

## Decision

Stop the Stage 9C threshold/gate route. The next route should be an
MSLNet-style coarse-to-fine binary foreground refiner:

- train on high-resolution patches;
- oversample phantom;
- include hard-negative patches from coarse predictions;
- use the refiner only as a foreground gate;
- keep Stage 9A probabilities for the final label_1 / label_2 assignment.

Promotion target:

- MSLNet-style F1 r3 above `0.9305`, or at least phantom F1 r3 above `0.916`;
- MSLNet-style Dice above `0.6251`;
- no material drop in official multiclass mean Dice.
