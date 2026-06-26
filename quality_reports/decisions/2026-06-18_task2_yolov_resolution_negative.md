# Decision Record — YOLOV Resolution Bump (576→768) is a NEGATIVE

**Date:** 2026-06-18. **Result: input resolution is NOT the localization lever; the stride (P2) is.**

## Setup
Retrained YOLOV-S two-class on the mv split (26 train / 6 valid videos) at **input/test 768** (vs 576 baseline), 24 epochs, same recipe. Eval on the trustworthy 6-video full-frame-GT panel.

## Result (panel, phantom)

| | per-case mAP50 | global mAP50 | per-case mAP50-95 |
|---|---:|---:|---:|
| Raw YOLOV 576 | 0.252 | 0.209 | 0.098 |
| Raw YOLOV 768 | 0.248 | 0.220 | 0.100 |

Flat (per-case slightly lower, mAP50-95 unchanged at ~0.10). Native COCOeval AP50 also plateaued ~0.10 (same as 576).

## Why
A ~13px tip at 576 → ~17px at 768, but both detect on stride-8 feature maps where the tip is ~2px — below what the head can localize precisely. More input pixels don't help if the detection feature level is too coarse. **The lever is the feature STRIDE (a P2/stride-4 head), not input resolution** — consistent with the bake-off diagnosis and the locR75=0.18 ceiling.

## Decision
Drop the resolution lever. Next localization attempt = **P2 (stride-4) head** on the YOLOX backbone inside YOLOV — the principled fix. Honest expectation remains bounded: mAP50-95 ~0.10 is near the annotation-precision floor for ~15px tips; realistic upside is incremental, not a doubling. If P2 also fails to move the panel, that is strong evidence we are at the practical ceiling of this approach on the phantom proxy, and effort should shift to domain (animal/human) + submission.
