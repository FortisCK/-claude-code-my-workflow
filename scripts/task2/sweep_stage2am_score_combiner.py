#!/usr/bin/env python3
"""Sweep conservative score-level combinations for Task2 Stage2AM.

This is a validation diagnostic on top of saved Stage2U candidate/prediction
rows. It does not train a CNN and does not generate new candidates. The search
keeps all candidates and only blends existing score columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import DetectionGroundTruth, DetectionPrediction, compute_class_ap, compute_detection_map
from scripts.task2.export_stage2ab_domain_policy_predictions import CLASS_SUFFIX, DEFAULT_SPLITS
from scripts.task2.sweep_stage2ae_calibrated_policy import transformed_box
from scripts.task2.export_stage2ab_domain_policy_predictions import load_box_transforms
from scripts.task2.sweep_stage2v_domain_policy import DEFAULT_SCORE_MODES


DEFAULT_BASE_POLICY = {
    0: {"phantom": "prob_iou75_source_rank_decay_roi", "animal": "prob_iou75_source_rank_decay_roi", "default": "prob_iou75_source_rank_decay_roi"},
    1: {"phantom": "rank_decay_roi", "animal": "prob_iou75_roi", "default": "rank_decay_roi"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--box-transform-json", type=Path, default=None)
    parser.add_argument("--optimize-split", default="valid_combined")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--score-mode", action="append", default=None)
    parser.add_argument("--alpha", action="append", type=float, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rows(run_dir: Path, split: str) -> tuple[list[DetectionGroundTruth], list[tuple[dict[str, str], dict[str, str]]]]:
    candidates = read_csv(run_dir / f"{split}_candidates_used.csv")
    predictions = read_csv(run_dir / f"{split}_eval_prediction_rows.csv")
    if len(candidates) != len(predictions):
        raise ValueError(f"{split}: row count mismatch: {len(candidates)} vs {len(predictions)}")
    for row_index, (candidate, prediction) in enumerate(zip(candidates, predictions), start=2):
        for field in ("sample_id", "source", "source_rank"):
            if str(candidate.get(field, "")) != str(prediction.get(field, "")):
                raise ValueError(
                    f"{split}: alignment mismatch row {row_index} for {field}: "
                    f"{candidate.get(field)!r} vs {prediction.get(field)!r}"
                )
    ground_truths: list[DetectionGroundTruth] = []
    seen: set[str] = set()
    for candidate in candidates:
        sample_id = candidate["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        ground_truths.append(
            DetectionGroundTruth(
                sample_id=sample_id,
                class_id=int(candidate["gt_class"]),
                box_xyxy=(
                    float(candidate["gt_x1"]),
                    float(candidate["gt_y1"]),
                    float(candidate["gt_x2"]),
                    float(candidate["gt_y2"]),
                ),
            )
        )
    return ground_truths, list(zip(candidates, predictions))


def policy_domain(candidate: dict[str, str]) -> str:
    domain = candidate.get("domain", "phantom")
    return domain if domain in {"phantom", "animal"} else "animal"


def score_value(prediction: dict[str, str], *, mode: str, class_id: int) -> float:
    key = f"{mode}_score_{CLASS_SUFFIX[class_id]}"
    if key not in prediction:
        raise KeyError(f"Missing score column {key}")
    return max(0.0, min(1.0, float(prediction[key])))


def blended_score(base: float, alt: float, *, alpha: float, blend_type: str) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    base = max(0.0, min(1.0, float(base)))
    alt = max(0.0, min(1.0, float(alt)))
    if blend_type == "linear":
        return (1.0 - alpha) * base + alpha * alt
    if blend_type == "geometric":
        eps = 1e-12
        return math.exp((1.0 - alpha) * math.log(base + eps) + alpha * math.log(alt + eps))
    if blend_type == "max":
        return max((1.0 - alpha) * base, alpha * alt)
    raise ValueError(f"Unknown blend_type {blend_type}")


def predictions_for_policy(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    policy: dict[str, Any],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> list[DetectionPrediction]:
    predictions: list[DetectionPrediction] = []
    class_policy = policy[str(class_id)]
    for candidate, prediction in rows:
        domain = policy_domain(candidate)
        domain_policy = class_policy.get(domain, class_policy["default"])
        base_mode = str(domain_policy["base_mode"])
        alt_mode = str(domain_policy["alt_mode"])
        alpha = float(domain_policy["alpha"])
        blend_type = str(domain_policy["blend_type"])
        base = score_value(prediction, mode=base_mode, class_id=class_id)
        alt = score_value(prediction, mode=alt_mode, class_id=class_id)
        predictions.append(
            DetectionPrediction(
                sample_id=candidate["sample_id"],
                class_id=class_id,
                score=blended_score(base, alt, alpha=alpha, blend_type=blend_type),
                box_xyxy=transformed_box(candidate, class_id=class_id, box_transforms=box_transforms),
            )
        )
    return predictions


def prediction_rows_for_policy(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    policy: dict[str, Any],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    class_policy = policy[str(class_id)]
    for candidate, prediction in rows:
        domain = policy_domain(candidate)
        domain_policy = class_policy.get(domain, class_policy["default"])
        base_mode = str(domain_policy["base_mode"])
        alt_mode = str(domain_policy["alt_mode"])
        alpha = float(domain_policy["alpha"])
        blend_type = str(domain_policy["blend_type"])
        base = score_value(prediction, mode=base_mode, class_id=class_id)
        alt = score_value(prediction, mode=alt_mode, class_id=class_id)
        x1, y1, x2, y2 = transformed_box(candidate, class_id=class_id, box_transforms=box_transforms)
        output.append(
            {
                "sample_id": candidate["sample_id"],
                "video_id": candidate.get("video_id", ""),
                "frame_index": candidate.get("frame_index", ""),
                "domain": candidate.get("domain", domain),
                "class_id": class_id,
                "score": blended_score(base, alt, alpha=alpha, blend_type=blend_type),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "source": candidate.get("source", ""),
                "source_rank": candidate.get("source_rank", ""),
                "score_mode": f"{blend_type}:{base_mode}:{alt_mode}:{alpha:.3f}",
                "policy_name": f"class{class_id}_{domain}",
            }
        )
    return output


def available_score_modes(rows: list[tuple[dict[str, str], dict[str, str]]], requested: list[str] | None) -> list[str]:
    fields = set(rows[0][1]) if rows else set()
    modes = list(requested or DEFAULT_SCORE_MODES)
    return [
        mode
        for mode in modes
        if all(f"{mode}_score_{suffix}" in fields for suffix in CLASS_SUFFIX.values())
    ]


def base_mode_for(class_id: int, domain: str) -> str:
    return DEFAULT_BASE_POLICY[class_id].get(domain, DEFAULT_BASE_POLICY[class_id]["default"])


def search_class_domain_policy(
    ground_truths: list[DetectionGroundTruth],
    rows: list[tuple[dict[str, str], dict[str, str]]],
    *,
    class_id: int,
    domain: str,
    score_modes: list[str],
    alphas: list[float],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    base_mode = base_mode_for(class_id, domain)
    blend_types = ("linear", "geometric", "max")
    for alt_mode in score_modes:
        for alpha in alphas:
            for blend_type in blend_types:
                domain_policy = {
                    "base_mode": base_mode,
                    "alt_mode": alt_mode,
                    "alpha": float(alpha),
                    "blend_type": blend_type,
                }
                class_policy = {
                    "phantom": {
                        "base_mode": base_mode_for(class_id, "phantom"),
                        "alt_mode": base_mode_for(class_id, "phantom"),
                        "alpha": 0.0,
                        "blend_type": "linear",
                    },
                    "animal": {
                        "base_mode": base_mode_for(class_id, "animal"),
                        "alt_mode": base_mode_for(class_id, "animal"),
                        "alpha": 0.0,
                        "blend_type": "linear",
                    },
                    "default": {
                        "base_mode": base_mode_for(class_id, "phantom"),
                        "alt_mode": base_mode_for(class_id, "phantom"),
                        "alpha": 0.0,
                        "blend_type": "linear",
                    },
                }
                class_policy[domain] = domain_policy
                policy = {
                    "0": class_policy if class_id == 0 else neutral_class_policy(0),
                    "1": class_policy if class_id == 1 else neutral_class_policy(1),
                }
                predictions = predictions_for_policy(
                    rows,
                    class_id=class_id,
                    policy=policy,
                    box_transforms=box_transforms,
                )
                ap_values = [
                    float(
                        compute_class_ap(
                            ground_truths,
                            predictions,
                            class_id=class_id,
                            iou_threshold=threshold,
                        )["ap"]
                    )
                    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
                ]
                row = {
                    "class_id": class_id,
                    "domain": domain,
                    "policy": domain_policy,
                    "ap50": ap_values[0],
                    "ap50_95": sum(ap_values) / len(ap_values),
                }
                if best is None or (float(row["ap50_95"]), float(row["ap50"])) > (
                    float(best["ap50_95"]),
                    float(best["ap50"]),
                ):
                    best = row
    if best is None:
        raise ValueError(f"No policy evaluated for class={class_id} domain={domain}")
    return best


def neutral_class_policy(class_id: int) -> dict[str, dict[str, Any]]:
    return {
        domain: {
            "base_mode": base_mode_for(class_id, domain),
            "alt_mode": base_mode_for(class_id, domain),
            "alpha": 0.0,
            "blend_type": "linear",
        }
        for domain in ("phantom", "animal")
    } | {
        "default": {
            "base_mode": base_mode_for(class_id, "phantom"),
            "alt_mode": base_mode_for(class_id, "phantom"),
            "alpha": 0.0,
            "blend_type": "linear",
        }
    }


def evaluate_policy(
    run_dir: Path,
    split: str,
    *,
    policy: dict[str, Any],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ground_truths, rows = load_rows(run_dir, split)
    predictions = [
        *predictions_for_policy(rows, class_id=0, policy=policy, box_transforms=box_transforms),
        *predictions_for_policy(rows, class_id=1, policy=policy, box_transforms=box_transforms),
    ]
    prediction_rows = [
        *prediction_rows_for_policy(rows, class_id=0, policy=policy, box_transforms=box_transforms),
        *prediction_rows_for_policy(rows, class_id=1, policy=policy, box_transforms=box_transforms),
    ]
    return compute_detection_map(ground_truths, predictions, class_ids=(0, 1)), prediction_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    box_transforms = load_box_transforms(args.box_transform_json)
    optimize_ground_truths, optimize_rows = load_rows(args.run_dir, args.optimize_split)
    score_modes = available_score_modes(optimize_rows, args.score_mode)
    alphas = args.alpha or [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
    if not score_modes:
        raise ValueError("No score modes available")

    best_by_class_domain: dict[str, dict[str, Any]] = {}
    final_policy = {"0": neutral_class_policy(0), "1": neutral_class_policy(1)}
    for class_id in (0, 1):
        best_by_class_domain[str(class_id)] = {}
        for domain in ("phantom", "animal"):
            best = search_class_domain_policy(
                optimize_ground_truths,
                optimize_rows,
                class_id=class_id,
                domain=domain,
                score_modes=score_modes,
                alphas=[float(value) for value in alphas],
                box_transforms=box_transforms,
            )
            best_by_class_domain[str(class_id)][domain] = best
            final_policy[str(class_id)][domain] = best["policy"]
        final_policy[str(class_id)]["default"] = final_policy[str(class_id)]["phantom"]

    metrics_by_split: dict[str, Any] = {}
    for split in args.splits:
        metrics, prediction_rows = evaluate_policy(
            args.run_dir,
            split,
            policy=final_policy,
            box_transforms=box_transforms,
        )
        metrics_by_split[split] = metrics
        if args.prediction_dir is not None:
            write_csv(args.prediction_dir / f"{split}_stage2am_predictions.csv", prediction_rows)

    output = {
        "artifact_type": "task2_stage2am_score_combiner_sweep",
        "run_dir": args.run_dir.as_posix(),
        "box_transform_json": args.box_transform_json.as_posix() if args.box_transform_json else None,
        "optimize_split": args.optimize_split,
        "score_modes": score_modes,
        "alphas": [float(value) for value in alphas],
        "base_policy": DEFAULT_BASE_POLICY,
        "best_by_class_domain": best_by_class_domain,
        "final_policy": final_policy,
        "metrics": metrics_by_split,
    }
    write_json(args.output_json, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
