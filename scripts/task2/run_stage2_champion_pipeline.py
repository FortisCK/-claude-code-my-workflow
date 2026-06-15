#!/usr/bin/env python3
"""Run the frozen Task2 champion/fallback export pipeline.

This script packages the current public-validation Task2 choices:

- Stage2AB: domain-aware score policy, no box calibration;
- Stage2AD: Stage2AB plus animal-only train-fitted box calibration strength 1.0;
- Stage2AE: Stage2AB plus animal-only box calibration strength 1.25.
- Stage2AI: Stage2AE plus calibrated high-IoU score policy;
- Stage2AN: Stage2AE plus class/domain global score scaling.
- Stage2AQ: Stage2AE plus phantom-class1-only Stage2X multi-source replacement.

It exports clean prediction CSVs for each requested variant, validates the
internal schema, and recomputes metrics when candidate rows include GT fields.
Hidden-test candidate rows usually do not contain GT fields; in that case the
pipeline still writes predictions and records that metrics are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import DetectionGroundTruth, DetectionPrediction, compute_detection_map
from scripts.task2.export_clean_predictions import CLEAN_FIELDS, apply_topk, write_rows
from scripts.task2.export_stage2ab_domain_policy_predictions import (
    DEFAULT_SPLITS,
    FORBIDDEN_INPUT_ONLY_FIELDS,
    export_split,
    load_box_transforms,
    load_score_policies,
)


DEFAULT_AD_TRANSFORM_JSON = Path(
    "outputs/task2/stage2ad_animal_only_box_calibration/"
    "yolo_only_trainfit_animal_only_iou30_valid_combined.json"
)
DEFAULT_AE_TRANSFORM_JSON = Path(
    "outputs/task2/stage2ae_calibration_strength_sweep/"
    "animal_only_strength_1.25_valid_combined.json"
)
DEFAULT_AI_POLICY_JSON = Path(
    "outputs/task2/stage2ai_calibrated_policy_sweep/stage2ae_box_policy_sweep.json"
)
DEFAULT_AN_SCALE_JSON = Path(
    "outputs/task2/stage2an_global_score_scale/stage2ae_fast_global_scale_mAP5095.json"
)
DEFAULT_AQ_MULTISOURCE_RUN_DIR = Path(
    "outputs/task2/stage2u_quality_ranker/stage2x_five_source_fullvalid_eval"
)
DEFAULT_AQ_SOURCE_POLICY = "yolo_stage2x"
DEFAULT_AQ_SCORE_MODE = "rank_decay_roi"
DEFAULT_AQ_TOPK = 5
DEFAULT_AQ_SCORE_SCALE = 0.75
AQ_SOURCE_POLICIES: dict[str, set[str]] = {
    "yolo_stage2x": {"yolo_stage2l", "stage2x_class1"},
    "yolo_geom_stage2x": {"yolo_stage2l", "task1_geometry_rect", "stage2x_class1"},
    "original_sources": {"yolo_stage2l", "task1_geometry_rect"},
}
VARIANT_ORDER = ("stage2ab", "stage2ad", "stage2ae", "stage2ai", "stage2an", "stage2aq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing {split}_candidates_used.csv and {split}_eval_prediction_rows.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument(
        "--variant",
        action="append",
        choices=VARIANT_ORDER,
        default=None,
        help="Variant to export. May be repeated. Defaults to all variants.",
    )
    parser.add_argument("--stage2ad-transform-json", type=Path, default=DEFAULT_AD_TRANSFORM_JSON)
    parser.add_argument("--stage2ae-transform-json", type=Path, default=DEFAULT_AE_TRANSFORM_JSON)
    parser.add_argument("--stage2ai-transform-json", type=Path, default=DEFAULT_AE_TRANSFORM_JSON)
    parser.add_argument("--stage2ai-policy-json", type=Path, default=DEFAULT_AI_POLICY_JSON)
    parser.add_argument("--stage2an-transform-json", type=Path, default=DEFAULT_AE_TRANSFORM_JSON)
    parser.add_argument("--stage2an-scale-json", type=Path, default=DEFAULT_AN_SCALE_JSON)
    parser.add_argument("--stage2aq-transform-json", type=Path, default=DEFAULT_AE_TRANSFORM_JSON)
    parser.add_argument("--stage2aq-multisource-run-dir", type=Path, default=DEFAULT_AQ_MULTISOURCE_RUN_DIR)
    parser.add_argument("--stage2aq-source-policy", choices=sorted(AQ_SOURCE_POLICIES), default=DEFAULT_AQ_SOURCE_POLICY)
    parser.add_argument("--stage2aq-score-mode", default=DEFAULT_AQ_SCORE_MODE)
    parser.add_argument("--stage2aq-topk", type=int, default=DEFAULT_AQ_TOPK)
    parser.add_argument("--stage2aq-score-scale", type=float, default=DEFAULT_AQ_SCORE_SCALE)
    parser.add_argument("--topk-per-sample-class", type=int, default=0)
    parser.add_argument(
        "--fallback-domain",
        default=None,
        help="Domain to use when candidate rows have missing/unknown domain metadata.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Do not recompute metrics even when GT fields are present.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def variant_transform_json(args: argparse.Namespace, variant: str) -> Path | None:
    if variant == "stage2ab":
        return None
    if variant == "stage2ad":
        return args.stage2ad_transform_json
    if variant == "stage2ae":
        return args.stage2ae_transform_json
    if variant == "stage2ai":
        return args.stage2ai_transform_json
    if variant == "stage2an":
        return args.stage2an_transform_json
    if variant == "stage2aq":
        return args.stage2aq_transform_json
    raise ValueError(f"Unknown variant: {variant}")


def variant_score_policy_json(args: argparse.Namespace, variant: str) -> Path | None:
    if variant == "stage2ai":
        return args.stage2ai_policy_json
    return None


def variant_scale_json(args: argparse.Namespace, variant: str) -> Path | None:
    if variant == "stage2an":
        return args.stage2an_scale_json
    return None


def is_phantom_class1(row: dict[str, str] | dict[str, Any]) -> bool:
    return row.get("domain", "").strip().lower() == "phantom" and int(row["class_id"]) == 1


def clean_prediction_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "video_id": row.get("video_id", ""),
        "frame_index": row.get("frame_index", ""),
        "domain": row.get("domain", "").strip().lower(),
        "class_id": int(row["class_id"]),
        "score": float(row["score"]),
        "x1": float(row["x1"]),
        "y1": float(row["y1"]),
        "x2": float(row["x2"]),
        "y2": float(row["y2"]),
        "source": row.get("source", ""),
        "source_rank": row.get("source_rank", ""),
        "score_mode": row.get("score_mode", ""),
        "policy_name": row.get("policy_name", ""),
    }


def stage2aq_replacement_rows(
    candidates: list[dict[str, str]],
    predictions: list[dict[str, str]],
    *,
    source_keep: set[str],
    score_mode: str,
    topk: int,
    score_scale: float,
) -> list[dict[str, Any]]:
    if len(candidates) != len(predictions):
        raise ValueError(
            f"Stage2AQ row mismatch: {len(candidates)} candidates vs {len(predictions)} predictions"
        )
    score_key = f"{score_mode}_score_collision"
    if predictions and score_key not in predictions[0]:
        raise KeyError(f"Stage2AQ missing score column {score_key}")
    rows: list[dict[str, Any]] = []
    for row_index, (candidate, prediction) in enumerate(zip(candidates, predictions), start=2):
        if candidate.get("domain", "").strip().lower() != "phantom":
            continue
        if candidate.get("source", "") not in source_keep:
            continue
        for key in ("sample_id", "source", "source_rank"):
            if str(candidate.get(key, "")) != str(prediction.get(key, "")):
                raise ValueError(
                    f"Stage2AQ alignment mismatch at row {row_index} for {key}: "
                    f"{candidate.get(key)!r} vs {prediction.get(key)!r}"
                )
        rows.append(
            {
                "sample_id": candidate["sample_id"],
                "video_id": candidate.get("video_id", ""),
                "frame_index": candidate.get("frame_index", ""),
                "domain": "phantom",
                "class_id": 1,
                "score": float(prediction[score_key]) * float(score_scale),
                "x1": float(candidate["x1"]),
                "y1": float(candidate["y1"]),
                "x2": float(candidate["x2"]),
                "y2": float(candidate["y2"]),
                "source": candidate.get("source", ""),
                "source_rank": candidate.get("source_rank", ""),
                "score_mode": score_mode,
                "policy_name": f"stage2aq_{score_mode}_top{topk}_x{score_scale:g}",
            }
        )
    return apply_topk(rows, int(topk))


def apply_stage2aq_replacement_to_csv(
    prediction_csv: Path,
    multisource_run_dir: Path,
    split: str,
    *,
    source_policy: str,
    score_mode: str,
    topk: int,
    score_scale: float,
) -> dict[str, Any]:
    if source_policy not in AQ_SOURCE_POLICIES:
        raise ValueError(f"Unknown Stage2AQ source policy: {source_policy}")
    baseline_rows = [clean_prediction_row(row) for row in read_csv(prediction_csv)]
    kept_rows = [row for row in baseline_rows if not is_phantom_class1(row)]
    candidates = read_csv(multisource_run_dir / f"{split}_candidates_used.csv")
    predictions = read_csv(multisource_run_dir / f"{split}_eval_prediction_rows.csv")
    replacement_rows = stage2aq_replacement_rows(
        candidates,
        predictions,
        source_keep=AQ_SOURCE_POLICIES[source_policy],
        score_mode=score_mode,
        topk=topk,
        score_scale=score_scale,
    )
    output_rows = sorted(
        [*kept_rows, *replacement_rows],
        key=lambda row: (str(row["sample_id"]), int(row["class_id"]), -float(row["score"])),
    )
    write_rows(prediction_csv, output_rows)
    return {
        "path": prediction_csv.as_posix(),
        "multisource_run_dir": multisource_run_dir.as_posix(),
        "source_policy": source_policy,
        "source_keep": sorted(AQ_SOURCE_POLICIES[source_policy]),
        "score_mode": score_mode,
        "topk": int(topk),
        "score_scale": float(score_scale),
        "baseline_rows": len(baseline_rows),
        "kept_baseline_rows": len(kept_rows),
        "replacement_rows": len(replacement_rows),
        "output_rows": len(output_rows),
    }


def load_score_scales(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected JSON object")
    raw_policy = loaded.get("best_policy", loaded)
    if not isinstance(raw_policy, dict):
        raise ValueError(f"{path}: expected best_policy object")
    scales: dict[tuple[str, str], float] = {}
    for key, value in raw_policy.items():
        parts = str(key).split("_", maxsplit=1)
        if len(parts) != 2 or not parts[0].startswith("class"):
            raise ValueError(f"{path}: invalid scale key {key!r}")
        class_id = str(int(parts[0].removeprefix("class")))
        domain = parts[1].strip().lower()
        scales[(class_id, domain)] = float(value)
    return scales


def apply_score_scales_to_csv(path: Path, scales: dict[tuple[str, str], float]) -> dict[str, Any]:
    if not scales:
        return {"path": path.as_posix(), "scaled_rows": 0, "scales": {}}
    rows = read_csv(path)
    scaled_rows = 0
    scale_counts: dict[str, int] = {}
    for row in rows:
        class_id = str(int(row["class_id"]))
        domain = row["domain"].strip().lower()
        scale = float(scales.get((class_id, domain), 1.0))
        if scale != 1.0:
            row["score"] = repr(float(row["score"]) * scale)
            row["policy_name"] = f"{row['policy_name']}_scale{scale:g}"
            scaled_rows += 1
            scale_key = f"class{class_id}_{domain}_x{scale:g}"
            scale_counts[scale_key] = scale_counts.get(scale_key, 0) + 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "path": path.as_posix(),
        "scaled_rows": scaled_rows,
        "scale_counts": dict(sorted(scale_counts.items())),
        "scales": {f"class{class_id}_{domain}": scale for (class_id, domain), scale in sorted(scales.items())},
    }


def candidate_has_gt(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    required = {"gt_class", "gt_x1", "gt_y1", "gt_x2", "gt_y2"}
    return required.issubset(rows[0])


def ground_truths_from_candidates(candidates: list[dict[str, str]]) -> list[DetectionGroundTruth]:
    ground_truths: list[DetectionGroundTruth] = []
    seen: set[str] = set()
    for row in candidates:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        ground_truths.append(
            DetectionGroundTruth(
                sample_id=sample_id,
                class_id=int(row["gt_class"]),
                box_xyxy=(
                    float(row["gt_x1"]),
                    float(row["gt_y1"]),
                    float(row["gt_x2"]),
                    float(row["gt_y2"]),
                ),
            )
        )
    return ground_truths


def predictions_from_clean_csv(path: Path) -> list[DetectionPrediction]:
    rows = read_csv(path)
    return [
        DetectionPrediction(
            sample_id=row["sample_id"],
            class_id=int(row["class_id"]),
            score=float(row["score"]),
            box_xyxy=(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ),
        )
        for row in rows
    ]


def recompute_metrics(
    run_dir: Path,
    prediction_dir: Path,
    split: str,
) -> dict[str, Any] | None:
    candidates = read_csv(run_dir / f"{split}_candidates_used.csv")
    if not candidate_has_gt(candidates):
        return None
    ground_truths = ground_truths_from_candidates(candidates)
    predictions = predictions_from_clean_csv(prediction_dir / f"{split}_domain_policy_predictions.csv")
    return compute_detection_map(ground_truths, predictions, class_ids=(0, 1))


def summarize_prediction_csv(path: Path) -> dict[str, Any]:
    rows = 0
    class_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    score_mode_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    min_score: float | None = None
    max_score: float | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if header != list(CLEAN_FIELDS):
            forbidden_present = sorted(set(header) & FORBIDDEN_INPUT_ONLY_FIELDS)
            if forbidden_present:
                raise ValueError(f"{path}: forbidden fields present: {forbidden_present}")
            raise ValueError(f"{path}: expected header {list(CLEAN_FIELDS)}, got {header}")
        for row_index, row in enumerate(reader, start=2):
            rows += 1
            class_id = str(int(row["class_id"]))
            domain = row["domain"]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            score_mode_counts[row["score_mode"]] = score_mode_counts.get(row["score_mode"], 0) + 1
            policy_counts[row["policy_name"]] = policy_counts.get(row["policy_name"], 0) + 1
            score = float(row["score"])
            min_score = score if min_score is None else min(min_score, score)
            max_score = score if max_score is None else max(max_score, score)
            x1, y1, x2, y2 = (float(row[name]) for name in ("x1", "y1", "x2", "y2"))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{path}: invalid box at row {row_index}: {(x1, y1, x2, y2)}")
    return {
        "path": path.as_posix(),
        "rows": rows,
        "class_counts": dict(sorted(class_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "score_mode_counts": dict(sorted(score_mode_counts.items())),
        "policy_counts": dict(sorted(policy_counts.items())),
        "min_score": min_score,
        "max_score": max_score,
    }


def export_variant(
    args: argparse.Namespace,
    variant: str,
) -> dict[str, Any]:
    transform_json = variant_transform_json(args, variant)
    score_policy_json = variant_score_policy_json(args, variant)
    scale_json = variant_scale_json(args, variant)
    box_transforms = load_box_transforms(transform_json)
    score_policies = load_score_policies(score_policy_json)
    score_scales = load_score_scales(scale_json)
    prediction_dir = args.output_dir / variant
    prediction_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, Any] = {}
    for split in args.splits:
        export_summary = export_split(
            args.run_dir,
            prediction_dir,
            split,
            topk_per_sample_class=int(args.topk_per_sample_class),
            box_transforms=box_transforms,
            score_policies=score_policies,
            fallback_domain=args.fallback_domain,
        )
        prediction_csv = prediction_dir / f"{split}_domain_policy_predictions.csv"
        scale_summary = apply_score_scales_to_csv(prediction_csv, score_scales)
        stage2aq_summary = None
        if variant == "stage2aq":
            stage2aq_summary = apply_stage2aq_replacement_to_csv(
                prediction_csv,
                args.stage2aq_multisource_run_dir,
                split,
                source_policy=args.stage2aq_source_policy,
                score_mode=args.stage2aq_score_mode,
                topk=int(args.stage2aq_topk),
                score_scale=float(args.stage2aq_score_scale),
            )
        schema_summary = summarize_prediction_csv(prediction_csv)
        metrics = None if args.skip_metrics else recompute_metrics(args.run_dir, prediction_dir, split)
        splits[split] = {
            "export": export_summary,
            "score_scaling": scale_summary,
            "stage2aq_replacement": stage2aq_summary,
            "schema": schema_summary,
            "metrics_available": metrics is not None,
            "metrics": metrics,
        }
    manifest = {
        "artifact_type": "task2_stage2_champion_variant_export",
        "variant": variant,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir.as_posix(),
        "prediction_dir": prediction_dir.as_posix(),
        "box_transform_json": transform_json.as_posix() if transform_json is not None else None,
        "score_policy_json": score_policy_json.as_posix() if score_policy_json is not None else None,
        "score_scale_json": scale_json.as_posix() if scale_json is not None else None,
        "fallback_domain": args.fallback_domain,
        "score_scales": {
            f"class{class_id}_{domain}": scale
            for (class_id, domain), scale in sorted(score_scales.items())
        },
        "stage2aq_multisource_run_dir": args.stage2aq_multisource_run_dir.as_posix()
        if variant == "stage2aq"
        else None,
        "stage2aq_source_policy": args.stage2aq_source_policy if variant == "stage2aq" else None,
        "stage2aq_score_mode": args.stage2aq_score_mode if variant == "stage2aq" else None,
        "stage2aq_topk": int(args.stage2aq_topk) if variant == "stage2aq" else None,
        "stage2aq_score_scale": float(args.stage2aq_score_scale) if variant == "stage2aq" else None,
        "topk_per_sample_class": int(args.topk_per_sample_class),
        "splits": splits,
    }
    manifest_path = prediction_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def compact_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"variant": manifest["variant"], "splits": {}}
    for split, split_info in manifest["splits"].items():
        metrics = split_info.get("metrics")
        result["splits"][split] = {
            "rows": split_info["schema"]["rows"],
            "metrics_available": metrics is not None,
            "mAP50": metrics.get("mAP50") if isinstance(metrics, dict) else None,
            "mAP50-95": metrics.get("mAP50-95") if isinstance(metrics, dict) else None,
        }
    return result


def main() -> int:
    args = parse_args()
    variants = args.variant or list(VARIANT_ORDER)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = [export_variant(args, variant) for variant in variants]
    summary = {
        "artifact_type": "task2_stage2_champion_pipeline_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "splits": list(args.splits),
        "variants": [compact_metrics(manifest) for manifest in manifests],
    }
    summary_path = args.output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
