#!/usr/bin/env python3
"""Verify upstream artifacts required by the Stage2AQ export policy.

The verifier is intentionally inference-safe: it does not require GT fields and
does not use labels to make predictions. When GT fields are present, it records
that fact for public-validation diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.task2.export_clean_predictions import CLEAN_FIELDS  # noqa: E402
from scripts.task2.run_stage2_champion_pipeline import (  # noqa: E402
    AQ_SOURCE_POLICIES,
    DEFAULT_AQ_MULTISOURCE_RUN_DIR,
    DEFAULT_AQ_SCORE_MODE,
    DEFAULT_AQ_SCORE_SCALE,
    DEFAULT_AQ_SOURCE_POLICY,
    DEFAULT_AQ_TOPK,
    summarize_prediction_csv,
)


DEFAULT_BASELINE_DIR = Path(
    "outputs/task2/stage2ar_champion_pipeline_with_aq/public_valid_ab_ad_ae_ai_an_aq/stage2ae"
)
DEFAULT_STAGE2AQ_DIR = Path(
    "outputs/task2/stage2ar_champion_pipeline_with_aq/public_valid_ab_ad_ae_ai_an_aq/stage2aq"
)
REQUIRED_CANDIDATE_FIELDS = {
    "sample_id",
    "video_id",
    "frame_index",
    "domain",
    "source",
    "source_rank",
    "x1",
    "y1",
    "x2",
    "y2",
}
ALIGNMENT_FIELDS = ("sample_id", "source", "source_rank")
GT_FIELDS = {"gt_class", "gt_x1", "gt_y1", "gt_x2", "gt_y2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--multisource-run-dir", type=Path, default=DEFAULT_AQ_MULTISOURCE_RUN_DIR)
    parser.add_argument("--stage2aq-dir", type=Path, default=DEFAULT_STAGE2AQ_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("valid_combined", "valid_phantom", "valid_animal"),
    )
    parser.add_argument("--source-policy", choices=sorted(AQ_SOURCE_POLICIES), default=DEFAULT_AQ_SOURCE_POLICY)
    parser.add_argument("--score-mode", default=DEFAULT_AQ_SCORE_MODE)
    parser.add_argument("--topk", type=int, default=DEFAULT_AQ_TOPK)
    parser.add_argument("--score-scale", type=float, default=DEFAULT_AQ_SCORE_SCALE)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/task2/stage2as_aq_upstream_verify/verify_summary.json"))
    parser.add_argument(
        "--allow-missing-stage2aq",
        action="store_true",
        help="Verify upstream baseline/multisource inputs even if Stage2AQ exported clean predictions are absent.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def require_fields(path: Path, fields: list[str], required: set[str]) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{path}: missing required fields: {missing}")


def count_rows(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def validate_alignment(
    candidates: list[dict[str, str]],
    predictions: list[dict[str, str]],
    *,
    split: str,
) -> None:
    if len(candidates) != len(predictions):
        raise ValueError(f"{split}: candidate/prediction row mismatch: {len(candidates)} vs {len(predictions)}")
    for row_index, (candidate, prediction) in enumerate(zip(candidates, predictions), start=2):
        for field in ALIGNMENT_FIELDS:
            if str(candidate.get(field, "")) != str(prediction.get(field, "")):
                raise ValueError(
                    f"{split}: candidate/prediction alignment mismatch at row {row_index} "
                    f"for {field}: {candidate.get(field)!r} vs {prediction.get(field)!r}"
                )


def validate_clean_csv(path: Path) -> dict[str, Any]:
    fields, _ = read_csv(path)
    if fields != list(CLEAN_FIELDS):
        raise ValueError(f"{path}: expected clean fields {list(CLEAN_FIELDS)}, got {fields}")
    return summarize_prediction_csv(path)


def summarize_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    baseline_csv = args.baseline_dir / f"{split}_domain_policy_predictions.csv"
    multisource_candidates_csv = args.multisource_run_dir / f"{split}_candidates_used.csv"
    multisource_predictions_csv = args.multisource_run_dir / f"{split}_eval_prediction_rows.csv"
    stage2aq_csv = args.stage2aq_dir / f"{split}_domain_policy_predictions.csv"

    baseline_summary = validate_clean_csv(baseline_csv)
    candidate_fields, candidate_rows = read_csv(multisource_candidates_csv)
    prediction_fields, prediction_rows = read_csv(multisource_predictions_csv)
    require_fields(multisource_candidates_csv, candidate_fields, REQUIRED_CANDIDATE_FIELDS)
    score_key = f"{args.score_mode}_score_collision"
    require_fields(multisource_predictions_csv, prediction_fields, set(ALIGNMENT_FIELDS) | {score_key})
    validate_alignment(candidate_rows, prediction_rows, split=split)

    source_keep = AQ_SOURCE_POLICIES[str(args.source_policy)]
    source_counts = Counter(row["source"] for row in candidate_rows)
    missing_sources = sorted(source for source in source_keep if source_counts.get(source, 0) == 0)
    if missing_sources:
        raise ValueError(f"{split}: Stage2AQ source policy {args.source_policy} missing sources: {missing_sources}")

    phantom_class1_candidates = [
        row
        for row in candidate_rows
        if row.get("domain", "").strip().lower() == "phantom" and row.get("source", "") in source_keep
    ]
    if not phantom_class1_candidates and split != "valid_animal":
        raise ValueError(f"{split}: no phantom candidates for Stage2AQ source policy {args.source_policy}")

    stage2aq_summary = None
    if stage2aq_csv.exists():
        stage2aq_summary = validate_clean_csv(stage2aq_csv)
    elif not args.allow_missing_stage2aq:
        raise FileNotFoundError(stage2aq_csv)

    return {
        "split": split,
        "baseline_csv": baseline_csv.as_posix(),
        "multisource_candidates_csv": multisource_candidates_csv.as_posix(),
        "multisource_predictions_csv": multisource_predictions_csv.as_posix(),
        "stage2aq_csv": stage2aq_csv.as_posix() if stage2aq_csv.exists() else None,
        "baseline_summary": baseline_summary,
        "stage2aq_summary": stage2aq_summary,
        "candidate_rows": len(candidate_rows),
        "prediction_rows": len(prediction_rows),
        "candidate_has_gt": GT_FIELDS.issubset(set(candidate_fields)),
        "required_score_key": score_key,
        "source_counts": count_rows(candidate_rows, "source"),
        "domain_counts": count_rows(candidate_rows, "domain"),
        "gt_class_counts": count_rows(candidate_rows, "gt_class") if "gt_class" in candidate_fields else None,
        "stage2aq_source_policy": args.source_policy,
        "stage2aq_source_keep": sorted(source_keep),
        "phantom_policy_candidate_rows": len(phantom_class1_candidates),
        "stage2aq_topk": int(args.topk),
        "stage2aq_score_scale": float(args.score_scale),
    }


def main() -> int:
    args = parse_args()
    split_summaries = {split: summarize_split(args, split) for split in args.splits}
    manifest = {
        "artifact_type": "task2_stage2as_aq_upstream_verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_dir": args.baseline_dir.as_posix(),
        "multisource_run_dir": args.multisource_run_dir.as_posix(),
        "stage2aq_dir": args.stage2aq_dir.as_posix(),
        "source_policy": args.source_policy,
        "score_mode": args.score_mode,
        "topk": int(args.topk),
        "score_scale": float(args.score_scale),
        "splits": split_summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
