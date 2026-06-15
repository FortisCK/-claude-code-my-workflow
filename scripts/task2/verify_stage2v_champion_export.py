#!/usr/bin/env python3
"""Verify the frozen Stage2V Task2 champion export.

This script checks the internal clean prediction package. It is intentionally
stricter than a generic CSV validator because it freezes the current champion
policy and catches the mistakes that matter before hidden-test adaptation:

- missing split CSVs;
- validation-only leakage columns;
- row-count drift between CSVs and ``eval_metrics.json``;
- accidental metric or score-policy changes.

It does not convert to the official CATHACTION hidden-test schema. That adapter
should be added once the challenge organizers publish the final result format.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.task2.export_clean_predictions import CLEAN_FIELDS


DEFAULT_EXPECTED_CLASS0_SCORE_MODE = "prob_iou75_source_rank_decay_roi"
DEFAULT_EXPECTED_CLASS1_SCORE_MODE = "roi"
DEFAULT_EXPECTED_VALID_COMBINED_MAP50 = 0.20096803092334323
DEFAULT_EXPECTED_VALID_COMBINED_MAP50_95 = 0.04924044637514719
DEFAULT_TOLERANCE = 1e-12
REQUIRED_SPLITS = ("valid_combined", "valid_phantom", "valid_animal")
FORBIDDEN_FIELDS = {
    "gt_class",
    "gt_iou",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "candidate_iou",
    "candidate_label",
    "verifier_label",
}


@dataclass(frozen=True)
class SplitCsvSummary:
    split: str
    path: str
    rows: int
    class_counts: dict[str, int]
    policy_counts: dict[str, int]
    score_mode_counts: dict[str, int]
    min_score: float | None
    max_score: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing *_predictions.csv and eval_metrics.json.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional metrics path. Defaults to PREDICTION_DIR/eval_metrics.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional verifier manifest path. Defaults to PREDICTION_DIR/champion_manifest.json.",
    )
    parser.add_argument(
        "--expected-class0-score-mode",
        default=DEFAULT_EXPECTED_CLASS0_SCORE_MODE,
    )
    parser.add_argument(
        "--expected-class1-score-mode",
        default=DEFAULT_EXPECTED_CLASS1_SCORE_MODE,
    )
    parser.add_argument(
        "--expected-valid-combined-map50",
        type=float,
        default=DEFAULT_EXPECTED_VALID_COMBINED_MAP50,
    )
    parser.add_argument(
        "--expected-valid-combined-map50-95",
        type=float,
        default=DEFAULT_EXPECTED_VALID_COMBINED_MAP50_95,
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--skip-frozen-metric-check",
        action="store_true",
        help=(
            "Skip exact valid_combined mAP checks. Use this when verifying an "
            "official validation/hidden-test export whose metrics are unavailable "
            "or intentionally different from the frozen public-valid champion."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def require_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name}: expected {expected}, got {actual}")


def count_csv_rows(path: Path, split: str) -> SplitCsvSummary:
    if not path.exists():
        raise FileNotFoundError(f"{split}: missing prediction CSV: {path}")

    rows = 0
    class_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    score_mode_counts: dict[str, int] = {}
    min_score: float | None = None
    max_score: float | None = None

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        expected_header = list(CLEAN_FIELDS)
        if header != expected_header:
            forbidden_present = sorted(set(header) & FORBIDDEN_FIELDS)
            if forbidden_present:
                raise ValueError(
                    f"{split}: forbidden validation/leakage fields present: {forbidden_present}"
                )
            raise ValueError(f"{split}: expected header {expected_header}, got {header}")

        for row_index, row in enumerate(reader, start=2):
            rows += 1
            class_id = row["class_id"]
            policy_name = row["policy_name"]
            score_mode = row["score_mode"]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            policy_counts[policy_name] = policy_counts.get(policy_name, 0) + 1
            score_mode_counts[score_mode] = score_mode_counts.get(score_mode, 0) + 1
            score = float(row["score"])
            min_score = score if min_score is None else min(min_score, score)
            max_score = score if max_score is None else max(max_score, score)
            if row["sample_id"] == "":
                raise ValueError(f"{split}: empty sample_id at row {row_index}")
            if int(row["class_id"]) not in (0, 1):
                raise ValueError(f"{split}: invalid class_id at row {row_index}: {row['class_id']}")
            x1, y1, x2, y2 = (float(row[name]) for name in ("x1", "y1", "x2", "y2"))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{split}: invalid box at row {row_index}: {(x1, y1, x2, y2)}")

    return SplitCsvSummary(
        split=split,
        path=path.as_posix(),
        rows=rows,
        class_counts=dict(sorted(class_counts.items())),
        policy_counts=dict(sorted(policy_counts.items())),
        score_mode_counts=dict(sorted(score_mode_counts.items())),
        min_score=min_score,
        max_score=max_score,
    )


def validate_metrics(
    metrics: dict[str, Any],
    summaries: dict[str, SplitCsvSummary],
    *,
    expected_class0_score_mode: str,
    expected_class1_score_mode: str,
    expected_valid_combined_map50: float,
    expected_valid_combined_map50_95: float,
    tolerance: float,
    check_frozen_metrics: bool = True,
) -> None:
    class0_score_mode = str(metrics.get("class0_score_mode", ""))
    class1_score_mode = str(metrics.get("class1_score_mode", ""))
    if class0_score_mode != expected_class0_score_mode:
        raise ValueError(
            f"class0_score_mode: expected {expected_class0_score_mode}, got {class0_score_mode}"
        )
    if class1_score_mode != expected_class1_score_mode:
        raise ValueError(
            f"class1_score_mode: expected {expected_class1_score_mode}, got {class1_score_mode}"
        )

    split_metrics = metrics.get("splits")
    if not isinstance(split_metrics, dict):
        raise ValueError("eval_metrics.json: missing object field 'splits'")

    for split, summary in summaries.items():
        split_info = split_metrics.get(split)
        if not isinstance(split_info, dict):
            raise ValueError(f"eval_metrics.json: missing split {split}")
        expected_rows = int(split_info.get("prediction_rows", -1))
        if summary.rows != expected_rows:
            raise ValueError(
                f"{split}: CSV row count {summary.rows} does not match metrics "
                f"prediction_rows {expected_rows}"
            )
        detection = split_info.get("detection")
        if not isinstance(detection, dict):
            raise ValueError(f"eval_metrics.json: missing detection metrics for {split}")
        for key in ("mAP50", "mAP50-95", "classes"):
            if key not in detection:
                raise ValueError(f"eval_metrics.json: missing {split}.detection.{key}")

    if check_frozen_metrics:
        combined_detection = split_metrics["valid_combined"]["detection"]
        require_close(
            "valid_combined mAP50",
            float(combined_detection["mAP50"]),
            expected_valid_combined_map50,
            tolerance,
        )
        require_close(
            "valid_combined mAP50-95",
            float(combined_detection["mAP50-95"]),
            expected_valid_combined_map50_95,
            tolerance,
        )


def build_manifest(
    prediction_dir: Path,
    metrics_path: Path,
    metrics: dict[str, Any],
    summaries: dict[str, SplitCsvSummary],
) -> dict[str, Any]:
    split_metrics = metrics["splits"]
    return {
        "artifact_type": "task2_stage2v_internal_champion_export",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_dir": prediction_dir.as_posix(),
        "metrics_json": metrics_path.as_posix(),
        "class0_run_dir": metrics.get("class0_run_dir", ""),
        "class0_score_mode": metrics.get("class0_score_mode", ""),
        "class1_run_dir": metrics.get("class1_run_dir", ""),
        "class1_score_mode": metrics.get("class1_score_mode", ""),
        "topk_per_sample_class": metrics.get("topk_per_sample_class", 0),
        "splits": {
            split: {
                "csv": summary.__dict__,
                "mAP50": split_metrics[split]["detection"]["mAP50"],
                "mAP50-95": split_metrics[split]["detection"]["mAP50-95"],
                "class_metrics": split_metrics[split]["detection"]["classes"],
            }
            for split, summary in summaries.items()
        },
        "official_schema_status": (
            "internal clean CSV only; official CATHACTION hidden-test formatter pending"
        ),
    }


def verify_export(
    prediction_dir: Path,
    metrics_path: Path,
    *,
    expected_class0_score_mode: str = DEFAULT_EXPECTED_CLASS0_SCORE_MODE,
    expected_class1_score_mode: str = DEFAULT_EXPECTED_CLASS1_SCORE_MODE,
    expected_valid_combined_map50: float = DEFAULT_EXPECTED_VALID_COMBINED_MAP50,
    expected_valid_combined_map50_95: float = DEFAULT_EXPECTED_VALID_COMBINED_MAP50_95,
    tolerance: float = DEFAULT_TOLERANCE,
    check_frozen_metrics: bool = True,
) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    summaries = {
        split: count_csv_rows(prediction_dir / f"{split}_predictions.csv", split)
        for split in REQUIRED_SPLITS
    }
    validate_metrics(
        metrics,
        summaries,
        expected_class0_score_mode=expected_class0_score_mode,
        expected_class1_score_mode=expected_class1_score_mode,
        expected_valid_combined_map50=expected_valid_combined_map50,
        expected_valid_combined_map50_95=expected_valid_combined_map50_95,
        tolerance=tolerance,
        check_frozen_metrics=check_frozen_metrics,
    )
    return build_manifest(prediction_dir, metrics_path, metrics, summaries)


def main() -> int:
    args = parse_args()
    prediction_dir = args.prediction_dir
    metrics_path = args.metrics_json or prediction_dir / "eval_metrics.json"
    output_json = args.output_json or prediction_dir / "champion_manifest.json"

    manifest = verify_export(
        prediction_dir,
        metrics_path,
        expected_class0_score_mode=args.expected_class0_score_mode,
        expected_class1_score_mode=args.expected_class1_score_mode,
        expected_valid_combined_map50=args.expected_valid_combined_map50,
        expected_valid_combined_map50_95=args.expected_valid_combined_map50_95,
        tolerance=args.tolerance,
        check_frozen_metrics=not args.skip_frozen_metric_check,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": output_json.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
