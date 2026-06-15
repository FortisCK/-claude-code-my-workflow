#!/usr/bin/env python3
"""Verify the frozen Stage2AB domain-aware Task2 export."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.task2.export_clean_predictions import CLEAN_FIELDS
from scripts.task2.verify_stage2v_champion_export import FORBIDDEN_FIELDS, REQUIRED_SPLITS


DEFAULT_EXPECTED_VALID_COMBINED_MAP50 = 0.21128164745000094
DEFAULT_EXPECTED_VALID_COMBINED_MAP50_95 = 0.050628135044670744
DEFAULT_TOLERANCE = 1e-12
EXPECTED_POLICIES = {
    "0": {
        "phantom": "prob_iou75_source_rank_decay_roi",
        "animal": "prob_iou75_source_rank_decay_roi",
    },
    "1": {
        "phantom": "rank_decay_roi",
        "animal": "prob_iou75_roi",
    },
}


@dataclass(frozen=True)
class DomainPolicyCsvSummary:
    split: str
    path: str
    rows: int
    class_counts: dict[str, int]
    domain_counts: dict[str, int]
    policy_counts: dict[str, int]
    score_mode_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing stage2ab_manifest.json and prediction CSV directory.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Defaults to ARTIFACT_DIR/stage2ab_manifest.json.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="Defaults to ARTIFACT_DIR/yolo_only_domain_policy_predictions.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
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
        help="Skip exact frozen public-valid mAP checks for non-public-valid exports.",
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


def prediction_path(prediction_dir: Path, split: str) -> Path:
    return prediction_dir / f"{split}_domain_policy_predictions.csv"


def summarize_csv(path: Path, split: str) -> DomainPolicyCsvSummary:
    if not path.exists():
        raise FileNotFoundError(f"{split}: missing prediction CSV: {path}")
    rows = 0
    class_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    score_mode_counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        expected_header = list(CLEAN_FIELDS)
        if header != expected_header:
            forbidden_present = sorted(set(header) & FORBIDDEN_FIELDS)
            if forbidden_present:
                raise ValueError(f"{split}: forbidden validation/leakage fields present: {forbidden_present}")
            raise ValueError(f"{split}: expected header {expected_header}, got {header}")
        for row_index, row in enumerate(reader, start=2):
            rows += 1
            class_id = str(int(row["class_id"]))
            domain = row["domain"]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            policy_counts[row["policy_name"]] = policy_counts.get(row["policy_name"], 0) + 1
            score_mode_counts[row["score_mode"]] = score_mode_counts.get(row["score_mode"], 0) + 1
            if class_id not in ("0", "1"):
                raise ValueError(f"{split}: invalid class_id at row {row_index}: {row['class_id']}")
            if domain not in ("phantom", "animal"):
                raise ValueError(f"{split}: invalid domain at row {row_index}: {domain!r}")
            expected_score_mode = EXPECTED_POLICIES[class_id][domain]
            if row["score_mode"] != expected_score_mode:
                raise ValueError(
                    f"{split}: row {row_index} expected score_mode {expected_score_mode} "
                    f"for class {class_id}/{domain}, got {row['score_mode']}"
                )
            x1, y1, x2, y2 = (float(row[name]) for name in ("x1", "y1", "x2", "y2"))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{split}: invalid box at row {row_index}: {(x1, y1, x2, y2)}")
    return DomainPolicyCsvSummary(
        split=split,
        path=path.as_posix(),
        rows=rows,
        class_counts=dict(sorted(class_counts.items())),
        domain_counts=dict(sorted(domain_counts.items())),
        policy_counts=dict(sorted(policy_counts.items())),
        score_mode_counts=dict(sorted(score_mode_counts.items())),
    )


def validate_manifest(
    manifest: dict[str, Any],
    summaries: dict[str, DomainPolicyCsvSummary],
    *,
    expected_valid_combined_map50: float,
    expected_valid_combined_map50_95: float,
    tolerance: float,
    check_frozen_metrics: bool,
) -> None:
    if manifest.get("artifact_type") != "task2_stage2ab_domain_aware_internal_export":
        raise ValueError(f"Unexpected artifact_type: {manifest.get('artifact_type')!r}")
    policies = manifest.get("best_class_policies")
    if not isinstance(policies, dict):
        raise ValueError("Manifest missing best_class_policies")
    for class_id, expected_domain_policy in EXPECTED_POLICIES.items():
        class_policy = policies.get(class_id, {}).get("policy")
        if not isinstance(class_policy, dict):
            raise ValueError(f"Manifest missing policy for class {class_id}")
        for domain, expected_score_mode in expected_domain_policy.items():
            if class_policy.get(domain) != expected_score_mode:
                raise ValueError(
                    f"Manifest class {class_id}/{domain}: expected {expected_score_mode}, "
                    f"got {class_policy.get(domain)}"
                )

    split_metrics = manifest.get("splits")
    if not isinstance(split_metrics, dict):
        raise ValueError("Manifest missing splits")
    for split, summary in summaries.items():
        split_info = split_metrics.get(split)
        if not isinstance(split_info, dict):
            raise ValueError(f"Manifest missing split {split}")
        if int(split_info.get("rows", -1)) != summary.rows:
            raise ValueError(
                f"{split}: CSV row count {summary.rows} does not match manifest rows {split_info.get('rows')}"
            )
        for key in ("mAP50", "mAP50-95", "classes"):
            if key not in split_info:
                raise ValueError(f"Manifest missing {split}.{key}")
    if check_frozen_metrics:
        combined = split_metrics["valid_combined"]
        require_close(
            "valid_combined mAP50",
            float(combined["mAP50"]),
            expected_valid_combined_map50,
            tolerance,
        )
        require_close(
            "valid_combined mAP50-95",
            float(combined["mAP50-95"]),
            expected_valid_combined_map50_95,
            tolerance,
        )


def verify_stage2ab(
    artifact_dir: Path,
    manifest_path: Path,
    prediction_dir: Path,
    *,
    expected_valid_combined_map50: float = DEFAULT_EXPECTED_VALID_COMBINED_MAP50,
    expected_valid_combined_map50_95: float = DEFAULT_EXPECTED_VALID_COMBINED_MAP50_95,
    tolerance: float = DEFAULT_TOLERANCE,
    check_frozen_metrics: bool = True,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    summaries = {
        split: summarize_csv(prediction_path(prediction_dir, split), split)
        for split in REQUIRED_SPLITS
    }
    validate_manifest(
        manifest,
        summaries,
        expected_valid_combined_map50=expected_valid_combined_map50,
        expected_valid_combined_map50_95=expected_valid_combined_map50_95,
        tolerance=tolerance,
        check_frozen_metrics=check_frozen_metrics,
    )
    return {
        "artifact_type": "task2_stage2ab_domain_policy_verification",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": artifact_dir.as_posix(),
        "manifest_json": manifest_path.as_posix(),
        "prediction_dir": prediction_dir.as_posix(),
        "base_run_dir": manifest.get("base_run_dir", ""),
        "best_class_policies": manifest["best_class_policies"],
        "splits": {
            split: {
                "csv": asdict(summary),
                "mAP50": manifest["splits"][split]["mAP50"],
                "mAP50-95": manifest["splits"][split]["mAP50-95"],
            }
            for split, summary in summaries.items()
        },
    }


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir
    manifest_path = args.manifest_json or artifact_dir / "stage2ab_manifest.json"
    prediction_dir = args.prediction_dir or artifact_dir / "yolo_only_domain_policy_predictions"
    output_json = args.output_json or artifact_dir / "stage2ab_verification.json"
    result = verify_stage2ab(
        artifact_dir,
        manifest_path,
        prediction_dir,
        expected_valid_combined_map50=args.expected_valid_combined_map50,
        expected_valid_combined_map50_95=args.expected_valid_combined_map50_95,
        tolerance=args.tolerance,
        check_frozen_metrics=not args.skip_frozen_metric_check,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "verification": output_json.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

