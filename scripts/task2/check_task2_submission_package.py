#!/usr/bin/env python3
"""Preflight checks for the Task2 submission inference package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("quality_reports/decisions/2026-06-14_task2_submission_package_preflight.json")
REQUIRED_FILES = (
    "scripts/task2/run_task2_submission_inference.py",
    "scripts/task2/run_stage2aq_hidden_pipeline.py",
    "scripts/task2/export_yolo_nogt_proposals.py",
    "scripts/task2/export_sequence_tip_nogt_proposals.py",
    "scripts/task2/build_nogt_candidate_pool.py",
    "scripts/task2/infer_stage2u_quality_ranker.py",
    "scripts/task2/run_stage2_champion_pipeline.py",
    "scripts/task2/export_stage2ab_domain_policy_predictions.py",
)
REQUIRED_WEIGHT_FILES = (
    "outputs/task2/yolo_stage2l_proposal/yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20/weights/best.pt",
    "outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt",
    "outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt",
)


@dataclass(frozen=True)
class Check:
    kind: str
    path: Path
    required: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--image-list-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checksum-manifest", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_checks(args: argparse.Namespace) -> list[Check]:
    checks = [Check("required_file", Path(path), required=True) for path in REQUIRED_FILES]
    checks.extend(Check("required_weight", Path(path), required=True) for path in REQUIRED_WEIGHT_FILES)
    if args.image_dir is not None:
        checks.append(Check("input_image_dir", args.image_dir, required=True))
    if args.image_list_csv is not None:
        checks.append(Check("input_image_list_csv", args.image_list_csv, required=True))
    if args.output_dir is not None:
        checks.append(Check("output_dir_creatable", args.output_dir, required=True))
    return checks


def nearest_existing_ancestor(path: Path) -> Path | None:
    current = path
    while True:
        if current.exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def evaluate_check(check: Check) -> dict[str, Any]:
    full = repo_path(check.path)
    exists = full.exists()
    if check.kind == "output_dir_creatable":
        ancestor = nearest_existing_ancestor(full)
        valid = bool(ancestor is not None and ancestor.is_dir() and os.access(ancestor, os.W_OK))
        return {
            "kind": check.kind,
            "path": display_path(full),
            "required": check.required,
            "exists": exists,
            "valid": valid,
            "size_bytes": None,
            "nearest_existing_ancestor": display_path(ancestor) if ancestor is not None else None,
        }
    if check.kind.endswith("_dir"):
        valid = exists and full.is_dir()
    elif check.kind.endswith("_csv"):
        valid = exists and full.is_file() and full.stat().st_size > 0
    else:
        valid = exists and full.is_file() and full.stat().st_size > 0
    return {
        "kind": check.kind,
        "path": display_path(full),
        "required": check.required,
        "exists": exists,
        "valid": valid,
        "size_bytes": full.stat().st_size if exists and full.is_file() else None,
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksum_manifest(path: Path) -> dict[str, dict[str, Any]]:
    with repo_path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    weights = manifest.get("weights")
    if not isinstance(weights, list):
        raise ValueError(f"{path}: expected weights list")
    records: dict[str, dict[str, Any]] = {}
    for item in weights:
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError(f"{path}: invalid weight record {item!r}")
        records[str(item["path"])] = item
    return records


def evaluate_checksum_manifest(path: Path) -> list[dict[str, Any]]:
    records = load_checksum_manifest(path)
    checks: list[dict[str, Any]] = []
    for path_text in REQUIRED_WEIGHT_FILES:
        full = repo_path(Path(path_text))
        expected = records.get(path_text)
        if expected is None:
            checks.append(
                {
                    "kind": "weight_checksum",
                    "path": path_text,
                    "required": True,
                    "exists": full.exists(),
                    "valid": False,
                    "reason": "missing_from_checksum_manifest",
                }
            )
            continue
        if not full.is_file():
            checks.append(
                {
                    "kind": "weight_checksum",
                    "path": path_text,
                    "required": True,
                    "exists": full.exists(),
                    "valid": False,
                    "reason": "weight_file_missing",
                }
            )
            continue
        actual_size = full.stat().st_size
        actual_sha256 = sha256_file(full)
        expected_size = int(expected.get("size_bytes", -1))
        expected_sha256 = str(expected.get("sha256", ""))
        checks.append(
            {
                "kind": "weight_checksum",
                "path": path_text,
                "required": True,
                "exists": True,
                "valid": actual_size == expected_size and actual_sha256 == expected_sha256,
                "size_bytes": actual_size,
                "expected_size_bytes": expected_size,
                "sha256": actual_sha256,
                "expected_sha256": expected_sha256,
            }
        )
    return checks


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    results = [evaluate_check(check) for check in build_checks(args)]
    checksum_manifest = getattr(args, "checksum_manifest", None)
    if checksum_manifest is not None:
        results.extend(evaluate_checksum_manifest(checksum_manifest))
    failures = [item for item in results if item["required"] and not item["valid"]]
    return {
        "artifact_type": "task2_submission_package_preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": REPO_ROOT.as_posix(),
        "checks": results,
        "failure_count": len(failures),
        "failures": failures,
        "official_format_status": (
            "Official Task2 result-file schema is not specified in the available "
            "2026 CATHACTION PDF; this preflight checks the current internal "
            "submission inference package only."
        ),
    }


def main() -> int:
    args = parse_args()
    if args.image_dir is not None and args.image_list_csv is not None:
        raise ValueError("Use only one of --image-dir or --image-list-csv")
    manifest = build_manifest(args)
    manifest_path = repo_path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.strict and int(manifest["failure_count"]) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
