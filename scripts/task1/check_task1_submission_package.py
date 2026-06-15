#!/usr/bin/env python3
"""Preflight check for the Task 1 (segmentation) submission package.

Mirrors scripts/task2/check_task2_submission_package.py: verifies required code,
config, and weight files exist, optionally validates weight checksums against a
manifest, and (with --strict) exits non-zero if any required item is invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.task1.run_task1_submission_inference import (  # noqa: E402
    REQUIRED_FILES,
    REQUIRED_WEIGHT_FILES,
)

DEFAULT_MANIFEST = REPO_ROOT / "quality_reports/decisions/2026-06-15_task1_submission_package_preflight.json"


@dataclass(frozen=True)
class Check:
    kind: str
    path: str
    required: bool


def display_path(value: str | Path) -> str:
    path = Path(value)
    abs_path = path if path.is_absolute() else (REPO_ROOT / path)
    try:
        return abs_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return abs_path.resolve().as_posix()


def _abs(path_text: str) -> Path:
    return (REPO_ROOT / path_text) if not Path(path_text).is_absolute() else Path(path_text)


def evaluate_check(check: Check) -> dict[str, object]:
    abs_path = _abs(check.path)
    exists = abs_path.exists()
    if check.kind.endswith("_dir"):
        valid = abs_path.is_dir()
    elif check.kind == "output_dir_creatable":
        ancestor = abs_path
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        valid = ancestor.is_dir()
        exists = ancestor.is_dir()
    else:  # file
        valid = abs_path.is_file() and abs_path.stat().st_size > 0
    return {
        "kind": check.kind,
        "path": display_path(check.path),
        "required": check.required,
        "exists": exists,
        "valid": valid,
        "size_bytes": abs_path.stat().st_size if abs_path.is_file() else None,
    }


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_checksum_manifest(manifest_path: Path) -> list[dict[str, object]]:
    records = json.loads(manifest_path.read_text(encoding="utf-8")).get("weights", [])
    by_path = {record["path"]: record for record in records}
    results: list[dict[str, object]] = []
    for weight in REQUIRED_WEIGHT_FILES:
        key = display_path(weight)
        abs_path = _abs(weight)
        record = by_path.get(key)
        if record is None:
            results.append({"path": key, "valid": False, "reason": "missing_from_checksum_manifest"})
            continue
        if not abs_path.is_file():
            results.append({"path": key, "valid": False, "reason": "weight_file_missing"})
            continue
        size_ok = abs_path.stat().st_size == record.get("size_bytes")
        sha_ok = sha256_file(abs_path) == record.get("sha256")
        results.append({
            "path": key,
            "valid": bool(size_ok and sha_ok),
            "size_match": size_ok,
            "sha256_match": sha_ok,
        })
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checksum-manifest", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = [Check("file", path, True) for path in REQUIRED_FILES]
    checks += [Check("weight_file", path, True) for path in REQUIRED_WEIGHT_FILES]
    if args.image_dir is not None:
        checks.append(Check("image_dir", str(args.image_dir), True))
    if args.output_dir is not None:
        checks.append(Check("output_dir_creatable", str(args.output_dir), True))

    results = [evaluate_check(check) for check in checks]
    failures = [item for item in results if item["required"] and not item["valid"]]

    checksum_results: list[dict[str, object]] = []
    if args.checksum_manifest is not None:
        manifest_path = _abs(str(args.checksum_manifest))
        checksum_results = evaluate_checksum_manifest(manifest_path)
        failures += [item for item in checksum_results if not item["valid"]]

    manifest = {
        "artifact_type": "task1_submission_package_preflight",
        "checks": results,
        "checksum_checks": checksum_results,
        "failure_count": len(failures),
        "failures": failures,
    }
    out = args.manifest if args.manifest.is_absolute() else (REPO_ROOT / args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))

    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
