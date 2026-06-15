#!/usr/bin/env python3
"""Write sha256 checksums for Task2 submission weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("quality_reports/decisions/2026-06-14_task2_weight_manifest.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.task2.check_task2_submission_package import REQUIRED_WEIGHT_FILES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_record(path_text: str) -> dict[str, Any]:
    path = repo_path(Path(path_text))
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": display_path(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def build_manifest() -> dict[str, Any]:
    return {
        "artifact_type": "task2_weight_checksum_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": REPO_ROOT.as_posix(),
        "weights": [weight_record(path) for path in REQUIRED_WEIGHT_FILES],
    }


def main() -> int:
    args = parse_args()
    manifest = build_manifest()
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
