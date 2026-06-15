#!/usr/bin/env python3
"""Submission-facing Task2 inference entrypoint.

This script wraps the hidden/no-GT Stage2AQ pipeline behind a stable interface
for future Docker use. It intentionally exports the current internal prediction
CSV schema only; the official challenge result schema is not defined in the
available 2026 PDF yet.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_FILENAME = "task2_predictions_internal.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image-dir", type=Path)
    input_group.add_argument("--image-list-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="hidden")
    parser.add_argument("--work-subdir", default="stage2aq_work")
    parser.add_argument("--result-filename", default=DEFAULT_RESULT_FILENAME)
    parser.add_argument("--fallback-domain", default="phantom")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--proposal-limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Pass strict preflight through to the hidden wrapper.")
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def final_internal_prediction_csv(output_dir: Path, work_subdir: str, split: str) -> Path:
    return output_dir / work_subdir / "hidden_stage2ae_stage2aq_predictions" / "stage2aq" / (
        f"{split}_domain_policy_predictions.csv"
    )


def result_csv_path(output_dir: Path, filename: str) -> Path:
    return output_dir / filename


def wrapper_manifest_path(output_dir: Path, work_subdir: str) -> Path:
    return output_dir / work_subdir / "stage2aq_hidden_pipeline_manifest.json"


def submission_manifest_path(output_dir: Path) -> Path:
    return output_dir / "task2_submission_inference_manifest.json"


def build_hidden_pipeline_command(args: argparse.Namespace) -> list[str]:
    output_dir = repo_path(args.output_dir)
    work_dir = output_dir / args.work_subdir
    command = [
        args.python.as_posix(),
        "scripts/task2/run_stage2aq_hidden_pipeline.py",
        "--generate-proposals",
        "--split",
        str(args.split),
        "--work-dir",
        work_dir.as_posix(),
        "--manifest",
        wrapper_manifest_path(output_dir, args.work_subdir).as_posix(),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--fallback-domain",
        str(args.fallback_domain),
    ]
    if args.image_dir is not None:
        command.extend(["--raw-image-dir", repo_path(args.image_dir).as_posix()])
    else:
        command.extend(["--raw-image-list-csv", repo_path(args.image_list_csv).as_posix()])
    if args.proposal_limit is not None:
        command.extend(["--proposal-limit", str(args.proposal_limit)])
    if args.strict:
        command.append("--strict")
    if args.execute:
        command.append("--execute")
    return command


def build_manifest(args: argparse.Namespace, command: list[str], *, executed: bool) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    internal_csv = final_internal_prediction_csv(output_dir, args.work_subdir, str(args.split))
    result_csv = result_csv_path(output_dir, str(args.result_filename))
    return {
        "artifact_type": "task2_submission_inference_entrypoint",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if executed else "dry_run",
        "repo_root": REPO_ROOT.as_posix(),
        "input": {
            "image_dir": display_path(repo_path(args.image_dir)) if args.image_dir is not None else None,
            "image_list_csv": display_path(repo_path(args.image_list_csv)) if args.image_list_csv is not None else None,
        },
        "split": str(args.split),
        "fallback_domain": str(args.fallback_domain),
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "proposal_limit": args.proposal_limit,
        "output_dir": display_path(output_dir),
        "wrapper_manifest": display_path(wrapper_manifest_path(output_dir, args.work_subdir)),
        "internal_stage2aq_prediction_csv": display_path(internal_csv),
        "stable_internal_result_csv": display_path(result_csv),
        "command": command,
        "official_format_status": (
            "The available 2026 CATHACTION PDF requires a predefined result file "
            "but does not specify the concrete Task2 schema. This entrypoint "
            "therefore exports the current internal prediction CSV until the "
            "official validation package or platform instructions define the "
            "required schema."
        ),
    }


def copy_internal_result(args: argparse.Namespace) -> None:
    output_dir = repo_path(args.output_dir)
    source = final_internal_prediction_csv(output_dir, args.work_subdir, str(args.split))
    destination = result_csv_path(output_dir, str(args.result_filename))
    if not source.exists():
        raise FileNotFoundError(f"Expected internal Stage2AQ prediction CSV was not produced: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main() -> int:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_hidden_pipeline_command(args)
    manifest = build_manifest(args, command, executed=bool(args.execute))
    submission_manifest_path(output_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.execute:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        copy_internal_result(args)
        manifest = build_manifest(args, command, executed=True)
        manifest["stable_internal_result_exists"] = str(
            result_csv_path(output_dir, str(args.result_filename)).exists()
        ).lower()
        submission_manifest_path(output_dir).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

