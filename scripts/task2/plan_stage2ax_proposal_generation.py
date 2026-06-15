#!/usr/bin/env python3
"""Plan hidden-test proposal generation sources for Stage2AQ.

This script records which proposal sources are currently hidden-ready and emits
the no-GT YOLO command. It is intentionally a planning/manifest helper: sources
that still require no-GT adapters are marked explicitly instead of silently
pretending they can run on hidden labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_WEIGHTS = Path(
    "outputs/task2/yolo_stage2l_proposal/"
    "yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20/weights/best.pt"
)
DEFAULT_STAGE2X_CHECKPOINT = Path(
    "outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt"
)


SOURCE_STATUS = {
    "yolo_stage2l": {
        "status": "hidden_ready",
        "script": "scripts/task2/export_yolo_nogt_proposals.py",
        "reason": "Runs directly from image directory or image-list CSV and does not require GT.",
    },
    "task1_geometry_rect": {
        "status": "needs_nogt_adapter",
        "script": "scripts/task2/evaluate_task1_geometry_proposals.py",
        "reason": "Current script loads labeled Task2Sample splits and writes GT diagnostics.",
    },
    "sequence_tip": {
        "status": "needs_nogt_adapter",
        "script": "scripts/task2/evaluate_sequence_tip_proposals.py",
        "reason": "Current script uses SequenceTipDataset with labeled samples and GT diagnostics.",
    },
    "stage2w_dense": {
        "status": "needs_nogt_adapter",
        "script": "scripts/task2/evaluate_sequence_tip_proposals.py",
        "reason": "Same sequence-tip evaluator path; needs hidden sequence dataset adapter.",
    },
    "stage2x_class1": {
        "status": "hidden_ready",
        "script": "scripts/task2/export_sequence_tip_nogt_proposals.py",
        "reason": "No-GT sequence-tip exporter can run the Stage2X class1 checkpoint from image sequences.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--split", default="hidden")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-dir", type=Path)
    group.add_argument("--image-list-csv", type=Path)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--stage2x-checkpoint", type=Path, default=DEFAULT_STAGE2X_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/stage2ax_proposal_generation"))
    parser.add_argument("--name", default="hidden_yolo_stage2l")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-chunk-size", type=int, default=64)
    parser.add_argument("--fallback-domain", default="unknown")
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def yolo_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python.as_posix(),
        "scripts/task2/export_yolo_nogt_proposals.py",
        "--weights",
        args.yolo_weights.as_posix(),
        "--split",
        str(args.split),
        "--output-dir",
        args.output_dir.as_posix(),
        "--name",
        str(args.name),
        "--device",
        str(args.device),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        str(args.max_det),
        "--batch-size",
        str(args.batch_size),
        "--source-chunk-size",
        str(args.source_chunk_size),
        "--fallback-domain",
        str(args.fallback_domain),
    ]
    if args.image_dir is not None:
        cmd.extend(["--image-dir", args.image_dir.as_posix()])
    else:
        cmd.extend(["--image-list-csv", args.image_list_csv.as_posix()])
    return cmd


def stage2x_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python.as_posix(),
        "scripts/task2/export_sequence_tip_nogt_proposals.py",
        "--checkpoint",
        args.stage2x_checkpoint.as_posix(),
        "--split",
        str(args.split),
        "--output-dir",
        args.output_dir.as_posix(),
        "--name",
        "hidden_stage2x_class1",
        "--device",
        str(args.device),
        "--fallback-domain",
        str(args.fallback_domain),
    ]
    if args.image_dir is not None:
        cmd.extend(["--image-dir", args.image_dir.as_posix()])
    else:
        cmd.extend(["--image-list-csv", args.image_list_csv.as_posix()])
    return cmd


def preflight(args: argparse.Namespace) -> list[dict[str, str]]:
    checks = [
        {
            "kind": "yolo_weights",
            "path": display_path(repo_path(args.yolo_weights)),
            "exists": str(repo_path(args.yolo_weights).exists()).lower(),
        },
        {
            "kind": "stage2x_checkpoint",
            "path": display_path(repo_path(args.stage2x_checkpoint)),
            "exists": str(repo_path(args.stage2x_checkpoint).exists()).lower(),
        }
    ]
    if args.image_dir is not None:
        checks.append(
            {
                "kind": "image_dir",
                "path": display_path(repo_path(args.image_dir)),
                "exists": str(repo_path(args.image_dir).exists()).lower(),
            }
        )
    else:
        checks.append(
            {
                "kind": "image_list_csv",
                "path": display_path(repo_path(args.image_list_csv)),
                "exists": str(repo_path(args.image_list_csv).exists()).lower(),
            }
        )
    return checks


def main() -> int:
    args = parse_args()
    output_run_dir = repo_path(args.output_dir) / args.name
    output_csv = output_run_dir / f"{args.split}_proposals.csv"
    stage2x_output_csv = repo_path(args.output_dir) / "hidden_stage2x_class1" / f"{args.split}_proposals.csv"
    manifest = {
        "artifact_type": "task2_stage2ax_proposal_generation_plan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_status": SOURCE_STATUS,
        "commands": [
            {
                "name": "export_yolo_stage2l_nogt_proposals",
                "argv": yolo_command(args),
                "outputs": [display_path(output_csv)],
                "source": "yolo_stage2l",
                "status": SOURCE_STATUS["yolo_stage2l"]["status"],
            },
            {
                "name": "export_stage2x_class1_nogt_proposals",
                "argv": stage2x_command(args),
                "outputs": [display_path(stage2x_output_csv)],
                "source": "stage2x_class1",
                "status": SOURCE_STATUS["stage2x_class1"]["status"],
            }
        ],
        "preflight_checks": preflight(args),
        "remaining_adapters": [
            name for name, info in SOURCE_STATUS.items() if info["status"] != "hidden_ready"
        ],
        "note": (
            "yolo_stage2l and stage2x_class1 are hidden-ready at this stage, "
            "which covers Stage2AQ's default replacement source policy. Geometry "
            "and other sequence-tip proposal variants still need adapters if we "
            "want the full public-validation five-source pool on hidden data."
        ),
    }
    manifest_path = repo_path(args.manifest) if args.manifest is not None else output_run_dir / "stage2ax_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
