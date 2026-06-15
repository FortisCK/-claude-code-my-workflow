#!/usr/bin/env python3
"""Orchestrate the frozen Stage2AQ Task2 pipeline for official-style splits.

The script is intentionally dry-run by default. It writes a manifest with the
exact commands needed to:

1. build a multi-source Stage2O candidate pool;
2. run the frozen Stage2U quality ranker in eval-only mode;
3. export Stage2AE and Stage2AQ clean predictions;
4. verify that Stage2AQ upstream artifacts are aligned.

This entry point assumes proposal CSVs already exist for every requested split
inside each proposal-source directory. It also assumes label-list splits are
available, because the current public-validation Stage2U evaluator computes
diagnostics from GT. A hidden-test Docker inference path should reuse the same
artifact contracts but cannot depend on GT-bearing candidate construction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPLITS = (
    "valid_combined=configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_combined_balanced_labels.txt",
    "valid_phantom=configs/task2/splits_stage2l_proposal/valid_phantom_balanced_small_labels.txt",
    "valid_animal=configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_animal_labels.txt",
)
DEFAULT_SOURCES = (
    "yolo_stage2l=outputs/task2/yolo_proposal_eval/yolo11s_1024_stage2l_combined_bal_e20_stage2l_panels",
    "task1_geometry_rect=outputs/task2/geometry_proposals/task1_stage5_geometry_d3_rect_stage2l_panels",
    "sequence_tip=outputs/task2/sequence_tip_proposals/convnext_tip384_lr1e4_noamp_e3_top20_templates",
    "stage2w_dense=outputs/task2/sequence_tip_proposals/stage2w_tip384_dense_templates_top30",
    "stage2x_class1=outputs/task2/sequence_tip_proposals/stage2x_class1_centernet_dense_templates_top30",
)
DEFAULT_TRAIN_CSV = Path(
    "outputs/task2/stage2o_candidate_pool/stage2o_train_subset_yolo_geometry_top50/valid_combined_candidates.csv"
)
DEFAULT_EVAL_CHECKPOINT = Path(
    "outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt"
)
DEFAULT_BASELINE_RUN_DIR = Path(
    "outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval"
)


@dataclass(frozen=True)
class NamedPath:
    name: str
    path: Path


def parse_named_path(value: str, *, label: str) -> NamedPath:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{label} must use name=path syntax: {value!r}")
    name, path = value.split("=", maxsplit=1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"{label} must use name=path syntax: {value!r}")
    return NamedPath(name=name, path=Path(path))


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument("--split", action="append", default=None, help="Split as name=label_list.")
    parser.add_argument("--source", action="append", default=None, help="Proposal source as name=directory.")
    parser.add_argument("--source-top-k", type=int, default=50)
    parser.add_argument("--global-top-k", default="1,5,10,20,50,100,150,200,250")
    parser.add_argument("--positive-iou", type=float, default=0.50)
    parser.add_argument("--background-iou", type=float, default=0.20)
    parser.add_argument("--baseline-source", default="yolo_stage2l")
    parser.add_argument("--candidate-output-dir", type=Path, default=Path("outputs/task2/stage2o_candidate_pool"))
    parser.add_argument("--candidate-name", default="stage2at_official_stage2x_five_source_top50")
    parser.add_argument("--stage2u-output-dir", type=Path, default=Path("outputs/task2/stage2u_quality_ranker"))
    parser.add_argument("--stage2u-name", default="stage2at_official_stage2x_eval")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--eval-checkpoint", type=Path, default=DEFAULT_EVAL_CHECKPOINT)
    parser.add_argument("--baseline-run-dir", type=Path, default=DEFAULT_BASELINE_RUN_DIR)
    parser.add_argument(
        "--champion-output-dir",
        type=Path,
        default=Path("outputs/task2/stage2at_aq_official_orchestration/champion_predictions"),
    )
    parser.add_argument(
        "--verify-output-json",
        type=Path,
        default=Path("outputs/task2/stage2at_aq_official_orchestration/stage2aq_verify_summary.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/task2/stage2at_aq_official_orchestration/manifest.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--execute", action="store_true", help="Run the generated commands.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail during dry-run when required input files are missing.",
    )
    return parser.parse_args()


def command_record(name: str, argv: list[str], *, outputs: list[Path]) -> dict[str, Any]:
    return {
        "name": name,
        "argv": argv,
        "outputs": [display_path(path) for path in outputs],
    }


def build_commands(args: argparse.Namespace, splits: list[NamedPath], sources: list[NamedPath]) -> list[dict[str, Any]]:
    split_names = [split.name for split in splits]
    candidate_run_dir = repo_path(args.candidate_output_dir) / args.candidate_name
    stage2u_run_dir = repo_path(args.stage2u_output_dir) / args.stage2u_name
    champion_output_dir = repo_path(args.champion_output_dir)
    verify_output_json = repo_path(args.verify_output_json)

    candidate_cmd = [
        args.python.as_posix(),
        "scripts/task2/build_stage2o_candidate_pool.py",
        "--data-root",
        args.data_root.as_posix(),
        "--source-top-k",
        str(args.source_top_k),
        "--global-top-k",
        str(args.global_top_k),
        "--positive-iou",
        str(args.positive_iou),
        "--background-iou",
        str(args.background_iou),
        "--baseline-source",
        str(args.baseline_source),
        "--output-dir",
        args.candidate_output_dir.as_posix(),
        "--name",
        str(args.candidate_name),
        "--seed",
        str(args.seed),
    ]
    for split in splits:
        candidate_cmd.extend(["--split", f"{split.name}={split.path.as_posix()}"])
    for source in sources:
        candidate_cmd.extend(["--source", f"{source.name}={source.path.as_posix()}"])

    ranker_cmd = [
        args.python.as_posix(),
        "scripts/task2/train_stage2u_quality_ranker.py",
        "--data-root",
        args.data_root.as_posix(),
        "--train-csv",
        args.train_csv.as_posix(),
        "--valid-combined-split",
        split_path(splits, "valid_combined").as_posix(),
        "--valid-phantom-split",
        split_path(splits, "valid_phantom").as_posix(),
        "--valid-animal-split",
        split_path(splits, "valid_animal").as_posix(),
        "--output-dir",
        args.stage2u_output_dir.as_posix(),
        "--name",
        str(args.stage2u_name),
        "--backend",
        "auto",
        "--model",
        "convnext_tiny",
        "--no-pretrained",
        "--input-size",
        "224",
        "--crop-scale",
        "8.0",
        "--min-crop-size",
        "224",
        "--max-crop-size",
        "512",
        "--epochs",
        "6",
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--lr",
        "0.0003",
        "--weight-decay",
        "0.0001",
        "--smooth-l1-beta",
        "0.05",
        "--class-loss-weight",
        "1.0",
        "--iou-loss-weight",
        "1.0",
        "--iou50-loss-weight",
        "0.5",
        "--iou75-loss-weight",
        "1.0",
        "--balanced-sampler",
        "--amp",
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--primary-score-mode",
        "prob_iou75_rank_decay_roi",
        "--primary-valid-split",
        "valid_combined",
        "--primary-map-key",
        "mAP50-95",
        "--max-candidates-per-sample",
        "0",
        "--candidate-budget-sort",
        "source_order",
        "--eval-checkpoint",
        args.eval_checkpoint.as_posix(),
    ]
    for split_name in split_names:
        ranker_cmd.extend(["--valid-csv", f"{split_name}={(candidate_run_dir / f'{split_name}_candidates.csv').as_posix()}"])

    champion_cmd = [
        args.python.as_posix(),
        "scripts/task2/run_stage2_champion_pipeline.py",
        "--run-dir",
        args.baseline_run_dir.as_posix(),
        "--output-dir",
        args.champion_output_dir.as_posix(),
        "--variant",
        "stage2ae",
        "--variant",
        "stage2aq",
        "--stage2aq-multisource-run-dir",
        stage2u_run_dir.as_posix(),
        "--splits",
        *split_names,
    ]

    verify_cmd = [
        args.python.as_posix(),
        "scripts/task2/verify_stage2aq_upstream.py",
        "--baseline-dir",
        (champion_output_dir / "stage2ae").as_posix(),
        "--multisource-run-dir",
        stage2u_run_dir.as_posix(),
        "--stage2aq-dir",
        (champion_output_dir / "stage2aq").as_posix(),
        "--output-json",
        verify_output_json.as_posix(),
        "--splits",
        *split_names,
    ]

    return [
        command_record(
            "build_stage2o_candidate_pool",
            candidate_cmd,
            outputs=[candidate_run_dir / f"{split.name}_candidates.csv" for split in splits],
        ),
        command_record(
            "eval_stage2u_quality_ranker",
            ranker_cmd,
            outputs=[stage2u_run_dir / f"{split.name}_eval_prediction_rows.csv" for split in splits],
        ),
        command_record(
            "export_stage2ae_stage2aq_predictions",
            champion_cmd,
            outputs=[
                champion_output_dir / "stage2ae" / f"{split.name}_domain_policy_predictions.csv"
                for split in splits
            ]
            + [
                champion_output_dir / "stage2aq" / f"{split.name}_domain_policy_predictions.csv"
                for split in splits
            ],
        ),
        command_record(
            "verify_stage2aq_upstream",
            verify_cmd,
            outputs=[verify_output_json],
        ),
    ]


def split_path(splits: list[NamedPath], name: str) -> Path:
    for split in splits:
        if split.name == name:
            return split.path
    raise ValueError(
        f"Stage2U currently requires split {name!r}; got {[split.name for split in splits]}. "
        "Keep valid_combined/valid_phantom/valid_animal names or generalize train_stage2u_quality_ranker.py first."
    )


def preflight(args: argparse.Namespace, splits: list[NamedPath], sources: list[NamedPath]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    def add(kind: str, path: Path, *, required: bool = True) -> None:
        full_path = repo_path(path)
        checks.append(
            {
                "kind": kind,
                "path": display_path(full_path),
                "exists": str(full_path.exists()).lower(),
                "required": str(required).lower(),
            }
        )

    add("data_root", args.data_root)
    add("train_csv", args.train_csv)
    add("eval_checkpoint", args.eval_checkpoint)
    add("baseline_run_dir", args.baseline_run_dir)
    for split in splits:
        add(f"split:{split.name}", split.path)
    for source in sources:
        add(f"source_dir:{source.name}", source.path)
        for split in splits:
            add(f"source_proposals:{source.name}:{split.name}", source.path / f"{split.name}_proposals.csv")
    return checks


def missing_required(checks: list[dict[str, str]]) -> list[dict[str, str]]:
    return [check for check in checks if check["required"] == "true" and check["exists"] != "true"]


def run_commands(commands: list[dict[str, Any]]) -> None:
    for command in commands:
        print(f"[Stage2AT] running {command['name']}", flush=True)
        subprocess.run(command["argv"], cwd=REPO_ROOT, check=True)


def main() -> int:
    args = parse_args()
    split_values = args.split if args.split is not None else list(DEFAULT_SPLITS)
    source_values = args.source if args.source is not None else list(DEFAULT_SOURCES)
    splits = [parse_named_path(value, label="split") for value in split_values]
    sources = [parse_named_path(value, label="source") for value in source_values]
    commands = build_commands(args, splits, sources)
    checks = preflight(args, splits, sources)
    missing = missing_required(checks)
    if (args.strict or args.execute) and missing:
        missing_text = "\n".join(f"- {item['kind']}: {item['path']}" for item in missing)
        raise FileNotFoundError(f"Stage2AT missing required inputs:\n{missing_text}")

    manifest = {
        "artifact_type": "task2_stage2at_aq_official_orchestration",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": REPO_ROOT.as_posix(),
        "mode": "execute" if args.execute else "dry_run",
        "note": (
            "Current Stage2AT covers official-style validation splits with labels and precomputed proposal CSVs. "
            "Hidden-test no-GT Docker inference still needs a dedicated artifact-compatible entry point."
        ),
        "splits": [{"name": split.name, "path": split.path.as_posix()} for split in splits],
        "sources": [{"name": source.name, "path": source.path.as_posix()} for source in sources],
        "preflight_checks": checks,
        "missing_required_inputs": missing,
        "commands": commands,
    }
    manifest_path = repo_path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    if args.execute:
        run_commands(commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
