from __future__ import annotations

import json
from pathlib import Path

from scripts.task2.run_stage2aq_official_pipeline import build_commands, main


def test_stage2at_build_commands_contains_expected_steps(tmp_path: Path) -> None:
    class Args:
        python = Path("/usr/bin/python")
        data_root = Path("datasets/collision_detection")
        source_top_k = 50
        global_top_k = "1,5,10"
        positive_iou = 0.5
        background_iou = 0.2
        baseline_source = "yolo_stage2l"
        candidate_output_dir = tmp_path / "candidate_pool"
        candidate_name = "official_candidates"
        stage2u_output_dir = tmp_path / "ranker"
        stage2u_name = "official_ranker_eval"
        train_csv = Path("train_candidates.csv")
        eval_checkpoint = Path("best.pt")
        baseline_run_dir = Path("baseline_run")
        champion_output_dir = tmp_path / "champion"
        verify_output_json = tmp_path / "verify.json"
        batch_size = 8
        workers = 0
        device = "cpu"
        seed = 2026

    splits = [
        type("NamedPath", (), {"name": "valid_combined", "path": Path("combined.txt")})(),
        type("NamedPath", (), {"name": "valid_phantom", "path": Path("phantom.txt")})(),
        type("NamedPath", (), {"name": "valid_animal", "path": Path("animal.txt")})(),
    ]
    sources = [
        type("NamedPath", (), {"name": "yolo_stage2l", "path": Path("yolo")})(),
        type("NamedPath", (), {"name": "stage2x_class1", "path": Path("stage2x")})(),
    ]

    commands = build_commands(Args, splits, sources)

    assert [command["name"] for command in commands] == [
        "build_stage2o_candidate_pool",
        "eval_stage2u_quality_ranker",
        "export_stage2ae_stage2aq_predictions",
        "verify_stage2aq_upstream",
    ]
    assert "scripts/task2/build_stage2o_candidate_pool.py" in commands[0]["argv"]
    assert "--eval-checkpoint" in commands[1]["argv"]
    assert commands[2]["argv"].count("--variant") == 2
    assert "stage2aq" in commands[2]["argv"]
    assert "--multisource-run-dir" in commands[3]["argv"]


def test_stage2at_main_writes_dry_run_manifest(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2aq_official_pipeline.py",
            "--manifest",
            manifest_path.as_posix(),
            "--split",
            f"valid_combined={tmp_path / 'combined.txt'}",
            "--split",
            f"valid_phantom={tmp_path / 'phantom.txt'}",
            "--split",
            f"valid_animal={tmp_path / 'animal.txt'}",
            "--source",
            f"yolo_stage2l={tmp_path / 'yolo'}",
            "--source",
            f"stage2x_class1={tmp_path / 'stage2x'}",
        ],
    )

    assert main() == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["mode"] == "dry_run"
    assert len(manifest["commands"]) == 4
    assert manifest["missing_required_inputs"]
