from __future__ import annotations

import json
from pathlib import Path

from scripts.task2.run_stage2aq_hidden_pipeline import build_commands, main, preflight


def test_stage2aw_build_commands_has_two_branch_hidden_flow(tmp_path: Path) -> None:
    class Args:
        python = Path("/usr/bin/python")
        split = ["hidden"]
        yolo_source = "yolo_stage2l=/proposal/yolo"
        source_top_k = 50
        image_dir = Path("/images")
        fallback_domain = "human"
        checkpoint = Path("best.pt")
        work_dir = tmp_path / "work"
        baseline_candidate_name = "baseline_candidates"
        multisource_candidate_name = "multi_candidates"
        baseline_ranker_name = "baseline_ranker"
        multisource_ranker_name = "multi_ranker"
        prediction_name = "preds"
        device = "cpu"
        batch_size = 2
        workers = 0
        generate_proposals = False

    commands = build_commands(
        Args,
        ["hidden"],
        ["yolo_stage2l=/proposal/yolo", "stage2x_class1=/proposal/stage2x"],
    )

    assert [command["name"] for command in commands] == [
        "build_yolo_only_nogt_candidate_pool",
        "build_multisource_nogt_candidate_pool",
        "infer_yolo_only_stage2u",
        "infer_multisource_stage2u",
        "export_stage2ae_stage2aq_hidden_predictions",
    ]
    assert "scripts/task2/build_nogt_candidate_pool.py" in commands[0]["argv"]
    assert commands[0]["argv"].count("--source") == 1
    assert commands[1]["argv"].count("--source") == 2
    assert "scripts/task2/infer_stage2u_quality_ranker.py" in commands[2]["argv"]
    assert "--stage2aq-multisource-run-dir" in commands[4]["argv"]
    assert "--skip-metrics" in commands[4]["argv"]
    assert commands[4]["argv"][commands[4]["argv"].index("--fallback-domain") + 1] == "human"


def test_stage2aw_main_writes_dry_run_manifest(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2aq_hidden_pipeline.py",
            "--manifest",
            manifest_path.as_posix(),
            "--work-dir",
            (tmp_path / "work").as_posix(),
            "--split",
            "hidden",
            "--yolo-source",
            f"yolo_stage2l={tmp_path / 'yolo'}",
            "--source",
            f"yolo_stage2l={tmp_path / 'yolo'}",
            "--source",
            f"stage2x_class1={tmp_path / 'stage2x'}",
        ],
    )

    assert main() == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["mode"] == "dry_run"
    assert len(manifest["commands"]) == 5
    assert manifest["missing_required_inputs"]


def test_stage2aw_generate_proposals_prepends_raw_image_commands(tmp_path: Path) -> None:
    class Args:
        python = Path("/usr/bin/python")
        split = ["hidden"]
        generate_proposals = True
        raw_image_dir = tmp_path / "images"
        raw_image_list_csv = None
        yolo_weights = tmp_path / "yolo.pt"
        stage2x_checkpoint = tmp_path / "stage2x.pt"
        proposal_output_name_yolo = "hidden_yolo_stage2l"
        proposal_output_name_stage2x = "hidden_stage2x_class1"
        proposal_limit = 2
        yolo_source = f"yolo_stage2l={tmp_path / 'work' / 'proposal_generation' / 'hidden_yolo_stage2l'}"
        source_top_k = 50
        image_dir = Path("/images")
        fallback_domain = "human"
        checkpoint = Path("best.pt")
        work_dir = tmp_path / "work"
        baseline_candidate_name = "baseline_candidates"
        multisource_candidate_name = "multi_candidates"
        baseline_ranker_name = "baseline_ranker"
        multisource_ranker_name = "multi_ranker"
        prediction_name = "preds"
        device = "cpu"
        batch_size = 2
        workers = 0

    Args.raw_image_dir.mkdir()
    Args.yolo_weights.write_text("weights", encoding="utf-8")
    Args.stage2x_checkpoint.write_text("checkpoint", encoding="utf-8")
    commands = build_commands(
        Args,
        ["hidden"],
        [
            f"yolo_stage2l={Args.work_dir / 'proposal_generation' / 'hidden_yolo_stage2l'}",
            f"stage2x_class1={Args.work_dir / 'proposal_generation' / 'hidden_stage2x_class1'}",
        ],
    )
    checks = preflight(
        Args,
        ["hidden"],
        [
            f"yolo_stage2l={Args.work_dir / 'proposal_generation' / 'hidden_yolo_stage2l'}",
            f"stage2x_class1={Args.work_dir / 'proposal_generation' / 'hidden_stage2x_class1'}",
        ],
    )

    assert [command["name"] for command in commands[:2]] == [
        "export_yolo_stage2l_nogt_proposals_hidden",
        "export_stage2x_class1_nogt_proposals_hidden",
    ]
    assert "scripts/task2/export_yolo_nogt_proposals.py" in commands[0]["argv"]
    assert "scripts/task2/export_sequence_tip_nogt_proposals.py" in commands[1]["argv"]
    assert commands[0]["argv"][commands[0]["argv"].index("--limit") + 1] == "2"
    assert commands[1]["argv"][commands[1]["argv"].index("--limit") + 1] == "2"
    assert commands[0]["argv"][commands[0]["argv"].index("--batch-size") + 1] == "2"
    assert commands[1]["argv"][commands[1]["argv"].index("--batch-size") + 1] == "2"
    assert commands[1]["argv"][commands[1]["argv"].index("--workers") + 1] == "0"
    assert "--limit" not in commands[2]["argv"]
    assert commands[2]["argv"][commands[2]["argv"].index("--image-dir") + 1] == Args.raw_image_dir.as_posix()
    assert commands[-1]["argv"][commands[-1]["argv"].index("--fallback-domain") + 1] == "human"
    assert all(check["kind"] != "image_dir" for check in checks)
    assert commands[-1]["name"] == "export_stage2ae_stage2aq_hidden_predictions"
    assert all(not check["kind"].startswith("source_proposals:") for check in checks)
