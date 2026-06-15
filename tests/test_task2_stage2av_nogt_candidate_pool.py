from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from scripts.task2.build_nogt_candidate_pool import main
from scripts.task2.build_nogt_candidate_pool import infer_domain
from scripts.task2.infer_stage2u_quality_ranker import load_inference_candidate_csv


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_build_nogt_candidate_pool_merges_sources_and_fills_metadata(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "sample_a.jpg"
    Image.new("RGB", (80, 60), color=(20, 30, 40)).save(image_path)
    yolo_dir = tmp_path / "yolo"
    geom_dir = tmp_path / "geom"
    write_csv(
        yolo_dir / "hidden_proposals.csv",
        [
            "sample_id",
            "video_id",
            "frame_index",
            "domain",
            "image_path",
            "image_width",
            "image_height",
            "has_proposal",
            "proposal_rank",
            "proposal_conf",
            "proposal_class",
            "proposal_x1",
            "proposal_y1",
            "proposal_x2",
            "proposal_y2",
        ],
        [
            {
                "sample_id": "sample_a",
                "video_id": "hidden_video",
                "frame_index": 7,
                "domain": "human",
                "image_path": image_path.as_posix(),
                "image_width": 80,
                "image_height": 60,
                "has_proposal": "True",
                "proposal_rank": 1,
                "proposal_conf": 0.8,
                "proposal_class": 1,
                "proposal_x1": 10,
                "proposal_y1": 11,
                "proposal_x2": 30,
                "proposal_y2": 31,
            }
        ],
    )
    write_csv(
        geom_dir / "hidden_proposals.csv",
        [
            "sample_id",
            "has_proposal",
            "proposal_rank",
            "proposal_score",
            "proposal_source",
            "proposal_x1",
            "proposal_y1",
            "proposal_x2",
            "proposal_y2",
        ],
        [
            {
                "sample_id": "sample_a",
                "has_proposal": "True",
                "proposal_rank": 2,
                "proposal_score": 0.4,
                "proposal_source": "geometry",
                "proposal_x1": 15,
                "proposal_y1": 16,
                "proposal_x2": 35,
                "proposal_y2": 36,
            },
            {
                "sample_id": "sample_a",
                "has_proposal": "True",
                "proposal_rank": 99,
                "proposal_score": 0.1,
                "proposal_source": "geometry",
                "proposal_x1": 1,
                "proposal_y1": 1,
                "proposal_x2": 2,
                "proposal_y2": 2,
            },
        ],
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_nogt_candidate_pool.py",
            "--split",
            "hidden",
            "--source",
            f"yolo_stage2l={yolo_dir}",
            "--source",
            f"task1_geometry_rect={geom_dir}",
            "--source-top-k",
            "10",
            "--output-dir",
            output_dir.as_posix(),
            "--name",
            "run",
        ],
    )

    assert main() == 0
    candidate_csv = output_dir / "run" / "hidden_candidates.csv"
    rows = list(csv.DictReader(candidate_csv.open(newline="", encoding="utf-8")))
    manifest = json.loads((output_dir / "run" / "args.json").read_text(encoding="utf-8"))
    loaded = load_inference_candidate_csv(candidate_csv, split="hidden")

    assert len(rows) == 2
    assert {row["source"] for row in rows} == {"yolo_stage2l", "task1_geometry_rect"}
    assert all(row["image_path"] for row in rows)
    assert all(row["image_width"] == "80" and row["image_height"] == "60" for row in rows)
    assert all(row["domain"] == "human" for row in rows)
    assert manifest["splits"]["hidden"]["rows"] == 2
    assert len(loaded) == 2
    assert loaded[1].source == "task1_geometry_rect"
    assert loaded[1].source_rank == 2


def test_infer_domain_uses_split_name_when_ids_are_generic() -> None:
    assert (
        infer_domain(
            "video_0_00000001",
            "video_0",
            {},
            split="valid_phantom",
            fallback_domain=None,
        )
        == "phantom"
    )
    assert (
        infer_domain(
            "video_5_00000001",
            "video_5",
            {},
            split="valid_animal",
            fallback_domain=None,
        )
        == "animal"
    )
