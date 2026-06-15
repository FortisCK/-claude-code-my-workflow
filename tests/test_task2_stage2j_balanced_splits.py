from __future__ import annotations

from pathlib import Path

from scripts.task2.prepare_stage2j_balanced_splits import prepare_stage2j_balanced_splits


def test_stage2j_balanced_splits_have_expected_panels(tmp_path: Path) -> None:
    repo_root = Path.cwd()

    summary = prepare_stage2j_balanced_splits(
        data_root=repo_root / "datasets/collision_detection",
        base_split_dir=repo_root / "configs/task2/splits",
        output_dir=tmp_path / "splits_stage2j_balanced",
        config_dir=tmp_path / "configs",
        repo_root=repo_root,
        class_names=("normal", "collision"),
        phantom_per_class=200,
        animal_repeat=7,
        phantom_balanced_val_per_class=50,
        seed=2026,
    )

    folds = summary["folds"]
    assert set(folds) == {"pilot_train_v1_val_v2", "pilot_train_v2_val_v1"}
    for fold in folds.values():
        assert set(fold["animal_train_videos"]).isdisjoint(fold["animal_val_videos"])
        assert fold["train"]["domain_counts"]["animal"] == (
            fold["train_unique"]["domain_counts"]["animal"] * 7
        )
        assert fold["train"]["duplicate_occurrences"] > 0
        assert fold["train"]["domain_counts"]["phantom"] == 400

        train_labels = read_names(repo_root / fold["train_labels"])
        valid_animal_labels = read_names(repo_root / fold["valid_animal_labels"])
        assert train_labels.isdisjoint(valid_animal_labels)

    panels = summary["shared_validation_panels"]
    balanced = panels["valid_phantom_balanced_small"]
    assert balanced["class_counts"] == {"0": 50, "1": 50}
    assert balanced["domain_counts"] == {"phantom": 100}

    yaml_paths = summary["yolo_yaml"]
    for fold_yaml in yaml_paths.values():
        assert set(fold_yaml) == {
            "valid_animal",
            "valid_phantom_full",
            "valid_phantom_balanced_small",
        }
        for path in fold_yaml.values():
            assert (repo_root / path).is_file() or Path(path).is_file()


def read_names(path: Path) -> set[str]:
    return {Path(line.strip()).name for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
