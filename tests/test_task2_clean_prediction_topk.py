from __future__ import annotations

from scripts.task2.evaluate_clean_prediction_topk import apply_animal_topk


def test_apply_animal_topk_keeps_non_animal_and_prunes_animal() -> None:
    rows = [
        {"sample_id": "p", "domain": "phantom", "class_id": 0, "score": 0.1},
        {"sample_id": "a", "domain": "animal", "class_id": 1, "score": 0.1},
        {"sample_id": "a", "domain": "animal", "class_id": 1, "score": 0.9},
        {"sample_id": "a", "domain": "animal", "class_id": 0, "score": 0.2},
        {"sample_id": "a", "domain": "animal", "class_id": 0, "score": 0.8},
    ]

    kept = apply_animal_topk(rows, 1)

    assert [(row["sample_id"], row["domain"], row["class_id"], row["score"]) for row in kept] == [
        ("p", "phantom", 0, 0.1),
        ("a", "animal", 0, 0.8),
        ("a", "animal", 1, 0.9),
    ]

