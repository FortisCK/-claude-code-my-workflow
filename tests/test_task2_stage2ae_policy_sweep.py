from __future__ import annotations

from scripts.task2.sweep_stage2ae_calibrated_policy import transformed_box


def test_transformed_box_uses_identity_when_transform_missing() -> None:
    candidate = {"domain": "animal", "x1": "1", "y1": "2", "x2": "5", "y2": "8"}

    assert transformed_box(candidate, class_id=1, box_transforms={}) == (1.0, 2.0, 5.0, 8.0)


def test_transformed_box_uses_class_transform() -> None:
    candidate = {"domain": "animal", "x1": "1", "y1": "2", "x2": "5", "y2": "8"}

    box = transformed_box(
        candidate,
        class_id=1,
        box_transforms={
            "animal": {
                "1": {"dx_center": 0.25, "dy_center": 0.0, "log_w": 0.0, "log_h": 0.0}
            }
        },
    )

    assert box == (2.0, 2.0, 6.0, 8.0)

