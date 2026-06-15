from __future__ import annotations

from scripts.task2.export_stage2p_refined_candidates import build_expanded_rows


def test_stage2p_export_appends_refined_candidate_without_leaking_iou_as_confidence() -> None:
    base_rows = [
        {
            "split": "valid_combined",
            "sample_id": "video_0_00000102",
            "video_id": "video_0",
            "frame_index": "102",
            "domain": "phantom",
            "image_path": "datasets/collision_detection/images/video_0_00000102.jpg",
            "label_path": "datasets/collision_detection/labels/video_0_00000102.txt",
            "image_width": "1107",
            "image_height": "842",
            "gt_class": "0",
            "gt_x1": "636.0",
            "gt_y1": "332.0",
            "gt_x2": "680.0",
            "gt_y2": "376.0",
            "source": "yolo_stage2l",
            "source_priority": "0",
            "source_rank": "1",
            "source_conf": "0.7692",
            "source_class": "0",
            "source_subtype": "",
            "candidate_x1": "635.0795288085938",
            "candidate_y1": "328.2948303222656",
            "candidate_x2": "679.5547485351562",
            "candidate_y2": "372.6944885253906",
            "candidate_cx": "657.3171",
            "candidate_cy": "350.4947",
            "candidate_width": "44.4752",
            "candidate_height": "44.3997",
            "candidate_iou": "0.8289",
            "matched_gt": "True",
            "verifier_label": "1",
            "verifier_label_name": "normal",
        }
    ]
    refined_rows = [
        {
            "sample_id": "video_0_00000102",
            "domain": "phantom",
            "gt_class": "0",
            "source": "yolo_stage2l",
            "source_rank": "1",
            "before_iou": "0.8289",
            "after_iou": "0.8413",
            "x1": "635.0795288085938",
            "y1": "328.2948303222656",
            "x2": "679.5547485351562",
            "y2": "372.6944885253906",
            "refined_x1": "634.5",
            "refined_y1": "329.7",
            "refined_x2": "677.9",
            "refined_y2": "374.1",
            "gt_x1": "636.0",
            "gt_y1": "332.0",
            "gt_x2": "680.0",
            "gt_y2": "376.0",
        }
    ]

    expanded, report = build_expanded_rows(
        base_rows=base_rows,
        refined_rows=refined_rows,
        split_name="valid_combined",
        source_name="stage2p_refined",
        source_priority=2,
        positive_iou=0.50,
        background_iou=0.10,
        include_base=True,
    )

    assert report["base_rows"] == 1
    assert report["refined_rows"] == 1
    assert len(expanded) == 2
    refined = expanded[1]
    assert refined["source"] == "stage2p_refined"
    assert refined["source_conf"] == "0.7692"
    assert refined["candidate_iou"] == 0.8413
    assert refined["verifier_label"] == 1
    assert refined["verifier_label_name"] == "normal"
