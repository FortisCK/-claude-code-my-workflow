from scripts.task2.evaluate_yolo_proposals import parse_top_k_values, summarize_proposals


class _Sample:
    def __init__(self, sample_id: str, class_id: int) -> None:
        self.sample_id = sample_id
        self.class_id = class_id


def test_parse_top_k_values_sorts_and_deduplicates() -> None:
    assert parse_top_k_values("5,1,3,5") == [1, 3, 5]


def test_summarize_proposals_reports_topk_recall() -> None:
    samples = [_Sample("a", 0), _Sample("b", 1)]
    rows = [
        {"sample_id": "a", "gt_class": 0, "has_proposal": True, "proposal_rank": 1, "proposal_gt_iou": 0.4},
        {"sample_id": "a", "gt_class": 0, "has_proposal": True, "proposal_rank": 2, "proposal_gt_iou": 0.8},
        {"sample_id": "b", "gt_class": 1, "has_proposal": True, "proposal_rank": 1, "proposal_gt_iou": 0.6},
    ]

    metrics = summarize_proposals(samples, rows, top_k_values=[1, 2])

    assert metrics["topk"]["1"]["recall_iou_0.50"] == 0.5
    assert metrics["topk"]["2"]["recall_iou_0.75"] == 0.5
    assert metrics["topk_by_gt_class"]["0"]["2"]["recall_iou_0.75"] == 1.0
