"""Metric correctness, checked against hand-computed expectations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from per_camera.metrics import average_precision, compute_ap, mean_iou, seg_iou_per_class


def test_mean_iou_is_one_for_a_perfect_prediction():
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, 2:5, 2:5] = 1
    assert mean_iou(target.clone(), target, num_classes=2) == pytest.approx(1.0)


def test_mean_iou_scores_a_perfect_background_only_frame_as_one():
    """Regression: the old metric divided 0 by ~0 and returned 0.0 for a
    frame where the model predicted background perfectly."""
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    assert mean_iou(target.clone(), target, num_classes=2) == pytest.approx(1.0)


def test_mean_iou_is_zero_for_a_completely_wrong_prediction():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :4] = 1
    pred = 1 - target
    assert mean_iou(pred, target, num_classes=2) == pytest.approx(0.0)


def test_seg_iou_per_class_marks_absent_classes():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    ious = seg_iou_per_class(target.clone(), target, num_classes=2)
    assert ious[0] == pytest.approx(1.0)
    assert ious[1] is None, "a class absent from both pred and GT must be skipped"


def test_mean_iou_hand_computed_value():
    # pred covers 6 px of a 4 px target, overlap 4 -> union 6 -> IoU 4/6.
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 0:2, 0:2] = 1
    pred = torch.zeros(1, 4, 4, dtype=torch.long)
    pred[0, 0:2, 0:3] = 1
    ious = seg_iou_per_class(pred, target, num_classes=2)
    assert ious[1] == pytest.approx(4 / 6)


def _det(xc, yc, w, h, score, cls=0):
    return np.array([xc, yc, w, h, score, cls], dtype=np.float64)


def _gt(xc, yc, w, h, cls=0):
    return np.array([cls, xc, yc, w, h], dtype=np.float64)


def test_ap_is_one_for_a_perfect_detector():
    dets = [np.array([_det(0.5, 0.5, 0.2, 0.2, 0.9)])]
    gts = [np.array([_gt(0.5, 0.5, 0.2, 0.2)])]
    ap, recall, precision = average_precision(dets, gts)
    assert ap == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)


def test_ap_is_zero_when_nothing_is_detected():
    gts = [np.array([_gt(0.5, 0.5, 0.2, 0.2)])]
    ap, recall, _ = average_precision([np.zeros((0, 6))], gts)
    assert ap == 0.0
    assert recall == 0.0


def test_ap_is_zero_when_there_is_nothing_to_detect_but_we_fire_anyway():
    dets = [np.array([_det(0.5, 0.5, 0.2, 0.2, 0.9)])]
    ap, _, _ = average_precision(dets, [np.zeros((0, 5))])
    assert ap == 0.0


def test_ap_is_zero_for_a_misplaced_detection():
    dets = [np.array([_det(0.1, 0.1, 0.1, 0.1, 0.9)])]
    gts = [np.array([_gt(0.9, 0.9, 0.1, 0.1)])]
    ap, recall, _ = average_precision(dets, gts)
    assert ap == 0.0
    assert recall == 0.0


def test_ap_penalises_a_false_positive_ranked_above_the_true_one():
    """A low-ranked FP does not reduce all-points AP (recall still hits 1.0 at
    precision 1.0), so the FP has to outrank the TP for AP to drop."""
    gts = [np.array([_gt(0.5, 0.5, 0.2, 0.2)])]
    dets = [
        np.array(
            [_det(0.05, 0.05, 0.05, 0.05, 0.95), _det(0.5, 0.5, 0.2, 0.2, 0.9)]
        )
    ]
    ap, recall, precision = average_precision(dets, gts)
    assert recall == pytest.approx(1.0)
    assert precision == pytest.approx(0.5)
    assert ap == pytest.approx(0.5)


def test_a_low_ranked_false_positive_leaves_ap_unchanged():
    gts = [np.array([_gt(0.5, 0.5, 0.2, 0.2)])]
    dets = [
        np.array(
            [_det(0.5, 0.5, 0.2, 0.2, 0.9), _det(0.05, 0.05, 0.05, 0.05, 0.8)]
        )
    ]
    ap, _, precision = average_precision(dets, gts)
    assert ap == pytest.approx(1.0)
    assert precision == pytest.approx(0.5), "the FP must still show up in precision"


def test_ap_does_not_double_count_a_matched_ground_truth():
    """Two detections on one GT: recall must saturate at 1.0, not 2.0."""
    gts = [np.array([_gt(0.5, 0.5, 0.2, 0.2)])]
    dets = [
        np.array(
            [_det(0.5, 0.5, 0.2, 0.2, 0.9), _det(0.5, 0.5, 0.2, 0.2, 0.8)]
        )
    ]
    _, recall, precision = average_precision(dets, gts)
    assert recall == pytest.approx(1.0)
    assert precision == pytest.approx(0.5)


def test_ap_counts_multiple_objects_per_frame():
    gts = [np.array([_gt(0.25, 0.25, 0.2, 0.2), _gt(0.75, 0.75, 0.2, 0.2)])]
    dets = [
        np.array([_det(0.25, 0.25, 0.2, 0.2, 0.9), _det(0.75, 0.75, 0.2, 0.2, 0.85)])
    ]
    ap, recall, precision = average_precision(dets, gts)
    assert recall == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)
    assert ap == pytest.approx(1.0)


def test_compute_ap_all_points_interpolation():
    ap = compute_ap(np.array([0.5, 1.0]), np.array([1.0, 0.5]))
    assert ap == pytest.approx(0.75)
