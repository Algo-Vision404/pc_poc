"""Detection target construction, decoding, and box maths."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from per_camera.detection import (
    box_iou,
    boxes_xywhn_to_xyxy,
    build_targets,
    decode_detections,
    focal_loss,
    gaussian_radius,
    nms_max_pool,
    reg_l1_loss,
)

OUT = 32


def _boxes(rows):
    return np.array(rows, dtype=np.float32)


def test_gaussian_radius_is_positive_and_grows_with_size():
    small = gaussian_radius(4, 4)
    large = gaussian_radius(16, 16)
    assert small > 0
    assert large > small


def test_build_targets_places_the_peak_correctly():
    boxes = _boxes([[0, 0.5, 0.5, 0.25, 0.25]])
    mask = np.array([1.0], dtype=np.float32)

    t = build_targets(boxes, mask, OUT, OUT, num_classes=1, max_objects=1)

    assert t["heatmap"].shape == (1, OUT, OUT)
    # xc = 0.5 * 32 = 16.0 -> integer index 16, zero sub-pixel offset.
    assert t["reg_index"][0] == 16 * OUT + 16
    assert t["heatmap"][0].max() == 1.0
    assert t["heatmap"][0, 16, 16] == 1.0
    assert t["offset"][0].tolist() == [0.0, 0.0]
    assert t["size"][0].tolist() == [8.0, 8.0]  # 0.25 * 32


def test_build_targets_handles_an_empty_frame():
    t = build_targets(_boxes(np.zeros((0, 5))), np.zeros((1,), np.float32), OUT, OUT, 1, 1)
    assert t["heatmap"].max() == 0.0
    assert t["reg_mask"].sum() == 0.0


def test_build_targets_handles_multiple_objects():
    boxes = _boxes([[0, 0.25, 0.25, 0.2, 0.2], [0, 0.75, 0.75, 0.2, 0.2]])
    t = build_targets(boxes, np.ones(2, np.float32), OUT, OUT, 1, 3)
    assert t["reg_mask"].sum() == 2.0
    assert t["heatmap"][0, 8, 8] == 1.0
    assert t["heatmap"][0, 24, 24] == 1.0


def _heads_for(boxes, out=OUT):
    """Build ideal head outputs that encode the given boxes."""
    heat = torch.full((1, 1, out, out), -10.0)
    off = torch.zeros(1, 2, out, out)
    size = torch.zeros(1, 2, out, out)
    for xc, yc, w, h in boxes:
        xi, yi = int(xc * out), int(yc * out)
        heat[0, 0, yi, xi] = 10.0
        off[0, 0, yi, xi] = xc * out - xi
        off[0, 1, yi, xi] = yc * out - yi
        size[0, 0, yi, xi] = w * out
        size[0, 1, yi, xi] = h * out
    return heat, off, size


def test_decode_recovers_the_encoded_box():
    heat, off, size = _heads_for([(0.5, 0.5, 0.25, 0.25)])
    dets = decode_detections(heat, off, size, topk=1)

    assert dets.shape == (1, 1, 6)
    xc, yc, w, h, score, cls = dets[0, 0].tolist()
    assert xc == pytest.approx(0.5)
    assert yc == pytest.approx(0.5)
    assert w == pytest.approx(0.25)
    assert h == pytest.approx(0.25)
    assert score > 0.99
    assert cls == 0.0


def test_decode_handles_sub_pixel_centres():
    heat, off, size = _heads_for([(0.515, 0.485, 0.2, 0.2)])
    dets = decode_detections(heat, off, size, topk=1)
    xc, yc, w, h, _, _ = dets[0, 0].tolist()
    assert abs(xc - 0.515) < 1e-4
    assert abs(yc - 0.485) < 1e-4


def test_decode_returns_multiple_objects_sorted_by_score():
    heat, off, size = _heads_for([(0.25, 0.25, 0.2, 0.2), (0.75, 0.75, 0.2, 0.2)])
    heat[0, 0, 24, 24] = 20.0  # make the second one more confident
    dets = decode_detections(heat, off, size, topk=4)

    real = dets[0][dets[0, :, 4] > 0.2]
    assert len(real) == 2
    assert real[0, 4] >= real[1, 4]


def test_decode_on_an_empty_heatmap_yields_nothing_above_threshold():
    heat = torch.full((1, 1, OUT, OUT), -10.0)
    off = torch.zeros(1, 2, OUT, OUT)
    size = torch.zeros(1, 2, OUT, OUT)
    dets = decode_detections(heat, off, size, topk=5)
    assert (dets[..., 4] < 0.01).all()


def test_nms_max_pool_suppresses_non_maxima():
    heat = torch.zeros(1, 1, 5, 5)
    heat[0, 0, 2, 2] = 1.0
    heat[0, 0, 2, 3] = 0.5
    out = nms_max_pool(heat)
    assert out[0, 0, 2, 2] == 1.0
    assert out[0, 0, 2, 3] == 0.0


def test_focal_loss_is_lower_for_a_correct_heatmap():
    gt = torch.zeros(1, 1, OUT, OUT)
    gt[0, 0, 16, 16] = 1.0
    bad = torch.zeros(1, 1, OUT, OUT)
    assert focal_loss(bad, gt).item() > focal_loss(gt * 10, gt).item()


def test_reg_l1_loss_only_counts_masked_positions():
    pred = torch.zeros(1, 2, 2, 2)
    index = torch.zeros(1, 1, dtype=torch.long)
    target = torch.zeros(1, 1, 2)

    zero_mask = torch.zeros(1, 1)
    assert reg_l1_loss(pred, zero_mask, index, target).item() == 0.0

    one_mask = torch.ones(1, 1)
    assert reg_l1_loss(pred, one_mask, index, target).item() == 0.0


def test_boxes_xywhn_to_xyxy():
    out = boxes_xywhn_to_xyxy(np.array([0.5, 0.5, 0.2, 0.4]))
    assert np.allclose(out, [0.4, 0.3, 0.6, 0.7])


def test_box_iou_known_values():
    assert box_iou(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0
    assert box_iou(np.array([0, 0, 1, 1]), np.array([2, 2, 3, 3])) == 0.0
    half = box_iou(np.array([0, 0, 2, 1]), np.array([1, 0, 3, 1]))
    assert abs(half - 1 / 3) < 1e-6


def _det(xc, yc, w, h, score, cls):
    return np.array([xc, yc, w, h, score, cls], dtype=np.float64)


def test_nms_drops_a_duplicate_class_at_the_same_location():
    """Regression: the per-channel max-pool let one object win in two class
    channels, so the API returned e.g. block 0.64 and triangle 0.21 on the same
    box — and drew the weaker one over the correct one."""
    from per_camera.detection import class_agnostic_nms

    dets = np.array(
        [
            _det(0.77, 0.90, 0.17, 0.16, 0.64, 0),  # block, correct
            _det(0.77, 0.90, 0.17, 0.16, 0.21, 2),  # triangle, same box
        ]
    )

    kept = class_agnostic_nms(dets, iou_thresh=0.5)

    assert len(kept) == 1
    assert kept[0][5] == 0, "kept the weaker duplicate instead of the winner"
    assert kept[0][4] == pytest.approx(0.64)


def test_nms_keeps_two_distinct_objects():
    from per_camera.detection import class_agnostic_nms

    dets = np.array(
        [
            _det(0.2, 0.2, 0.1, 0.1, 0.9, 0),
            _det(0.8, 0.8, 0.1, 0.1, 0.7, 1),
        ]
    )

    assert len(class_agnostic_nms(dets, iou_thresh=0.5)) == 2


def test_nms_returns_results_in_descending_score_order():
    from per_camera.detection import class_agnostic_nms

    dets = np.array(
        [
            _det(0.2, 0.2, 0.1, 0.1, 0.3, 0),
            _det(0.5, 0.5, 0.1, 0.1, 0.9, 1),
            _det(0.8, 0.8, 0.1, 0.1, 0.6, 2),
        ]
    )

    scores = class_agnostic_nms(dets)[:, 4]

    assert list(scores) == sorted(scores, reverse=True)


def test_nms_handles_an_empty_input():
    from per_camera.detection import class_agnostic_nms

    assert len(class_agnostic_nms(np.zeros((0, 6)))) == 0


def test_nms_is_class_agnostic():
    """Two classes on the same box is the case being fixed; two of the SAME
    class on the same box is the ordinary NMS case. Both must collapse."""
    from per_camera.detection import class_agnostic_nms

    same_class = np.array(
        [
            _det(0.5, 0.5, 0.2, 0.2, 0.8, 1),
            _det(0.5, 0.5, 0.2, 0.2, 0.4, 1),
        ]
    )

    assert len(class_agnostic_nms(same_class, iou_thresh=0.5)) == 1
