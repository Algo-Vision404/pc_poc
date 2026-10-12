"""`evaluate()` — the function that produces every published number.

It was only ever exercised indirectly, through a full training run, so nothing
pinned *what it measures*. These tests drive it with a stub model that emits a
known heatmap, which makes the metric's behaviour directly assertable.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from per_camera.classes import NUM_CLASSES, NUM_SEG_CLASSES
from per_camera.metrics import evaluate
from per_camera.paths import IMG_SIZE

#: Regressed box size in feature cells -> 8/32 = 0.25 normalised. The decoder
#: maps cell (y, x) to a centre of (x/W, y/H) with no half-cell offset, so the
#: ground-truth boxes below are written to match exactly what the stub decodes
#: to.
BOX = 8.0
SIZE_N = BOX / 32.0


def centre(cx: int) -> float:
    return cx / 32.0


class StubNet(nn.Module):
    """Emits a fixed heatmap from an explicit peak list.

    ``peaks`` is ``(class, y, x, logit)``. Note that two peaks in the SAME
    channel within a 3x3 window cannot both survive: ``decode_detections``
    applies a 3x3 max-pool per channel first. Duplicates therefore only ever
    arise ACROSS channels, which is what this stub reproduces.
    """

    def __init__(self, peaks: list[tuple[int, int, int, float]]) -> None:
        super().__init__()
        self.peaks = peaks

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        heatmap = torch.full((b, NUM_CLASSES, 32, 32), -6.0)
        for cls, y, cx, logit in self.peaks:
            heatmap[:, cls, y, cx] = logit
        return {
            "seg": torch.zeros(b, NUM_SEG_CLASSES, IMG_SIZE, IMG_SIZE),
            "depth": torch.zeros(b, 1, IMG_SIZE, IMG_SIZE),
            "heatmap": heatmap,
            "offset": torch.zeros(b, 2, 32, 32),
            "size": torch.full((b, 2, 32, 32), BOX),
        }


def _loader(gt_boxes: list[tuple[int, int, int]]):
    """One batch item whose GT is ``[(cls, y, x), ...]`` at the stub geometry."""
    boxes = torch.zeros(1, 3, 5)
    mask = torch.zeros(1, 3)
    for i, (cls, y, cx) in enumerate(gt_boxes):
        boxes[0, i] = torch.tensor([float(cls), centre(cx), centre(y), SIZE_N, SIZE_N])
        mask[0, i] = 1.0
    return [
        {
            "image": torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),
            "seg": torch.zeros(1, IMG_SIZE, IMG_SIZE, dtype=torch.long),
            "depth": torch.zeros(1, 1, IMG_SIZE, IMG_SIZE),
            "ann": {"boxes": boxes, "box_mask": mask},
        }
    ]


def test_nms_removes_a_cross_class_duplicate_from_the_metrics():
    """Regression: the API suppressed cross-class duplicates but `evaluate`
    scored raw decoder output, so reported precision described detections no
    client would ever receive.

    Setup: a block and a sphere sit far apart, and the decoder also fires a
    weak sphere peak on top of the block. Class 1 therefore carries one true
    detection and one spurious one, so its precision reads 0.5 raw and 1.0 once
    the duplicate is suppressed. Class 0 is unaffected either way.

    Two subtleties this test had to be written around, both of which are real
    properties of the metric rather than bugs:

    - The duplicate must be in a class that also has ground truth. If the
      spurious class had no GT anywhere, `mean_average_precision` skips it and
      nothing moves.
    - It shows up in *precision*, not in AP. The duplicate scores below the
      true positive, and all-points AP does not penalise a false positive
      ranked below every true positive. Asserting on AP here passes either way
      and would be testing nothing.
    """
    model = StubNet(
        [
            (0, 5, 5, 6.0),    # block, the real one
            (1, 5, 5, 4.0),    # sphere duplicate on top of the block
            (1, 20, 20, 5.0),  # sphere, the real one
        ]
    )
    loader = _loader([(0, 5, 5), (1, 20, 20)])
    device = torch.device("cpu")
    kw = dict(num_seg_classes=NUM_SEG_CLASSES, num_det_classes=NUM_CLASSES, score_thresh=0.2)

    raw = evaluate(model, loader, device, nms_iou_thresh=None, **kw)
    with_nms = evaluate(model, loader, device, nms_iou_thresh=0.5, **kw)

    # Class 0 sees one clean detection either way.
    assert raw["ap50_class0"] == pytest.approx(1.0)
    assert with_nms["ap50_class0"] == pytest.approx(1.0)

    # Class 1 carries the duplicate, so its precision is where it shows.
    assert raw["precision50"] == pytest.approx(0.75)  # mean of 1.0 and 0.5
    assert with_nms["precision50"] == pytest.approx(1.0)

    # AP is insensitive to it, and recall is unaffected: the true positives
    # are found in both cases.
    assert raw["map50"] == with_nms["map50"]
    assert raw["recall50"] == with_nms["recall50"]


def test_nms_is_on_by_default():
    """The default has to match the API. A default of None would silently
    restore the old behaviour for any caller that does not pass the argument."""
    model = StubNet([(0, 5, 5, 6.0)])

    result = evaluate(
        model, _loader([(0, 5, 5)]), torch.device("cpu"),
        num_seg_classes=NUM_SEG_CLASSES, num_det_classes=NUM_CLASSES, score_thresh=0.2,
    )

    assert result["precision50"] == pytest.approx(1.0)


def test_an_empty_batch_returns_zeroes_not_a_crash():
    model = StubNet([(0, 5, 5, 6.0)])

    result = evaluate(
        model, [], torch.device("cpu"),
        num_seg_classes=NUM_SEG_CLASSES, num_det_classes=NUM_CLASSES,
    )

    assert result == {"mIoU": 0.0, "rmse": 0.0, "abs_rel": 0.0, "map50": 0.0}


def test_val_score_keys_exist_in_the_evaluate_output():
    """`val_score` reads mIoU, map50 and rmse by name. If evaluate ever renames
    one, selection breaks with a KeyError at the end of the first epoch."""
    from per_camera.training import val_score

    model = StubNet([(0, 5, 5, 6.0)])
    result = evaluate(
        model, _loader([(0, 5, 5)]), torch.device("cpu"),
        num_seg_classes=NUM_SEG_CLASSES, num_det_classes=NUM_CLASSES, score_thresh=0.2,
    )

    assert isinstance(val_score(result), float)


def test_a_class_absent_from_ground_truth_is_skipped_not_scored_zero():
    """With one GT box of class 0, the other two classes must not drag mAP
    down — otherwise a small validation split tanks the headline number."""
    model = StubNet([(0, 5, 5, 6.0)])

    result = evaluate(
        model, _loader([(0, 5, 5)]), torch.device("cpu"),
        num_seg_classes=NUM_SEG_CLASSES, num_det_classes=NUM_CLASSES, score_thresh=0.2,
    )

    assert "ap50_class0" in result
    assert result["map50"] == pytest.approx(result["ap50_class0"])
