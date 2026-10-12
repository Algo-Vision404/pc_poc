"""The demo-video entry point (`pc-poc-infer`).

This was at 0% coverage. It is a documented entry point that loads a checkpoint,
generates frames and writes a video — everything that can break quietly.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from per_camera import inference
from per_camera.classes import NUM_SEG_CLASSES
from per_camera.data import NORM_MEAN, NORM_STD
from per_camera.paths import RESULTS_DIR


def test_unnormalize_inverts_the_training_normalisation():
    """Regression guard: NORM_MEAN/NORM_STD must stay identical between the
    dataset and this helper, or every demo frame is silently mis-scaled."""
    original = torch.rand(3, 8, 8)
    normalised = (original - torch.tensor(NORM_MEAN).view(3, 1, 1)) / torch.tensor(
        NORM_STD
    ).view(3, 1, 1)

    recovered = inference.unnormalize(normalised)

    assert torch.allclose(recovered, original, atol=1e-6)


def test_seg_colours_are_distinct_per_class():
    """Regression: labels were scaled by 255 through a JET colormap, so classes
    1, 2 and 3 all clipped to the same saturated colour and a four-class mask
    came back single-coloured."""
    labels = np.arange(NUM_SEG_CLASSES, dtype=np.uint8)
    bgr = inference._seg_to_bgr(labels)

    assert bgr.shape == (NUM_SEG_CLASSES, 3)
    colours = {tuple(int(c) for c in bgr[i]) for i in range(NUM_SEG_CLASSES)}
    assert len(colours) == NUM_SEG_CLASSES, f"classes share a colour: {colours}"


def test_seg_colouring_clamps_out_of_range_labels():
    """A stale checkpoint emitting an unexpected label must still render."""
    assert inference._seg_to_bgr(np.array([[250]], dtype=np.uint8)).shape == (1, 1, 3)


def test_main_writes_a_video(trained_artifacts):
    code = inference.main(["--frames", "4", "--fps", "4"])

    assert code == 0
    video = RESULTS_DIR / "inference_stream.mp4"
    assert video.exists() and video.stat().st_size > 0


def test_main_does_not_touch_the_training_dataset(trained_artifacts):
    """Regression: an earlier version regenerated the training manifest in
    place, truncating it and overwriting the first N training images.

    The bug itself was never reproduced, so this asserts the invariant the fix
    established rather than the original symptom: running inference must leave
    the training manifest and its images byte-identical.
    """
    from per_camera.paths import MANIFEST_PATH

    before = MANIFEST_PATH.read_bytes()
    first_image = next(csv.DictReader(before.decode().splitlines()))["image"]
    image_path = MANIFEST_PATH.parent / first_image
    image_before = image_path.read_bytes()

    inference.main(["--frames", "4"])

    assert MANIFEST_PATH.read_bytes() == before, "inference rewrote the training manifest"
    assert image_path.read_bytes() == image_before, "inference overwrote a training image"

    # Demo frames live in their own directory instead.
    assert (RESULTS_DIR / "demo_manifest.csv").exists()
    assert (RESULTS_DIR / "demo_frames").is_dir()


def test_frames_must_be_positive():
    with pytest.raises(SystemExit) as exc:
        inference.main(["--frames", "0"])

    assert exc.value.code != 0
