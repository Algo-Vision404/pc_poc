"""The figures entry point (`pc-poc-visualize`).

Also at 0% coverage. Runs headless via matplotlib's Agg backend, so it is
safe to exercise in CI without a display.
"""

from __future__ import annotations

import pytest
import torch

from per_camera import visualize
from per_camera.classes import SEG_PALETTE
from per_camera.data import NORM_MEAN, NORM_STD, PerCameraDataset, build_transform
from per_camera.model import BigPerCameraNet
from per_camera.paths import CHECKPOINT_PATH, MANIFEST_PATH, RESULTS_DIR
from per_camera.training import load_checkpoint


def test_unnormalize_inverts_the_training_normalisation():
    original = torch.rand(3, 8, 8)
    normalised = (original - torch.tensor(NORM_MEAN).view(3, 1, 1)) / torch.tensor(
        NORM_STD
    ).view(3, 1, 1)

    assert torch.allclose(visualize.unnormalize(normalised), original, atol=1e-6)


def test_segmentation_colormap_has_one_colour_per_class():
    """Regression: the panels used cmap="gray", which renders labels 0..3 as
    near-identical shades of grey. The palette must be one entry per class."""
    assert visualize._SEG_CMAP.N == len(SEG_PALETTE)


def test_main_writes_one_figure_per_sample(trained_artifacts):
    code = visualize.main(["--count", "2"])

    assert code == 0
    for i in range(2):
        path = RESULTS_DIR / f"sample_{i:03d}.png"
        assert path.exists() and path.stat().st_size > 0, f"missing {path.name}"


def test_count_is_clamped_to_the_dataset_size(trained_artifacts):
    """Asking for more samples than exist must not crash or silently report
    figures that were never written."""
    device = torch.device("cpu")
    model = BigPerCameraNet().to(device)
    load_checkpoint(CHECKPOINT_PATH, model=model, device=device)
    dataset = PerCameraDataset(MANIFEST_PATH, transform=build_transform(), max_samples=2)

    rendered = visualize.render_samples(model, dataset, device, count=99)

    assert rendered == len(dataset)


def test_count_must_be_positive():
    with pytest.raises(SystemExit) as exc:
        visualize.main(["--count", "0"])

    assert exc.value.code != 0


def test_rendering_does_not_warn_about_an_empty_legend(trained_artifacts):
    """Regression: the "GT" label was attached with `box is gt_boxes[0]`, which
    is always False because numpy indexing returns a fresh view object. The
    legend was therefore always empty and matplotlib warned on every figure."""
    import warnings

    device = torch.device("cpu")
    model = BigPerCameraNet().to(device)
    load_checkpoint(CHECKPOINT_PATH, model=model, device=device)
    dataset = PerCameraDataset(MANIFEST_PATH, transform=build_transform(), max_samples=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        visualize.render_samples(model, dataset, device, count=3)

    legend_warnings = [
        str(w.message) for w in caught if "No artists with labels" in str(w.message)
    ]
    assert not legend_warnings, f"empty-legend warning still raised: {legend_warnings}"
