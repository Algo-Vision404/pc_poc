"""Training-loop argument validation, seeding, and model selection."""

from __future__ import annotations

import pytest

from per_camera.data import generate_synthetic_dataset
from per_camera.model import ARCH_VERSION
from per_camera.training import seed_everything, train, val_score


@pytest.fixture
def manifest(tmp_path):
    return generate_synthetic_dataset(
        24, data_dir=tmp_path / "synthetic_data", manifest_path=tmp_path / "manifest.csv", seed=3
    )


def test_rejects_zero_epochs(manifest):
    """Regression: `--epochs 0` died with UnboundLocalError on ckpt_p."""
    with pytest.raises(ValueError, match="epochs"):
        train(manifest, epochs=0, batch_size=4, max_samples=24)


def test_rejects_negative_epochs(manifest):
    with pytest.raises(ValueError, match="epochs"):
        train(manifest, epochs=-3, batch_size=4, max_samples=24)


def test_rejects_zero_batch(manifest):
    with pytest.raises(ValueError, match="batch_size"):
        train(manifest, epochs=1, batch_size=0, max_samples=24)


def test_rejects_a_dataset_too_small_to_split(manifest):
    """Regression: `--samples 1` crashed inside random_split."""
    with pytest.raises(ValueError, match="at least 2 samples"):
        train(manifest, epochs=1, batch_size=4, max_samples=1)


def test_results_report_the_new_metric_set(manifest):
    results = train(manifest, epochs=1, batch_size=4, max_samples=24, seed=0)

    assert results["arch_version"] == ARCH_VERSION
    assert results["best_epoch"] == 1
    assert results["config"]["seed"] == 0

    metrics = results["best_metrics"]
    for key in ("mIoU", "ap50", "rmse", "abs_rel", "det_iou"):
        assert key in metrics, key
    assert len(results["history"]) == 1


def test_same_seed_reproduces_the_validation_split(manifest):
    """Regression: the train/val split was unseeded, so metrics were not
    comparable between runs."""
    g1 = seed_everything(123)
    g2 = seed_everything(123)

    import torch

    from per_camera.data import PerCameraDataset, build_transform

    ds = PerCameraDataset(manifest, transform=build_transform())
    a = torch.utils.data.random_split(ds, [19, 5], generator=g1)[1].indices
    b = torch.utils.data.random_split(ds, [19, 5], generator=g2)[1].indices
    assert list(a) == list(b)


def test_val_score_prefers_higher_iou_and_ap_and_lower_rmse():
    base = {"mIoU": 0.5, "ap50": 0.5, "rmse": 0.1}
    assert val_score({**base, "mIoU": 0.9}) > val_score(base)
    assert val_score({**base, "ap50": 0.9}) > val_score(base)
    assert val_score({**base, "rmse": 0.01}) > val_score(base)


def test_best_checkpoint_is_written_even_when_the_last_epoch_is_worse(manifest):
    """`poc_checkpoint.pt` must hold the best validation epoch, not the last."""
    from per_camera.training import load_checkpoint

    results = train(manifest, epochs=2, batch_size=4, max_samples=24, seed=0)
    ckpt = load_checkpoint(results["checkpoint"])
    assert ckpt["epoch"] == results["best_epoch"]
    assert 1 <= results["best_epoch"] <= 2
