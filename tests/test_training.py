"""Training-loop argument validation and result reporting."""

from __future__ import annotations

import pytest

from per_camera.data import generate_synthetic_dataset
from per_camera.model import ARCH_VERSION
from per_camera.training import train


@pytest.fixture
def manifest(tmp_path):
    return generate_synthetic_dataset(
        12, data_dir=tmp_path / "synthetic_data", manifest_path=tmp_path / "manifest.csv"
    )


def test_rejects_zero_epochs(manifest):
    """Regression: `--epochs 0` died with UnboundLocalError on ckpt_p."""
    with pytest.raises(ValueError, match="epochs"):
        train(manifest, epochs=0, batch_size=4, max_samples=12)


def test_rejects_negative_epochs(manifest):
    with pytest.raises(ValueError, match="epochs"):
        train(manifest, epochs=-3, batch_size=4, max_samples=12)


def test_rejects_zero_batch(manifest):
    with pytest.raises(ValueError, match="batch_size"):
        train(manifest, epochs=1, batch_size=0, max_samples=12)


def test_rejects_a_dataset_too_small_to_split(manifest):
    """Regression: `--samples 1` crashed inside random_split."""
    with pytest.raises(ValueError, match="at least 2 samples"):
        train(manifest, epochs=1, batch_size=4, max_samples=1)


def test_results_carry_arch_version_and_metrics(manifest):
    results = train(manifest, epochs=1, batch_size=4, max_samples=12)

    assert results["arch_version"] == ARCH_VERSION
    assert results["checkpoint"]
    assert "val_mIoU" in results["best_metrics"]
    assert "val_rmse" in results["best_metrics"]
    assert "val_det_iou" in results["best_metrics"]
    assert isinstance(results["torch_version"], str)
