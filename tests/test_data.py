"""Dataset generation and the manifest format."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from per_camera.data import (
    PerCameraDataset,
    build_transform,
    collate_fn,
    generate_synthetic_dataset,
)
from per_camera.paths import IMG_SIZE


@pytest.fixture
def manifest(tmp_path):
    return generate_synthetic_dataset(
        6, data_dir=tmp_path / "synthetic_data", manifest_path=tmp_path / "manifest.csv"
    )


def test_manifest_stores_relative_paths(manifest):
    """Regression: the manifest used to store absolute Windows paths, which
    made it unusable on any other machine."""
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 6
    for row in rows:
        for field in ("image", "seg", "depth"):
            value = row[field]
            assert not Path(value).is_absolute(), value
            assert "\\" not in value, value
            assert ":" not in value, value


def test_manifest_is_portable_between_directories(manifest, tmp_path):
    """Moving the whole artifact tree must not break the manifest."""
    import shutil

    moved = tmp_path / "relocated"
    shutil.copytree(tmp_path / "synthetic_data", moved / "synthetic_data")
    shutil.copy(manifest, moved / "manifest.csv")

    dataset = PerCameraDataset(moved / "manifest.csv", transform=build_transform())
    assert len(dataset) == 6
    assert dataset[0]["image"].shape == (3, IMG_SIZE, IMG_SIZE)


def test_dataset_shapes(manifest):
    dataset = PerCameraDataset(manifest, transform=build_transform())
    sample = dataset[0]

    assert sample["image"].shape == (3, IMG_SIZE, IMG_SIZE)
    assert sample["seg"].shape == (IMG_SIZE, IMG_SIZE)
    assert sample["depth"].shape == (1, IMG_SIZE, IMG_SIZE)
    assert sample["ann"]["bbox"].shape == (4,)
    assert sample["seg"].dtype == torch.long


def test_depth_and_seg_support_agree(manifest):
    dataset = PerCameraDataset(manifest, transform=build_transform())
    sample = dataset[0]
    assert torch.equal(sample["seg"] > 0, sample["depth"].squeeze(0) > 0)


def test_collate_stacks_a_batch(manifest):
    dataset = PerCameraDataset(manifest, transform=build_transform())
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, IMG_SIZE, IMG_SIZE)
    assert batch["ann"]["bbox"].shape == (2, 4)


def test_labels_are_in_valid_ranges(manifest):
    with open(manifest, newline="") as fh:
        for row in csv.DictReader(fh):
            for field in ("xc", "yc", "w", "h"):
                assert 0.0 <= float(row[field]) <= 1.0


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_sample_count(tmp_path, bad):
    with pytest.raises(ValueError):
        generate_synthetic_dataset(
            bad, data_dir=tmp_path / "d", manifest_path=tmp_path / "m.csv"
        )


def test_rejects_empty_manifest(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("image,seg,depth,class,xc,yc,w,h\n")
    with pytest.raises(ValueError):
        PerCameraDataset(empty)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PerCameraDataset(tmp_path / "nope.csv")


def test_max_samples_must_be_positive(manifest):
    with pytest.raises(ValueError):
        PerCameraDataset(manifest, max_samples=0)
