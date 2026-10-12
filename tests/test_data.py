"""Dataset generation, manifest format, and ground-truth consistency."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from per_camera.classes import NUM_CLASSES, NUM_SEG_CLASSES
from per_camera.data import (
    BG_DEPTH_NEAR,
    MAX_BOXES_PER_FRAME,
    PerCameraDataset,
    build_transform,
    class_mask,
    collate_fn,
    generate_synthetic_dataset,
    pad_boxes,
    parse_boxes,
)
from per_camera.paths import IMG_SIZE


@pytest.fixture
def manifest(tmp_path):
    return generate_synthetic_dataset(
        24, data_dir=tmp_path / "synthetic_data", manifest_path=tmp_path / "manifest.csv", seed=7
    )


def _rows(manifest):
    with open(manifest, newline="") as fh:
        return list(csv.DictReader(fh))


def test_manifest_stores_relative_paths(manifest):
    """Regression: the manifest used to store absolute Windows paths, which
    made it unusable on any other machine."""
    for row in _rows(manifest):
        for field in ("image", "seg", "depth"):
            value = row[field]
            assert not Path(value).is_absolute(), value
            assert "\\" not in value, value
            assert ":" not in value, value


def test_manifest_is_portable_between_directories(manifest, tmp_path):
    import shutil

    moved = tmp_path / "relocated"
    shutil.copytree(tmp_path / "synthetic_data", moved / "synthetic_data")
    shutil.copy(manifest, moved / "manifest.csv")

    dataset = PerCameraDataset(moved / "manifest.csv", transform=build_transform())
    assert len(dataset) == 24
    assert dataset[0]["image"].shape == (3, IMG_SIZE, IMG_SIZE)


def test_dataset_shapes_and_keys(manifest):
    sample = PerCameraDataset(manifest, transform=build_transform())[0]

    assert sample["image"].shape == (3, IMG_SIZE, IMG_SIZE)
    assert sample["seg"].shape == (IMG_SIZE, IMG_SIZE)
    assert sample["depth"].shape == (1, IMG_SIZE, IMG_SIZE)
    assert sample["seg"].dtype == torch.long

    ann = sample["ann"]
    assert ann["boxes"].shape == (MAX_BOXES_PER_FRAME, 5)
    assert ann["box_mask"].shape == (MAX_BOXES_PER_FRAME,)
    assert ann["heatmap"].shape[0] == NUM_CLASSES
    assert ann["reg_index"].dtype == torch.int64


def test_collate_stacks_a_batch(manifest):
    dataset = PerCameraDataset(manifest, transform=build_transform())
    batch = collate_fn([dataset[0], dataset[1]])

    assert batch["image"].shape == (2, 3, IMG_SIZE, IMG_SIZE)
    assert batch["ann"]["boxes"].shape == (2, MAX_BOXES_PER_FRAME, 5)
    assert batch["ann"]["heatmap"].shape[0] == 2


def test_seed_makes_the_dataset_reproducible(tmp_path):
    """Regression: nothing was seeded, so the same command produced different
    data every run."""
    a = generate_synthetic_dataset(
        6, data_dir=tmp_path / "a", manifest_path=tmp_path / "a.csv", seed=42
    )
    b = generate_synthetic_dataset(
        6, data_dir=tmp_path / "b", manifest_path=tmp_path / "b.csv", seed=42
    )

    assert a.read_bytes().replace(b"a/", b"b/") == b.read_bytes()
    assert (tmp_path / "a" / "img_0000.png").read_bytes() == (
        tmp_path / "b" / "img_0000.png"
    ).read_bytes()
    assert np.array_equal(
        np.load(tmp_path / "a" / "depth_0000.npy"), np.load(tmp_path / "b" / "depth_0000.npy")
    )


def test_different_seeds_differ(tmp_path):
    generate_synthetic_dataset(6, data_dir=tmp_path / "a", manifest_path=tmp_path / "a.csv", seed=1)
    generate_synthetic_dataset(6, data_dir=tmp_path / "b", manifest_path=tmp_path / "b.csv", seed=2)
    assert (tmp_path / "a" / "img_0000.png").read_bytes() != (
        tmp_path / "b" / "img_0000.png"
    ).read_bytes()


def test_generator_produces_empty_frames(manifest):
    """Empty frames are what make the detection score meaningful."""
    counts = [len(json.loads(r["boxes"])) for r in _rows(manifest)]
    assert 0 in counts, "expected some frames with no objects"
    assert max(counts) <= MAX_BOXES_PER_FRAME


def test_mask_stays_inside_the_box_label(manifest):
    """The segmentation silhouette must be contained in the amodal box.

    Regression: PIL's inclusive rectangle painted (w+1)x(h+1) while the box
    label claimed w x h, so segmentation and detection GT disagreed. Now that
    each class has its own silhouette the box is an outer bound rather than an
    exact match, but it still must not leak.
    """
    for row in _rows(manifest):
        boxes = json.loads(row["boxes"])
        if len(boxes) != 1:
            continue  # occlusion makes the union non-additive
        seg = np.array(Image.open(manifest.parent / row["seg"]))
        box = boxes[0]
        w = round(box["w"] * IMG_SIZE)
        h = round(box["h"] * IMG_SIZE)
        x = int(round(box["xc"] * IMG_SIZE - w / 2))
        y = int(round(box["yc"] * IMG_SIZE - h / 2))

        inside = np.zeros_like(seg, dtype=bool)
        inside[y : y + h, x : x + w] = True
        assert (seg > 0).sum() > 0, "object produced no mask at all"
        assert not (seg[~inside] > 0).any(), "mask leaked outside its box label"


def test_depth_is_a_real_field_not_a_mask_copy(manifest):
    """Regression: depth used to be a constant inside the object and zero
    outside, making it the segmentation mask plus one scalar."""
    for row in _rows(manifest):
        depth = np.load(manifest.parent / row["depth"])
        seg = np.array(Image.open(manifest.parent / row["seg"]))

        # The background now carries a receding-plane depth, so depth is dense.
        assert depth.min() >= 0.0 and depth.max() <= 1.0
        assert depth[seg == 0].min() >= BG_DEPTH_NEAR - 1e-6
        # And it varies spatially rather than being one value per object.
        assert len(np.unique(np.round(depth[seg == 0], 3))) > 5


def test_object_depth_is_not_a_function_of_area(manifest):
    """Two objects of the same size must be able to sit at different depths."""
    by_area: dict[int, set[float]] = {}
    for row in _rows(manifest):
        for box in json.loads(row["boxes"]):
            area = round(box["w"] * box["h"], 3)
            by_area.setdefault(area, set()).add(round(box["depth"], 3))

    repeated = [v for v in by_area.values() if len(v) > 1]
    assert repeated, "area alone appears to determine depth — the shortcut is back"


def test_labels_are_in_valid_ranges(manifest):
    for row in _rows(manifest):
        for box in json.loads(row["boxes"]):
            for field in ("xc", "yc", "w", "h", "depth"):
                assert 0.0 <= box[field] <= 1.0
            assert 0 <= box["cls"] < NUM_CLASSES


def test_parse_boxes_handles_empty():
    assert parse_boxes("[]").shape == (0, 5)
    assert parse_boxes("").shape == (0, 5)


def test_pad_boxes_truncates_and_masks():
    boxes = np.arange(20, dtype=np.float32).reshape(4, 5)
    padded, mask = pad_boxes(boxes)
    assert padded.shape == (MAX_BOXES_PER_FRAME, 5)
    assert mask.sum().item() == MAX_BOXES_PER_FRAME


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_sample_count(tmp_path, bad):
    with pytest.raises(ValueError):
        generate_synthetic_dataset(bad, data_dir=tmp_path / "d", manifest_path=tmp_path / "m.csv")


def test_rejects_empty_manifest(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("image,seg,depth,boxes\n")
    with pytest.raises(ValueError):
        PerCameraDataset(empty)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PerCameraDataset(tmp_path / "nope.csv")


def test_max_samples_must_be_positive(manifest):
    with pytest.raises(ValueError):
        PerCameraDataset(manifest, max_samples=0)


def test_all_three_classes_are_actually_generated(tmp_path):
    """Regression: the head supported N classes but the generator only ever
    emitted class 0, so multi-class support was untested and unusable."""
    manifest = generate_synthetic_dataset(
        80, data_dir=tmp_path / "d", manifest_path=tmp_path / "m.csv", seed=7
    )
    seen: set[int] = set()
    for row in _rows(manifest):
        for box in json.loads(row["boxes"]):
            seen.add(int(box["cls"]))

    assert seen == set(range(NUM_CLASSES)), f"only generated classes {sorted(seen)}"


def test_segmentation_labels_cover_every_class(tmp_path):
    manifest = generate_synthetic_dataset(
        80, data_dir=tmp_path / "d", manifest_path=tmp_path / "m.csv", seed=7
    )
    values: set[int] = set()
    for row in _rows(manifest):
        values |= set(np.unique(np.array(Image.open(manifest.parent / row["seg"]))).tolist())

    assert 0 in values, "background must be present"
    assert max(values) <= NUM_SEG_CLASSES - 1, f"label {max(values)} out of range"
    # 0..NUM_CLASSES: background plus every object class.
    assert values == set(range(NUM_SEG_CLASSES)), f"missing labels: {sorted(values)}"


def test_each_class_has_a_distinct_silhouette():
    """If the shapes coincided, the classes would be indistinguishable and
    per-class AP would be meaningless."""
    masks = [class_mask(c, 32, 32) for c in range(NUM_CLASSES)]

    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            assert not np.array_equal(masks[i], masks[j]), f"{i} and {j} are identical"
            inter = (masks[i] & masks[j]).sum()
            union = (masks[i] | masks[j]).sum()
            assert inter / union < 0.9, f"classes {i} and {j} are near-identical"

    # A block fills its box; the others are strict subsets.
    assert masks[0].all()
    assert not masks[1].all()
    assert not masks[2].all()
