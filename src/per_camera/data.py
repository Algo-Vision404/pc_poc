"""Synthetic dataset generation and the :class:`torch.utils.data.Dataset` wrapper.

Frames contain 0-3 rectangles (empty frames included) over a receding ground
plane. Ground truth is written as image / mask / depth-map triples plus a JSON
box list, with paths stored relative to the manifest so it stays portable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .detection import build_targets
from .model import DET_STRIDE
from .paths import DATA_DIR, IMG_SIZE, MANIFEST_PATH, ensure_artifact_dirs, resolve_manifest_path

MANIFEST_FIELDS = ["image", "seg", "depth", "boxes"]

#: Mean/std used at train time and at serve time. Must stay identical.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

#: Detection classes. One class keeps the PoC honest; the head is written for N.
NUM_CLASSES = 1
CLASS_NAMES = ("object",)

#: Scene composition.
MAX_OBJECTS = 3
P_EMPTY_FRAME = 0.2

#: Depth ranges. Background is a receding plane; objects sit in front of it.
BG_DEPTH_NEAR = 0.55
BG_DEPTH_FAR = 0.95
OBJ_DEPTH_MIN = 0.05
OBJ_DEPTH_MAX = 0.50

#: Boxes are padded to this width so a batch can be stacked.
MAX_BOXES_PER_FRAME = MAX_OBJECTS


def build_transform(resize: bool = False) -> transforms.Compose:
    """The one preprocessing pipeline. ``resize=True`` for arbitrary uploads."""
    ops: list = []
    if resize:
        ops.append(transforms.Resize((IMG_SIZE, IMG_SIZE)))
    ops.extend([transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
    return transforms.Compose(ops)


def _background(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """RGB background and its depth field: a plane receding with image row."""
    rows = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    depth = np.broadcast_to(BG_DEPTH_NEAR + (BG_DEPTH_FAR - BG_DEPTH_NEAR) * rows, (size, size))
    depth = np.ascontiguousarray(depth.copy())

    # Slight vertical shading so the frame is not perfectly flat.
    shade = (26 + 14 * (1.0 - rows[:, 0])).astype(np.uint8)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[:, :, 0] = shade[:, None]
    rgb[:, :, 1] = shade[:, None]
    rgb[:, :, 2] = (shade + rng.integers(0, 6)).astype(np.uint8)[:, None]
    return rgb, depth


def _object_color(rng: np.random.Generator, depth: float) -> tuple[int, int, int]:
    """Brighter and more saturated when near; hazier when far.

    This gives the depth head an appearance cue that is independent of size, so
    depth cannot be recovered from bounding-box area alone.
    """
    base = rng.uniform(90, 255, size=3)
    haze = 1.0 - 0.45 * depth
    return tuple(int(np.clip(c * haze, 0, 255)) for c in base)


def generate_synthetic_dataset(
    n_samples: int,
    data_dir: Path | None = None,
    manifest_path: Path | None = None,
    seed: int | None = None,
) -> Path:
    """Render ``n_samples`` frames and write a manifest next to them.

    With ``seed`` given, the dataset is bit-for-bit reproducible.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rng = np.random.default_rng(seed)

    data_dir = Path(data_dir) if data_dir else DATA_DIR
    manifest_path = Path(manifest_path) if manifest_path else MANIFEST_PATH
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_artifact_dirs()

    manifest_dir = manifest_path.parent.resolve()
    rows: list[dict[str, object]] = []

    for i in range(int(n_samples)):
        rgb, depth = _background(rng, IMG_SIZE)
        seg = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        n_objects = 0 if rng.random() < P_EMPTY_FRAME else int(rng.integers(1, MAX_OBJECTS + 1))

        # Sample depth first so objects can be painted far-to-near (occlusion).
        obj_depth = rng.uniform(OBJ_DEPTH_MIN, OBJ_DEPTH_MAX, size=n_objects)
        boxes: list[dict[str, float]] = []

        specs = []
        for d in obj_depth:
            # Nearer objects are larger, but with enough per-axis noise that
            # area alone does not determine depth.
            scale = (1.10 - 0.70 * float(d)) * rng.uniform(0.85, 1.15, size=2)
            w = int(np.clip(12 * scale[0] * 1.6, 8, IMG_SIZE // 2))
            h = int(np.clip(12 * scale[1] * 1.6, 8, IMG_SIZE // 2))
            x = int(rng.integers(0, IMG_SIZE - w + 1))
            y = int(rng.integers(0, IMG_SIZE - h + 1))
            specs.append((float(d), x, y, w, h))

        # Paint farthest first so nearer objects occlude them.
        for d, x, y, w, h in sorted(specs, key=lambda s: -s[0]):
            # Exclusive upper bound -> the rectangle is exactly w x h pixels,
            # which is what the box label claims.
            rgb[y : y + h, x : x + w] = _object_color(rng, d)
            seg[y : y + h, x : x + w] = 1
            depth[y : y + h, x : x + w] = d

        for d, x, y, w, h in specs:
            boxes.append(
                {
                    "cls": 0,
                    "xc": (x + w / 2) / IMG_SIZE,
                    "yc": (y + h / 2) / IMG_SIZE,
                    "w": w / IMG_SIZE,
                    "h": h / IMG_SIZE,
                    "depth": round(float(d), 5),
                }
            )

        img_path = (data_dir / f"img_{i:04d}.png").resolve()
        seg_path = (data_dir / f"seg_{i:04d}.png").resolve()
        depth_path = (data_dir / f"depth_{i:04d}.npy").resolve()

        Image.fromarray(rgb, "RGB").save(img_path)
        Image.fromarray(seg, "L").save(seg_path)
        np.save(depth_path, depth.astype(np.float32))

        rows.append(
            {
                "image": img_path.relative_to(manifest_dir).as_posix(),
                "seg": seg_path.relative_to(manifest_dir).as_posix(),
                "depth": depth_path.relative_to(manifest_dir).as_posix(),
                "boxes": json.dumps(boxes),
            }
        )

    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def parse_boxes(raw: str) -> np.ndarray:
    """Parse the manifest's JSON box list into an ``(N, 5)`` array.

    Columns are ``[cls, xc, yc, w, h]``; returns shape ``(0, 5)`` when empty.
    """
    boxes = json.loads(raw) if raw else []
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(
        [[b["cls"], b["xc"], b["yc"], b["w"], b["h"]] for b in boxes], dtype=np.float32
    )


def pad_boxes(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate to ``MAX_BOXES_PER_FRAME`` and return ``(boxes, mask)``."""
    padded = np.zeros((MAX_BOXES_PER_FRAME, 5), dtype=np.float32)
    mask = np.zeros((MAX_BOXES_PER_FRAME,), dtype=np.float32)
    n = min(len(boxes), MAX_BOXES_PER_FRAME)
    if n:
        padded[:n] = boxes[:n]
        mask[:n] = 1.0
    return padded, mask


class PerCameraDataset(Dataset):
    """Reads the frames, masks, depth maps, and box lists described by a manifest."""

    def __init__(
        self,
        manifest_csv: str | Path,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_csv).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")

        with open(self.manifest_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"manifest is empty: {self.manifest_path}")
        if max_samples is not None:
            if int(max_samples) <= 0:
                raise ValueError(f"max_samples must be positive, got {max_samples}")
            rows = rows[: int(max_samples)]

        self.rows = rows
        self.manifest_dir = self.manifest_path.parent
        self.transform = transform or build_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[int(idx)]
        img = Image.open(resolve_manifest_path(row["image"], self.manifest_dir)).convert("RGB")
        seg = Image.open(resolve_manifest_path(row["seg"], self.manifest_dir)).convert("L")
        depth = np.load(resolve_manifest_path(row["depth"], self.manifest_dir))
        boxes, box_mask = pad_boxes(parse_boxes(row["boxes"]))
        targets = build_targets(
            boxes,
            box_mask,
            out_h=IMG_SIZE // DET_STRIDE,
            out_w=IMG_SIZE // DET_STRIDE,
            num_classes=NUM_CLASSES,
            max_objects=MAX_BOXES_PER_FRAME,
        )

        return {
            "image": self.transform(img),
            "seg": torch.from_numpy(np.array(seg, dtype=np.uint8, copy=True)).long(),
            "depth": torch.from_numpy(np.array(depth, dtype=np.float32, copy=True))
            .unsqueeze(0)
            .float(),
            "ann": {
                "boxes": torch.from_numpy(boxes),
                "box_mask": torch.from_numpy(box_mask),
                "heatmap": torch.from_numpy(targets["heatmap"]),
                "reg_mask": torch.from_numpy(targets["reg_mask"]),
                "reg_index": torch.from_numpy(targets["reg_index"]),
                "offset": torch.from_numpy(targets["offset"]),
                "size": torch.from_numpy(targets["size"]),
            },
        }


def collate_fn(batch: list[dict]) -> dict:
    keys = ("boxes", "box_mask", "heatmap", "reg_mask", "reg_index", "offset", "size")
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "seg": torch.stack([b["seg"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "ann": {k: torch.stack([b["ann"][k] for b in batch]) for k in keys},
    }
