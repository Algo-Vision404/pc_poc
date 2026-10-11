"""Synthetic dataset generation and the :class:`torch.utils.data.Dataset` wrapper."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from .paths import DATA_DIR, IMG_SIZE, MANIFEST_PATH, ensure_artifact_dirs, resolve_manifest_path

MANIFEST_FIELDS = ["image", "seg", "depth", "class", "xc", "yc", "w", "h"]

#: Mean/std used at train time and at serve time. Must stay identical.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def build_transform(resize: bool = False) -> transforms.Compose:
    """The one preprocessing pipeline. ``resize=True`` for arbitrary uploads."""
    ops: list = []
    if resize:
        ops.append(transforms.Resize((IMG_SIZE, IMG_SIZE)))
    ops.extend([transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
    return transforms.Compose(ops)


def generate_synthetic_dataset(
    n_samples: int,
    data_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> Path:
    """Render ``n_samples`` frames of one coloured rectangle on a dark ground.

    Images are written to ``data_dir`` (default ``artifacts/synthetic_data``)
    and a manifest is written to ``manifest_path``. Manifest entries are
    **relative to the manifest's own directory** so the file stays portable.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    data_dir = Path(data_dir) if data_dir else DATA_DIR
    manifest_path = Path(manifest_path) if manifest_path else MANIFEST_PATH
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_artifact_dirs()

    manifest_dir = manifest_path.parent.resolve()
    rows: list[dict[str, object]] = []

    for i in range(int(n_samples)):
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (30, 30, 30))
        seg = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        depth = Image.new("F", (IMG_SIZE, IMG_SIZE), 0.0)

        draw = ImageDraw.Draw(img)
        draw_seg = ImageDraw.Draw(seg)
        draw_depth = ImageDraw.Draw(depth)

        w, h = np.random.randint(12, 41), np.random.randint(12, 41)
        x = np.random.randint(0, IMG_SIZE - w)
        y = np.random.randint(0, IMG_SIZE - h)
        color = tuple(int(c) for c in np.random.uniform(60, 255, size=3))

        # Depth correlates with object area to simulate a perspective cue.
        area = int(w) * int(h)
        area_norm = (area - 12 * 12) / (40 * 40 - 12 * 12 + 1e-5)
        dval = 0.1 + 0.8 * area_norm

        draw.rectangle([x, y, x + w, y + h], fill=color)
        draw_seg.rectangle([x, y, x + w, y + h], fill=1)
        for yy in range(int(y), int(y) + int(h) + 1):
            draw_depth.line([(x, yy), (x + w, yy)], fill=float(dval))

        img_path = (data_dir / f"img_{i:04d}.png").resolve()
        seg_path = (data_dir / f"seg_{i:04d}.png").resolve()
        depth_path = (data_dir / f"depth_{i:04d}.npy").resolve()

        img.save(img_path)
        seg.save(seg_path)
        np.save(depth_path, np.asarray(depth, dtype=np.float32))

        rows.append(
            {
                "image": img_path.relative_to(manifest_dir).as_posix(),
                "seg": seg_path.relative_to(manifest_dir).as_posix(),
                "depth": depth_path.relative_to(manifest_dir).as_posix(),
                "class": 1,
                "xc": (x + w / 2) / IMG_SIZE,
                "yc": (y + h / 2) / IMG_SIZE,
                "w": w / IMG_SIZE,
                "h": h / IMG_SIZE,
            }
        )

    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


class PerCameraDataset(Dataset):
    """Reads the frames, masks, depth maps, and boxes described by a manifest."""

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

        return {
            "image": self.transform(img),
            "seg": torch.from_numpy(np.array(seg, dtype=np.uint8, copy=True)).long(),
            "depth": torch.from_numpy(np.array(depth, dtype=np.float32, copy=True))
            .unsqueeze(0)
            .float(),
            "ann": {
                "class": torch.tensor(int(row["class"]), dtype=torch.long),
                "bbox": torch.tensor(
                    [float(row["xc"]), float(row["yc"]), float(row["w"]), float(row["h"])],
                    dtype=torch.float,
                ),
            },
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "seg": torch.stack([b["seg"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "ann": {
            "class": torch.stack([b["ann"]["class"] for b in batch]),
            "bbox": torch.stack([b["ann"]["bbox"] for b in batch]),
        },
    }
