"""Render per-sample prediction figures against ground truth.

Usage::

    python -m per_camera.visualize --count 5
    pc-poc-visualize --count 5
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # Headless-safe: never requires a display.

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .data import PerCameraDataset, build_transform  # noqa: E402
from .model import BigPerCameraNet  # noqa: E402
from .paths import CHECKPOINT_PATH, IMG_SIZE, MANIFEST_PATH, RESULTS_DIR, ensure_artifact_dirs  # noqa: E402
from .training import load_checkpoint  # noqa: E402


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 0.5 + 0.5


def render_samples(
    model: torch.nn.Module, dataset: PerCameraDataset, device: torch.device, count: int
) -> int:
    count = min(count, len(dataset))
    for i in range(count):
        sample = dataset[i]
        img = sample["image"].unsqueeze(0).to(device)
        gt_seg = sample["seg"].numpy()
        gt_depth = sample["depth"].squeeze(0).numpy()
        gt_bbox = sample["ann"]["bbox"].numpy()

        with torch.no_grad():
            out = model(img)
            seg_logits = F.interpolate(
                out["seg"], size=gt_seg.shape, mode="bilinear", align_corners=False
            )
            pred_seg = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
            depth_up = F.interpolate(
                out["depth"], size=gt_depth.shape, mode="bilinear", align_corners=False
            )
            pred_depth = depth_up.squeeze(0).squeeze(0).cpu().numpy()
            pred_bbox = out["det"][:, 1:5].squeeze(0).cpu().numpy()

        print(f"Sample {i}")
        print(" GT Seg unique:", np.unique(gt_seg))
        print(" Pred Seg unique:", np.unique(pred_seg))

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        disp_img = np.clip(unnormalize(sample["image"]).permute(1, 2, 0).numpy(), 0, 1)
        axes[0].imshow(disp_img)
        axes[0].set_title("Input Image & Detection")
        axes[0].axis("off")

        gt_xc, gt_yc, gt_w, gt_h = gt_bbox
        axes[0].add_patch(
            patches.Rectangle(
                ((gt_xc - gt_w / 2) * IMG_SIZE, (gt_yc - gt_h / 2) * IMG_SIZE),
                gt_w * IMG_SIZE,
                gt_h * IMG_SIZE,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
                label="GT",
            )
        )

        pred_xc, pred_yc, pred_w, pred_h = pred_bbox
        axes[0].add_patch(
            patches.Rectangle(
                ((pred_xc - pred_w / 2) * IMG_SIZE, (pred_yc - pred_h / 2) * IMG_SIZE),
                pred_w * IMG_SIZE,
                pred_h * IMG_SIZE,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
                label="Pred",
                linestyle="--",
            )
        )
        axes[0].legend()

        axes[1].imshow(gt_seg, cmap="gray")
        axes[1].set_title("Ground Truth Segmentation")
        axes[1].axis("off")

        axes[2].imshow(pred_seg, cmap="gray")
        axes[2].set_title("Predicted Segmentation")
        axes[2].axis("off")

        im = axes[3].imshow(pred_depth, cmap="viridis")
        axes[3].set_title("Predicted Depth")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        save_path = RESULTS_DIR / f"sample_{i:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pc-poc-visualize", description="Render prediction figures for sample frames."
    )
    parser.add_argument("--count", type=int, default=5, help="number of samples to render")
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error("--count must be >= 1")

    ensure_artifact_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BigPerCameraNet().to(device)
    load_checkpoint(CHECKPOINT_PATH, model=model, device=device)
    model.eval()

    dataset = PerCameraDataset(MANIFEST_PATH, transform=build_transform(), max_samples=args.count)
    rendered = render_samples(model, dataset, device, args.count)
    print(f"Wrote {rendered} figures to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
