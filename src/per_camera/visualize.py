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
from .detection import decode_detections  # noqa: E402
from .model import BigPerCameraNet  # noqa: E402
from .paths import CHECKPOINT_PATH, IMG_SIZE, MANIFEST_PATH, RESULTS_DIR, ensure_artifact_dirs  # noqa: E402
from .training import load_checkpoint  # noqa: E402

SCORE_THRESHOLD = 0.2


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 0.5 + 0.5


def _rect(ax, box, color, linestyle="-", label=None):  # noqa: ANN001
    xc, yc, w, h = box
    ax.add_patch(
        patches.Rectangle(
            ((xc - w / 2) * IMG_SIZE, (yc - h / 2) * IMG_SIZE),
            w * IMG_SIZE,
            h * IMG_SIZE,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
            label=label,
        )
    )


def render_samples(
    model: torch.nn.Module, dataset: PerCameraDataset, device: torch.device, count: int
) -> int:
    count = min(count, len(dataset))
    for i in range(count):
        sample = dataset[i]
        img = sample["image"].unsqueeze(0).to(device)
        gt_seg = sample["seg"].numpy()
        gt_depth = sample["depth"].squeeze(0).numpy()
        gt_mask = sample["ann"]["box_mask"].numpy() > 0.5
        gt_boxes = sample["ann"]["boxes"].numpy()[gt_mask][:, 1:5]

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
            dets = decode_detections(out["heatmap"], out["offset"], out["size"])[0].cpu().numpy()

        dets = dets[dets[:, 4] >= SCORE_THRESHOLD]

        print(
            f"Sample {i}: {len(gt_boxes)} GT box(es), {len(dets)} detection(s), "
            f"seg unique {np.unique(pred_seg)}"
        )

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        disp_img = np.clip(unnormalize(sample["image"]).permute(1, 2, 0).numpy(), 0, 1)
        axes[0].imshow(disp_img)
        axes[0].set_title(f"Input ({len(gt_boxes)} GT / {len(dets)} pred)")
        axes[0].axis("off")
        for box in gt_boxes:
            _rect(axes[0], box, "g", label="GT" if box is gt_boxes[0] else None)
        for j, det in enumerate(dets):
            _rect(
                axes[0],
                det[:4],
                "r",
                linestyle="--",
                label=f"Pred {det[4]:.2f}" if j == 0 else None,
            )
        axes[0].legend(fontsize=8)

        axes[1].imshow(gt_seg, cmap="gray")
        axes[1].set_title("GT Segmentation")
        axes[1].axis("off")

        axes[2].imshow(pred_seg, cmap="gray")
        axes[2].set_title("Pred Segmentation")
        axes[2].axis("off")

        im_gt = axes[3].imshow(gt_depth, cmap="viridis", vmin=0, vmax=1)
        axes[3].set_title("GT Depth")
        axes[3].axis("off")
        plt.colorbar(im_gt, ax=axes[3], fraction=0.046, pad=0.04)

        im_pred = axes[4].imshow(pred_depth, cmap="viridis", vmin=0, vmax=1)
        axes[4].set_title("Pred Depth")
        axes[4].axis("off")
        plt.colorbar(im_pred, ax=axes[4], fraction=0.046, pad=0.04)

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
