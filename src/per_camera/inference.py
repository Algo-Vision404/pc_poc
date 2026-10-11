"""Offline batch inference rendered to an MP4 grid.

Usage::

    python -m per_camera.inference --frames 60 --fps 10
    pc-poc-infer --frames 60
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import PerCameraDataset, build_transform, generate_synthetic_dataset
from .model import BigPerCameraNet
from .paths import (
    CHECKPOINT_PATH,
    IMG_SIZE,
    RESULTS_DIR,
    ensure_artifact_dirs,
)
from .training import load_checkpoint


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo the (0.5, 0.5, 0.5) normalisation back to [0, 1]."""
    return tensor * 0.5 + 0.5


def _label(image: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def run_inference_video(
    model: torch.nn.Module,
    dataset: PerCameraDataset,
    device: torch.device,
    output_video_path,
    fps: int = 5,
) -> None:
    """Run inference over a frame sequence and write a 2x2 grid video."""
    print(f"Starting inference on {len(dataset)} frames...")

    out_size = (IMG_SIZE * 2, IMG_SIZE * 2)
    writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter could not open {output_video_path} — "
            "the 'mp4v' codec is unavailable in this OpenCV build."
        )

    try:
        for i in range(len(dataset)):
            sample = dataset[i]
            img_tensor = sample["image"].unsqueeze(0).to(device)
            gt_seg = sample["seg"].numpy()

            with torch.no_grad():
                out = model(img_tensor)
                seg_logits = F.interpolate(
                    out["seg"], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
                )
                pred_seg = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
                depth_up = F.interpolate(
                    out["depth"], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
                )
                pred_depth = depth_up.squeeze(0).squeeze(0).cpu().numpy()
                pred_bbox = out["det"][:, 1:5].squeeze(0).cpu().numpy()

            disp = np.clip(unnormalize(sample["image"]).permute(1, 2, 0).numpy(), 0, 1)
            disp = cv2.cvtColor((disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            pred_xc, pred_yc, pred_w, pred_h = pred_bbox
            x1 = int((pred_xc - pred_w / 2) * IMG_SIZE)
            y1 = int((pred_yc - pred_h / 2) * IMG_SIZE)
            x2 = int((pred_xc + pred_w / 2) * IMG_SIZE)
            y2 = int((pred_yc + pred_h / 2) * IMG_SIZE)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                disp, "Det", (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
            )

            depth_color = cv2.applyColorMap(
                (np.clip(pred_depth, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
            )

            gt_seg_color = cv2.cvtColor((gt_seg * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gt_seg_color[:, :, 0] = 0
            gt_seg_color[:, :, 2] = 0

            pred_seg_color = cv2.cvtColor((pred_seg * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            pred_seg_color[:, :, 0] = 0
            pred_seg_color[:, :, 1] = 0

            grid = np.vstack(
                (np.hstack((disp, depth_color)), np.hstack((gt_seg_color, pred_seg_color)))
            )
            _label(grid, "Input + BBox", 5, 15)
            _label(grid, "Depth Map", IMG_SIZE + 5, 15)
            _label(grid, "GT Segmentation", 5, IMG_SIZE + 15)
            _label(grid, "Pred Segmentation", IMG_SIZE + 5, IMG_SIZE + 15)
            writer.write(grid)
    finally:
        writer.release()

    print(f"Video saved to {output_video_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pc-poc-infer", description="Render an inference demo video from synthetic frames."
    )
    parser.add_argument("--frames", type=int, default=60, help="frames in the demo sequence")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args(argv)

    if args.frames < 1:
        parser.error("--frames must be >= 1")

    ensure_artifact_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = BigPerCameraNet().to(device)
    load_checkpoint(CHECKPOINT_PATH, model=model, device=device)
    model.eval()

    # Demo frames go to their own directory and their own manifest. The previous
    # version regenerated the training manifest in place, which truncated it and
    # overwrote the first N training images.
    print("Generating demo sequence...")
    demo_dir = RESULTS_DIR / "demo_frames"
    demo_manifest = RESULTS_DIR / "demo_manifest.csv"
    generate_synthetic_dataset(args.frames, data_dir=demo_dir, manifest_path=demo_manifest)

    dataset = PerCameraDataset(
        demo_manifest, transform=build_transform(), max_samples=args.frames
    )
    run_inference_video(model, dataset, device, RESULTS_DIR / "inference_stream.mp4", fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
