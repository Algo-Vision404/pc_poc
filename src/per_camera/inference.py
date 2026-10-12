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

from .classes import SEG_PALETTE
from .data import PerCameraDataset, build_transform, generate_synthetic_dataset
from .detection import decode_detections
from .model import BigPerCameraNet
from .paths import CHECKPOINT_PATH, IMG_SIZE, RESULTS_DIR, ensure_artifact_dirs
from .training import load_checkpoint

SCORE_THRESHOLD = 0.2


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo the (0.5, 0.5, 0.5) normalisation back to [0, 1]."""
    return tensor * 0.5 + 0.5


def _seg_to_bgr(seg: np.ndarray) -> np.ndarray:
    """Map class indices to distinct BGR colours using the shared palette.

    Scaling the label image by 255 instead would clip every class above 0 to
    the same saturated colour, so a four-class mask looked single-colour.
    """
    lut = np.array(SEG_PALETTE, dtype=np.uint8)[:, ::-1]  # RGB -> BGR
    return lut[np.clip(seg, 0, len(lut) - 1).astype(int)]


def _label(image: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _draw_boxes(
    image: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], scores=None
) -> None:
    for i, (xc, yc, w, h) in enumerate(np.atleast_2d(boxes)):
        if w <= 0 or h <= 0:
            continue
        x1 = int((xc - w / 2) * IMG_SIZE)
        y1 = int((yc - h / 2) * IMG_SIZE)
        x2 = int((xc + w / 2) * IMG_SIZE)
        y2 = int((yc + h / 2) * IMG_SIZE)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        tag = f"{scores[i]:.2f}" if scores is not None else ""
        cv2.putText(image, tag, (x1, max(y1 - 4, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


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
            gt_mask = sample["ann"]["box_mask"].numpy() > 0.5
            gt_boxes = sample["ann"]["boxes"].numpy()[gt_mask][:, 1:5]

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
                dets = decode_detections(
                    out["heatmap"], out["offset"], out["size"]
                )[0].cpu().numpy()

            dets = dets[dets[:, 4] >= SCORE_THRESHOLD]

            disp = np.clip(unnormalize(sample["image"]).permute(1, 2, 0).numpy(), 0, 1)
            disp = cv2.cvtColor((disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            _draw_boxes(disp, gt_boxes, (0, 200, 0))
            if len(dets):
                _draw_boxes(disp, dets[:, :4], (0, 0, 255), dets[:, 4])

            depth_color = cv2.applyColorMap(
                (np.clip(pred_depth, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
            )

            gt_seg_color = _seg_to_bgr(gt_seg)
            pred_seg_color = _seg_to_bgr(pred_seg)

            grid = np.vstack(
                (np.hstack((disp, depth_color)), np.hstack((gt_seg_color, pred_seg_color)))
            )
            _label(grid, "Input + depth / GT seg + Pred seg", 5, 15)
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
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    if args.frames < 1:
        parser.error("--frames must be >= 1")

    ensure_artifact_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = BigPerCameraNet().to(device)
    load_checkpoint(CHECKPOINT_PATH, model=model, device=device)
    model.eval()

    # Demo frames go to their own directory and manifest. An earlier version
    # regenerated the training manifest in place, truncating it and overwriting
    # the first N training images.
    print("Generating demo sequence...")
    demo_dir = RESULTS_DIR / "demo_frames"
    demo_manifest = RESULTS_DIR / "demo_manifest.csv"
    generate_synthetic_dataset(
        args.frames, data_dir=demo_dir, manifest_path=demo_manifest, seed=args.seed
    )

    dataset = PerCameraDataset(demo_manifest, transform=build_transform(), max_samples=args.frames)
    run_inference_video(model, dataset, device, RESULTS_DIR / "inference_stream.mp4", fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
