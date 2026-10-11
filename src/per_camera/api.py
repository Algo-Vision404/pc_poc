"""FastAPI inference server.

Usage::

    uvicorn per_camera.api:app --host 0.0.0.0 --port 8000
    pc-poc-api --port 8000
"""

from __future__ import annotations

import argparse
import base64
import io
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .data import build_transform
from .model import ARCH_VERSION, BigPerCameraNet
from .paths import CHECKPOINT_PATH, IMG_SIZE
from .training import load_checkpoint

logger = logging.getLogger("per_camera.api")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fail loudly. The previous version did `if model_path.exists(): load(...)`
# with no else, so a missing checkpoint produced a silently untrained network
# that still answered 200 on /predict and reported ok on /. load_checkpoint
# raises FileNotFoundError on a missing file and RuntimeError on an
# architecture mismatch, so this module cannot start in a broken state.
model = BigPerCameraNet().to(device)
CHECKPOINT_META = load_checkpoint(CHECKPOINT_PATH, model=model, device=device)
model.eval()

# Single source of truth for preprocessing, shared with training.
transform = build_transform(resize=True)

app = FastAPI(
    title="Per-Camera PoC Inference API",
    version=ARCH_VERSION,
    description="Multi-task perception: segmentation, monocular depth, and detection.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_b64_png(array: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buffer).decode("utf-8")


def _colorize(mask: np.ndarray, colormap: int) -> np.ndarray:
    return cv2.applyColorMap(np.clip(mask * 255, 0, 255).astype(np.uint8), colormap)


@app.get("/health")
def health() -> dict:
    """Readiness probe. Reports the loaded model, not a hardcoded ok."""
    return {
        "status": "ready",
        "model": "BigPerCameraNet",
        "arch_version": ARCH_VERSION,
        "checkpoint_arch_version": CHECKPOINT_META.get("arch_version"),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": CHECKPOINT_META.get("epoch"),
        "device": str(device),
        "input_size": IMG_SIZE,
    }


@app.get("/")
def root() -> dict:
    return {"status": "ok", "model": "BigPerCameraNet", "docs": "/docs", "health": "/health"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:  # noqa: B008 - FastAPI DI idiom
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)
        seg_logits = F.interpolate(
            out["seg"], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        pred_seg = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        depth_up = F.interpolate(
            out["depth"], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        pred_depth = depth_up.squeeze(0).squeeze(0).cpu().numpy()
        pred_bbox = out["det"][:, 1:5].squeeze(0).cpu().numpy()
        cls_score = torch.sigmoid(out["det"][:, 0]).item()

    canvas = cv2.cvtColor(np.array(img.resize((IMG_SIZE, IMG_SIZE))), cv2.COLOR_RGB2BGR)

    pred_xc, pred_yc, pred_w, pred_h = pred_bbox
    x1 = int((pred_xc - pred_w / 2) * IMG_SIZE)
    y1 = int((pred_yc - pred_h / 2) * IMG_SIZE)
    x2 = int((pred_xc + pred_w / 2) * IMG_SIZE)
    y2 = int((pred_yc + pred_h / 2) * IMG_SIZE)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        canvas,
        f"Det: {cls_score:.2f}",
        (x1, max(y1 - 5, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1,
    )

    return {
        "input_b64": _to_b64_png(canvas),
        "seg_b64": _to_b64_png(_colorize(pred_seg, cv2.COLORMAP_JET)),
        "depth_b64": _to_b64_png(_colorize(pred_depth, cv2.COLORMAP_VIRIDIS)),
        "bbox": {
            "x": float(pred_xc),
            "y": float(pred_yc),
            "w": float(pred_w),
            "h": float(pred_h),
        },
        "score": float(cls_score),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pc-poc-api", description="Serve the inference API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    logger.info("artifacts root: %s", CHECKPOINT_PATH.parent.parent)
    logger.info("checkpoint:     %s", CHECKPOINT_PATH)

    import uvicorn

    uvicorn.run(
        "per_camera.api:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
