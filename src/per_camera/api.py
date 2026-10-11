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
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .classes import CLASS_NAMES, SEG_PALETTE, class_name
from .data import build_transform
from .detection import decode_detections
from .model import ARCH_VERSION, BigPerCameraNet
from .paths import CHECKPOINT_PATH, IMG_SIZE
from .training import load_checkpoint

logger = logging.getLogger("per_camera.api")

#: Comma-separated allowlist. Wildcard-plus-credentials lets any site call this
#: API, so the default is the local dashboard only.
CORS_ORIGINS_ENV_VAR = "PER_CAMERA_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
SCORE_THRESHOLD = 0.2
MAX_DETECTIONS = 20

_MAGIC = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpeg"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fail loudly. An earlier version did `if path.exists(): load(...)` with no
# else, so a missing checkpoint produced a silently untrained network that
# still answered 200 on /predict and reported ok on /.
model = BigPerCameraNet().to(device)
CHECKPOINT_META = load_checkpoint(CHECKPOINT_PATH, model=model, device=device)
model.eval()

transform = build_transform()

app = FastAPI(
    title="Per-Camera PoC Inference API",
    version=ARCH_VERSION,
    description="Multi-task perception: segmentation, monocular depth, and detection.",
)

_raw_origins = os.environ.get(CORS_ORIGINS_ENV_VAR, DEFAULT_CORS_ORIGINS)
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _to_b64_png(array: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encoding failed")
    return base64.b64encode(buffer).decode("utf-8")


def _colorize(mask: np.ndarray, colormap: int) -> np.ndarray:
    """Colour a continuous field normalised to ``[0, 1]``."""
    return cv2.applyColorMap(np.clip(mask * 255, 0, 255).astype(np.uint8), colormap)


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Map class indices to BGR through the shared palette.

    Routing labels through ``_colorize`` instead would multiply the class index
    by 255 and clip, so classes 1, 2 and 3 all rendered as the same saturated
    colour and a four-class mask came back single-coloured.
    """
    lut = np.array(SEG_PALETTE, dtype=np.uint8)[:, ::-1]  # RGB -> BGR
    return lut[np.clip(labels, 0, len(lut) - 1).astype(int)]


def _sniff_format(contents: bytes) -> str:
    for magic, name in _MAGIC.items():
        if contents.startswith(magic):
            return name
    if contents[:4] == b"RIFF" and contents[8:12] == b"WEBP":
        return "webp"
    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Send a PNG, JPEG, or WebP image.",
    )


def letterbox(img: Image.Image, size: int) -> tuple[Image.Image, float, int, int]:
    """Resize preserving aspect ratio and pad to a square.

    The network needs a fixed square input. Squashing instead would distort
    the frame, and boxes predicted in that space would not map back onto the
    original image.
    """
    w, h = img.size
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def box_to_original(
    xc: float, yc: float, w: float, h: float, *, scale: float, pad_x: int, pad_y: int, size: int
) -> tuple[float, float, float, float]:
    """Map a box from the letterboxed network space to original-image pixels.

    Input box is normalised over the padded square; the result is
    ``(x, y, w, h)`` in original-image pixels, top-left origin.
    """
    cx = (xc * size - pad_x) / scale
    cy = (yc * size - pad_y) / scale
    bw = w * size / scale
    bh = h * size / scale
    return cx - bw / 2, cy - bh / 2, bw, bh


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
        "classes": list(CLASS_NAMES),
        "cors_origins": _origins,
    }


@app.get("/")
def root() -> dict:
    return {"status": "ok", "model": "BigPerCameraNet", "docs": "/docs", "health": "/health"}


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict:  # noqa: B008 - FastAPI DI idiom
    """Run multi-task inference on one image.

    Declared ``def`` rather than ``async def`` so Starlette runs it in a
    threadpool. With ``async def`` the blocking forward pass occupied the
    event loop, capping the server at one concurrent request.
    """
    contents = file.file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(contents) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Upload exceeds the {limit_mb} MB limit.")

    fmt = _sniff_format(contents)
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode {fmt} image: {exc}") from exc

    if img.width * img.height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="Image resolution is too large.")

    orig_w, orig_h = img.size
    canvas, scale, pad_x, pad_y = letterbox(img, IMG_SIZE)
    img_t = transform(canvas).unsqueeze(0).to(device)

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
        dets = decode_detections(
            out["heatmap"], out["offset"], out["size"], topk=MAX_DETECTIONS
        )[0].cpu().numpy()

    kept = dets[dets[:, 4] >= SCORE_THRESHOLD]
    detections = []
    for xc, yc, w, h, score, cls in kept:
        x, y, bw, bh = box_to_original(
            float(xc), float(yc), float(w), float(h),
            scale=scale, pad_x=pad_x, pad_y=pad_y, size=IMG_SIZE,
        )
        detections.append(
            {
                # Pixels in the original image, plus normalised coords for overlays.
                "x_px": round(x, 2),
                "y_px": round(y, 2),
                "w_px": round(bw, 2),
                "h_px": round(bh, 2),
                "x": round(x / orig_w, 6),
                "y": round(y / orig_h, 6),
                "w": round(bw / orig_w, 6),
                "h": round(bh / orig_h, 6),
                "score": round(float(score), 4),
                "class": class_name(int(cls)),
                "class_id": int(cls),
            }
        )
    detections.sort(key=lambda d: -d["score"])

    overlay = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1 = int(det["x"] * orig_w * scale) + pad_x
        y1 = int(det["y"] * orig_h * scale) + pad_y
        x2 = int((det["x"] + det["w"]) * orig_w * scale) + pad_x
        y2 = int((det["y"] + det["h"]) * orig_h * scale) + pad_y
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"{det['score']:.2f}",
            (x1, max(y1 - 4, 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

    return {
        "input_b64": _to_b64_png(overlay),
        "seg_b64": _to_b64_png(_colorize_labels(pred_seg)),
        "depth_b64": _to_b64_png(_colorize(pred_depth, cv2.COLORMAP_VIRIDIS)),
        "detections": detections,
        "score": detections[0]["score"] if detections else 0.0,
        "image_size": {"width": orig_w, "height": orig_h},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pc-poc-api", description="Serve the inference API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    logger.info("checkpoint: %s", CHECKPOINT_PATH)
    logger.info("cors origins: %s", _origins)

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
