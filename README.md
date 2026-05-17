# Per-Camera Proof-of-Concept — Production Vision System

A production-grade, multi-task autonomous perception framework built on a ResNet-inspired backbone. The system performs simultaneous **object detection**, **semantic segmentation**, and **monocular depth estimation** from a single camera frame, served through a high-performance FastAPI inference backend and a professional React/Vite SaaS dashboard.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Roadmap](#roadmap)

---

## Overview

This project evolved from a lightweight single-task PoC (`TinyPerCameraNet`) into a scalable, production-grade multi-task vision system (`BigPerCameraNet`). Key engineering upgrades include:

- **Spatial-preserving detection head** — replaced destructive `AdaptiveAvgPool2d` with spatial convolutions to fix bounding box coordinate collapse
- **Depth-aware synthetic dataset** — ground-truth depth is deterministically derived from object area (simulating real-world perspective cues), providing a learnable signal
- **Loss weighting** — depth loss weighted at `30×` relative to segmentation to prevent gradient starvation on sparse targets
- **Production stack** — FastAPI inference API + React/Vite dashboard with live streaming support

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT FRAME (3×128×128)                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Stem Conv 7×7  │  stride=2, 64ch
                    │  BatchNorm+ReLU │
                    │  MaxPool 3×3    │  stride=2
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   ResBlock ×2   │  64ch  (layer1)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   ResBlock ×2   │  128ch, stride=2 (layer2)
                    └───────┬─────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
┌────────▼───────┐ ┌────────▼───────┐ ┌────────▼───────┐
│  Segmentation  │ │   Depth Head   │ │ Detection Head │
│     Head       │ │                │ │                │
│ ConvT 2×2 ×3  │ │ ConvT 2×2 ×3  │ │ Conv 3×3 (det) │
│ Conv 1×1       │ │ Conv 1×1+Sig  │ │ AdaptAvgPool   │
│ 2-class logits │ │ [0,1] depth   │ │ FC cls + bbox  │
└────────┬───────┘ └────────┬───────┘ └────────┬───────┘
         │                  │                  │
    Seg Mask (H×W)    Depth Map (H×W)   [cls_score, x, y, w, h]
```

### Residual Block

```
Input ──► Conv3×3 ──► BN ──► ReLU ──► Conv3×3 ──► BN ──► (+) ──► ReLU ──► Output
    │                                                        ▲
    └──────────────────── Downsample (1×1 Conv + BN) ────────┘
                          (only if stride≠1 or channel mismatch)
```

### Detection Head (Spatial-Preserving)

```
Feature Map (128ch)
    │
    ▼  Conv3×3 + BN + ReLU   →  64ch
    │
    ▼  Conv3×3 + BN + ReLU   →  64ch
    │
    ▼  AdaptiveAvgPool2d(4×4)
    │
    ▼  Flatten  →  1024
    │
    ├──► FC(1024→1)   →  cls score (BCEWithLogits)
    └──► FC(1024→4)   →  [xc, yc, w, h] normalized (SmoothL1)
```

### Multi-Task Loss

```
L_total = L_seg  +  30 × L_depth  +  L_cls  +  2 × L_bbox
            │             │              │            │
       CrossEntropy      MSE       BCEWithLogits  SmoothL1
```

> **Why 30× depth weight?** The depth targets are spatially sparse (object occupies ~10–15% of pixels). Without upweighting, the optimizer ignores depth gradients entirely in favour of the dominant segmentation loss.

### End-to-End Stack

```
┌──────────────────────────────────────────────────────────────┐
│                  React / Vite Dashboard                      │
│  ┌────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │ File Upload│  │ Live Synthetic  │  │  Webcam Stream   │  │
│  │  + Static  │  │   Stream (2fps) │  │     (1fps)       │  │
│  └──────┬─────┘  └────────┬────────┘  └────────┬─────────┘  │
│         └────────────────►│◄───────────────────┘            │
│                      POST /predict                           │
└──────────────────────────────────────────────────────────────┘
                            │ HTTP
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              FastAPI Inference Backend (port 8000)           │
│   Decode JPEG/PNG  ──►  Preprocess  ──►  BigPerCameraNet     │
│   Postprocess  ──►  base64 encode  ──►  JSON response        │
└──────────────────────────────────────────────────────────────┘
                            │ PyTorch
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     BigPerCameraNet                          │
│               (ResNet backbone, 3 heads)                     │
│          Checkpoint: artifacts/models/poc_checkpoint.pt      │
└──────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
per_camera_poc/
├── train and vis/
│   ├── per_camera_poc.py     # Dataset generation + BigPerCameraNet + training
│   ├── api.py                # FastAPI inference server
│   ├── inference_poc.py      # Offline batch inference + telemetry video
│   ├── visualize_poc.py      # Ground-truth visualization utilities
│   └── artifacts/
│       ├── models/           # .pt checkpoints, .onnx exports
│       ├── results/          # poc_results.json (KPI metrics)
│       └── synthetic_data/   # Generated images, masks, depth .npy
├── ui/
│   ├── src/
│   │   ├── App.jsx           # Main dashboard (upload/stream/webcam)
│   │   └── index.css         # SaaS design system tokens
│   ├── index.html
│   └── package.json
├── requirements.txt
└── README.md
```

---

## Performance

Latest training run (2 048 samples, 30 epochs, batch 32, AdamW lr=1e-3):

| Metric | Value |
|---|---|
| **Segmentation mIoU** | **0.927** |
| **Depth RMSE** | **0.096** |
| **Detection IoU** | 0.387 |
| **Best Val Loss** | 0.0267 |

> Detection IoU is bounded by the spatial resolution of the 128×128 synthetic dataset. Real-world deployment with higher resolution and real camera data will significantly improve this metric.

---

## Quick Start

### 1. Clone & setup

```bash
git clone https://github.com/Algo-Vision404/Per-Camera-Proof-of-Concept-PoC-.git
cd Per-Camera-Proof-of-Concept-PoC-
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
cd "train and vis"
python per_camera_poc.py --samples 2048 --epochs 30 --batch 32
```

| Argument | Default | Description |
|---|---|---|
| `--samples` | 128 | Number of synthetic frames to generate |
| `--epochs` | 20 | Training epochs |
| `--batch` | 16 | Batch size |

Artifacts are written to `train and vis/artifacts/`.

### 3. Start the inference API

```bash
cd "train and vis"
uvicorn api:app --port 8000
```

API will be live at `http://127.0.0.1:8000`. Docs: `http://127.0.0.1:8000/docs`.

### 4. Start the dashboard

```bash
cd ui
npm install
npm run dev
```

Dashboard at `http://localhost:3000`.

---

## API Reference

### `POST /predict`

Accepts a single image file and returns multi-task inference results.

**Request**: `multipart/form-data` with field `file` (JPEG or PNG).

**Response** (JSON):

```json
{
  "score": 0.87,
  "bbox": { "x": 0.42, "y": 0.51, "w": 0.18, "h": 0.22 },
  "depth_b64": "<base64-encoded PNG — viridis colormap>",
  "seg_b64": "<base64-encoded PNG — binary mask>",
  "input_b64": "<base64-encoded PNG — original frame>"
}
```

| Field | Description |
|---|---|
| `score` | Detection confidence `[0, 1]` |
| `bbox` | Normalized bounding box (cx, cy, w, h) |
| `depth_b64` | Depth map rendered via Matplotlib viridis colormap |
| `seg_b64` | Binary segmentation mask |
| `input_b64` | Echo of the input frame |

---

## Dashboard

The React dashboard provides three input modes and three output panels:

| Feature | Description |
|---|---|
| **Upload Frame** | Run single-image inference with manual trigger |
| **Live Synthetic Stream** | Auto-generates frames and runs inference at 2fps |
| **Webcam** | Streams from hardware camera at 1fps via `getUserMedia` |
| **Confidence Threshold** | Slider to suppress low-confidence detections |
| **Detection Overlay** | Bounding box drawn over the input with score label |
| **Depth Map** | Viridis-coloured depth from the network |
| **Segmentation Mask** | Binary foreground mask |

---

## Roadmap

| Priority | Item |
|---|---|
| High | Replace synthetic data with real RGBD camera streams |
| High | Add FPN neck for multi-scale detection on real scenes |
| Medium | Integrate TensorRT / ONNX Runtime for edge deployment |
| Medium | Docker Compose manifest (API + UI) for one-command deploy |
| Low | Multi-camera synchronization and stereo depth fusion |
| Low | Continuous training loop with online hard example mining |

---

## Disclaimer

`per_camera_poc.py` auto-creates the `artifacts/` directory tree on first run — no manual folder setup needed.

The ONNX export step requires stdout UTF-8 support. On Windows the script handles this automatically via a safe encoding redirect.
