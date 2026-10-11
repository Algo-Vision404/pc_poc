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
│ Conv3×3 ×2    │ │ Conv3×3 ×2    │ │ Conv3×3 s2 ×2  │
│ Upsample ×4/×2 │ │ Upsample ×4/×2 │ │ Flatten → FC   │
│ 2-class logits │ │ Conv1×1+Sigmoid│ │ FC cls + bbox  │
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
Feature Map (128ch, 16×16)
    │
    ▼  Conv3×3 s2 + BN + ReLU   →  64ch, 8×8
    │
    ▼  Conv3×3 s2 + BN + ReLU   →  32ch, 4×4
    │
    ▼  Flatten  →  512
    │
    ▼  FC(512→256) + ReLU + Dropout(0.3)
    │
    ├──► FC(256→1)              →  cls score (BCEWithLogits)
    └──► FC(256→4) + Sigmoid    →  [xc, yc, w, h] normalized (SmoothL1)
```

> No `AdaptiveAvgPool2d` anywhere in the head — pooling destroys the spatial
> signal the box regression depends on. Strided convolutions keep it.

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
pc_poc/
├── src/per_camera/           # Installable Python package
│   ├── paths.py              # Artifact resolution (repo-anchored, env-overridable)
│   ├── model.py              # BigPerCameraNet + ARCH_VERSION
│   ├── data.py               # Dataset generation, Dataset, collate, transform
│   ├── metrics.py            # IoU / RMSE evaluation
│   ├── training.py           # Training loop + checkpoint I/O
│   ├── api.py                # FastAPI inference server
│   ├── inference.py          # Offline batch inference -> MP4 grid
│   └── visualize.py          # Per-sample prediction figures
├── artifacts/                # GENERATED, gitignored. Created at the repo root.
│   ├── models/               # .pt checkpoints, .onnx exports
│   ├── results/              # poc_results.json, figures, demo video
│   ├── synthetic_data/       # Generated images, masks, depth .npy
│   └── dataset_manifest.csv  # Relative paths — portable across machines
├── ui/
│   ├── src/App.jsx           # Dashboard (upload / stream / webcam)
│   ├── src/index.css         # Design system tokens
│   └── package.json
├── tests/
├── pyproject.toml
├── requirements.txt          # Pinned runtime deps
├── requirements-export.txt   # Optional ONNX export deps
└── README.md
```

Artifact location is resolved once, in `src/per_camera/paths.py`, anchored to the
repository — **never** to the current working directory. Override it with
`PC_POC_ARTIFACTS=/some/where`.


---

## Performance

Metrics are written to `artifacts/results/poc_results.json` by every training
run, and that file — not this README — is the source of truth. Checkpoint
metadata (epoch, config, metrics, architecture version, torch version) is
embedded in the `.pt` file itself, so you can always tell which model you have.

```json
{
  "arch_version": "big-per-camera-net-v1",
  "best_metrics": { "val_mIoU": 0.0, "val_rmse": 0.0738, "val_det_iou": 0.1265 },
  "config": { "epochs": 5, "batch_size": 32, "dataset_size": 128 },
  "torch_version": "2.14.1+cu130"
}
```

> **These numbers do not reproduce run to run, and that is a known defect.**
> Nothing in the training path is seeded — not the dataset, not the train/val
> split, not the weight init — so two runs of the same command diverge. Two
> 128-sample / 5-epoch runs measured `val_mIoU` 0.208 and 0.000. Adding
> `--seed` is tracked in `AUDIT.md`.
>
> Two further caveats, also in `AUDIT.md`: "mIoU" is foreground-only rather
> than a true mean over classes, and "Detection IoU" is localisation error on
> a single-object synthetic task, not AP.


---

## Quick Start

Requires Python 3.10+.

### 1. Clone & setup

```bash
git clone https://github.com/Algo-Vision404/pc_poc.git
cd pc_poc
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt           # runtime deps
pip install -r requirements-export.txt    # optional: ONNX export
pip install -e .                          # installs the per_camera package + CLI
```

### 2. Train the model

```bash
python -m per_camera.training --samples 2048 --epochs 30 --batch 32
# or, equivalently:
pc-poc-train --samples 2048 --epochs 30 --batch 32
```

| Argument | Default | Description |
|---|---|---|
| `--samples` | 128 | Synthetic frames to generate |
| `--epochs` | 20 | Training epochs |
| `--batch` | 16 | Batch size |
| `--max_samples` | all | Cap the dataset size |

Artifacts are written to `artifacts/` at the repo root (override with
`PC_POC_ARTIFACTS`). This step is **required** before starting the API — there
is no committed checkpoint.

### 3. Start the inference API

```bash
python -m uvicorn per_camera.api:app --host 0.0.0.0 --port 8000
# or:
pc-poc-api --port 8000
```

Live at `http://127.0.0.1:8000` — docs at `/docs`, readiness at `/health`.

The server refuses to start without a valid checkpoint. `GET /health` reports
the loaded architecture version and training epoch, so you can always confirm
which model is actually serving.

### 4. Optional: demo video and figures

```bash
pc-poc-infer --frames 60 --fps 10     # -> artifacts/results/inference_stream.mp4
pc-poc-visualize --count 5            # -> artifacts/results/sample_*.png
```

Both write to their own directories and never touch the training set.

### 5. Start the dashboard

```bash
cd ui
npm install
npm run dev
```

Dashboard at `http://localhost:3000`.


---

## Development

```bash
pip install -e ".[dev]"

pytest          # 32 tests — path resolution, manifest portability,
                # checkpoint guards, training validation, API contract
ruff check src tests
```

The suite covers the failure modes this project previously shipped with: cwd-
dependent artifact paths, manifests carrying another machine's absolute paths,
checkpoints from a different architecture loading silently, and the trainer
crashing on ordinary arguments.

| Test module | Covers |
|---|---|
| `tests/test_paths.py` | Repo-anchored artifacts, env override, no import side effects |
| `tests/test_data.py` | Relative manifest paths, portability, label ranges |
| `tests/test_checkpoint.py` | Metadata, round-trip, stale-architecture rejection |
| `tests/test_training.py` | Argument validation, result reporting |
| `tests/test_api.py` | `/health` readiness, `/predict` response contract |

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

`artifacts/` is created automatically on first run and is gitignored — generated
data, checkpoints, and exports are never committed, because they drift out of
sync with the code. Train locally.

Every checkpoint records its `arch_version`. Loading a checkpoint written by a
different model version raises immediately with a clear message instead of
silently producing garbage:

```
RuntimeError: Checkpoint architecture mismatch at .../poc_checkpoint.pt.
  checkpoint arch_version: None
  code expects           : 'big-per-camera-net-v1'
```

The ONNX export is optional. If `onnx`/`onnxscript` are not installed the
training run prints a warning to stderr and still exits 0 — the `.pt`
checkpoint remains valid for the API.

