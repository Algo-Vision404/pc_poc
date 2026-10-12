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

This project evolved from a lightweight single-task PoC (`TinyPerCameraNet`) into a multi-task vision system (`BigPerCameraNet`). Current design:

- **Anchor-free detection head** — a CenterNet-style heatmap plus sub-pixel offset and size regression at stride 4, so the model emits 0..N boxes per frame and a confidence score worth thresholding. (The earlier head flattened the feature map and regressed one box through an FC layer, which structurally capped it at a single detection.)
- **A real depth field** — the background carries a receding ground plane and each object has its own depth, cued by size *and* aerial-perspective brightness. Previously depth was a constant inside the object and zero outside, which made it the segmentation mask plus one scalar that was a pure function of box area — a shortcut, not a depth signal.
- **Multi-object, multi-class scenes** — 0–3 objects per frame, ~20% empty, painted far-to-near for occlusion. Three classes (`block`, `sphere`, `triangle`) each have a distinct silhouette and hue family, so detection is genuinely class-aware and the segmentation head has four labels rather than two. Empty frames are what give the classifier a negative class to learn.
- **Honest metrics** — true per-class mIoU (not foreground-only), class-aware **mAP@0.5** with per-class AP and precision/recall, mean IoU of matched detections for localisation quality, and AbsRel for depth.
- **Reproducible runs** — `--seed` seeds the dataset, the train/val split, and the weight init. Checkpoints are selected on validation quality, with early stopping.
- **Production stack** — FastAPI inference API + React/Vite dashboard with live streaming support.

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
┌────────▼───────┐ ┌────────▼───────┐ ┌────────▼────────┐
│  Segmentation  │ │   Depth Head   │ │ Detection Head  │
│     Head       │ │                │ │ (anchor-free)   │
│ Conv3×3 ×2    │ │ Conv3×3 ×2    │ │ Conv3×3, up ×2  │
│ Upsample ×4/×2 │ │ Upsample ×4/×2 │ │ 3 × Conv1×1     │
│ 2-class logits │ │ Conv1×1+Sigmoid│ │ heat/off/size   │
└────────┬───────┘ └────────┬───────┘ └────────┬────────┘
         │                  │                  │
    Seg Mask (H×W)    Depth Map (H×W)   N × [x, y, w, h, score]
```

### Residual Block

```
Input ──► Conv3×3 ──► BN ──► ReLU ──► Conv3×3 ──► BN ──► (+) ──► ReLU ──► Output
    │                                                        ▲
    └──────────────────── Downsample (1×1 Conv + BN) ────────┘
                          (only if stride≠1 or channel mismatch)
```

### Detection Head (anchor-free, CenterNet-style)

```
Feature Map (128ch, 16×16, stride 8)
    │
    ▼  Conv3×3 + BN + ReLU            →  64ch
    │
    ▼  Upsample ×2                    →  64ch, 32×32 (stride 4)
    │
    ▼  Conv3×3 + BN + ReLU            →  32ch
    │
    ├──► Conv1×1  →  heatmap  (C, 32×32)   per-pixel class score
    ├──► Conv1×1  →  offset   (2, 32×32)   sub-pixel centre residual
    └──► Conv1×1  →  size     (2, 32×32)   object w, h in feature units
```

Decoding applies a 3×3 max-pool NMS to the heatmap, takes the top-K peaks, and
lifts each back to image coordinates using its offset and size.

> **Why replace the old head?** The previous version flattened the feature map
> and regressed a single box through an FC layer. That structurally could not
> emit more than one detection, and because every frame contained exactly one
> object with a hardcoded class label of `1`, the classifier was trained on a
> constant target and its "confidence" carried no information. The heatmap head
> handles 0..N objects and produces a score worth thresholding.

The head has always taken a class count `C`, but until v3 the dataset only ever
emitted class 0 — so `C` was a parameter with one legal value and per-class AP
was untestable. The generator now emits three classes with distinct shapes, the
heatmap has three channels, and the segmentation head has four (background
plus three).

### Multi-Task Loss

```
L_total = L_seg  +  5 × L_depth  +  L_heatmap  +  L_offset  +  0.1 × L_size
            │           │              │             │              │
       CrossEntropy    MSE       focal (CenterNet)  L1@peaks    L1@peaks
```

> **Why 5× and not 30×?** The old depth target was sparse — a constant inside
> the object and zero everywhere else — so it needed heavy upweighting to
> compete with the segmentation loss. Depth is now a dense field (a receding
> background plane plus per-object values), so it carries gradient everywhere
> and needs far less help.

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
│   ├── detection.py          # Heatmap targets, focal loss, NMS decoding, AP
│   ├── metrics.py            # mIoU / RMSE / AbsRel / AP@0.5 evaluation
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

Reference run — `--samples 512 --epochs 20 --batch 32 --seed 0`, CPU, torch 2.14.1,
architecture `big-per-camera-net-v3`. Source: `artifacts/results/poc_results.json`.

| Metric | Value | Notes |
|---|---|---|
| **Detection mAP@0.5** | **0.995** | recall 0.995, precision 0.952 |
| — AP `block` | 1.000 | per-class |
| — AP `sphere` | 0.984 | the hardest: curved silhouette, no box-aligned edges |
| — AP `triangle` | 1.000 | per-class |
| **Matched IoU** | **0.808** | mean IoU of the detections that matched a GT box |
| **Segmentation mIoU** | **0.874** | true mean over **4** labels (background + 3 classes) |
| **Depth RMSE** | **0.037** | on a dense depth field in `[0, 1]` |
| **Depth AbsRel** | **0.039** | scale-invariant relative error |

Model selection picked **epoch 19** of 20.

> **mIoU is not comparable with the pre-v3 figure of 0.952.** That number was a
> mean over two labels (background + one object class); this one is a mean over
> four. Averaging in three object classes instead of one mechanically lowers the
> mean — the drop is a change in what the metric measures, not a regression.

Every run writes `artifacts/results/poc_results.json` with the full per-epoch
history, and that file — not this README — is the source of truth. Checkpoint
metadata (epoch, config, metrics, architecture version, torch version) is
embedded in the `.pt` file, so you can always tell which model you have.

These numbers are reproducible: `--seed` seeds the dataset, the train/val
split, and the weight init. Verified by running the same command twice and
getting an identical dataset hash and identical metrics to six decimals.

The mAP curve shows the detection head actually learning rather than sitting at
a constant, and doing so for all three classes rather than one:

```
epoch    1   2   3   4    5    6    7    8    9   10   11   12 ... 19   20
mAP50    0   0   0  .63  .87  .93  .96  .98  .99  .99  .99 1.00 ... .99 .99
mIoU   .24 .24 .32  .47  .62  .65  .67  .67  .84  .85  .85  .86 ... .87 .86
```

> The `det_iou` metric from earlier revisions is gone. It compared only the
> top-scoring detection against the *first* ground-truth box, so a detector
> that found every object correctly still read 0.44 while AP was 0.99 — a
> number that looked like a localisation failure but measured nothing.
> `matched_iou` replaces it: the mean IoU of the detections that actually
> matched a ground-truth box, which is the localisation quality that metric was
> trying to express. On this run it reads 0.808, which is what a detector at
> mAP 0.995 actually localises like.


---

## Out-of-Distribution Guardrail

**This model has never seen a real photograph.** It is trained entirely on
synthetic frames from this repository's own generator, and given one it does
not merely perform poorly — it performs *confidently* poorly, emitting 11–19
boxes per real image at scores of 0.2–0.6, none of them real.

That failure mode is the most disqualifying thing a perception system can
have, because nothing downstream can distinguish an invented box from a real
one. So the system detects the situation and refuses.

`per_camera/ood.py` fits robust appearance statistics (median and median
absolute deviation over eight cheap image features) on the raw training
frames during `train()`, and scores every upload against them before decoding
detections.

Measured on eight real photographs — four warehouse interiors, four street
scenes — against the training distribution:

| | OOD score | outcome |
|---|---|---|
| Real photographs (n=8) | 66.6 – 244.8 | **8/8 rejected**, 118 invented boxes suppressed |
| Synthetic frames (n=300) | 0.4 – 11.8 | 299/300 accepted |

The threshold is 8.0, so the margin between the most legitimate training
frame and the least outlandish real photograph is roughly **5×**. There is no
overlap.

Detections are **suppressed, not merely flagged** — a client that ignores a
flag still draws the boxes, which is the exact failure this exists to prevent.
The response reports `ood_score`, `ood_threshold`, `ood_rejected` and
`ood_suppressed_detections`, and `/health` reports whether the guardrail is
active at all.

### Limits, stated plainly

This is a guardrail, not a capability. It does not make the model see better;
it stops the model lying. Specifically:

- The gap it exploits is partly an artefact of how artificial the training data
  is — dark, flat, textureless. **Improving the generator with domain
  randomisation will narrow it**, and the statistics must be re-fitted when
  that happens (training does this automatically).
- It will not catch an input engineered to mimic the training statistics.
- It is fitted per checkpoint. Statistics from one training run say nothing
  about another.

The honest summary: this makes the system *trustworthy within its known
limits*, which is a prerequisite for deployment, not a substitute for real
training data.

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
python -m per_camera.training --samples 512 --epochs 20 --batch 32 --seed 0
# or, equivalently:
pc-poc-train --samples 512 --epochs 20 --batch 32 --seed 0
```

| Argument | Default | Description |
|---|---|---|
| `--samples` | 128 | Synthetic frames to generate |
| `--epochs` | 20 | Training epochs |
| `--batch` | 16 | Batch size |
| `--max_samples` | all | Cap the dataset size |
| `--seed` | 0 | Seeds the dataset, the split, and the weight init |
| `--patience` | 8 | Early-stopping patience, in epochs without improvement |

The train/val split is a fixed 80/20. Expect roughly 20 minutes on CPU for
512 samples over 20 epochs.

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

### 6. Alternative: run it all with Docker

```bash
docker compose up --build api ui          # API on :8000, dashboard on :8080
docker compose --profile train run train  # one-shot training run
```

`artifacts/` is a named volume shared by the API and the training container, so
a checkpoint trained by the `train` service is picked up by the API without
copying files around. The images install the CPU build of PyTorch and run as an
unprivileged user; weights are **not** baked into the image, so you must run
`train` (or mount a checkpoint) before the API will serve.

> The container files are committed but could not be built in the environment
> where they were written — Docker was unavailable. `docker compose config`
> parses cleanly and every path the Dockerfiles reference exists, but treat the
> first build as unverified.


---

## Development

```bash
pip install -e ".[dev]"

pytest          # 128 tests, 97% statement coverage of src/per_camera
ruff check src tests

cd ui && npm test   # 34 more, covering the dashboard
```

`httpx` is a hard requirement of the test suite (starlette's `TestClient`
imports it) and is declared in the `dev` extra. It previously resolved only as a
transitive dependency of something else, so a clean `pip install -e ".[dev]"`
failed at collection — which is exactly the sort of breakage a fresh-venv check
catches and a long-lived one hides.

The ONNX parity test needs `onnxruntime`, declared in
`requirements-export.txt`. Without it the test skips, and an export that parses
correctly but computes the wrong numbers would go unnoticed — the published
metrics would describe a model nobody is actually running.

The suite covers the failure modes this project previously shipped with: cwd-
dependent artifact paths, manifests carrying another machine's absolute paths,
checkpoints from a different architecture loading silently, the trainer
crashing on ordinary arguments, and malformed uploads returning 500.

| Test module | Covers |
|---|---|
| `tests/test_paths.py` | Repo-anchored artifacts, env override, no import side effects |
| `tests/test_data.py` | Relative manifest paths, seeding, GT/box agreement, depth realism, all three classes generated |
| `tests/test_detection.py` | Heatmap targets, NMS incl. class-agnostic suppression, decode round-trip, box maths |
| `tests/test_metrics.py` | Per-class mIoU incl. the 0/0 case, class-aware mAP against hand-computed values |
| `tests/test_checkpoint.py` | Metadata, round-trip, stale-architecture rejection |
| `tests/test_training.py` | Argument validation, split seeding, val-based selection, early stopping, the CLI end to end |
| `tests/test_api.py` | `/health` readiness, response contract, 400/413/422 paths, letterbox round-trip |
| `tests/test_onnx.py` | Export validity, output names, shapes carrying the real class count — and that the exported graph **computes the same numbers as PyTorch** |
| `tests/test_inference.py` | Demo-video entry point, and that it leaves the training dataset untouched |
| `tests/test_visualize.py` | Figure entry point, per-class colormap, clamped sample count |

The two entry points that were previously at 0% coverage — `inference.py` and
`visualize.py` — are now exercised for real, which is how the "GT" legend label
bug was found: it had been attached with `box is gt_boxes[0]`, which is always
False because numpy indexing returns a fresh view object.

### Detection post-processing

The heatmap NMS runs per class channel, so a single object can win in two
channels at once and come back as two detections at the same location with
different labels. `class_agnostic_nms` suppresses the weaker one before the
response is built. Measured on 40 frames: 85 raw detections → 78, removing 7
duplicates across 4 frames, and the exact per-frame class match went from
36/40 to 40/40. Without it the server-rendered overlay drew the duplicate over
the correct box, so a block could appear outlined in the triangle's colour.

---

## API Reference

### `POST /predict`

Accepts a single image file and returns multi-task inference results.

**Request**: `multipart/form-data` with field `file` (PNG, JPEG, or WebP).

The frame is **letterboxed** into the 128×128 model input rather than squashed,
so returned geometry maps back onto the original image without distortion.

**Response** (JSON):

```json
{
  "detections": [
    {
      "x_px": 120.5, "y_px": 88.0, "w_px": 64.0, "h_px": 48.0,
      "x": 0.188, "y": 0.183, "w": 0.1, "h": 0.1,
      "score": 0.87, "class": "block", "class_id": 0
    }
  ],
  "score": 0.87,
  "image_size": { "width": 640, "height": 480 },
  "depth_b64": "<base64 PNG — viridis>",
  "seg_b64": "<base64 PNG — one colour per class>",
  "input_b64": "<base64 PNG — letterboxed frame with boxes drawn>"
}
```

| Field | Description |
|---|---|
| `detections` | 0..N boxes, sorted by descending score |
| `detections[].x_px` … | Box in **original image pixels** (top-left origin) |
| `detections[].x` … | Same box normalised to `[0,1]` of the original image |
| `detections[].score` | Detection confidence `[0, 1]` |
| `detections[].class` | Class name: `block`, `sphere`, or `triangle` |
| `detections[].class_id` | Same class as an integer, for colour lookups |
| `score` | Convenience: the top detection's score, or `0.0` if none |
| `image_size` | Original upload dimensions, so clients can scale overlays |
| `ood_score` | Robust z-score of the input against the training distribution. `null` when no guardrail is fitted |
| `ood_threshold` | Rejection threshold; `null` when unguarded |
| `ood_rejected` | `true` when the input was judged out-of-distribution and detections were suppressed |
| `ood_suppressed_detections` | How many boxes were withheld, so a rejection is distinguishable from an empty scene |

**Errors**

| Status | Cause |
|---|---|
| `400` | Empty body, unknown magic bytes, or an undecodable image |
| `413` | Over 10 MB, or over 16 megapixels |
| `422` | Missing the `file` field |

### `GET /health`

Readiness probe. Reports the checkpoint that is actually loaded, not a
hardcoded `ok`:

```json
{
  "status": "ready",
  "arch_version": "big-per-camera-net-v3",
  "checkpoint_arch_version": "big-per-camera-net-v3",
  "checkpoint_epoch": 14,
  "device": "cpu",
  "input_size": 128,
  "classes": ["block", "sphere", "triangle"],
  "cors_origins": ["http://localhost:3000"]
}
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PER_CAMERA_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated CORS allowlist |
| `PC_POC_ARTIFACTS` | `<repo>/artifacts` | Where checkpoints and data live |
| `VITE_API_URL` (UI) | `/api` (Vite proxy) | API base URL for the dashboard |

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
| Medium | Integrate TensorRT / ONNX Runtime for edge deployment |
| Low | Multi-camera synchronization and stereo depth fusion |
| Low | Continuous training loop with online hard example mining |

### On the FPN neck

"Add an FPN for multi-scale detection" was on this list and has been **removed
as not justified by the data**. Measured with `metrics.ap_by_size`:

| band | area range | GT boxes | mAP@0.5 |
|---|---|---|---|
| small | < 0.005 | 0 | — |
| medium | 0.005 – 0.015 | 113 | 0.955 |
| large | > 0.015 | 212 | 0.963 |

Every object in the dataset is 12–21 px on a side at 128×128, so there is no
small-object regime at all, and performance is flat across the sizes that do
exist (0.955 vs 0.963). A multi-scale neck would add parameters and complexity
to solve a problem this data does not have. **Revisit this if real camera
footage replaces the synthetic generator** — that is when object scale actually
starts to vary.

Done and no longer on this list: repo-anchored artifact paths, fail-loud
checkpoint loading, pinned dependencies, seeded reproducible runs,
validation-based model selection, multi-object **and multi-class** detection
with class-aware mAP, input validation on the API, per-frame stream
cancellation with an in-flight guard, CI on every PR, containerised API + UI,
and a clean dependency audit on both stacks. See `AUDIT.md` for the full
history.

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

