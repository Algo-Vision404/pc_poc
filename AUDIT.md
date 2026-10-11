# Project Audit — `pc_poc` (Per-Camera Proof-of-Concept)

Audited at commit `14d8cc0` on branch `arena/7dbe8a79-pc-poc`.

**Method:** every claim below was produced by *executing* the code in a fresh
Python 3.11.2 venv (torch 2.14.1, torchvision 0.29.1, fastapi 0.143.0,
opencv 5.0.0) and in a fresh `npm install`. Nothing here is inferred from
reading alone. Where I could not execute a path, it is labelled **[unverified]**.

---

## Headline

**Not one of the four entry points in this repository runs successfully on a
clean clone.** The committed model checkpoint was trained by an older
architecture and cannot be loaded by the current code; the training script
always exits with a non-zero status; the documented API command silently serves
a randomly-initialised network; and `requirements.txt` is missing the packages
the API imports.

The README describes a "production-grade, multi-task autonomous perception
framework." The code underneath is a genuine, working single-file PoC — but it
is not reproducible, not installable, not tested, and the artifacts committed
to git do not correspond to the code committed to git.

---

## P0 — Blocking: nothing runs

### 1. The shipped checkpoint cannot be loaded by the shipped code

`artifacts/models/poc_checkpoint.pt` was produced by the old `TinyPerCameraNet`,
not by the current `BigPerCameraNet`.

```
checkpoint tensors: 16        current model tensors: 92
keys missing from checkpoint : 84      unexpected keys in checkpoint: 8
SHAPE MISMATCHES: 8
    seg.0.weight      ckpt(32, 64, 3, 3) != code(64, 128, 3, 3)
    depth.3.weight    ckpt(1, 32, 1, 1)  != code(32, 64, 3, 3)
load_state_dict: RuntimeError
```

The checkpoint's keys are `conv.0.weight`, `conv.3.weight`, … — a flat
`nn.Sequential` that does not exist in `BigPerCameraNet` (which has `conv1`,
`bn1`, `layer1`, `layer2`).

Consequences, all executed:

| Entry point | Result |
|---|---|
| `python "train and vis/visualize_poc.py"` | `RuntimeError`, **exit 1** |
| `import api` from repo root | `RuntimeError` at import — the app never boots |
| `import api` from `train and vis/` (as documented) | boots **with random weights** — see #4 |
| `python "train and vis/inference_poc.py"` | `Checkpoint not found` (from `train and vis/`) or `RuntimeError` (from root) |

**Fix:** delete the stale checkpoint from git history, retrain, and add an
architecture fingerprint to the checkpoint so this can never happen silently
again (see #24).

---

### 2. `per_camera_poc.py` always exits 1 — the training run never completes

The "Windows UTF-8 safety" wrapper around the ONNX export destroys stdout:

```python
_orig_stdout = sys.stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
```

Reassigning `sys.stdout` orphans the original `TextIOWrapper`. When it is
garbage-collected, `TextIOWrapper.__del__` closes the underlying buffer — the
*same* buffer the new wrapper is writing to. Every subsequent `print` then
raises `ValueError: I/O operation on closed file`.

This fires on **both** paths:

```
# ONNX export FAILS:
  File "per_camera_poc.py", line 371, in train
    print(f"[WARNING] ONNX export failed (non-critical): {e}")
ValueError: I/O operation on closed file.
EXIT CODE = 1

# ONNX export SUCCEEDS (after installing onnx + onnxscript):
  File "per_camera_poc.py", line 367, in train
    print(f"ONNX model saved to {onnx_path}")
ValueError: I/O operation on closed file.
EXIT CODE = 1
```

Note the irony: the `try/except` that exists to make ONNX failure
*"non-critical"* is what converts it into a fatal error. `train()` never
returns, and `__main__`'s `print("Training complete. Best metrics:")` is
unreachable. **README Quick Start step 2 is broken on every platform, not just
Windows.**

**Fix:** delete the wrapper entirely. Use `PYTHONIOENCODING=utf-8`, or
`sys.stdout.reconfigure(encoding="utf-8")`, and never reassign `sys.stdout`.

---

### 3. ONNX export fails out of the box

```
ModuleNotFoundError: No module named 'onnxscript'
```

`onnx` and `onnxscript` are not in `requirements.txt`, so the export the README
advertises can never succeed on a fresh install — and because of #2 it takes
the whole script down with it.

---

### 4. `ARTIFACTS` is CWD-relative, so the documented commands find nothing

```python
ARTIFACTS = Path("./artifacts").resolve()   # resolved at IMPORT time, against cwd
```

The README says `cd "train and vis" && uvicorn api:app --port 8000`. Executed
exactly that way:

```
ARTIFACTS resolved to : /home/user/pc_poc/train and vis/artifacts
checkpoint path       : .../train and vis/artifacts/models/poc_checkpoint.pt
checkpoint exists?    : False
```

`api.py` then hits:

```python
if model_path.exists():
    ...load_state_dict(...)
model.eval()          # <- runs regardless
```

There is no `else`. **The API starts cleanly, reports `{"status":"ok"}` on `/`,
and serves predictions from a randomly-initialised network.** No exception, no
warning, no log line. This is the single most dangerous bug in the repo: it
fails *silently* and produces plausible-looking garbage.

Two related problems:
- `import per_camera_poc` **creates a directory tree as a side effect.** Merely
  importing `api.py` created `train and vis/artifacts/{models,results,synthetic_data}`
  on disk.
- `inference_poc.py` run as documented prints `Checkpoint not found. Run training first.`
  even though a checkpoint is sitting in the repository.

**Fix:** anchor paths to the file, not the process:
`ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"`, overridable
by a `PC_POC_ARTIFACTS` env var. Move directory creation into an explicit
`ensure_dirs()` called from `main()`. Raise on a missing checkpoint.

---

### 5. `requirements.txt` is missing the API's dependencies

`api.py` imports `fastapi` and needs `uvicorn` + `python-multipart` (the latter
for `UploadFile`). None are listed:

```
fastapi      in requirements.txt? NO
uvicorn      in requirements.txt? NO
multipart    in requirements.txt? NO
onnx         in requirements.txt? NO
onnxscript   in requirements.txt? NO
```

README step 3 (`uvicorn api:app --port 8000`) therefore fails immediately with
`ModuleNotFoundError` on a clean install.

Additionally: **zero pinned versions** (`grep -c "==" requirements.txt` → `0`).
Every install resolves to whatever is newest, which is exactly how #1-class
drift happens unnoticed.

---

### 6. The committed manifest contains hardcoded Windows paths

```
image,seg,depth,class,xc,yc,w,h
C:\Users\hp\Per-Camera Proof-of-Concept\artifacts\synthetic_data\img_0000.png,...
```

On this Linux checkout that path resolves to
`C:UsershpPer-Camera Proof-of-Conceptartifacts...` — **MISSING**. The committed
manifest is unusable on any machine other than the author's laptop.

It is also internally inconsistent: the manifest has **64 rows** while
`artifacts/synthetic_data/` contains **512** `img_`/`seg_`/`depth_` triples.

**Fix:** never persist absolute paths. Store paths relative to the manifest, or
better, derive filenames from an index and a single `--data-dir`.

---

## P1 — Correctness

### 7. "Best" checkpoint selection compares training loss against a variable named `best_val_loss`

```python
best_val_loss = float('inf')
...
if avg_train_loss < best_val_loss:      # <- TRAIN loss
    best_val_loss = avg_train_loss
    torch.save(...)
```

Training loss decreases monotonically, so the guard is always true.
Instrumenting `torch.save` over a 4-epoch run:

```
Epoch 1/4 - Loss: 3.8964 ...   >>> torch.save() fired
Epoch 2/4 - Loss: 0.4241 ...   >>> torch.save() fired
Epoch 3/4 - Loss: 0.3397 ...   >>> torch.save() fired
Epoch 4/4 - Loss: 0.1782 ...   >>> torch.save() fired
saved 4 times in 4 epochs -> saves EVERY epoch
```

**"Best" always means "last".** There is no model selection and no early
stopping. This is not hypothetical — a real 1024-sample run shows validation
metrics regressing while the checkpoint keeps being overwritten:

```
Epoch 2/20 - Loss: 0.1168 - Val mIoU: 0.8600 - Val RMSE: 0.0418 - Val Det IoU: 0.4195
Epoch 3/20 - Loss: 0.0816 - Val mIoU: 0.8130 - Val RMSE: 0.0448 - Val Det IoU: 0.1467
                            ^^^^^^^ worse      ^^^^^^^ worse       ^^^^^^^ 3x worse
```

Epoch 3 is strictly worse on every validation metric yet was saved as "best".

**Fix:** select on a validation composite (e.g. `val_mIoU - val_rmse`), keep
`best` and `last` checkpoints separately, add patience-based early stopping.

---

### 8. Ordinary CLI arguments crash the trainer

```
epochs=0              : UnboundLocalError: cannot access local variable 'ckpt_p'
samples=1             : ValueError: num_samples should be a positive integer value, but got num_samples=0
```

`ckpt_p` is only bound inside the save branch, but is dereferenced
unconditionally afterwards. `--samples 1` makes the 80/20 split round the
validation set to zero.

**Fix:** validate arguments up front; bind `ckpt_p` before the loop.

---

### 9. No seeding anywhere — nothing is reproducible

`grep -rn "manual_seed\|random.seed\|np.random.seed\|deterministic"` across
`train and vis/` returns **nothing**.

```
same command twice -> depth[0] run1=0.26813 run2=0.56593   identical=False
```

The dataset, the train/val split (`torch.utils.data.random_split` with no
generator), and the weight init are all unseeded. Two runs of the same command
produce different data *and* different splits.

This also means the README's claim that depth is "**deterministically
derived**" is misleading — the *formula* is deterministic, but the boxes it is
applied to are not.

**Fix:** add `--seed`, call `random.seed`/`np.random.seed`/`torch.manual_seed`,
pass an explicit `torch.Generator` to `random_split`, and record the seed in the
results JSON.

---

### 10. Ground-truth labels are internally inconsistent (off-by-one)

```python
draw_seg.rectangle([x, y, x + w, y + h], fill=1)   # PIL is inclusive -> (w+1) x (h+1)
w_norm, h_norm = w / IMG_SIZE, h / IMG_SIZE        # label says w x h
```

Measured on 5/5 generated samples:

```
sample 0: bbox label 36 x 26   actual mask 37 x 27   OFF-BY-ONE: +1 x +1
sample 1: bbox label 19 x 21   actual mask 20 x 22   OFF-BY-ONE: +1 x +1
sample 2: bbox label 34 x 33   actual mask 35 x 34   OFF-BY-ONE: +1 x +1
sample 3: bbox label 25 x 26   actual mask 26 x 27   OFF-BY-ONE: +1 x +1
sample 4: bbox label 29 x 23   actual mask 30 x 24   OFF-BY-ONE: +1 x +1
```

The segmentation/depth targets and the detection target disagree by
construction, on every single sample. It caps the achievable IoU and teaches
the two heads contradictory geometry.

**Fix:** `draw.rectangle([x, y, x + w - 1, y + h - 1])`, and label the box as
`((x+0.5)/S, (y+0.5)/S, w/S, h/S)` in pixel-centre convention.

---

### 11. The "depth" task is not depth — and seg/depth are the same task

Depth GT is `0.1 + 0.8 * f(bbox_area)`, constant inside the object, `0.0`
outside:

```
sample 0: depth unique values: [0.         0.53516483]
sample 1: depth unique values: [0.         0.24010989]
sample 2: depth unique values: [0.         0.63736266]
```

Verified the formula reproduces the files exactly (8/8 samples, `< 1e-5`), and
that depth's support is *identical* to the segmentation mask's:

```
pixels where seg==1  : 999
pixels where depth>0 : 999
identical support?   True
```

So the three "tasks" are one task counted three ways:
- **segmentation** = find the rectangle
- **depth** = the same rectangle, scaled by a scalar that is a pure function of
  its area
- **detection** = the rectangle's box

A model can drive depth RMSE to near zero by copying its segmentation mask and
multiplying by a scalar it can infer from the box it already predicts. No
geometry, no occlusion, no perspective is learned. **"Depth RMSE 0.096"
measures area regression**, and the 30× loss weight is tuning the balance
between two copies of the same signal.

**Fix:** if depth is meant to be a real capability, generate it from something
the model cannot shortcut — e.g. a smooth gradient plane per object, occlusion
ordering between overlapping objects, and background depth that varies.
Otherwise, drop the depth head and stop calling this multi-task.

---

### 12. `mIoU` is misnamed and has an unguarded 0/0

```python
intersection = torch.logical_and(preds_seg, seg_t).float().sum()
union = torch.logical_or(preds_seg, seg_t).float().sum()
iou_seg = intersection / (union + 1e-6)
total_iou_seg += iou_seg.item()
```

This is a **batch-global, foreground-only IoU** — not *mean* over classes. The
background class IoU is never computed, and averaging a batch-level IoU over
batches is biased by a partial final batch. Calling it "mIoU" in the README
overstates what is measured.

Feeding exact ground truth through the real `evaluate()` shows the metric is
otherwise sane:

```
perfect model -> mIoU=1.0000  rmse=0.0000  det_IoU=1.0000   (expected 1.0/0.0/1.0)
```

But the degenerate case is unguarded:

```
model predicts background perfectly and GT IS background -> mIoU=0.0000
```

A **perfect** prediction on a background-only frame scores **0.0**. Harmless on
today's dataset (every frame has an object) but a live trap the moment
empty/negative frames are introduced — which is exactly what #13 requires.

**Fix:** per-class IoU with a proper ignore-index, averaged over classes;
rename to what it is.

---

### 13. There is no detection task — only single-box regression, and the class label is a constant

The generator draws exactly **one** rectangle per frame (a single
`draw.rectangle` call, no loop), and the class label is hardcoded:

```
=== class label distribution in the generated manifest ===
  Counter({'1': 200})
```

`det_cls` is trained with `BCEWithLogitsLoss` against a constant positive
label. There is no negative class and no empty frame, so the head converges to
"always output ≈1.0".

Two consequences:
- **"Detection IoU 0.387" is a localisation regression error, not a detection
  metric.** There is no AP, no recall/precision, no thresholding, no NMS.
- **The UI's "Confidence Threshold" slider filters on a score that carries no
  information.** "Suppressed (Low Confidence)" can only ever appear while the
  model is untrained.

Also note `det_bbox` ends in `Sigmoid`, so `xc`/`yc` saturate toward 0/1 for
objects near the frame edge — vanishing gradients exactly where boxes are
hardest.

**Fix:** emit 0–3 objects per frame including empty frames; use a real
class-balanced label; report AP@0.5 and recall; consider a per-pixel
centerness/heatmap head instead of global box regression.

---

### 14. The API returns HTTP 500 on any malformed input

Executed against the real app:

```
1. valid PNG        -> 200
2. corrupt PNG      -> 500  Internal Server Error
3. text file        -> 500  Internal Server Error
4. empty file       -> 500  Internal Server Error
5. wrong field name -> 422  (FastAPI's own validation — the only good one)
6. 1x1 px           -> 200
7. 640x480 upload   -> 200
```

`Image.open(io.BytesIO(contents)).convert("RGB")` is completely unguarded.
There is no content-type check, no size cap, and `cv2.imencode`'s success flag
is discarded (`_, buffer = cv2.imencode(...)`) — a failed encode yields
`base64.b64encode(None)` → `TypeError`.

**Fix:** validate magic bytes + content type, cap upload size, return `400`
with a useful message, check the `imencode` flag.

---

### 15. Bounding boxes are never mapped back to the input image

```python
original_size = img.size      # line 61 — assigned, NEVER referenced again
```

The frame is squashed to 128×128, the box is predicted in that space, and it is
returned unmodified:

```
7. 640x480 upload -> 200   returned input_b64 is (128, 128) (aspect lost)
   bbox={'x': 0.514, 'y': 0.494, 'w': 0.500, 'h': 0.485}
```

`original_size` exists precisely to undo this and was never wired up.

---

### 16. The frontend overlay is geometrically wrong for non-square images

`.grid-item img` is `width: 100%` with **no `aspect-ratio` and no
`object-fit`**, and the overlay is positioned as percentages of that element
using coordinates from the squashed 128×128 space:

```jsx
left:   `${(results.bbox.x - results.bbox.w/2) * 100}%`,
width:  `${results.bbox.w * 100}%`,
```

For any non-square upload the rendered image keeps its natural aspect ratio
while the box assumes a square — the overlay drifts, increasingly so the
further the aspect ratio is from 1:1. Combined with #15, boxes are wrong twice
over.

---

## P2 — Security & operations

### 17. CORS allows any origin with credentials

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, ...)
```

Verified against the real app:

```
--- PREFLIGHT from https://evil.example ---
  access-control-allow-origin: https://evil.example
  access-control-allow-credentials: true
```

Starlette reflects the requesting origin rather than sending `*`, so **any
website can call this API**. `allow_origins=["*"]` + `allow_credentials=True`
is also a spec-invalid combination.

**Fix:** explicit allowlist from an env var; drop credentials (nothing here
uses them).

---

### 18. Blocking inference inside `async def`

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ...
    out = model(img_t)        # blocking, on the event loop
```

Measured with a 1 s forward pass:

```
3 sequential requests : 3.18s
```

Because the handler is `async def`, the forward pass runs on the event loop
thread. Request #2 cannot begin until #1 finishes, and the health endpoint is
blocked too. Under uvicorn this is a hard concurrency ceiling of **1**.

**Fix:** make the handler `def` (Starlette runs it in a threadpool), or
`await run_in_executor(...)`.

---

### 19. No upload size limit or decompression-bomb guard

Nothing caps the request body. PIL's bomb guard only trips at ~89 M pixels.
A large upload will happily exhaust memory.

---

### 20. Frontend: 8 vulnerabilities on a clean install, and lint is red

```
npm install -> 8 vulnerabilities (1 low, 1 moderate, 6 high)
npm run lint -> EXIT=1
  /ui/src/App.jsx  1:8  error  'React' is defined but never used  no-unused-vars
npm run build -> succeeds (196.65 kB JS, 3.74 kB CSS)
```

High-severity advisories include `brace-expansion`, `browserslist`, `nanoid`
and `postcss`. `npm run lint` fails on a pristine checkout, so any CI that runs
it is red from day one.

---

### 21. The backend URL is hardcoded — the UI cannot be deployed

```
ui/src/App.jsx:68:  fetch('http://127.0.0.1:8000/predict', ...)
```

No `import.meta.env`, no `VITE_*` variable, no `.env`, and `vite.config.js` has
no `server.proxy`. The dashboard only works when the browser is on the same
machine as the API. It cannot be served from any other host, container, or
preview environment.

**Fix:** `import.meta.env.VITE_API_URL ?? '/api'` plus a Vite dev proxy; add
`.env.example`.

---

### 22. 38 MB of generated data and a stale checkpoint are committed to git

```
tracked files: 1567   of which artifacts/: 1544
tracked blob total: 38M
  1536 artifacts/synthetic_data
```

`.gitignore` intends to exclude them:

```
train and vis/artifacts/synthetic_data/
train and vis/artifacts/models/*.pt
```

…but the artifacts actually live at the **repo root** `artifacts/`, so the
patterns never match:

```
$ git check-ignore -v artifacts/synthetic_data/depth_0000.npy
NOT IGNORED -> committed
```

The `.gitignore` is written for a directory layout the code does not produce
(see #4 and #29).

---

### 23. No tests, no CI, no licence, no packaging

```
no .github            -> no CI at all
no test_*.py          -> no Python tests
no conftest.py
no pyproject.toml     -> not installable, no dependency metadata
no Dockerfile
no LICENSE
```

For a project whose README opens with "production-grade", there is no automated
verification of anything.

---

## P3 — Structure & hygiene

### 24. Checkpoints carry no metadata

```
top-level keys: ['model_state', 'opt_state']
```

No epoch, no validation metrics, no image size, no class names, no config, no
architecture version, no torch version. **This is precisely why the stale
checkpoint in #1 went unnoticed** — there was nothing in the file to compare
against the code.

**Fix:** always save `{model_state, epoch, val_metrics, config, arch_version,
img_size, torch_version, created_utc}`, and assert `arch_version` on load.

### 25. `train and vis/` — a directory with a space in its name

It cannot be a Python package, so the code resorts to
`sys.path.append(os.path.dirname(os.path.abspath(__file__)))`, and every shell
command in the README must be quoted. Rename to `src/` (or
`per_camera/`) and add a `pyproject.toml`.

### 26. Module-level side effects

- `import per_camera_poc` creates four directories.
- `import visualize_poc` loads a checkpoint, builds a dataset, and renders
  figures — there is no `if __name__ == "__main__":` guard, so it is
  unimportable and untestable.

### 27. `inference_poc.py` overwrites the training dataset

It calls `generate_synthetic_dataset(args.frames)`, which rewrites
`dataset_manifest.csv` and overwrites `img_0000..N`/`seg_0000..N`/`depth_0000..N`
in place. Running the documented demo-video step truncates the manifest and
clobbers the first N training images. **[unverified at runtime]** — the
checkpoint load (#1) crashed before reaching that line — but the code path is
unambiguous.

**Fix:** generate demo frames into a separate directory, or add a
`--out-dir` that never touches the training set.

### 28. The README's Performance table is not backed by any artifact

README claims (2048 samples, 30 epochs):

| Metric | README |
|---|---|
| Segmentation mIoU | 0.927 |
| Depth RMSE | 0.096 |
| Detection IoU | 0.387 |
| Best Val Loss | 0.0267 |

The committed `artifacts/results/poc_results.json` contains:

```json
{
  "ckpt": "C:\\Users\\hp\\Per-Camera Proof-of-Concept\\artifacts\\models\\poc_checkpoint.pt",
  "avg_loss": 1.8979230262339115
}
```

No metrics at all — just a *training* loss of 1.898 and a Windows path. Worse,
that schema (`avg_loss`) does not match what the current code writes
(`best_metrics`), proving the results file predates the current script. And
"Best Val Loss 0.0267" cannot be a validation loss at all, given #7.

For reference, a real run I executed (1024 samples, 4 epochs, batch 32) reached
`mIoU 0.9133 / RMSE 0.0369 / Det IoU 0.3923` — the same order of magnitude, so
the numbers are plausible. But they are **unreproducible** (#9) and
**unsupported by anything in the repository**.

### 29. README architecture diagram contradicts the code

README's Detection Head diagram:

```
▼  AdaptiveAvgPool2d(4×4)  →  Flatten  →  1024
├──► FC(1024→1)   →  cls score
└──► FC(1024→4)   →  [xc, yc, w, h]
```

The actual head (`1,092,744` params total):

```
(0) Conv2d(128, 64, 3x3, stride=2)  (1) BN  (2) ReLU
(3) Conv2d(64, 32, 3x3, stride=2)   (4) BN  (5) ReLU
(6) Flatten  (7) Linear(512 -> 256)  (8) ReLU  (9) Dropout(0.3)
cls : Linear(256 -> 1)
bbox: Linear(256 -> 4) + Sigmoid

AdaptiveAvgPool2d present anywhere? False
layer2 feature map: (2, 128, 16, 16)
flattened det feats: 512   (README says 1024)
```

The prose ("replaced destructive `AdaptiveAvgPool2d` with spatial
convolutions") is correct — the *diagram* directly below it was never updated.
The README also states artifacts are written to `train and vis/artifacts/`;
they are written to `<cwd>/artifacts`.

### 30. Dead code and template cruft

| Item | Status |
|---|---|
| `ui/src/App.css` (184 lines) | never imported — dead |
| `ui/src/assets/hero.png` | orphan, never referenced |
| `ui/src/assets/react.svg` | orphan |
| `ui/src/assets/vite.svg` | orphan |
| `ui/public/icons.svg` | orphan |
| `ui/README.md` | stock Vite template boilerplate |
| `ui/index.html` `<title>` | `"ui"` — the Vite default |
| `inference_poc.py` | unused import `Path` |
| `visualize_poc.py` | unused imports `argparse`, `json` |
| `api.py:61` | `original_size` assigned, never used |

### 31. Frontend robustness gaps

- **No request cancellation.** The 2 fps stream uses `setInterval` with an
  `async` callback and no in-flight guard or `AbortController`. If a round trip
  exceeds 500 ms, requests pile up and resolve out of order, so a stale
  response can overwrite a fresh one.
- **Memory leak.** `URL.createObjectURL(file)` (line 59) is never paired with
  `URL.revokeObjectURL`.
- **`alert()` for errors**, and silent `return null` on stream failures — no
  visible "backend unreachable" state.
- **Webcam constraints are ideals, not requirements.**
  `getUserMedia({ video: { width: 128, height: 128 } })` — real cameras return
  640×480 and the frame is squashed.
- `useEffect(() => () => stopAllStreams(), [])` closes over a non-memoised
  function; `videoRef.current.srcObject` is never nulled after stopping tracks.
- No TypeScript, no tests, no error boundary.

### 32. Miscellaneous

- `print` everywhere instead of `logging`; no request IDs, no metrics, no
  structured errors.
- `inference_poc.py` uses `exit(1)` instead of `sys.exit(1)`, and never checks
  `cv2.VideoWriter.isOpened()` — a missing `mp4v` codec silently produces an
  empty file.
- `api.py`'s health endpoint is `GET /` and reports
  `{"status":"ok"}` even when no checkpoint was loaded (#4).
- `per_camera_poc.py`'s `_default_transform` returns `[0,1]` while training and
  serving normalise to `[-1,1]` — an inconsistency waiting to bite anyone who
  omits `transform`.

---

## How to make this solid — prioritised

### Phase 0 — Make it run (a day)

1. Delete the `sys.stdout` wrapper; use `sys.stdout.reconfigure(encoding="utf-8")`.
2. Anchor `ARTIFACTS` to `Path(__file__)`, not `Path(".")`; add a
   `PC_POC_ARTIFACTS` override.
3. **Raise** if the checkpoint is missing; never serve random weights. Add a
   real `/health` that reports model readiness.
4. Complete `requirements.txt` (`fastapi`, `uvicorn[standard]`,
   `python-multipart`, `onnx`, `onnxscript`) and **pin** everything.
5. `git rm -r --cached artifacts/`, fix the `.gitignore` paths to match
   reality, and stop storing absolute paths in the manifest.
6. Rename `train and vis/` → `src/per_camera/`; add `pyproject.toml`.

### Phase 1 — Make it correct (2–3 days)

7. Seed everything; add `--seed`; pass a `Generator` to `random_split`.
8. Select checkpoints on validation metrics; add early stopping.
9. Fix the off-by-one so seg/depth and detection GT agree.
10. Guard the 0/0 in IoU; compute true per-class mIoU.
11. Decide the depth question: give it a non-shortcuttable signal, or remove it.
12. Add empty frames and 0–3 objects so detection is a real task with real AP.
13. Map boxes back to the input resolution; fix the frontend overlay geometry.
14. Save full metadata in the checkpoint and assert `arch_version` on load.
15. Re-run training and regenerate `poc_results.json`; make the README quote
    that file rather than hardcoded numbers.

### Phase 2 — Make it safe to serve (2–3 days)

16. CORS allowlist from env; drop `allow_credentials`.
17. `def` handler (or `run_in_executor`) so inference leaves the event loop.
18. Validate uploads: magic bytes, content type, size cap; check `imencode`.
19. Return `400`/`422` with real messages instead of `500`.
20. `logging` + request IDs + `/metrics`.

### Phase 3 — Make it maintainable (ongoing)

21. `pytest` suite: dataset GT invariants (mask/bbox agreement, depth range),
    model shape contracts, metric correctness against an oracle, API error
    matrix, checkpoint round-trip.
22. GitHub Actions: lint + typecheck + test + `vite build` on every PR.
23. Frontend: `VITE_API_URL` + Vite proxy, `AbortController` per stream tick,
    in-flight guard, revoke object URLs, visible connection state, error
    boundary.
24. Fix `npm audit` (6 high); delete the dead CSS/SVGs; set a real `<title>`
    and write a real `ui/README.md`.
25. `Dockerfile` + `docker-compose.yml` (API + UI) — the README already lists
    this on the roadmap.
26. Add `LICENSE` and `CONTRIBUTING.md`.

---

## The two things I would fix first

If nothing else gets done:

1. **#4 — the silent random-weight API.** It is the only bug here that produces
   *confident wrong answers* instead of an error. Everything else announces
   itself with a traceback.
2. **#1 + #24 — the stale checkpoint with no metadata.** Until checkpoints
   self-describe their architecture, this class of bug will keep recurring, and
   you will keep discovering it at inference time rather than at save time.
