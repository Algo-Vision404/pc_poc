# Per-Camera PoC — Dashboard

React + Vite single-page dashboard for the `per_camera` inference API. Three
input modes (file upload, synthetic stream, webcam) and three output panels
(annotated input, depth map, segmentation mask).

## Run it

```bash
npm install
npm run dev        # http://localhost:3000
```

The dashboard talks to the API at **`/api`**, which the Vite dev server proxies
to the backend. Nothing in the browser calls the API origin directly — that
would break the moment the two are served from different hosts.

Start the backend first:

```bash
# from the repository root
python -m uvicorn per_camera.api:app --host 0.0.0.0 --port 8000
```

## Configuration

Copy `.env.example` to `.env.local` to override.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_URL` | `/api` | API base URL. Set it only when the dashboard is served from a different origin than the API and there is no reverse proxy in front. |
| `PER_CAMERA_API_TARGET` | `http://127.0.0.1:8000` | Where the dev proxy forwards `/api`. Read by `vite.config.js` (Node side, not exposed to the browser). |
| `PER_CAMERA_ALLOWED_HOSTS` | any | Comma-separated `Host` allowlist for the dev server. Vite rejects unknown hosts by default to guard against DNS rebinding; set this when running behind a proxy with a fixed hostname. |

Only `VITE_`-prefixed variables reach browser code.

## Scripts

| Script | Purpose |
|---|---|
| `npm run dev` | Dev server with HMR on port 3000 |
| `npm run build` | Production bundle to `dist/` |
| `npm run preview` | Serve the built bundle locally |
| `npm run lint` | ESLint |
| `npm test` | Vitest, single run (CI) |
| `npm run test:watch` | Vitest in watch mode |

## Tests

`npm test` runs 34 tests across three files:

| File | Covers |
|---|---|
| `src/lib/api.test.js` | The request contract: multipart shape, `ApiError` carrying the HTTP status, the API's `detail` message surfacing, non-JSON error bodies falling back to the status line, and aborts propagating unwrapped so a superseded frame stays quiet |
| `src/lib/synthetic.js` tests | That each class draws its own silhouette — a block a rect, a sphere an inscribed ellipse, a triangle an apex-top path — that frames stay in bounds, that empty frames occur, and that all three classes appear over many frames |
| `src/App.test.jsx` | The dashboard end to end through a stubbed `fetch`: the checkpoint badge, error banners, the confidence filter, per-class counts, and box colours |

The synthetic-frame generator mirrors `per_camera.data`, so those shape tests
are what keep the demo stream in-distribution. If a class is added to
`classes.py`, the generator and its tests must follow or the demo silently
feeds the model a distribution it was never trained on.

jsdom has no canvas backend. Rather than depend on the native `canvas` package
(which needs a C compiler at install time), `src/test/setup.js` substitutes a
2D context that records calls. The tests therefore assert *which drawing calls
were made* — that a sphere issues `ellipse`, a triangle issues `moveTo`/`lineTo`
/`closePath` — rather than pixel output, which would test the rasteriser and
not this code.

## How it talks to the backend

`POST {API_URL}/predict` with `multipart/form-data` field `file`. The response
carries a `detections` array (0..N boxes, sorted by score) plus base64 PNGs for
the annotated input, depth, and segmentation panels. Box coordinates come back
normalised to the **original** image, so the overlay aligns at any aspect ratio.

The streaming modes abort the previous request when a newer frame arrives and
skip a tick entirely while one is still in flight. Without that, a backend
slower than the frame rate queues requests and they resolve out of order, so a
stale frame overwrites a newer one on screen.

## Production

`Dockerfile` builds the bundle and serves it from nginx, which proxies `/api`
to the API service — the same arrangement as the dev proxy. See
`docker-compose.yml` in the repository root.
