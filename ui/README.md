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
