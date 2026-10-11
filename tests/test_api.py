"""The inference API: readiness, response contract, and input validation."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from per_camera.paths import IMG_SIZE


@pytest.fixture(scope="module")
def client(trained_artifacts):
    # Imported lazily: the module loads its checkpoint at import time, so the
    # session fixture has to have created one first.
    from per_camera.api import app

    return TestClient(app)


def _png(size=(64, 48), color=(10, 200, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_health_reports_the_loaded_model(client):
    """Regression: `/` used to return a hardcoded ok even when no checkpoint
    had been loaded, hiding a silently untrained network."""
    body = client.get("/health").json()

    assert body["status"] == "ready"
    assert body["arch_version"] == body["checkpoint_arch_version"]
    assert body["checkpoint_epoch"] >= 1
    assert body["input_size"] == 128
    assert body["cors_origins"]
    assert "*" not in body["cors_origins"]


def test_predict_returns_all_three_heads(client):
    response = client.post("/predict", files={"file": ("a.png", _png(), "image/png")})
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"input_b64", "seg_b64", "depth_b64", "detections", "score", "image_size"}
    assert isinstance(body["detections"], list)
    assert 0.0 <= body["score"] <= 1.0
    assert body["image_size"] == {"width": 64, "height": 48}


def test_detection_geometry_is_in_original_image_units(client):
    """Regression: boxes were returned in the squashed 128x128 space and the
    original size was read then thrown away."""
    response = client.post("/predict", files={"file": ("w.png", _png((640, 480)), "image/png")})
    body = response.json()

    assert body["image_size"] == {"width": 640, "height": 480}
    for det in body["detections"]:
        assert {"x", "y", "w", "h", "x_px", "y_px", "w_px", "h_px", "score", "class"} <= set(det)
        assert det["x_px"] == pytest.approx(det["x"] * 640, abs=0.5)
        assert det["y_px"] == pytest.approx(det["y"] * 480, abs=0.5)


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("corrupt.png", b"\x89PNG\r\n\x1a\nGARBAGE_NOT_AN_IMAGE"),
        ("note.txt", b"hello world this is a text file"),
        ("truncated.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"),
    ],
)
def test_malformed_input_returns_400_not_500(client, name, payload):
    """Regression: every one of these used to raise and return 500."""
    response = client.post("/predict", files={"file": (name, payload, "image/png")})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_empty_upload_returns_400(client):
    response = client.post("/predict", files={"file": ("e.png", b"", "image/png")})
    assert response.status_code == 400


def test_oversized_resolution_returns_413(client):
    buf = io.BytesIO()
    Image.new("RGB", (5000, 5000), (1, 2, 3)).save(buf, "PNG")  # 25 MP > the 16 MP cap
    response = client.post("/predict", files={"file": ("big.png", buf.getvalue(), "image/png")})
    assert response.status_code == 413


def test_oversized_body_returns_413(client):
    payload = b"\x89PNG" + b"x" * (11 * 1024 * 1024)
    response = client.post("/predict", files={"file": ("h.png", payload, "image/png")})
    assert response.status_code == 413


def test_missing_field_returns_422(client):
    response = client.post("/predict", files={"image": ("a.png", _png(), "image/png")})
    assert response.status_code == 422


def test_segmentation_colours_are_distinct_per_class():
    """Regression: labels were passed through `mask * 255` + a JET colormap, so
    classes 1, 2 and 3 all clipped to the same saturated colour and a
    four-class mask came back single-coloured."""
    from per_camera.api import _colorize_labels
    from per_camera.classes import NUM_SEG_CLASSES

    labels = np.arange(NUM_SEG_CLASSES, dtype=np.uint8).reshape(1, -1)
    bgr = _colorize_labels(labels)

    assert bgr.shape == (1, NUM_SEG_CLASSES, 3)
    colours = [tuple(int(c) for c in bgr[0, i]) for i in range(NUM_SEG_CLASSES)]
    assert len(set(colours)) == NUM_SEG_CLASSES, f"classes share a colour: {colours}"

    # Out-of-range indices must clamp, not raise, so a stale checkpoint that
    # emits an unexpected label still renders.
    assert _colorize_labels(np.array([[99]], dtype=np.uint8)).shape == (1, 1, 3)


def test_health_reports_the_loaded_checkpoint(client):
    """/health is the readiness probe and the only way to tell a stale
    checkpoint from a good one without reading server logs."""
    body = client.get("/health").json()

    assert body["status"] == "ready"
    assert body["arch_version"] == body["checkpoint_arch_version"], (
        "the loaded checkpoint was built for a different architecture"
    )
    assert body["checkpoint_epoch"] is not None
    assert body["input_size"] == IMG_SIZE
    assert isinstance(body["cors_origins"], list)


def test_health_lists_the_class_vocabulary(client):
    """The dashboard labels boxes with these names, so they are part of the
    contract and not an internal detail."""
    from per_camera.classes import CLASS_NAMES

    body = client.get("/health").json()

    assert body["classes"] == list(CLASS_NAMES)


def test_root_points_at_the_docs(client):
    body = client.get("/").json()

    assert body["status"] == "ok"
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"


def test_webp_is_accepted():
    """Regression guard on the magic-byte sniff: WebP is RIFF....WEBP, which a
    naive prefix check against RIFF alone would misclassify."""
    from per_camera.api import _sniff_format

    header = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 8
    assert _sniff_format(header) == "webp"


def test_unknown_magic_bytes_are_rejected():
    import pytest
    from fastapi import HTTPException

    from per_camera.api import _sniff_format

    with pytest.raises(HTTPException) as exc:
        _sniff_format(b"not an image at all")

    assert exc.value.status_code == 400


def test_a_png_encoding_failure_is_a_500_not_a_crash(monkeypatch):
    """`_to_b64_png` must convert an OpenCV failure into a real HTTP error
    rather than returning an empty string that renders as a broken image."""
    import pytest
    from fastapi import HTTPException

    from per_camera import api

    monkeypatch.setattr(api.cv2, "imencode", lambda *a, **k: (False, None))

    with pytest.raises(HTTPException) as exc:
        api._to_b64_png(np.zeros((4, 4, 3), dtype=np.uint8))

    assert exc.value.status_code == 500


def test_main_configures_uvicorn_from_argv(client, monkeypatch):
    """The CLI entry point must pass host/port through, and must use the import
    string when --reload is set (uvicorn requires that for reloading)."""
    from per_camera import api

    seen = {}

    def fake_run(target, host, port, reload):  # noqa: A002 - mirrors uvicorn
        seen.update(target=target, host=host, port=port, reload=reload)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    assert api.main(["--host", "127.0.0.1", "--port", "9999"]) == 0
    assert seen == {"target": api.app, "host": "127.0.0.1", "port": 9999, "reload": False}

    assert api.main(["--reload"]) == 0
    assert seen["target"] == "per_camera.api:app"
    assert seen["reload"] is True


def _forward_box(x0, y0, bw, bh, *, scale, pad_x, pad_y, size):
    """Where a box in original-image pixels lands in normalised letterbox space.

    This is the transform the network sees, written out independently of
    `box_to_original` so the test is a genuine round trip rather than a
    re-derivation of the same formula.
    """
    cx = (x0 + bw / 2) * scale + pad_x
    cy = (y0 + bh / 2) * scale + pad_y
    return cx / size, cy / size, bw * scale / size, bh * scale / size


@pytest.mark.parametrize(
    ("orig_w", "orig_h"),
    [(128, 128), (640, 480), (480, 640), (1920, 1080), (100, 100), (333, 77)],
)
def test_box_to_original_inverts_the_letterbox(orig_w, orig_h):
    """Regression: predicted boxes used to be returned in network space, so on
    any non-square upload the overlay landed in the wrong place (audit #15)."""
    from PIL import Image as PILImage

    from per_camera.api import box_to_original, letterbox
    from per_camera.paths import IMG_SIZE

    _, scale, pad_x, pad_y = letterbox(
        PILImage.new("RGB", (orig_w, orig_h)), IMG_SIZE
    )

    for frac in (0.0, 0.25, 0.5):
        bw, bh = orig_w * 0.2, orig_h * 0.2
        x0, y0 = orig_w * frac, orig_h * frac
        # Skip boxes that would overflow the frame.
        if x0 + bw > orig_w or y0 + bh > orig_h:
            continue

        xc, yc, wn, hn = _forward_box(
            x0, y0, bw, bh, scale=scale, pad_x=pad_x, pad_y=pad_y, size=IMG_SIZE
        )
        rx, ry, rw, rh = box_to_original(
            xc, yc, wn, hn, scale=scale, pad_x=pad_x, pad_y=pad_y, size=IMG_SIZE
        )

        assert rx == pytest.approx(x0, abs=1e-6)
        assert ry == pytest.approx(y0, abs=1e-6)
        assert rw == pytest.approx(bw, abs=1e-6)
        assert rh == pytest.approx(bh, abs=1e-6)


def test_letterbox_never_distorts_the_aspect_ratio():
    """Squashing to square would make every returned box wrong in one axis."""
    from PIL import Image as PILImage

    from per_camera.api import letterbox
    from per_camera.paths import IMG_SIZE

    canvas, scale, pad_x, pad_y = letterbox(PILImage.new("RGB", (640, 480)), IMG_SIZE)

    assert canvas.size == (IMG_SIZE, IMG_SIZE)
    # The content keeps 4:3, so the horizontal padding must be zero and the
    # vertical padding non-zero for a landscape frame.
    assert pad_x == 0
    assert pad_y > 0
    assert round(640 * scale) == IMG_SIZE
