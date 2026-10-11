"""The inference API: readiness, response contract, and input validation."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


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
