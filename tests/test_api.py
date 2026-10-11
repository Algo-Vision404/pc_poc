"""The inference API must never start in a broken state."""

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


def test_predict_returns_all_three_heads(client):
    response = client.post("/predict", files={"file": ("a.png", _png(), "image/png")})
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"input_b64", "seg_b64", "depth_b64", "bbox", "score"}
    assert 0.0 <= body["score"] <= 1.0
    assert set(body["bbox"]) == {"x", "y", "w", "h"}
    for value in body["bbox"].values():
        assert 0.0 <= value <= 1.0


def test_predict_survives_a_non_square_upload(client):
    response = client.post("/predict", files={"file": ("w.png", _png((640, 480)), "image/png")})
    assert response.status_code == 200


def test_predict_rejects_a_missing_field(client):
    response = client.post("/predict", files={"image": ("a.png", _png(), "image/png")})
    assert response.status_code == 422
