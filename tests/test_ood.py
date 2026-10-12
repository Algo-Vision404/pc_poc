"""Tests for the out-of-distribution guardrail.

The behaviour under test is that the system can recognise input it was never
trained on and refuse to emit detections, rather than hallucinating
confidently. Both halves matter: it must catch genuine outsiders, and it must
not cry wolf on legitimate frames.
"""

import json

import numpy as np
import pytest

from per_camera.ood import (
    FEATURE_NAMES,
    MAX_FEATURE_Z,
    OODDetector,
    extract_features,
)


def flat_frame(seed: int = 0) -> np.ndarray:
    """A stand-in for the synthetic training data.

    Dark, flat, low texture: one solid bright rectangle on a near-black
    background. This is the appearance distribution the detector is fitted on.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:] = rng.integers(8, 24, size=3, dtype=np.uint8)
    x, y = rng.integers(10, 40, size=2)
    img[y : y + 14, x : x + 14] = rng.integers(150, 230, size=3, dtype=np.uint8)
    return img


def noisy_frame(seed: int = 100) -> np.ndarray:
    """A stand-in for a real photograph: bright, high contrast, textured everywhere."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def detector() -> OODDetector:
    return OODDetector.fit(np.stack([extract_features(flat_frame(s)) for s in range(40)]))


def test_extract_features_has_the_expected_shape_and_names():
    f = extract_features(flat_frame())
    assert f.shape == (len(FEATURE_NAMES),)
    assert f.dtype == np.float64
    assert len(FEATURE_NAMES) == 8


def test_extract_features_is_deterministic():
    img = flat_frame(7)
    assert np.array_equal(extract_features(img), extract_features(img.copy()))


def test_extract_features_rejects_non_rgb():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        extract_features(np.zeros((64, 64), dtype=np.uint8))


def test_features_scale_independently_of_input_resolution():
    """Block averaging means a 64px and a 512px version of the same picture
    must produce comparable statistics, or resolution alone would trip the
    detector in production."""
    small = flat_frame(3)
    large = np.repeat(np.repeat(small, 8, axis=0), 8, axis=1)
    a, b = extract_features(small), extract_features(large)
    # Edge density is the most resolution-sensitive statistic; it must still
    # land in the same order of magnitude.
    assert a[4] == pytest.approx(b[4], rel=0.5)
    assert a[0] == pytest.approx(b[0], rel=0.1)


def test_fit_rejects_too_few_samples():
    feats = np.stack([extract_features(flat_frame(s)) for s in range(4)])
    with pytest.raises(ValueError, match="at least 8 samples"):
        OODDetector.fit(feats)


def test_fit_rejects_wrong_feature_count():
    with pytest.raises(ValueError, match=r"\(N, 8\)"):
        OODDetector.fit(np.zeros((20, 3)))


def test_fit_threshold_is_never_below_the_floor():
    """A constant training set would otherwise give a threshold of 0 and
    reject every input, including legitimate ones."""
    feats = np.stack([extract_features(flat_frame(0)) for _ in range(20)])
    det = OODDetector.fit(feats, min_threshold=8.0)
    assert det.threshold >= 8.0


def test_in_distribution_input_is_accepted(detector):
    for seed in range(200, 230):
        assert not detector.is_ood(flat_frame(seed)), f"seed {seed} falsely rejected"


def test_out_of_distribution_input_is_caught(detector):
    """The core guarantee: textured, well-lit input is not what the model saw."""
    for seed in range(300, 310):
        assert detector.is_ood(noisy_frame(seed)), f"seed {seed} wrongly accepted"


def test_ood_margin_is_wide(detector):
    """Separation must be a gap, not a knife edge. Measured on real
    photographs the margin is roughly 5x; on these stand-ins it should still
    be decisive."""
    inside = [detector.score(flat_frame(s)) for s in range(100, 120)]
    outside = [detector.score(noisy_frame(s)) for s in range(400, 420)]
    assert max(inside) * 2 < min(outside)


def test_worst_feature_names_a_real_feature(detector):
    name = detector.worst_feature(noisy_frame(1))
    assert name in FEATURE_NAMES


def test_save_load_round_trip(detector, tmp_path):
    path = detector.save(tmp_path / "ood.json")
    assert path.exists()
    rt = OODDetector.load(path)
    assert np.array_equal(rt.median, detector.median)
    assert np.array_equal(rt.scale, detector.scale)
    assert rt.threshold == detector.threshold
    assert rt.n_fit == detector.n_fit


def test_from_dict_rejects_a_mismatched_feature_set(detector):
    """If the feature vector ever changes, stale statistics must fail loudly
    rather than silently comparing against the wrong columns."""
    payload = detector.to_dict()
    payload["features"] = ["something_else"] + list(FEATURE_NAMES[1:])
    with pytest.raises(ValueError, match="different feature set"):
        OODDetector.from_dict(payload)


def test_score_is_clipped_rather_than_infinite(detector):
    """A feature that is constant across the training set has an MAD of zero;
    without flooring the scale, any deviation divides into infinity."""
    assert np.isfinite(detector.score(noisy_frame(1)))
    assert detector.score(noisy_frame(1)) <= MAX_FEATURE_Z


def test_score_features_validates_shape(detector):
    with pytest.raises(ValueError, match="8 features"):
        detector.score_features(np.zeros(3))


def test_to_dict_is_json_serialisable(detector):
    json.dumps(detector.to_dict())
