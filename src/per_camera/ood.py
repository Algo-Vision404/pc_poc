"""Out-of-distribution detection.

Why this module exists
----------------------
A detector trained on synthetic frames does not merely perform poorly on real
photographs - it performs *confidently* poorly. Measured on eight ordinary
photographs (warehouse interiors, street scenes) the network emitted 15-16
boxes per image at scores of 0.22-0.56, overwhelmingly labelled "triangle",
none of them real. A system that hallucinates with conviction is worse than
one that is merely inaccurate, because nothing downstream can tell the
difference.

This module gives the system the ability to say "this is not what I was
trained on" and refuse. It does not improve perception; it bounds the damage
from bad perception, which is what a deployable system actually requires.

Method
------
Rather than reaching for the network's internals, it measures cheap image
statistics that describe the *appearance distribution* of the training data.
On this dataset the separation is not subtle - mean edge density is
0.0002-0.0025 for synthetic frames against 0.0348-0.0606 for real
photographs, a 14x gap with no overlap whatsoever. Real scenes are textured
and well lit; the synthetic generator produces flat shapes on a dark
background.

Statistics are summarised robustly (median and median absolute deviation
rather than mean and standard deviation) because a handful of atypical
training frames should not widen the acceptance region enough to admit real
photographs.

Honest limitation
-----------------
The gap this exploits is partly an artefact of how artificial the training
data is. Improving the generator with domain randomisation - real textures,
varied lighting, sensor noise - will narrow it, and this detector must be
re-fitted when that happens. It is a guardrail for the current data
distribution, not a general-purpose novelty detector, and it will not catch
an adversarial input engineered to mimic the training statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .paths import IMG_SIZE

#: Feature names, in the order produced by :func:`extract_features`.
FEATURE_NAMES: tuple[str, ...] = (
    "mean_r",
    "mean_g",
    "mean_b",
    "std",
    "edge_density",
    "texture",
    "dynamic_range",
    "clipped_highlight_fraction",
)

#: Floor on the robust scale. Without it a feature that is constant across the
#: training set (for example the clipped-highlight fraction, which is exactly
#: zero for every synthetic frame) has an MAD of 0 and any non-zero test value
#: divides into infinity. Flooring keeps the score finite and comparable.
MIN_SCALE = 1e-4

#: Cap on a single feature's contribution, so one degenerate feature cannot
#: dominate the aggregate score.
MAX_FEATURE_Z = 1e3


def extract_features(image: np.ndarray) -> np.ndarray:
    """Reduce an RGB image to a small vector of appearance statistics.

    Accepts ``uint8`` or float arrays of shape ``(H, W, 3)``. The image is
    resized to the network input size so statistics are computed at a
    consistent scale regardless of the camera resolution.

    Returns a 1-D float64 array of length ``len(FEATURE_NAMES)``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB array, got {image.shape}")

    img = _resize(image)
    a = img.astype(np.float64)
    if a.max() > 1.5:  # uint8-scale input
        a /= 255.0

    gray = a.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))

    return np.array(
        [
            a[..., 0].mean(),
            a[..., 1].mean(),
            a[..., 2].mean(),
            a.std(),
            (gx.mean() + gy.mean()) / 2.0,
            (gx.std() + gy.std()) / 2.0,
            a.max() - a.min(),
            (gray > 0.98).mean(),
        ],
        dtype=np.float64,
    )


def _resize(image: np.ndarray) -> np.ndarray:
    """Resize to ``(IMG_SIZE, IMG_SIZE)`` without depending on OpenCV.

    Block averaging is enough for distributional statistics and keeps this
    module importable in environments that only have numpy.
    """
    h, w = image.shape[:2]
    if (h, w) == (IMG_SIZE, IMG_SIZE):
        return image

    # Crop-pad to square so aspect ratio does not bias the statistics, then
    # block-average down to the target size.
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    sq = image[y0 : y0 + side, x0 : x0 + side]

    # Trim to a multiple of IMG_SIZE so the reshape is exact.
    step = side // IMG_SIZE
    if step < 1:
        # Upsampling path: nearest-neighbour repeat. Only reached for inputs
        # smaller than the network resolution, which the API rejects earlier.
        idx = (np.arange(IMG_SIZE) * side // IMG_SIZE)
        return sq[np.ix_(idx, idx)]
    usable = step * IMG_SIZE
    sq = sq[:usable, :usable]
    return sq.reshape(IMG_SIZE, step, IMG_SIZE, step, 3).mean(axis=(1, 3))


def _robust_scale(values: np.ndarray) -> np.ndarray:
    """Median absolute deviation scaled to be comparable to a standard deviation."""
    med = np.median(values, axis=0)
    mad = np.median(np.abs(values - med), axis=0)
    return np.maximum(mad * 1.4826, MIN_SCALE)


class OODDetector:
    """Decides whether an input looks like the data the model was trained on.

    Fit on features drawn from the training distribution, then score new
    inputs. The score is the largest robust z-score across features: how far
    the most surprising single statistic sits from the training set. Higher is
    more out-of-distribution.
    """

    def __init__(
        self,
        median: np.ndarray,
        scale: np.ndarray,
        threshold: float,
        n_fit: int,
    ) -> None:
        self.median = np.asarray(median, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.threshold = float(threshold)
        self.n_fit = int(n_fit)

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        quantile: float = 0.999,
        min_threshold: float = 8.0,
    ) -> OODDetector:
        """Fit on an ``(N, len(FEATURE_NAMES))`` array of training features.

        The threshold is the given quantile of the training set's own scores,
        so a fixed fraction of legitimate frames are expected to be flagged.
        ``min_threshold`` stops the threshold collapsing towards zero when the
        training set is small or unusually uniform, which would make the
        detector reject almost everything.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected an (N, {len(FEATURE_NAMES)}) array, got {features.shape}"
            )
        if features.shape[0] < 8:
            raise ValueError(
                f"need at least 8 samples to fit robustly, got {features.shape[0]}"
            )

        median = np.median(features, axis=0)
        scale = _robust_scale(features)
        scores = np.abs(features - median) / scale
        per_sample = scores.max(axis=1)
        threshold = max(float(np.quantile(per_sample, quantile)), min_threshold)
        return cls(median=median, scale=scale, threshold=threshold, n_fit=len(features))

    def score(self, image: np.ndarray) -> float:
        """Largest robust z-score across features. Higher means more unusual."""
        return self.score_features(extract_features(image))

    def score_features(self, features: np.ndarray) -> float:
        f = np.asarray(features, dtype=np.float64)
        if f.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {f.shape}")
        z = np.abs(f - self.median) / self.scale
        return float(np.clip(z, 0.0, MAX_FEATURE_Z).max())

    def is_ood(self, image: np.ndarray) -> bool:
        """True when the input should not be trusted."""
        return self.score(image) > self.threshold

    def worst_feature(self, image: np.ndarray) -> str:
        """Name of the feature driving the score - useful for diagnostics."""
        f = extract_features(image)
        z = np.abs(f - self.median) / self.scale
        return FEATURE_NAMES[int(np.argmax(z))]

    def to_dict(self) -> dict:
        return {
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "threshold": self.threshold,
            "n_fit": self.n_fit,
            "features": list(FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> OODDetector:
        if payload.get("features") != list(FEATURE_NAMES):
            raise ValueError(
                "fitted OOD statistics use a different feature set than this "
                f"build: {payload.get('features')}"
            )
        return cls(
            median=np.asarray(payload["median"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            threshold=float(payload["threshold"]),
            n_fit=int(payload["n_fit"]),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> OODDetector:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
