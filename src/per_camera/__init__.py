"""Per-Camera PoC — multi-task perception (segmentation, depth, detection)."""

from __future__ import annotations

from .model import ARCH_VERSION, BigPerCameraNet
from .paths import ARTIFACTS, IMG_SIZE

__all__ = ["ARCH_VERSION", "ARTIFACTS", "BigPerCameraNet", "IMG_SIZE"]
__version__ = "0.2.0"
