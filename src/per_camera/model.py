"""BigPerCameraNet — ResNet-style backbone with three task heads.

Detection is anchor-free and CenterNet-style: a per-pixel class heatmap plus
sub-pixel offset and object-size regression at stride 4. That supports 0..N
objects per frame and yields a real confidence score, unlike the previous
single global box which could only ever regress one rectangle.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Bumped whenever the module layout or shapes change. Stored in every
#: checkpoint and asserted on load so a stale checkpoint fails loudly at
#: startup instead of returning plausible-looking garbage at inference time.
#:
#: v2: detection head replaced (single box -> heatmap + offset + size).
ARCH_VERSION = "big-per-camera-net-v2"

#: Detection head output stride relative to the input.
DET_STRIDE = 4

#: Prior bias for the heatmap so it starts near-zero under a sigmoid.
HEATMAP_BIAS_INIT = -2.19


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an optional 1x1 downsample on the identity."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class BigPerCameraNet(nn.Module):
    """Multi-task network producing segmentation, depth, and detections."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.arch_version = ARCH_VERSION
        self.num_classes = num_classes
        self.det_stride = DET_STRIDE
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)

        self.seg = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 2, 1),
        )

        self.depth = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # Shared detection trunk, upsampled from stride 8 to stride 4 so the
        # heatmap has enough resolution for small objects.
        self.det_trunk = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.det_heatmap = nn.Conv2d(32, num_classes, 1)
        self.det_offset = nn.Conv2d(32, 2, 1)
        self.det_size = nn.Conv2d(32, 2, 1)

        nn.init.constant_(self.det_heatmap.bias, HEATMAP_BIAS_INIT)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer2(self.layer1(x))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        det = self.det_trunk(features)
        return {
            "seg": self.seg(features),
            "depth": self.depth(features),
            "heatmap": self.det_heatmap(det),
            "offset": self.det_offset(det),
            "size": self.det_size(det),
        }
