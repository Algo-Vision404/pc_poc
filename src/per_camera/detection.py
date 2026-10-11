"""Anchor-free detection targets, losses, and decoding (CenterNet-style)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

#: Gaussian peak overlap used when sizing the heatmap target.
MIN_OVERLAP = 0.7
#: Loss weight for the size regression relative to the offset regression.
SIZE_LOSS_WEIGHT = 0.1


def gaussian_radius(width: float, height: float, min_overlap: float = MIN_OVERLAP) -> float:
    """Radius of the heatmap gaussian for an object of the given size."""
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(max(b1**2 - 4 * a1 * c1, 0.0))) / 2

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + np.sqrt(max(b2**2 - 4 * a2 * c2, 0.0))) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    disc = b3**2 - 4 * a3 * c3
    r3 = (b3 + np.sqrt(max(disc, 0.0))) / 2 if disc >= 0 else 0.0

    return max(float(min(r1, r2, r3)), 0.0)


def _gaussian_kernel(radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    sigma = max(radius / 3.0, 1e-3)
    g = np.exp(-(x**2) / (2 * sigma**2))
    return np.outer(g, g)


def draw_gaussian(heatmap: np.ndarray, cx: float, cy: float, radius: float) -> None:
    """Stamp a 2D gaussian centred at floor(cx), floor(cy), taking the max.

    Clipped at the heatmap edges. The peak lands exactly on the integer cell
    that ``build_targets`` records in ``reg_index``.
    """
    h, w = heatmap.shape
    x0, y0 = int(np.floor(cx)), int(np.floor(cy))
    if not (0 <= x0 < w and 0 <= y0 < h):
        return

    radius = int(radius)
    if radius <= 0:
        heatmap[y0, x0] = max(heatmap[y0, x0], 1.0)
        return

    kernel = _gaussian_kernel(radius)

    left = min(x0, radius)
    right = min(w - x0, radius + 1)
    top = min(y0, radius)
    bottom = min(h - y0, radius + 1)
    if right <= left or bottom <= top:
        return

    region = heatmap[y0 - top : y0 + bottom, x0 - left : x0 + right]
    patch = kernel[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(region, patch, out=region)


def build_targets(
    boxes: np.ndarray,
    box_mask: np.ndarray,
    out_h: int,
    out_w: int,
    num_classes: int,
    max_objects: int,
) -> dict[str, np.ndarray]:
    """Build heatmap/offset/size targets for one frame.

    ``boxes`` is ``(N, 5)`` of ``[cls, xc, yc, w, h]`` in normalised coords.
    """
    heatmap = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
    reg_mask = np.zeros((max_objects,), dtype=np.float32)
    reg_index = np.zeros((max_objects,), dtype=np.int64)
    offset_target = np.zeros((max_objects, 2), dtype=np.float32)
    size_target = np.zeros((max_objects, 2), dtype=np.float32)

    for i in range(max_objects):
        if i >= len(boxes) or box_mask[i] < 0.5:
            continue
        cls, xc, yc, w, h = boxes[i]
        cxf, cyf = float(xc) * out_w, float(yc) * out_h
        wf, hf = max(float(w) * out_w, 1.0), max(float(h) * out_h, 1.0)

        draw_gaussian(heatmap[int(cls)], cxf, cyf, gaussian_radius(wf, hf))

        xi, yi = int(np.floor(cxf)), int(np.floor(cyf))
        xi = min(max(xi, 0), out_w - 1)
        yi = min(max(yi, 0), out_h - 1)

        reg_mask[i] = 1.0
        reg_index[i] = yi * out_w + xi
        offset_target[i] = (cxf - xi, cyf - yi)
        size_target[i] = (wf, hf)

    return {
        "heatmap": heatmap,
        "reg_mask": reg_mask,
        "reg_index": reg_index,
        "offset": offset_target,
        "size": size_target,
    }


def nms_max_pool(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keep only local maxima — the standard 3x3 NMS for a heatmap."""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    return heat * (hmax == heat).float()


def decode_detections(
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    size: torch.Tensor,
    topk: int = 20,
    score_thresh: float = 0.0,
) -> torch.Tensor:
    """Decode heads into detections.

    Returns ``(B, K, 6)`` of ``[xc, yc, w, h, score, cls]`` with geometry
    normalised to ``[0, 1]`` in input-image space.
    """
    heat = nms_max_pool(torch.sigmoid(heatmap))
    b, c, h, w = heat.shape
    k = min(topk, h * w)

    scores, flat = torch.topk(heat.view(b, c * h * w), k, dim=1)
    classes = (flat // (h * w)).float()
    flat = flat % (h * w)

    xs = (flat % w).float()
    ys = (flat // w).float()

    off = offset.permute(0, 2, 3, 1).reshape(b, h * w, 2)
    off = torch.gather(off, 1, flat.unsqueeze(-1).expand(b, k, 2))
    xc = (xs + off[..., 0]) / w
    yc = (ys + off[..., 1]) / h

    sz = size.permute(0, 2, 3, 1).reshape(b, h * w, 2)
    sz = torch.gather(sz, 1, flat.unsqueeze(-1).expand(b, k, 2))
    bw = sz[..., 0] / w
    bh = sz[..., 1] / h

    out = torch.stack([xc, yc, bw, bh, scores, classes], dim=-1)
    if score_thresh > 0:
        keep = scores >= score_thresh
        out = out * keep.unsqueeze(-1)
    return out


def focal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """CenterNet's modified focal loss for the heatmap."""
    pred = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()

    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred**2 * (1 - gt) ** 4 * neg

    num_pos = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def reg_l1_loss(
    pred: torch.Tensor, mask: torch.Tensor, index: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """L1 on the regressed channels, evaluated only at ground-truth peaks."""
    pred = pred.permute(0, 2, 3, 1).reshape(pred.size(0), -1, pred.size(1))
    pred = torch.gather(pred, 1, index.unsqueeze(-1).expand(-1, -1, pred.size(2)))
    mask = mask.unsqueeze(-1).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    return loss / mask.sum().clamp(min=1.0)


def boxes_xywhn_to_xyxy(boxes: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert ``[xc, yc, w, h]`` to ``[x1, y1, x2, y2]``."""
    xc, yc, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    out = [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]
    rest = [boxes[..., i] for i in range(4, boxes.shape[-1])]
    return torch.stack(out + rest, dim=-1) if torch.is_tensor(boxes) else np.stack(out + rest, -1)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two ``[x1, y1, x2, y2]`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0
