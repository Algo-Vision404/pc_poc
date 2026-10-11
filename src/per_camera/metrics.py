"""Evaluation metrics for the three task heads."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .detection import box_iou, boxes_xywhn_to_xyxy, decode_detections


def seg_iou_per_class(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> list[float | None]:
    """Per-class IoU over a whole batch.

    Returns ``None`` for a class that is absent from both prediction and
    ground truth, so callers can skip it instead of scoring a perfect
    background-only frame as 0.0 (the 0/0 hole the old metric had).
    """
    out: list[float | None] = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        out.append(None if union == 0 else inter / union)
    return out


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """Mean over the classes actually present in the batch."""
    present = [v for v in seg_iou_per_class(pred, target, num_classes) if v is not None]
    return float(np.mean(present)) if present else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Average precision using the all-points (VOC 2010+) interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(
    detections: list[np.ndarray],
    ground_truths: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """AP at a single IoU threshold.

    ``detections`` / ``ground_truths`` are per-frame ``(N, 6)`` / ``(M, 5)``
    arrays of ``[xc, yc, w, h, score(, cls)]`` in normalised coords.
    """
    dets = [(i, d) for i, ds in enumerate(detections) for d in ds]
    if not dets:
        return 0.0, 0.0, 0.0

    dets.sort(key=lambda t: -float(t[1][4]))
    total_gt = sum(len(g) for g in ground_truths)
    if total_gt == 0:
        return 0.0, 0.0, 0.0

    matched: dict[int, set[int]] = {i: set() for i in range(len(ground_truths))}
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))

    for rank, (frame, det) in enumerate(dets):
        gts = ground_truths[frame]
        if len(gts) == 0:
            fp[rank] = 1.0
            continue

        det_xyxy = boxes_xywhn_to_xyxy(det[:4].astype(np.float64))
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            # Ground truth rows are [cls, xc, yc, w, h] — the box is 1:5.
            iou = box_iou(det_xyxy, boxes_xywhn_to_xyxy(gt[1:5].astype(np.float64)))
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j not in matched[frame]:
            tp[rank] = 1.0
            matched[frame].add(best_j)
        else:
            fp[rank] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    return compute_ap(recalls, precisions), float(recalls[-1]), float(precisions[-1])


def detection_iou(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean IoU of the top-scoring detection against the first GT box.

    Kept as a coarse localisation sanity check; ``average_precision`` is the
    metric that matters.
    """
    n = int(mask.sum().item())
    if n == 0:
        return 0.0
    best = pred[:, 0, :4]
    gt = target[:, 0, 1:5]
    p = boxes_xywhn_to_xyxy(best.detach().cpu().numpy().astype(np.float64))
    g = boxes_xywhn_to_xyxy(gt.detach().cpu().numpy().astype(np.float64))
    return float(np.mean([box_iou(a, b) for a, b in zip(p, g, strict=True)]))


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int = 2,
    score_thresh: float = 0.2,
    topk: int = 10,
) -> dict[str, float]:
    """Run the model over ``loader`` and return every tracked metric."""
    model.eval()
    iou_sum = 0.0
    rmse_sum = 0.0
    abs_rel_sum = 0.0
    diou_sum = 0.0
    all_dets: list[np.ndarray] = []
    all_gts: list[np.ndarray] = []
    batches = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            seg_t = batch["seg"].to(device)
            depth_t = batch["depth"].to(device)
            boxes_t = batch["ann"]["boxes"].to(device)
            box_mask = batch["ann"]["box_mask"].to(device)
            out = model(imgs)

            seg_logits = F.interpolate(
                out["seg"], size=seg_t.shape[-2:], mode="bilinear", align_corners=False
            )
            iou_sum += mean_iou(torch.argmax(seg_logits, dim=1), seg_t, num_classes)

            depth_pred = F.interpolate(
                out["depth"], size=depth_t.shape[-2:], mode="bilinear", align_corners=False
            )
            rmse_sum += torch.sqrt(F.mse_loss(depth_pred, depth_t)).item()
            abs_rel_sum += (
                (depth_pred - depth_t).abs() / depth_t.clamp(min=1e-3)
            ).mean().item()

            dets = decode_detections(out["heatmap"], out["offset"], out["size"], topk=topk)
            diou_sum += detection_iou(dets, boxes_t, box_mask)

            scores = dets[..., 4]
            for b in range(dets.shape[0]):
                keep = scores[b] >= score_thresh
                all_dets.append(dets[b][keep].cpu().numpy())
                m = box_mask[b] > 0.5
                all_gts.append(boxes_t[b][m].cpu().numpy())

            batches += 1

    if batches == 0:
        return {"mIoU": 0.0, "rmse": 0.0, "abs_rel": 0.0, "det_iou": 0.0, "ap50": 0.0}

    ap50, recall, precision = average_precision(all_dets, all_gts)
    return {
        "mIoU": iou_sum / batches,
        "rmse": rmse_sum / batches,
        "abs_rel": abs_rel_sum / batches,
        "det_iou": diou_sum / batches,
        "ap50": ap50,
        "recall50": recall,
        "precision50": precision,
    }
