"""Evaluation metrics for the three task heads.

Detection is reported as **mAP@0.5** averaged over the classes present in the
validation split, plus per-class AP. Localisation quality is reported
separately as the mean IoU of the detections that actually matched a ground
truth box.

The previous ``det_iou`` metric compared the single top detection against the
*first* GT box in the batch, so a detector that found every object correctly
still scored ~0.44 while its AP was 0.99. It was worse than no metric at all,
because it looked like a localisation failure. ``matched_iou`` is the number
that metric was trying to be.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .detection import box_iou, boxes_xywhn_to_xyxy, decode_detections

#: Detections are (N, 6): [xc, yc, w, h, score, cls].
DET_CLS_COL = 5
#: Ground truth rows are (M, 5): [cls, xc, yc, w, h] — the box is 1:5.
GT_CLS_COL = 0


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
) -> tuple[float, float, float, float]:
    """AP at a single IoU threshold, for detections of one class.

    ``detections`` / ``ground_truths`` are per-frame ``(N, 6)`` / ``(M, 5)``
    arrays in normalised coords. Matching is greedy by descending score, and a
    ground-truth box may only be matched once.

    Returns ``(ap, recall, precision, mean_matched_iou)``. The last value is
    the mean IoU of the true positives — localisation quality, independent of
    how many boxes were found. It is 0.0 when nothing matched.
    """
    dets = [(i, d) for i, ds in enumerate(detections) for d in ds]
    if not dets:
        return 0.0, 0.0, 0.0, 0.0

    dets.sort(key=lambda t: -float(t[1][4]))
    total_gt = sum(len(g) for g in ground_truths)
    if total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0

    matched: dict[int, set[int]] = {i: set() for i in range(len(ground_truths))}
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    tp_ious: list[float] = []

    for rank, (frame, det) in enumerate(dets):
        gts = ground_truths[frame]
        if len(gts) == 0:
            fp[rank] = 1.0
            continue

        det_xyxy = boxes_xywhn_to_xyxy(det[:4].astype(np.float64))
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            iou = box_iou(det_xyxy, boxes_xywhn_to_xyxy(gt[1:5].astype(np.float64)))
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j not in matched[frame]:
            tp[rank] = 1.0
            matched[frame].add(best_j)
            tp_ious.append(best_iou)
        else:
            fp[rank] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    mean_tp_iou = float(np.mean(tp_ious)) if tp_ious else 0.0
    return compute_ap(recalls, precisions), float(recalls[-1]), float(precisions[-1]), mean_tp_iou


def _by_class(arrays: list[np.ndarray], col: int, cls: int) -> list[np.ndarray]:
    return [a[a[:, col] == cls] if len(a) else a for a in arrays]


def mean_average_precision(
    detections: list[np.ndarray],
    ground_truths: list[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Class-aware mAP plus per-class AP, recall, precision, and matched IoU.

    A class absent from the ground truth is skipped rather than scored 0, so a
    small validation split cannot tank mAP for a class that never appeared.
    """
    aps: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    ious: list[float] = []
    per_class: dict[str, float] = {}

    for c in range(num_classes):
        d_c = _by_class(detections, DET_CLS_COL, c)
        g_c = _by_class(ground_truths, GT_CLS_COL, c)
        has_gt = sum(len(g) for g in g_c) > 0
        has_det = sum(len(d) for d in d_c) > 0

        if not has_gt and not has_det:
            continue  # class not present at all — skip it

        ap, rec, prec, miou = average_precision(d_c, g_c, iou_threshold)
        per_class[f"ap50_class{c}"] = ap
        if has_gt:
            aps.append(ap)
            recalls.append(rec)
            precisions.append(prec)
        if miou > 0.0:
            ious.append(miou)

    return {
        "map50": float(np.mean(aps)) if aps else 0.0,
        "recall50": float(np.mean(recalls)) if recalls else 0.0,
        "precision50": float(np.mean(precisions)) if precisions else 0.0,
        "matched_iou": float(np.mean(ious)) if ious else 0.0,
        **per_class,
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_seg_classes: int = 2,
    num_det_classes: int = 1,
    score_thresh: float = 0.2,
    topk: int = 10,
) -> dict[str, float]:
    """Run the model over ``loader`` and return every tracked metric."""
    model.eval()
    iou_sum = 0.0
    rmse_sum = 0.0
    abs_rel_sum = 0.0
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
            iou_sum += mean_iou(torch.argmax(seg_logits, dim=1), seg_t, num_seg_classes)

            depth_pred = F.interpolate(
                out["depth"], size=depth_t.shape[-2:], mode="bilinear", align_corners=False
            )
            rmse_sum += torch.sqrt(F.mse_loss(depth_pred, depth_t)).item()
            abs_rel_sum += (
                (depth_pred - depth_t).abs() / depth_t.clamp(min=1e-3)
            ).mean().item()

            dets = decode_detections(out["heatmap"], out["offset"], out["size"], topk=topk)
            scores = dets[..., 4]
            for b in range(dets.shape[0]):
                keep = scores[b] >= score_thresh
                all_dets.append(dets[b][keep].cpu().numpy())
                m = box_mask[b] > 0.5
                all_gts.append(boxes_t[b][m].cpu().numpy())

            batches += 1

    if batches == 0:
        return {"mIoU": 0.0, "rmse": 0.0, "abs_rel": 0.0, "map50": 0.0}

    det_metrics = mean_average_precision(all_dets, all_gts, num_det_classes)
    return {
        "mIoU": iou_sum / batches,
        "rmse": rmse_sum / batches,
        "abs_rel": abs_rel_sum / batches,
        **det_metrics,
    }
