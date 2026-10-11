"""Evaluation metrics for the three task heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_iou(pred_box: torch.Tensor, gt_box: torch.Tensor) -> float:
    """Mean IoU over a batch. Both boxes are ``[xc, yc, w, h]`` normalised."""
    pred_x1 = pred_box[:, 0] - pred_box[:, 2] / 2
    pred_y1 = pred_box[:, 1] - pred_box[:, 3] / 2
    pred_x2 = pred_box[:, 0] + pred_box[:, 2] / 2
    pred_y2 = pred_box[:, 1] + pred_box[:, 3] / 2

    gt_x1 = gt_box[:, 0] - gt_box[:, 2] / 2
    gt_y1 = gt_box[:, 1] - gt_box[:, 3] / 2
    gt_x2 = gt_box[:, 0] + gt_box[:, 2] / 2
    gt_y2 = gt_box[:, 1] + gt_box[:, 3] / 2

    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    return (inter_area / torch.clamp(union_area, min=1e-6)).mean().item()


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, float]:
    """Return ``(foreground IoU, depth RMSE, detection IoU)`` averaged over batches."""
    model.eval()
    total_iou_seg = 0.0
    total_rmse = 0.0
    total_iou_det = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            seg_t = batch["seg"].to(device)
            depth_t = batch["depth"].to(device)
            bbox_t = batch["ann"]["bbox"].to(device)
            out = model(imgs)

            seg_logits = F.interpolate(out["seg"], size=seg_t.shape[-2:], mode="bilinear")
            preds_seg = torch.argmax(seg_logits, dim=1)
            intersection = torch.logical_and(preds_seg, seg_t).float().sum()
            union = torch.logical_or(preds_seg, seg_t).float().sum()
            total_iou_seg += (intersection / (union + 1e-6)).item()

            depth_pred = F.interpolate(out["depth"], size=depth_t.shape[-2:], mode="bilinear")
            total_rmse += torch.sqrt(F.mse_loss(depth_pred, depth_t)).item()

            total_iou_det += compute_iou(out["det"][:, 1:5], bbox_t)
            batches += 1

    if batches == 0:
        return 0.0, 0.0, 0.0
    return total_iou_seg / batches, total_rmse / batches, total_iou_det / batches
