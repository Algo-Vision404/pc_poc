"""Training entry point for BigPerCameraNet.

Usage::

    python -m per_camera.training --samples 2048 --epochs 30 --batch 32
    pc-poc-train --samples 2048 --epochs 30 --batch 32
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import PerCameraDataset, build_transform, collate_fn, generate_synthetic_dataset
from .metrics import evaluate
from .model import ARCH_VERSION, BigPerCameraNet
from .paths import (
    CHECKPOINT_PATH,
    IMG_SIZE,
    MODEL_DIR,
    ONNX_PATH,
    RESULTS_JSON,
    ensure_artifact_dirs,
)

# Relative loss weights. Depth is upweighted because its target is spatially
# sparse (the object covers ~10-15% of pixels).
DEPTH_LOSS_WEIGHT = 30.0
BBOX_LOSS_WEIGHT = 2.0


def _configure_stdout() -> None:
    """Force UTF-8 on stdout without replacing the stream object.

    The previous implementation reassigned ``sys.stdout`` to a new
    ``TextIOWrapper``. That orphaned the original wrapper, whose finaliser
    closed the shared buffer, so every later ``print`` raised
    ``ValueError: I/O operation on closed file`` and the run exited 1.
    ``reconfigure`` mutates the existing stream in place and is safe.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # Stream does not support it (already closed / redirected).


def make_checkpoint_payload(
    model: nn.Module,
    optimizer: optim.Optimizer,
    *,
    epoch: int,
    loss: float,
    metrics: dict,
    config: dict,
) -> dict:
    """Checkpoint with enough metadata to detect a stale artifact on load."""
    return {
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "arch_version": ARCH_VERSION,
        "img_size": IMG_SIZE,
        "epoch": epoch,
        "loss": loss,
        "metrics": metrics,
        "config": config,
        # str() is required: torch.__version__ is a TorchVersion subclass, and
        # torch.load(..., weights_only=True) refuses to unpickle unknown globals.
        "torch_version": str(torch.__version__),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def load_checkpoint(
    path: Path,
    model: nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Load a checkpoint and refuse to return one built for another architecture."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train one first: "
            f"`python -m per_camera.training --samples 2048 --epochs 30 --batch 32`"
        )

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    found = checkpoint.get("arch_version")
    if found != ARCH_VERSION:
        raise RuntimeError(
            f"Checkpoint architecture mismatch at {path}.\n"
            f"  checkpoint arch_version: {found!r}\n"
            f"  code expects           : {ARCH_VERSION!r}\n"
            "This checkpoint was written by a different version of the model. "
            "Retrain with `python -m per_camera.training`."
        )

    if model is not None:
        model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def export_onnx(model: nn.Module, device: torch.device, onnx_path: Path) -> bool:
    """Best-effort ONNX export. Returns True on success; never raises."""
    _configure_stdout()
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["seg", "depth", "det"],
        )
    except Exception as exc:  # noqa: BLE001 - export is optional, must not kill a run
        print(f"[warn] ONNX export skipped ({type(exc).__name__}: {exc})", file=sys.stderr)
        print(
            "[warn] Install the export extras if you need it: pip install onnx onnxscript",
            file=sys.stderr,
        )
        return False
    print(f"ONNX model saved to {onnx_path}")
    return True


def train(
    manifest_csv: Path,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    max_samples: int | None = 64,
    device: torch.device | None = None,
) -> dict:
    """Train the network and write a checkpoint plus a results JSON."""
    epochs = int(epochs)
    batch_size = int(batch_size)
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    ensure_artifact_dirs()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PerCameraDataset(manifest_csv, transform=build_transform(), max_samples=max_samples)
    if len(dataset) < 2:
        raise ValueError(
            f"Need at least 2 samples to form a train/val split, got {len(dataset)}"
        )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BigPerCameraNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.MSELoss()
    det_cls_loss_fn = nn.BCEWithLogitsLoss()
    det_bbox_loss_fn = nn.SmoothL1Loss()

    checkpoint_path = CHECKPOINT_PATH
    best_loss = float("inf")
    best_metrics: dict = {}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            seg_t = batch["seg"].to(device)
            depth_t = batch["depth"].to(device)
            cls_t = batch["ann"]["class"].to(device).float().unsqueeze(1)
            bbox_t = batch["ann"]["bbox"].to(device)

            optimizer.zero_grad()
            out = model(imgs)

            seg_logits = nn.functional.interpolate(
                out["seg"], size=seg_t.shape[-2:], mode="bilinear"
            )
            depth_pred = nn.functional.interpolate(
                out["depth"], size=depth_t.shape[-2:], mode="bilinear"
            )

            loss = (
                seg_loss_fn(seg_logits, seg_t)
                + DEPTH_LOSS_WEIGHT * depth_loss_fn(depth_pred, depth_t)
                + det_cls_loss_fn(out["det"][:, 0:1], cls_t)
                + BBOX_LOSS_WEIGHT * det_bbox_loss_fn(out["det"][:, 1:5], bbox_t)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        val_miou, val_rmse, val_det_iou = evaluate(model, val_loader, device)
        avg_train_loss = epoch_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.4f} "
            f"- Val mIoU: {val_miou:.4f} - Val RMSE: {val_rmse:.4f} "
            f"- Val Det IoU: {val_det_iou:.4f}"
        )

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_metrics = {
                "val_mIoU": val_miou,
                "val_rmse": val_rmse,
                "val_det_iou": val_det_iou,
                "loss": avg_train_loss,
            }
            torch.save(
                make_checkpoint_payload(
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    loss=avg_train_loss,
                    metrics=best_metrics,
                    config={
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "max_samples": max_samples,
                        "manifest": str(manifest_csv),
                        "dataset_size": len(dataset),
                    },
                ),
                checkpoint_path,
            )

    results = {
        "arch_version": ARCH_VERSION,
        "checkpoint": str(checkpoint_path),
        "best_metrics": best_metrics,
        "config": {"epochs": epochs, "batch_size": batch_size, "dataset_size": len(dataset)},
        "torch_version": str(torch.__version__),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(RESULTS_JSON, "w") as fh:
        json.dump(results, fh, indent=2)

    print("Exporting model to ONNX format...")
    load_checkpoint(checkpoint_path, model=model, device=device)
    if export_onnx(model, device, ONNX_PATH):
        results["onnx_path"] = str(ONNX_PATH)
        with open(RESULTS_JSON, "w") as fh:
            json.dump(results, fh, indent=2)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pc-poc-train", description="Train BigPerCameraNet on the synthetic dataset."
    )
    parser.add_argument("--samples", type=int, default=128, help="synthetic frames to generate")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None, help="cap the dataset size")
    args = parser.parse_args(argv)

    if args.samples < 1:
        parser.error("--samples must be >= 1")

    _configure_stdout()
    ensure_artifact_dirs()

    manifest = generate_synthetic_dataset(args.samples)
    print(f"Generated {args.samples} samples -> {manifest}")
    print(f"Artifacts root: {MODEL_DIR.parent}")

    results = train(
        manifest, epochs=args.epochs, batch_size=args.batch, max_samples=args.max_samples
    )

    print("Training complete. Best metrics:")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
