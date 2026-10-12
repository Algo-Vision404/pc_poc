"""Training entry point for BigPerCameraNet.

Usage::

    python -m per_camera.training --samples 2048 --epochs 30 --batch 32 --seed 0
    pc-poc-train --samples 2048 --epochs 30 --batch 32 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .classes import CLASS_NAMES, NUM_CLASSES, NUM_SEG_CLASSES
from .data import PerCameraDataset, build_transform, collate_fn, generate_synthetic_dataset
from .detection import SIZE_LOSS_WEIGHT, focal_loss, reg_l1_loss
from .metrics import evaluate
from .model import ARCH_VERSION, BigPerCameraNet
from .paths import (
    CHECKPOINT_PATH,
    IMG_SIZE,
    ONNX_PATH,
    RESULTS_JSON,
    ensure_artifact_dirs,
)

# Relative loss weights. Depth is dense now (a full depth field, not a sparse
# object mask), so it no longer needs the 30x upweighting the previous sparse
# target required.
DEPTH_LOSS_WEIGHT = 5.0


def seed_everything(seed: int) -> torch.Generator:
    """Seed every RNG the pipeline touches and return a split generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def val_score(metrics: dict[str, float]) -> float:
    """Single scalar used for model selection and early stopping.

    Higher is better. Combines the two detection/segmentation quality signals
    and penalises depth error.
    """
    return metrics["mIoU"] + metrics["map50"] - metrics["rmse"]


def _configure_stdout() -> None:
    """Force UTF-8 on stdout without replacing the stream object.

    An earlier version reassigned ``sys.stdout`` to a new ``TextIOWrapper``.
    That orphaned the original wrapper, whose finaliser closed the shared
    buffer, so every later ``print`` raised ``ValueError: I/O operation on
    closed file`` and the run exited 1. ``reconfigure`` mutates in place.
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
            "`python -m per_camera.training --samples 2048 --epochs 30 --batch 32 --seed 0`"
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
            output_names=["seg", "depth", "heatmap", "offset", "size"],
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
    seed: int = 0,
    patience: int = 8,
    val_split: float = 0.2,
    device: torch.device | None = None,
) -> dict:
    """Train the network and write a checkpoint plus a results JSON."""
    epochs = int(epochs)
    batch_size = int(batch_size)
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    ensure_artifact_dirs()
    generator = seed_everything(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PerCameraDataset(manifest_csv, transform=build_transform(), max_samples=max_samples)
    if len(dataset) < 2:
        raise ValueError(
            f"Need at least 2 samples to form a train/val split, got {len(dataset)}"
        )

    train_size = int((1.0 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BigPerCameraNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.MSELoss()

    best_path = CHECKPOINT_PATH
    last_path = CHECKPOINT_PATH.with_name("poc_checkpoint_last.pt")
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            ann = batch["ann"]
            imgs = batch["image"].to(device)
            seg_t = batch["seg"].to(device)
            depth_t = batch["depth"].to(device)

            heatmap_t = ann["heatmap"].to(device)
            reg_mask = ann["reg_mask"].to(device)
            reg_index = ann["reg_index"].to(device)
            offset_t = ann["offset"].to(device)
            size_t = ann["size"].to(device)

            optimizer.zero_grad()
            out = model(imgs)

            seg_logits = nn.functional.interpolate(
                out["seg"], size=seg_t.shape[-2:], mode="bilinear", align_corners=False
            )
            depth_pred = nn.functional.interpolate(
                out["depth"], size=depth_t.shape[-2:], mode="bilinear", align_corners=False
            )

            seg_loss = seg_loss_fn(seg_logits, seg_t)
            depth_loss = depth_loss_fn(depth_pred, depth_t)
            hm_loss = focal_loss(out["heatmap"], heatmap_t)
            off_loss = reg_l1_loss(out["offset"], reg_mask, reg_index, offset_t)
            size_loss = reg_l1_loss(out["size"], reg_mask, reg_index, size_t)

            loss = (
                seg_loss
                + DEPTH_LOSS_WEIGHT * depth_loss
                + hm_loss
                + off_loss
                + SIZE_LOSS_WEIGHT * size_loss
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        metrics = evaluate(
            model,
            val_loader,
            device,
            num_seg_classes=NUM_SEG_CLASSES,
            num_det_classes=NUM_CLASSES,
        )
        avg_train_loss = epoch_loss / len(train_loader)
        score = val_score(metrics)

        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, **metrics})
        per_class = " ".join(
            f"{CLASS_NAMES[c]}={metrics[f'ap50_class{c}']:.3f}"
            for c in range(NUM_CLASSES)
            if f"ap50_class{c}" in metrics
        )
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.4f} "
            f"- Val mIoU: {metrics['mIoU']:.4f} - Val mAP50: {metrics['map50']:.4f} "
            f"({per_class}) "
            f"- Val RMSE: {metrics['rmse']:.4f} - Val AbsRel: {metrics['abs_rel']:.4f}"
        )

        # Select on validation quality, not on training loss. The previous
        # version compared training loss against a variable named
        # `best_val_loss`, so "best" was always simply the last epoch.
        torch.save(
            make_checkpoint_payload(
                model,
                optimizer,
                epoch=epoch + 1,
                loss=avg_train_loss,
                metrics=metrics,
                config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "max_samples": max_samples,
                    "seed": seed,
                    "manifest": str(manifest_csv),
                    "dataset_size": len(dataset),
                },
            ),
            last_path,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = dict(metrics)
            epochs_without_improvement = 0
            torch.save(
                make_checkpoint_payload(
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    loss=avg_train_loss,
                    metrics=metrics,
                    config={
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "max_samples": max_samples,
                        "seed": seed,
                        "manifest": str(manifest_csv),
                        "dataset_size": len(dataset),
                    },
                ),
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: no validation "
                    f"improvement for {patience} epochs (best was epoch {best_epoch})."
                )
                break

    results = {
        "arch_version": ARCH_VERSION,
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_score": best_score,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "dataset_size": len(dataset),
            "train_size": train_size,
            "val_size": val_size,
            "val_split": val_split,
            "seed": seed,
            "patience": patience,
        },
        "history": history,
        "torch_version": str(torch.__version__),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(RESULTS_JSON, "w") as fh:
        json.dump(results, fh, indent=2)

    print("Exporting model to ONNX format...")
    load_checkpoint(best_path, model=model, device=device)
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
    parser.add_argument("--seed", type=int, default=0, help="makes the whole run reproducible")
    parser.add_argument("--patience", type=int, default=8, help="early-stopping patience")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="fraction of the dataset held out for validation (default 0.2)",
    )
    args = parser.parse_args(argv)

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.patience < 1:
        parser.error("--patience must be >= 1")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1")

    _configure_stdout()
    ensure_artifact_dirs()

    manifest = generate_synthetic_dataset(args.samples, seed=args.seed)
    print(f"Generated {args.samples} samples (seed={args.seed}) -> {manifest}")

    results = train(
        manifest,
        epochs=args.epochs,
        batch_size=args.batch,
        max_samples=args.max_samples,
        seed=args.seed,
        patience=args.patience,
        val_split=args.val_split,
    )

    print(f"Training complete. Best metrics (epoch {results['best_epoch']}):")
    print(json.dumps(results["best_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
