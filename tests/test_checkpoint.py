"""Checkpoint metadata and the guards that stop a stale model from serving."""

from __future__ import annotations

import pytest
import torch

from per_camera.model import ARCH_VERSION, BigPerCameraNet
from per_camera.paths import CHECKPOINT_PATH
from per_camera.training import load_checkpoint, make_checkpoint_payload


def test_trained_checkpoint_exists(trained_artifacts):
    assert CHECKPOINT_PATH.is_file()


def test_checkpoint_carries_metadata(trained_artifacts):
    """Regression: checkpoints used to hold only model_state + opt_state, so a
    stale artifact from a different architecture was undetectable."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)

    assert ckpt["arch_version"] == ARCH_VERSION
    assert ckpt["img_size"] == 128
    assert isinstance(ckpt["epoch"], int)
    assert "metrics" in ckpt and "config" in ckpt
    assert isinstance(ckpt["torch_version"], str)
    assert "created_utc" in ckpt


def test_checkpoint_round_trips(trained_artifacts):
    model = BigPerCameraNet()
    ckpt = load_checkpoint(CHECKPOINT_PATH, model=model)
    assert ckpt["arch_version"] == ARCH_VERSION

    reference = BigPerCameraNet()
    reference.load_state_dict(ckpt["model_state"])
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.equal(a, b)


def test_missing_checkpoint_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_checkpoint(tmp_path / "does_not_exist.pt")
    assert "per_camera.training" in str(exc.value)


def test_stale_architecture_is_rejected(tmp_path):
    """Regression: the repo shipped a TinyPerCameraNet checkpoint that
    BigPerCameraNet could not load; it now fails loudly instead of silently."""
    stale = tmp_path / "stale.pt"
    torch.save({"model_state": {}, "opt_state": {}}, stale)

    with pytest.raises(RuntimeError) as exc:
        load_checkpoint(stale, model=BigPerCameraNet())

    message = str(exc.value)
    assert "architecture mismatch" in message
    assert ARCH_VERSION in message


def test_forged_state_dict_is_rejected(tmp_path):
    stale = tmp_path / "wrong_shape.pt"
    torch.save(
        {"model_state": {"conv1.weight": torch.zeros(1)}, "arch_version": ARCH_VERSION},
        stale,
    )
    with pytest.raises(RuntimeError):
        load_checkpoint(stale, model=BigPerCameraNet())


def test_payload_helper_embeds_arch_version(trained_artifacts):
    model = BigPerCameraNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    payload = make_checkpoint_payload(
        model, opt, epoch=3, loss=0.5, metrics={"val_mIoU": 0.1}, config={"epochs": 3}
    )
    assert payload["arch_version"] == ARCH_VERSION
    assert payload["epoch"] == 3
    assert isinstance(payload["torch_version"], str)
