"""Artifact path resolution must be repo-anchored, never cwd-dependent."""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager

import pytest

from per_camera import paths


@contextmanager
def artifacts_at(value: str | None):
    """Point ``per_camera.paths`` at ``value`` (or no override) and reload.

    Restores the environment and reloads again on exit, so a test cannot leak
    a rebinded module state into the next one.
    """
    original = os.environ.get(paths.ARTIFACTS_ENV_VAR)
    if value is None:
        os.environ.pop(paths.ARTIFACTS_ENV_VAR, None)
    else:
        os.environ[paths.ARTIFACTS_ENV_VAR] = value
    try:
        importlib.reload(paths)
        yield paths
    finally:
        if original is None:
            os.environ.pop(paths.ARTIFACTS_ENV_VAR, None)
        else:
            os.environ[paths.ARTIFACTS_ENV_VAR] = original
        importlib.reload(paths)


def test_env_var_overrides_artifact_root(tmp_path):
    with artifacts_at(str(tmp_path)) as reloaded:
        assert reloaded.ARTIFACTS == tmp_path.resolve()
        assert reloaded.DATA_DIR == tmp_path.resolve() / "synthetic_data"
        assert reloaded.CHECKPOINT_PATH.parent == tmp_path.resolve() / "models"


def test_artifacts_are_not_resolved_against_cwd(monkeypatch, tmp_path):
    """Regression: ARTIFACTS used to be Path('./artifacts'), so every entry
    point silently found nothing when run from a different directory."""
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with artifacts_at(None):
        resolved = paths._default_artifacts_root()

    assert resolved != elsewhere / "artifacts"
    assert resolved.name == "artifacts"
    # Either the repo root (when running from a checkout) or the cwd fallback —
    # never a path that depends on where the interpreter happened to start.
    assert resolved.is_absolute()


def test_ensure_artifact_dirs_creates_tree(tmp_path):
    with artifacts_at(str(tmp_path / "fresh")) as reloaded:
        reloaded.ensure_artifact_dirs()
        for directory in (
            reloaded.ARTIFACTS,
            reloaded.DATA_DIR,
            reloaded.MODEL_DIR,
            reloaded.RESULTS_DIR,
        ):
            assert directory.is_dir(), directory


def test_importing_paths_has_no_side_effects(tmp_path):
    """Regression: importing the module used to mkdir the whole tree."""
    target = tmp_path / "untouched"
    with artifacts_at(str(target)):
        assert not target.exists()


def test_module_state_is_restored_after_reload(tmp_path):
    """The context manager must leave the session-wide artifact root intact."""
    before = paths.ARTIFACTS
    with artifacts_at(str(tmp_path)):
        assert paths.ARTIFACTS != before
    assert paths.ARTIFACTS == before


def test_manifest_path_resolver_handles_legacy_absolute_paths(tmp_path):
    """Old manifests stored another machine's absolute Windows paths.

    A Windows path is not "absolute" on POSIX, so this needs an explicit
    drive-letter/backslash check or it gets joined verbatim.
    """
    (tmp_path / "synthetic_data").mkdir()
    real = tmp_path / "synthetic_data" / "img_0000.png"
    real.write_bytes(b"x")

    legacy = "C:\\Users\\someone\\project\\artifacts\\synthetic_data\\img_0000.png"
    expected = tmp_path / "synthetic_data" / "img_0000.png"
    assert paths.resolve_manifest_path(legacy, tmp_path) == expected

    relative = "synthetic_data/img_0000.png"
    assert paths.resolve_manifest_path(relative, tmp_path) == real.resolve()


def test_resolve_manifest_path_accepts_a_real_absolute_path(tmp_path):
    real = tmp_path / "elsewhere.png"
    real.write_bytes(b"x")
    assert paths.resolve_manifest_path(str(real), tmp_path) == real


@pytest.mark.parametrize("stored", ["synthetic_data/a.png", "C:\\x\\a.png", "/tmp/a.png"])
def test_resolver_never_returns_a_path_containing_a_backslash(stored, tmp_path):
    resolved = paths.resolve_manifest_path(stored, tmp_path)
    assert "\\" not in str(resolved)
