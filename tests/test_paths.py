"""Artifact path resolution must be repo-anchored, never cwd-dependent."""

from __future__ import annotations

import importlib

from per_camera import paths


def test_env_var_overrides_artifact_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ARTIFACTS_ENV_VAR, str(tmp_path))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.ARTIFACTS == tmp_path.resolve()
        assert reloaded.DATA_DIR == tmp_path.resolve() / "synthetic_data"
        assert reloaded.CHECKPOINT_PATH.parent == tmp_path.resolve() / "models"
    finally:
        importlib.reload(paths)


def test_artifacts_are_not_resolved_against_cwd(monkeypatch, tmp_path):
    """Regression: ARTIFACTS used to be Path('./artifacts'), so every entry
    point silently found nothing when run from a different directory."""
    monkeypatch.delenv(paths.ARTIFACTS_ENV_VAR, raising=False)

    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = paths._default_artifacts_root()

    assert resolved != elsewhere / "artifacts"
    assert resolved.name == "artifacts"
    # Either the repo root (when running from a checkout) or the cwd fallback —
    # never a path that depends on where the interpreter happened to start.
    assert resolved.is_absolute()


def test_ensure_artifact_dirs_creates_tree(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ARTIFACTS_ENV_VAR, str(tmp_path / "fresh"))
    reloaded = importlib.reload(paths)
    try:
        reloaded.ensure_artifact_dirs()
        for directory in (
            reloaded.ARTIFACTS,
            reloaded.DATA_DIR,
            reloaded.MODEL_DIR,
            reloaded.RESULTS_DIR,
        ):
            assert directory.is_dir(), directory
    finally:
        importlib.reload(paths)


def test_importing_paths_has_no_side_effects(monkeypatch, tmp_path):
    """Regression: importing the module used to mkdir the whole tree."""
    target = tmp_path / "untouched"
    monkeypatch.setenv(paths.ARTIFACTS_ENV_VAR, str(target))
    importlib.reload(paths)
    try:
        assert not target.exists()
    finally:
        importlib.reload(paths)


def test_manifest_path_resolver_handles_legacy_absolute_paths(tmp_path):
    """Old manifests stored another machine's absolute Windows paths."""
    (tmp_path / "synthetic_data").mkdir()
    real = tmp_path / "synthetic_data" / "img_0000.png"
    real.write_bytes(b"x")

    legacy = "C:\\Users\\someone\\project\\artifacts\\synthetic_data\\img_0000.png"
    expected = tmp_path / "synthetic_data" / "img_0000.png"
    assert paths.resolve_manifest_path(legacy, tmp_path) == expected

    relative = "synthetic_data/img_0000.png"
    assert paths.resolve_manifest_path(relative, tmp_path) == real.resolve()
