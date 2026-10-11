"""Artifact location resolution.

Paths are anchored to the repository, never to the process working directory.
Override the location with the ``PC_POC_ARTIFACTS`` environment variable.

Importing this module has no filesystem side effects; call
:func:`ensure_artifact_dirs` from an entry point when you intend to write.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Name of the environment variable that overrides the artifact root.
ARTIFACTS_ENV_VAR = "PC_POC_ARTIFACTS"

#: Directory that contains ``src/`` — two levels up from this file.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_artifacts_root() -> Path:
    """Resolve the artifact root: env var, then repo, then cwd."""
    override = os.environ.get(ARTIFACTS_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    # Only treat REPO_ROOT as the repo if it really looks like one, so an
    # installed copy of this package does not write next to site-packages.
    if (REPO_ROOT / "pyproject.toml").is_file():
        return (REPO_ROOT / "artifacts").resolve()
    return (Path.cwd() / "artifacts").resolve()


ARTIFACTS: Path = _default_artifacts_root()
DATA_DIR: Path = ARTIFACTS / "synthetic_data"
MODEL_DIR: Path = ARTIFACTS / "models"
RESULTS_DIR: Path = ARTIFACTS / "results"

CHECKPOINT_PATH: Path = MODEL_DIR / "poc_checkpoint.pt"
ONNX_PATH: Path = MODEL_DIR / "poc_model.onnx"
MANIFEST_PATH: Path = ARTIFACTS / "dataset_manifest.csv"
RESULTS_JSON: Path = RESULTS_DIR / "poc_results.json"

#: Network input edge length in pixels.
IMG_SIZE: int = 128


def ensure_artifact_dirs() -> None:
    """Create the artifact tree. Explicit, so imports stay side-effect free."""
    for directory in (ARTIFACTS, DATA_DIR, MODEL_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_manifest_path(stored: str, manifest_dir: Path) -> Path:
    """Turn a path from the manifest into an absolute path.

    The manifest stores paths relative to its own directory so that it stays
    portable across machines and operating systems. A foreign absolute path is
    still resolved by filename, which keeps manifests written elsewhere readable.
    """
    candidate = Path(stored)

    # A Windows path is NOT "absolute" on POSIX — Path("C:\\x\\y").is_absolute()
    # is False there — so a drive letter or backslash has to be detected
    # explicitly or the value gets joined onto manifest_dir verbatim.
    looks_foreign = "\\" in stored or (len(stored) > 1 and stored[1] == ":")

    if candidate.is_absolute() or looks_foreign:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        return manifest_dir / "synthetic_data" / Path(stored.replace("\\", "/")).name

    return (manifest_dir / candidate).resolve()
