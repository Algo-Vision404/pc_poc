"""Shared pytest configuration.

The artifact root must be redirected to a throwaway directory *before*
``per_camera.paths`` is first imported, because it resolves the location at
import time. Setting the environment variable at conftest module scope runs
before any test module is collected.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="pc_poc_tests_"))
os.environ["PC_POC_ARTIFACTS"] = str(_TMP_ROOT)

import pytest  # noqa: E402

from per_camera.data import generate_synthetic_dataset  # noqa: E402
from per_camera.paths import ARTIFACTS, ensure_artifact_dirs  # noqa: E402
from per_camera.training import train  # noqa: E402


@pytest.fixture(scope="session")
def artifacts_root() -> Path:
    return ARTIFACTS


@pytest.fixture(scope="session")
def trained_artifacts() -> Path:
    """Generate a tiny dataset and run one training epoch.

    Session-scoped because the API module loads its checkpoint at import time.
    """
    ensure_artifact_dirs()
    manifest = generate_synthetic_dataset(16)
    train(manifest, epochs=1, batch_size=4, max_samples=16)
    return ARTIFACTS


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001, ARG001
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)
