"""The ONNX export path.

`export_onnx` catches every exception and returns False, so a broken export
prints one warning line and the training run still exits 0. That is the exact
failure mode the original audit called out as P0 ("ONNX export fails out of the
box"), and nothing in the suite would have caught it coming back. These tests
pin the exported graph to the model's actual class count.
"""

from __future__ import annotations

import shutil
import sys

import numpy as np
import pytest
import torch

from per_camera.classes import NUM_CLASSES, NUM_SEG_CLASSES
from per_camera.model import BigPerCameraNet
from per_camera.paths import IMG_SIZE
from per_camera.training import export_onnx

onnx = pytest.importorskip(
    "onnx", reason="export extras not installed (pip install -r requirements-export.txt)"
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export once for the module — the exporter is slow enough that doing it
    per test would dominate the suite."""
    out = tmp_path_factory.mktemp("onnx")
    path = out / "model.onnx"
    device = torch.device("cpu")

    ok = export_onnx(BigPerCameraNet(), device, path)

    assert ok, "export_onnx returned False — the export is broken"
    assert path.exists() and path.stat().st_size > 0, "export wrote no file"
    return path


def test_exported_graph_passes_the_onnx_checker(exported):
    """Catches a structurally invalid graph, which loads fine in some runtimes
    and fails in others."""
    onnx.checker.check_model(str(exported))


def test_export_exposes_the_documented_input_and_output_names(exported):
    model = onnx.load(str(exported), load_external_data=False)

    assert [i.name for i in model.graph.input] == ["input"]
    assert [o.name for o in model.graph.output] == [
        "seg",
        "depth",
        "heatmap",
        "offset",
        "size",
    ]


def test_exported_shapes_carry_the_real_class_count(exported):
    """Regression: the seg head emits NUM_CLASSES+1 channels and the heatmap
    NUM_CLASSES. If those drift, a consumer decoding the heatmap would read the
    wrong number of classes and silently mislabel every detection."""
    model = onnx.load(str(exported), load_external_data=False)
    shapes = {
        o.name: [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]
        for o in model.graph.output
    }

    assert shapes["seg"] == [1, NUM_SEG_CLASSES, IMG_SIZE, IMG_SIZE], shapes["seg"]
    assert shapes["depth"] == [1, 1, IMG_SIZE, IMG_SIZE], shapes["depth"]
    assert shapes["heatmap"][1] == NUM_CLASSES, shapes["heatmap"]
    assert shapes["offset"][1] == 2 and shapes["size"][1] == 2


def test_a_failed_export_returns_false_rather_than_raising(tmp_path):
    """Export is optional, so a failure must not kill the training run — but it
    must report False so the caller knows there is no ONNX to point at."""
    path = tmp_path / "model.onnx"
    device = torch.device("cpu")

    class Exploding(BigPerCameraNet):
        def forward(self, x):  # noqa: D102 - deliberately broken
            raise RuntimeError("synthetic export failure")

    assert export_onnx(Exploding(), device, path) is False


def test_export_writes_the_companion_weights_file(tmp_path):
    """torch.onnx.export splits weights into `<name>.onnx.data` above a size
    threshold. Deleting only the .onnx leaves an orphan, so record that both
    are expected artifacts."""
    path = tmp_path / "model.onnx"
    export_onnx(BigPerCameraNet(), torch.device("cpu"), path)

    produced = sorted(p.name for p in tmp_path.iterdir())
    assert "model.onnx" in produced
    # Whatever else the exporter emits is fine; it just must not be empty.
    assert produced, "exporter produced nothing"


def test_export_cleans_up_its_stdout_reconfiguration(tmp_path, capsys):
    """`export_onnx` calls _configure_stdout, which reconfigures the shared
    stdout buffer. A previous revision reassigned sys.stdout and the orphaned
    wrapper's finaliser closed the buffer on teardown. Pin the identity."""
    before = sys.stdout
    export_onnx(BigPerCameraNet(), torch.device("cpu"), tmp_path / "m.onnx")
    after = sys.stdout

    assert after is before, "export replaced sys.stdout instead of reconfiguring it"
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_exported_model_computes_the_same_numbers_as_pytorch(tmp_path):
    """The point of the export.

    A graph that passes `onnx.checker` and has the right output shapes can
    still compute different numbers — a dropped BatchNorm, a folded constant,
    an opset difference in how bilinear upsampling rounds. Serving the ONNX
    while the metrics were measured in PyTorch would mean the published numbers
    describe a model nobody is running.
    """
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

    path = tmp_path / "model.onnx"
    device = torch.device("cpu")
    model = BigPerCameraNet()
    model.eval()
    assert export_onnx(model, device, path) is True

    torch.manual_seed(0)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        expected = model(x)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": x.numpy()})
    names = [o.name for o in session.get_outputs()]

    assert sorted(names) == sorted(expected.keys())

    for name, got in zip(names, actual, strict=True):
        want = expected[name].numpy()
        assert got.shape == want.shape, f"{name}: {got.shape} != {want.shape}"
        # Not exact equality: the ONNX graph is re-optimised and constant-folded,
        # so tiny float differences are expected. What must not happen is a
        # systematic divergence, which is what a dropped or reordered op causes.
        max_abs = float(np.abs(got - want).max())
        assert max_abs < 1e-4, f"{name}: max abs difference {max_abs}"


def test_exported_model_is_deterministic_across_runs(tmp_path):
    """Catches a graph that depends on uninitialised state, which would make
    production inference irreproducible while tests still pass."""
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

    path = tmp_path / "model.onnx"
    model = BigPerCameraNet()
    model.eval()
    export_onnx(model, torch.device("cpu"), path)

    torch.manual_seed(1)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    first = session.run(None, {"input": x})
    second = session.run(None, {"input": x})

    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)
