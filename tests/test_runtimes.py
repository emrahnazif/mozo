"""Tests for runtime selection.

What these protect is a judgement call: ``auto`` prefers torch on every device because that is
what measurement showed, and a future edit that promotes ONNX on a hunch should have to change
a test that says so.
"""

from __future__ import annotations

import pytest

from mozo.runtimes import RuntimeError_, providers_for, runnable, select_runtime

BOTH = ["onnx-fp32", "torch-fp32"]


class TestSelectRuntime:
    @pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "mps"])
    def test_auto_prefers_torch_on_every_device(self, device):
        assert select_runtime(device, BOTH) == "torch-fp32"

    def test_auto_takes_what_exists_when_torch_is_absent(self):
        assert select_runtime("cpu", ["onnx-fp32"]) == "onnx-fp32"

    def test_auto_falls_through_to_anything_published(self):
        assert select_runtime("cpu", ["coreml-fp16"]) == "coreml-fp16"

    def test_a_split_runtime_is_named_once_and_chosen_as_a_whole(self):
        """SAM 2 publishes an encoder and a decoder that compose one runtime. A caller choosing
        a runtime must see ``onnx-fp32``, never a half it would have to know to rejoin."""
        split = ["onnx-fp32-decoder", "onnx-fp32-encoder", "torch-fp32"]
        assert runnable(split) == ["onnx-fp32", "torch-fp32"]
        assert select_runtime("cpu", split) == "torch-fp32"
        assert select_runtime("cpu", split, requested="onnx-fp32") == "onnx-fp32"

    def test_asking_for_half_a_runtime_by_name_is_refused(self):
        """Otherwise a typo selects an encoder and fails much later, at load time."""
        with pytest.raises(RuntimeError_, match="not published"):
            select_runtime("cpu", ["onnx-fp32-encoder", "onnx-fp32-decoder"],
                           requested="onnx-fp32-encoder")

    def test_a_missing_library_drops_a_split_runtime_whole(self, monkeypatch):
        """Not half of it, or ``auto`` offers an encoder this machine cannot execute."""
        from mozo import runtimes

        # Pointing the requirement at a module that does not exist is what makes ``find_spec``
        # return None; patching ``find_spec`` itself would not, because ``executable`` imports
        # it inside the function.
        monkeypatch.setitem(runtimes._REQUIRES, "onnx", "definitely_not_installed")
        assert "onnx-fp32" not in runtimes.executable(
            ["onnx-fp32-encoder", "onnx-fp32-decoder", "torch-fp32"])

    def test_explicit_request_is_honoured(self):
        assert select_runtime("cpu", BOTH, requested="onnx-fp32") == "onnx-fp32"

    def test_explicit_request_for_something_unpublished_fails_loudly(self):
        with pytest.raises(RuntimeError_, match="onnx-fp16"):
            select_runtime("cpu", BOTH, requested="onnx-fp16")

    def test_nothing_published_is_an_error(self):
        with pytest.raises(RuntimeError_, match="nothing runnable"):
            select_runtime("cpu", [])

    def test_auto_skips_a_runtime_whose_library_is_missing(self, monkeypatch):
        """`auto` must not hand back an artifact this machine cannot execute."""
        import mozo.runtimes as module
        monkeypatch.setitem(module._REQUIRES, "coreml", "a_module_that_does_not_exist")
        assert select_runtime("mps", ["coreml-fp32", "torch-fp32"]) == "torch-fp32"

    def test_asking_by_name_for_a_missing_library_still_returns_the_key(self, monkeypatch):
        """An explicit request is honoured up to the point the runner explains what is missing."""
        import mozo.runtimes as module
        monkeypatch.setitem(module._REQUIRES, "coreml", "a_module_that_does_not_exist")
        assert select_runtime("mps", ["coreml-fp32", "torch-fp32"], requested="coreml-fp32") == "coreml-fp32"


#: OWLv2's entry, spelled out so these tests do not move when the registry does.
DERIVES = {"torch-fp16": "torch-fp32", "tensorrt-fp16": "onnx-fp32"}


class TestDerivedRuntimes:
    """Runtimes built at load time from another published artifact rather than downloaded."""

    def test_offered_only_where_the_source_is_published(self):
        assert runnable(["torch-fp32"], DERIVES) == ["torch-fp32", "torch-fp16"]
        assert runnable(BOTH, DERIVES) == ["onnx-fp32", "torch-fp32", "torch-fp16",
                                           "tensorrt-fp16"]
        assert runnable(["labels"], DERIVES) == []

    def test_a_family_without_derives_is_untouched(self):
        assert runnable(BOTH) == runnable(BOTH, None) == runnable(BOTH, {}) == BOTH

    def test_asking_for_one_on_the_wrong_device_names_the_device(self):
        with pytest.raises(RuntimeError_, match="runs only on cuda.*'cpu'"):
            select_runtime("cpu", BOTH, "torch-fp16", derives=DERIVES)
        with pytest.raises(RuntimeError_, match="runs only on cuda.*'mps'"):
            select_runtime("mps", BOTH, "tensorrt-fp16", derives=DERIVES)

    @pytest.mark.parametrize("device", ["cuda", "cuda:1"])
    def test_on_cuda_fp16_is_preferred_and_either_can_be_named(self, device):
        assert select_runtime(device, BOTH, derives=DERIVES) == "torch-fp16"
        assert select_runtime(device, BOTH, "torch-fp16", derives=DERIVES) == "torch-fp16"
        assert select_runtime(device, BOTH, "tensorrt-fp16", derives=DERIVES) == "tensorrt-fp16"

    def test_without_a_device_a_named_runtime_is_only_checked_for_existence(self):
        """Which files a runtime is made of does not depend on where it would run."""
        assert select_runtime(None, BOTH, "tensorrt-fp16", derives=DERIVES) == "tensorrt-fp16"

    def test_auto_never_lands_on_a_runtime_the_device_cannot_run(self, monkeypatch):
        """Even through the last-resort branch, which takes whatever is executable."""
        import mozo.runtimes as module
        monkeypatch.setitem(module._REQUIRES, "tensorrt", "json")  # importable everywhere
        with pytest.raises(RuntimeError_, match="nothing this model publishes can run here"):
            select_runtime("cpu", ["onnx-fp32"], executes=("tensorrt",), derives=DERIVES)
        assert select_runtime("cuda", ["onnx-fp32"], executes=("tensorrt",),
                              derives=DERIVES) == "tensorrt-fp16"

    @pytest.mark.parametrize("device", ["cpu", "mps"])
    def test_auto_keeps_fp32_off_cuda(self, device):
        assert select_runtime(device, BOTH, derives=DERIVES) == "torch-fp32"

    def test_auto_does_not_prefer_tensorrt(self, monkeypatch):
        """TensorRT fp16 moves about 2% of confident OWLv2 detections; a user has to choose it."""
        import mozo.runtimes as module
        monkeypatch.setitem(module._REQUIRES, "tensorrt", "json")
        assert select_runtime("cuda", BOTH, derives=DERIVES).startswith("torch")

    def test_the_family_filter_applies_to_derived_runtimes_too(self):
        with pytest.raises(RuntimeError_, match="can only execute torch"):
            select_runtime("cuda", BOTH, "tensorrt-fp16", executes=("torch",), derives=DERIVES)


class TestProviders:
    def test_cuda_asks_for_the_gpu_first_and_keeps_cpu_as_a_floor(self):
        assert providers_for("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_mps_does_not_ask_for_coreml(self):
        """The CoreML provider placed 608 of 1415 RF-DETR nodes and then failed at run time."""
        assert providers_for("mps") == ["CPUExecutionProvider"]

    def test_cpu(self):
        assert providers_for("cpu") == ["CPUExecutionProvider"]


def _tiny_graph(directory, forward):
    """Export *forward* as a one-input ONNX graph, the shape of what mozo publishes."""
    import torch

    class Graph(torch.nn.Module):
        def forward(self, x):
            return forward(x)

    path = directory / "onnx-fp32.onnx"
    torch.onnx.export(Graph(), (torch.ones(1, 8),), str(path), input_names=["x"],
                      output_names=["y"], opset_version=17, dynamo=False)
    return path


@pytest.fixture
def cuda_tensorrt():
    from importlib.util import find_spec

    import torch

    if not torch.cuda.is_available() or find_spec("tensorrt") is None:
        pytest.skip("needs CUDA and tensorrt")
    return torch


class TestTensorRTRunner:
    def test_the_engine_is_built_once_and_kept_beside_the_graph(self, cuda_tensorrt, tmp_path):
        from mozo.runtimes import TensorRTRunner

        torch = cuda_tensorrt
        graph = _tiny_graph(tmp_path, lambda x: x * 2 + 1)
        first = TensorRTRunner(graph, device="cuda", fp16=False)
        assert first.built and first.engine_path.parent == tmp_path / "onnx-fp32.engines"
        assert not TensorRTRunner(graph, device="cuda", fp16=False).built

        x = torch.arange(8.0, device="cuda")[None]
        (y,) = first(x)
        assert y.device.type == "cuda" and torch.equal(y, x * 2 + 1)

    def test_precision_and_gpu_are_in_the_engines_name(self, cuda_tensorrt, tmp_path):
        """An engine built for one GPU or TensorRT version must never be loaded by another."""
        import tensorrt

        from mozo.runtimes import TensorRTRunner

        name = TensorRTRunner(_tiny_graph(tmp_path, lambda x: x + 1), device="cuda").engine_path.name
        assert name.startswith("fp16-") and f"trt{tensorrt.__version__}" in name and "-sm" in name

    def test_an_engine_that_returns_non_finite_values_is_refused_and_deleted(
            self, cuda_tensorrt, tmp_path):
        from mozo.runtimes import RuntimeError_, TensorRTRunner

        graph = _tiny_graph(tmp_path, lambda x: (x * 0).log())  # log(0) on the zeros probe
        with pytest.raises(RuntimeError_, match="non-finite y"):
            TensorRTRunner(graph, device="cuda", fp16=False)
        assert not list((tmp_path / "onnx-fp32.engines").glob("*.engine"))

    def test_concurrent_calls_each_get_their_own_answer(self, cuda_tensorrt, tmp_path):
        from concurrent.futures import ThreadPoolExecutor

        from mozo.runtimes import TensorRTRunner

        torch = cuda_tensorrt
        run = TensorRTRunner(_tiny_graph(tmp_path, lambda x: x * 3), device="cuda", fp16=False)
        inputs = [torch.full((1, 8), float(i), device="cuda") for i in range(64)]
        with ThreadPoolExecutor(8) as pool:
            outputs = list(pool.map(lambda x: run(x)[0], inputs))
        torch.cuda.synchronize()
        assert all(torch.equal(y, x * 3) for x, y in zip(inputs, outputs))

    def test_cpu_is_refused(self, cuda_tensorrt, tmp_path):
        from mozo.runtimes import RuntimeError_, TensorRTRunner

        with pytest.raises(RuntimeError_, match="only on CUDA"):
            TensorRTRunner(tmp_path / "missing.onnx", device="cpu")


def test_the_missing_tensorrt_message_names_the_matching_wheel(monkeypatch):
    """TensorRT ships one wheel per CUDA major, and it has to match torch's."""
    import torch

    from mozo.runtimes import _tensorrt_install_hint

    monkeypatch.setattr(torch.version, "cuda", "11.8")
    assert _tensorrt_install_hint() == "pip install tensorrt-cu11"
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    assert _tensorrt_install_hint() == "pip install tensorrt-cu13"


def test_building_an_engine_leaves_the_callers_locale_alone(cuda_tensorrt, tmp_path):
    """TensorRT's builder switches its thread's C locale to ASCII. On the caller's thread that
    would break every later ``open()`` of a UTF-8 file without an explicit encoding."""
    import locale

    from mozo.runtimes import TensorRTRunner

    before = locale.getpreferredencoding(False)
    TensorRTRunner(_tiny_graph(tmp_path, lambda x: x - 1), device="cuda", fp16=False)
    assert locale.getpreferredencoding(False) == before
