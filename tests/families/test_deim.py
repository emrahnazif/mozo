"""Does the DEIM family work.

These load real checkpoints and run real inference, so they need the published artifacts and are
skipped when those are absent. Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_deim.py -q

What they protect: the registry and the adapter agree; every variant ships its terms and its
vocabulary; the geometry written down is the checkpoint's (the strict, dtype-checked load); the
refusals; the result shape; a recorded answer on the fixture photograph, so drift shows up without
upstream installed; and that every runtime gives the torch answer. The exact comparison against
the authors' repository is ``tools/verify/deim.py``, which needs a clone of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import COORD_STEP, published, require_weights
from mozo.runtimes import select_runtime
from mozo.weights import WeightsError, artifacts

FAMILY = "deim"
ALL = ["nano", "small", "medium", "large", "xlarge", "xlarge-obj365"]
#: Variants publishing an fp16 graph. nano's TensorRT fp16 engine was slower than its fp32 one
#: (2.4 against 1.9 ms on an RTX 3090), so it publishes none.
FP16 = set(ALL) - {"nano"}
RECORDED = Path(__file__).resolve().parent.parent / "fixtures" / "deim_example.json"

#: The fixture photograph is two people at a desk; COCO calls what is on it these things. A model
#: that reads it as anything outside this set at the default threshold has regressed.
EXPECTED_NAMES = {"person", "laptop", "chair", "cup", "tv", "dining table", "book", "cell phone",
                  "keyboard", "mouse", "bottle", "potted plant"}

#: What a graph runtime may move against torch, after PixelFlow's rounding: what
#: ``tools/export/deim.py`` holds the fp32 graph to, plus one rounding step.
BOX_TOLERANCE = 1e-2 + COORD_STEP


def _worst(want, got) -> float:
    """The largest box-coordinate difference between two detection lists, pair by pair."""
    return max((max(abs(a - b) for a, b in zip(x.bbox, y.bbox)) for x, y in zip(want, got)),
               default=0.0)


@pytest.fixture(scope="module")
def predictor_for():
    """Predictors built once per (variant, runtime), dropped when the variant changes."""
    from mozo.adapters.deim import DEIMPredictor

    cache: dict[str, object] = {}
    loaded = None

    def build(variant: str, runtime: str = "torch-fp32", device: str = "cpu"):
        nonlocal loaded
        if variant != loaded:
            cache.clear()
            loaded = variant
        key = f"{runtime}@{device}"
        if key not in cache:
            try:
                cache[key] = DEIMPredictor(variant, device=device, runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"{FAMILY}/{variant} weights unavailable: {error}")
        return cache[key]

    return build


class TestRegistry:
    def test_registry_agrees_with_the_adapter(self):
        from mozo.adapters.deim import DEIMPredictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY[FAMILY]
        assert entry["adapter_class"] == DEIMPredictor.__name__
        assert entry["module"] == "mozo.adapters.deim"
        assert entry["task_type"] == "object_detection"
        assert list(entry["variants"]) == list(DEIMPredictor.VARIANTS) == ALL

    def test_the_node_and_the_adapter_share_a_default_threshold(self):
        import inspect

        from mozo.adapters.deim import DEIMPredictor
        from mozo.workflow.nodes import model as nodes

        adapter = inspect.signature(DEIMPredictor.predict).parameters["threshold"].default
        node = inspect.signature(inspect.unwrap(nodes.deim)).parameters["threshold"].default
        assert adapter == node == 0.4

    def test_the_vendor_imports_nothing_from_mozo(self):
        """Rule 1 of VENDORING.md: every import inside the vendor is relative or third-party."""
        root = Path(__file__).resolve().parents[2] / "mozo" / "vendors" / "deim_deploy"
        for path in root.glob("*.py"):
            imports = [line.strip() for line in path.read_text().splitlines()
                       if line.strip().startswith(("import ", "from "))]
            for line in imports:
                assert "mozo" not in line and "_deploy" not in line, f"{path.name}: {line}"


class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_publishes_its_checkpoint_graph_and_vocabulary(self, variant):
        keys = published(FAMILY, variant)
        if not keys:
            pytest.skip(f"{FAMILY}/{variant} is not in the manifest")
        assert {"torch-fp32", "onnx-fp32", "labels"} <= set(keys)
        assert ("onnx-fp16" in keys) == (variant in FP16)

    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_ships_its_terms(self, variant):
        from mozo.weights import license_of

        if not published(FAMILY, variant):
            pytest.skip(f"{FAMILY}/{variant} is not in the manifest")
        assert license_of(FAMILY, variant) == "Apache-2.0"

    def test_the_vocabulary_is_cocos_80_in_contiguous_order(self):
        require_weights(FAMILY, "nano", "labels")
        from mozo.labels import resolve

        rows = resolve(FAMILY, "nano", published=True)
        assert [row["id"] for row in rows] == list(range(80))
        assert rows[0]["name"] == "person" and rows[79]["name"] == "toothbrush"


class TestGeometry:
    @pytest.mark.parametrize("variant", ALL)
    def test_the_checkpoint_loads_strictly_as_its_variant(self, variant):
        """Keys, shapes and dtypes; the loader raises on any disagreement."""
        require_weights(FAMILY, variant)
        from mozo.vendors.deim_deploy import load
        from mozo.weights import resolve

        model = load(variant, resolve(FAMILY, variant, "torch-fp32"))
        assert model.decoder.reg_scale.dtype.is_floating_point is False

    def test_a_checkpoint_of_another_size_is_refused(self):
        require_weights(FAMILY, "nano")
        from mozo.vendors.deim_deploy import load
        from mozo.weights import resolve

        with pytest.raises(ValueError, match="not a DEIM 'small' checkpoint"):
            load("small", resolve(FAMILY, "nano", "torch-fp32"))

    def test_a_float_reg_scale_is_refused(self, monkeypatch):
        """``load_state_dict`` would cast it silently; the dtype check is what catches it."""
        require_weights(FAMILY, "nano")
        import dataclasses

        from mozo.vendors.deim_deploy import config, load, model
        from mozo.weights import resolve

        spec = dataclasses.replace(config.SPECS["nano"], reg_scale=4.0)
        monkeypatch.setattr(model, "get_spec", lambda variant: spec)
        with pytest.raises(ValueError, match="dtypes differ"):
            load("nano", resolve(FAMILY, "nano", "torch-fp32"))

    def test_the_two_xlarges_differ_only_where_a_shape_cannot_show(self):
        from mozo.vendors.deim_deploy import SPECS

        a, b = SPECS["xlarge"], SPECS["xlarge-obj365"]
        differ = {f for f in a.__dataclass_fields__ if getattr(a, f) != getattr(b, f)}
        assert differ == {"config", "freeze_norm", "dec_act"}


class TestRefusals:
    def test_an_unknown_variant(self):
        from mozo.adapters.deim import DEIMPredictor

        with pytest.raises(ValueError, match="Unsupported variant"):
            DEIMPredictor("huge")

    def test_an_image_that_is_not_rgb(self):
        from mozo.vendors.deim_deploy import preprocess

        with pytest.raises(ValueError, match="uint8 RGB"):
            preprocess(np.zeros((10, 10), np.uint8))


class TestDetections:
    @pytest.mark.parametrize("variant", ALL)
    def test_finds_the_scene(self, predictor_for, image, variant):
        require_weights(FAMILY, variant)
        found = predictor_for(variant).predict(image)
        names = {d.class_name for d in found}
        assert "person" in names
        assert names <= EXPECTED_NAMES, f"unexpected classes: {names - EXPECTED_NAMES}"
        assert all(d.masks is None for d in found)

    def test_a_threshold_filters(self, predictor_for, image):
        require_weights(FAMILY, "nano")
        low = predictor_for("nano").predict(image, threshold=0.05)
        high = predictor_for("nano").predict(image, threshold=0.6)
        assert len(low) > len(high) > 0
        assert all(d.confidence > 0.6 for d in high)

    def test_at_most_300_and_none_on_a_blank_frame(self, predictor_for):
        require_weights(FAMILY, "nano")
        grey = np.full((480, 640, 3), 114, np.uint8)
        assert len(predictor_for("nano").predict(grey, threshold=0.0)) == 300
        assert len(predictor_for("nano").predict(grey)) == 0

    @pytest.mark.parametrize("variant", ["nano", "xlarge-obj365"])
    def test_matches_the_recorded_answer(self, predictor_for, image, variant):
        """Recorded from this vendor after ``tools/verify/deim.py`` held it equal to upstream at
        every module. Exact: same weights, same CPU arithmetic."""
        require_weights(FAMILY, variant)
        recorded = json.loads(RECORDED.read_text())[variant]
        found = predictor_for(variant).predict(image, threshold=recorded["threshold"])
        assert [d.class_id for d in found] == recorded["class_id"]
        assert [d.bbox for d in found] == recorded["bbox"]
        assert [d.confidence for d in found] == recorded["confidence"]


class TestRuntimes:
    def test_without_onnxruntime_the_cpu_stays_on_torch(self, monkeypatch):
        import mozo.runtimes as runtimes

        require_weights(FAMILY, "nano")
        monkeypatch.setitem(runtimes._REQUIRES, "onnx", "no_such_module_for_this_test")
        assert TestTensorRT._choose("cpu", "nano") == "torch-fp32"

    @pytest.mark.parametrize("variant", ALL)
    def test_onnx_gives_the_torch_answer(self, predictor_for, image, variant):
        require_weights(FAMILY, variant, "onnx-fp32")
        pytest.importorskip("onnxruntime")
        want = predictor_for(variant).predict(image)
        got = predictor_for(variant, "onnx-fp32").predict(image)
        assert [d.class_id for d in want] == [d.class_id for d in got]
        assert _worst(want, got) <= BOX_TOLERANCE


class TestTensorRT:
    """On CUDA: the fp16 engine chosen by ``auto`` where its graph is published, else fp32."""

    @staticmethod
    def _choose(device, variant, runtime="auto"):
        from mozo.registry import DERIVES, PREFERS

        return select_runtime(device, artifacts(FAMILY, variant), runtime,
                              derives=DERIVES[FAMILY], prefers=PREFERS[FAMILY])

    def test_auto_picks_the_fp16_engine_on_cuda_where_published(self, monkeypatch):
        import mozo.runtimes as runtimes

        require_weights(FAMILY, "nano", "onnx-fp32")
        monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: True)
        for variant in ALL:
            want = "tensorrt-fp16" if variant in FP16 else "tensorrt-fp32"
            assert self._choose("cuda", variant) == want, variant
            assert self._choose("cuda", variant, "tensorrt-fp32") == "tensorrt-fp32", variant
        with pytest.raises(Exception, match="not published"):
            self._choose("cuda", "nano", "tensorrt-fp16")
        assert self._choose("cpu", "nano") == "onnx-fp32"

    def test_without_tensorrt_cuda_stays_on_torch(self, monkeypatch):
        import mozo.runtimes as runtimes

        require_weights(FAMILY, "nano", "onnx-fp32")
        monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: False)
        assert self._choose("cuda", "nano") == "torch-fp32"

    @staticmethod
    def _on_cuda(predictor_for, variant, runtime):
        torch = pytest.importorskip("torch")
        from mozo.registry import DERIVES
        from mozo.runtimes import _tensorrt_unusable

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        if runtime.startswith("tensorrt") and (problem := _tensorrt_unusable()) is not None:
            pytest.skip(problem)
        require_weights(FAMILY, variant, DERIVES[FAMILY].get(runtime, runtime))
        return predictor_for(variant, runtime, "cuda")

    def test_the_fp32_engine_matches_torch(self, predictor_for, image):
        """Against torch on the CPU: the engine is built with TF32 off, so it is IEEE fp32."""
        want = predictor_for("nano").predict(image)
        got = self._on_cuda(predictor_for, "nano", "tensorrt-fp32").predict(image)
        assert len(want) == len(got) > 0
        assert [d.class_id for d in want] == [d.class_id for d in got]
        assert _worst(want, got) <= 0.1

    def test_the_fp16_engine_keeps_the_confident_detections(self, predictor_for, image):
        want = predictor_for("small").predict(image, threshold=0.7)
        got = self._on_cuda(predictor_for, "small", "tensorrt-fp16").predict(image, threshold=0.4)
        assert len(want) > 0
        for found in want:
            distance, nearest = min((float(np.abs(np.subtract(g.bbox, found.bbox)).max()), i)
                                    for i, g in enumerate(got))
            assert got[nearest].class_name == found.class_name
            assert distance < 5.0


class TestCoreML:
    """On Apple silicon: ``auto`` takes the Core ML fp32 package (the shared order puts it first
    on ``mps``), held to the export tool's fp32 tolerances. No fp16 package is published."""

    def test_auto_picks_the_fp32_package_on_apple_silicon(self):
        pytest.importorskip("coremltools")
        require_weights(FAMILY, "nano", "coreml-fp32")
        for variant in ALL:
            assert TestTensorRT._choose("mps", variant) == "coreml-fp32", variant

    @pytest.mark.parametrize("variant", ["nano", "small"])
    def test_the_package_matches_torch(self, predictor_for, image, variant):
        pytest.importorskip("coremltools")
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("needs Apple silicon")
        require_weights(FAMILY, variant, "coreml-fp32")
        got = predictor_for(variant, "coreml-fp32", "mps").predict(image)
        want = predictor_for(variant).predict(image)
        assert len(want) == len(got) > 0
        assert [d.class_id for d in want] == [d.class_id for d in got]
        assert _worst(want, got) <= BOX_TOLERANCE
