"""Does the Mayaku family work (so far: the street models).

These load real checkpoints and run real inference, so they need the published artifacts and are
skipped when those are absent. Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_mayaku.py -q

What they protect: the registry and the adapter agree; every variant ships its terms; the
geometry the vendor writes down is the checkpoint's; the refusals; the result shape; a recorded
answer on the fixture photograph, so drift shows up without ``mayaku`` installed; and that every
runtime gives the torch answer. The exact comparison against ``mayaku`` itself is
``tools/verify/mayaku.py``, which needs the reference installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import COORD_STEP, published, require_weights
from mozo.runtimes import select_runtime
from mozo.vendors.mayaku_deploy import SPECS
from mozo.weights import WeightsError, artifacts

FAMILY = "mayaku"
# The street models, read from the vendor's own table. Every assertion below about the scene and
# the 22-class vocabulary is about them; another Mayaku model gets lists of its own.
STREET = [v for v in SPECS if v.startswith("street-")]
DETECTION = [v for v in STREET if not SPECS[v].seg]
SEGMENTATION = [v for v in STREET if SPECS[v].seg]
ALL = STREET
RECORDED = Path(__file__).resolve().parent.parent / "fixtures" / "mayaku_example.json"

#: The classes in the fixture photograph (two people at a desk), and the two this family exists
#: to add to mozo: a model that reads it as anything else has regressed.
EXPECTED_NAMES = {"person", "face"}

#: What a graph runtime may move against torch, after PixelFlow's rounding: what
#: ``tools/export/mayaku.py`` holds the graph to at full precision, plus one rounding step.
BOX_TOLERANCE = 1e-2 + COORD_STEP


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


@pytest.fixture(scope="module")
def predictor_for():
    """Predictors built once per (variant, runtime), dropped when the variant changes."""
    from mozo.adapters.mayaku import MayakuPredictor

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
                cache[key] = MayakuPredictor(variant, device=device, runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"{FAMILY}/{variant} weights unavailable: {error}")
        return cache[key]

    return build


class TestRegistry:
    def test_registry_agrees_with_the_adapter(self):
        from mozo.adapters.mayaku import MayakuPredictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY[FAMILY]
        assert entry["adapter_class"] == MayakuPredictor.__name__
        assert entry["module"] == "mozo.adapters.mayaku"
        assert entry["task_type"] == "object_detection"
        assert list(entry["variants"]) == list(MayakuPredictor.VARIANTS)
        assert set(ALL) == set(MayakuPredictor.VARIANTS)

    def test_the_default_threshold_is_the_checkpoints_own(self):
        """0.4 is written in the adapter and the node too; the vendor's spec owns it."""
        import inspect

        from mozo.adapters.mayaku import MayakuPredictor
        from mozo.vendors.mayaku_deploy import SPECS
        from mozo.workflow.nodes import model as nodes

        owned = {spec.conf for spec in SPECS.values()}
        assert owned == {inspect.signature(MayakuPredictor.predict).parameters["threshold"].default}
        assert owned == {inspect.signature(inspect.unwrap(nodes.mayaku)).parameters["threshold"].default}

    def test_the_vendor_specs_cover_every_variant(self):
        from mozo.adapters.mayaku import MayakuPredictor
        from mozo.vendors.mayaku_deploy import SPECS

        assert set(SPECS) == set(MayakuPredictor.VARIANTS)
        assert {v for v, s in SPECS.items() if s.seg} == set(SEGMENTATION)


class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_publishes_its_graphs_and_vocabulary(self, variant):
        keys = published(FAMILY, variant)
        if not keys:
            pytest.skip(f"{FAMILY}/{variant} is not in the manifest")
        assert {"torch-fp32", "onnx-fp32", "onnx-fp16", "labels"} <= set(keys)

    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_ships_its_terms(self, variant):
        from mozo.weights import license_of

        if not published(FAMILY, variant):
            pytest.skip(f"{FAMILY}/{variant} is not in the manifest")
        assert license_of(FAMILY, variant) == "Apache-2.0"

    @pytest.mark.parametrize("variant", ALL)
    def test_the_published_vocabulary_is_the_checkpoints(self, predictor_for, variant):
        """22 classes, the checkpoint's own order, faces and licence plates among them."""
        require_weights(FAMILY, variant, "labels")
        from mozo.labels import resolve

        published_names = [row["name"] for row in resolve(FAMILY, variant, published=True)]
        assert published_names == predictor_for(variant)._detector.names
        assert len(published_names) == 22
        assert {"face", "license plate"} <= set(published_names)


class TestGeometry:
    @pytest.mark.parametrize("variant", ALL)
    def test_the_checkpoint_agrees_with_the_written_spec(self, variant):
        """The loader refuses a disagreeing checkpoint; this loads each one, which is the check."""
        require_weights(FAMILY, variant)
        from mozo.vendors.mayaku_deploy.model import check_sidecar, get_spec
        from mozo.vendors.mayaku_deploy.network import read_checkpoint
        from mozo.weights import resolve

        try:
            path = resolve(FAMILY, variant, "torch-fp32")
        except WeightsError as error:
            pytest.skip(str(error))
        check_sidecar(read_checkpoint(path)[1], get_spec(variant), str(path))

    def test_a_checkpoint_of_the_other_kind_is_refused(self):
        """A detection checkpoint loaded as its segmentation variant would run without masks."""
        require_weights(FAMILY, "street-nano")
        from mozo.vendors.mayaku_deploy import Detector
        from mozo.weights import resolve

        try:
            path = resolve(FAMILY, "street-nano", "torch-fp32")
        except WeightsError as error:
            pytest.skip(str(error))
        with pytest.raises(ValueError, match="not the Mayaku checkpoint"):
            Detector(path, "street-seg-nano")
        with pytest.raises(ValueError, match="not the Mayaku checkpoint"):
            Detector(path, "street-medium")


class TestRefusals:
    def test_an_unknown_variant(self):
        from mozo.adapters.mayaku import MayakuPredictor

        with pytest.raises(ValueError, match="Unsupported variant"):
            MayakuPredictor("large")

    @pytest.mark.parametrize("threshold", [-0.1, 1.0, 1.5])
    def test_a_threshold_outside_zero_to_one(self, predictor_for, image, threshold):
        require_weights(FAMILY, "street-nano")
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            predictor_for("street-nano").predict(image, threshold=threshold)


class TestDetections:
    @pytest.mark.parametrize("variant", ALL)
    def test_finds_the_scene(self, predictor_for, image, variant):
        require_weights(FAMILY, variant)
        found = predictor_for(variant).predict(image)
        names = {d.class_name for d in found}
        assert names <= EXPECTED_NAMES, f"unexpected classes: {names - EXPECTED_NAMES}"
        assert names == EXPECTED_NAMES

    @pytest.mark.parametrize("variant", SEGMENTATION)
    def test_segmentation_variants_return_masks(self, predictor_for, image, variant):
        require_weights(FAMILY, variant)
        found = predictor_for(variant).predict(image)
        assert len(found) > 0
        for detection in found:
            assert detection.masks[0].shape == image.shape[:2]
            assert detection.masks[0].dtype == bool
            assert detection.masks[0].any()

    @pytest.mark.parametrize("variant", DETECTION)
    def test_detection_variants_return_no_masks(self, predictor_for, image, variant):
        require_weights(FAMILY, variant)
        found = predictor_for(variant).predict(image)
        assert len(found) > 0
        assert all(d.masks is None for d in found)

    @pytest.mark.parametrize("size", ["nano", "medium"])
    def test_segmentation_boxes_are_the_detection_boxes(self, predictor_for, image, size):
        """The seg checkpoint is the det checkpoint plus a mask branch, so the boxes agree."""
        detection, segmentation = f"street-{size}", f"street-seg-{size}"
        require_weights(FAMILY, detection)
        require_weights(FAMILY, segmentation)
        boxes = [d.bbox for d in predictor_for(detection).predict(image)]
        assert boxes == [d.bbox for d in predictor_for(segmentation).predict(image)]

    def test_an_empty_frame_finds_nothing(self, predictor_for):
        require_weights(FAMILY, "street-seg-nano")
        grey = np.full((480, 640, 3), 114, np.uint8)
        assert len(predictor_for("street-seg-nano").predict(grey, threshold=0.05)) == 0

    @pytest.mark.parametrize("variant", ["street-nano", "street-seg-nano"])
    def test_matches_the_recorded_answer(self, predictor_for, image, variant):
        """Recorded from this vendor after ``tools/verify/mayaku.py`` held it equal to
        ``mayaku`` 3.1.1 at every stage. Exact: same weights, same CPU arithmetic."""
        require_weights(FAMILY, variant)
        recorded = json.loads(RECORDED.read_text())[variant]
        found = predictor_for(variant).predict(image, threshold=recorded["threshold"])
        assert [d.bbox for d in found] == recorded["bbox"]
        assert [d.confidence for d in found] == recorded["confidence"]
        assert [d.class_id for d in found] == recorded["class_id"]
        if "mask_pixels" in recorded:
            assert [int(d.masks[0].sum()) for d in found] == recorded["mask_pixels"]


class TestRuntimes:
    def test_auto_picks_torch_on_the_cpu(self):
        require_weights(FAMILY, "street-nano")
        assert select_runtime("cpu", artifacts(FAMILY, "street-nano")) == "torch-fp32"

    @pytest.mark.parametrize("variant", ALL)
    def test_onnx_gives_the_torch_answer(self, predictor_for, image, variant):
        require_weights(FAMILY, variant, "onnx-fp32")
        pytest.importorskip("onnxruntime")
        want = predictor_for(variant).predict(image, threshold=0.05)
        got = predictor_for(variant, "onnx-fp32").predict(image, threshold=0.05)
        assert [d.class_id for d in want] == [d.class_id for d in got]
        worst = max((max(abs(a - b) for a, b in zip(x.bbox, y.bbox)) for x, y in zip(want, got)),
                    default=0)
        assert worst <= BOX_TOLERANCE
        if variant in SEGMENTATION:
            for x, y in zip(want, got):
                assert _mask_iou(x.masks[0], y.masks[0]) >= 0.999


class TestTensorRT:
    """On CUDA: the exact fp32 engine chosen by ``auto``, fp16 on request, for all four."""

    @staticmethod
    def _choose(device, variant, runtime="auto"):
        from mozo.registry import DERIVES

        return select_runtime(device, artifacts(FAMILY, variant), runtime,
                              derives=DERIVES[FAMILY])

    def test_auto_picks_the_fp32_engine_on_cuda_when_tensorrt_is_usable(self, monkeypatch):
        import mozo.runtimes as runtimes

        require_weights(FAMILY, "street-nano", "onnx-fp32")
        monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: True)
        for variant in ALL:
            assert self._choose("cuda", variant) == "tensorrt-fp32", variant
            assert self._choose("cuda", variant, "tensorrt-fp16") == "tensorrt-fp16", variant
        assert self._choose("cpu", "street-nano") == "torch-fp32"

    def test_without_tensorrt_cuda_stays_on_torch(self, monkeypatch):
        import mozo.runtimes as runtimes

        require_weights(FAMILY, "street-nano", "onnx-fp32")
        monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: False)
        assert self._choose("cuda", "street-nano") == "torch-fp32"

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
        want = predictor_for("street-seg-nano").predict(image, threshold=0.05)
        got = self._on_cuda(predictor_for, "street-seg-nano", "tensorrt-fp32").predict(image,
                                                                                 threshold=0.05)
        assert len(want) == len(got) > 0
        assert [d.class_id for d in want] == [d.class_id for d in got]
        worst = max(max(abs(a - b) for a, b in zip(x.bbox, y.bbox)) for x, y in zip(want, got))
        assert worst <= 0.1
        for x, y in zip(want, got):
            assert _mask_iou(x.masks[0], y.masks[0]) >= 0.99

    def test_the_fp16_engine_keeps_the_confident_detections(self, predictor_for, image):
        want = predictor_for("street-nano").predict(image, threshold=0.7)
        got = self._on_cuda(predictor_for, "street-nano", "tensorrt-fp16").predict(image, threshold=0.4)
        assert len(want) > 0
        for found in want:
            nearest = min(got, key=lambda g: float(np.abs(np.subtract(g.bbox, found.bbox)).max()))
            assert nearest.class_name == found.class_name
            assert float(np.abs(np.subtract(nearest.bbox, found.bbox)).max()) < 5.0
