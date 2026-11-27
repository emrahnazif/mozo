"""Depth Anything 3, from the variant table down to an inverse-depth map on a real photograph.

Tests that need weights skip when the artifact cannot be obtained. Agreement with upstream is not
tested here: it needs the authors' repository at the pinned commit, and lives in
``tools/verify/depth_anything_v3.py``, where the commit check can refuse a wrong baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import FIXTURE, published, require_weights
from mozo.vendors.depth_anything_v3_deploy import (MODEL_SPECS, DepthAnything3Net, fill_sky,
                                                   get_spec, processed_size, shape_inputs)
from mozo.vendors.depth_anything_v3_deploy.graph import input_names, output_names, profile
from mozo.runtimes import select_runtime
from mozo.weights import WeightsError, artifacts

FAMILY = "depth_anything_v3"
ALL = ("small", "base", "mono-large")


@pytest.fixture(scope="module")
def predictor_for():
    from mozo.adapters.depth_anything_v3 import DepthAnythingV3Predictor

    cache: dict[tuple, object] = {}

    def build(variant: str, runtime: str = "torch-fp32"):
        if (variant, runtime) not in cache:
            try:
                cache[variant, runtime] = DepthAnythingV3Predictor(variant, device="cpu",
                                                                   runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"{FAMILY}/{variant} {runtime} unavailable: {error}")
        return cache[variant, runtime]

    return build


@pytest.fixture(scope="module")
def image():
    from mozo.image import load_image

    return load_image(str(FIXTURE))


class TestVariantTable:
    def test_three_variants(self):
        assert set(MODEL_SPECS) == set(ALL)

    def test_any_view_models_alternate_attention_and_have_no_sky(self):
        for variant in ("small", "base"):
            spec = get_spec(variant)
            assert (spec.alt_start, spec.qknorm_start, spec.rope_start) == (4, 4, 4)
            assert spec.cat_token and not spec.sky

    def test_the_monocular_model_is_plain_dinov2_with_a_sky_head(self):
        spec = get_spec("mono-large")
        assert (spec.alt_start, spec.qknorm_start, spec.rope_start) == (-1, -1, -1)
        assert spec.sky and not spec.cat_token

    @pytest.mark.parametrize("variant", ALL)
    def test_geometry_builds_the_published_tensor_count(self, variant):
        # The strict load against the checkpoint is the real check; this keeps the specs honest
        # on a machine without weights.
        assert len(DepthAnything3Net(get_spec(variant)).state_dict()) == (
            406 if variant == "mono-large" else 269)

    def test_unknown_variant_names_the_alternatives(self):
        with pytest.raises(KeyError, match="mono-large"):
            get_spec("giant")


class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_publishes_torch(self, variant):
        keys = published(FAMILY, variant)
        if not keys:
            pytest.skip(f"{FAMILY}/{variant} is not published locally")
        assert "torch-fp32" in keys

    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_is_apache(self, variant):
        from mozo.weights import license_of

        if not published(FAMILY, variant):
            pytest.skip(f"{FAMILY}/{variant} is not published locally")
        assert license_of(FAMILY, variant) == "Apache-2.0"

    @pytest.mark.parametrize("variant", ALL)
    def test_a_graph_brings_the_position_embedding(self, variant):
        keys = published(FAMILY, variant)
        if not any(k.startswith("onnx") for k in keys):
            pytest.skip(f"{FAMILY}/{variant} publishes no graph")
        assert "pos_embed" in keys


class TestPreprocessing:
    @pytest.mark.parametrize("shape,expected", [
        ((1281, 1920), (336, 504)),   # the fixture photograph
        ((480, 640), (378, 504)),
        ((640, 480), (504, 378)),
        ((300, 504), (294, 504)),     # long side already 504: only the multiple-of-14 resize
        ((217, 504), (224, 504)),     # 217 is halfway between 210 and 224: ties round up
        ((60, 90), (336, 504)),       # enlarged
    ])
    def test_long_side_504_and_each_side_its_nearest_multiple_of_14(self, shape, expected):
        assert processed_size(*shape) == expected

    def test_extreme_aspect_ratio_is_refused_not_crashed(self, predictor_for):
        require_weights(FAMILY, "small")
        with pytest.raises(ValueError, match="too elongated"):
            predictor_for("small").predict(np.zeros((5, 2000, 3), np.uint8))


class TestSkyFill:
    def test_sky_becomes_the_99th_percentile_of_the_rest(self):
        depth = torch.arange(1, 201, dtype=torch.float32).reshape(1, 1, 10, 20)
        sky = torch.zeros_like(depth)
        sky[..., :2, :] = 1.0  # the top two rows are sky
        filled = fill_sky(depth, sky, torch.Generator().manual_seed(0))
        rest = depth[sky < 0.3]
        assert torch.equal(filled[sky < 0.3], rest)
        assert torch.all(filled[sky >= 0.3] == torch.quantile(rest, 0.99))

    def test_ten_sky_pixels_or_fewer_change_nothing(self):
        depth = torch.rand(1, 1, 10, 20)
        sky = torch.zeros_like(depth)
        sky.view(-1)[:10] = 1.0
        assert torch.equal(fill_sky(depth, sky, torch.Generator().manual_seed(0)), depth)

    def test_a_large_map_samples_from_its_own_generator_not_the_global_one(self):
        depth = torch.rand(1, 1, 400, 400)
        sky = torch.zeros_like(depth)
        sky[..., :50, :] = 1.0
        state = torch.get_rng_state()
        a = fill_sky(depth, sky, torch.Generator().manual_seed(0))
        assert torch.equal(torch.get_rng_state(), state)
        assert torch.equal(a, fill_sky(depth, sky, torch.Generator().manual_seed(0)))


class TestGraphInterface:
    @pytest.mark.parametrize("variant", ALL)
    def test_profile_covers_every_input_from_3_to_36_patches(self, variant):
        spec = get_spec(variant)
        shapes = profile(spec)
        assert tuple(shapes) == input_names(spec)
        low, _, high = shapes["pixels"]
        assert low[2:] == (42, 42) and high[2:] == (504, 504)

    @pytest.mark.parametrize("variant", ALL)
    def test_shape_inputs_match_the_profile(self, variant):
        spec = get_spec(variant)
        net = DepthAnything3Net(spec)
        tensors = shape_inputs(spec, net.backbone.pretrained.pos_embed, 378, 504)
        _, preferred, _ = zip(*profile(spec).values())
        assert [tuple(t.shape) for t in tensors] == list(preferred[1:])
        assert all(t.is_contiguous() for t in tensors)

    @pytest.mark.parametrize("variant", ALL)
    def test_precomputed_shape_inputs_give_the_same_depth(self, variant):
        """What a graph is handed must be exactly what the network computes for itself."""
        torch.manual_seed(0)
        net = DepthAnything3Net(get_spec(variant)).eval()
        x = torch.randn(1, 1, 3, 42, 98)
        with torch.inference_mode():
            own = net(x)
            given = net(x, net.shape_inputs(42, 98))
        assert set(own) == set(output_names(get_spec(variant))) | (
            set() if get_spec(variant).sky else {"depth_conf"})
        assert all(torch.equal(own[k], given[k]) for k in own)


class TestDepthMaps:
    @pytest.mark.parametrize("variant", ALL)
    def test_inverse_depth_at_the_input_resolution(self, predictor_for, image, variant):
        require_weights(FAMILY, variant)
        inverse = predictor_for(variant).predict(image)
        assert inverse.shape == image.shape[:2]
        assert inverse.dtype == np.float32
        assert inverse.min() > 0.0 and inverse.max() > inverse.min()

    def test_accepts_a_path_as_well_as_an_array(self, predictor_for, image):
        require_weights(FAMILY, "small")
        model = predictor_for("small")
        assert np.array_equal(model.predict(str(FIXTURE)), model.predict(image))

    def test_unitless_and_larger_is_nearer(self, predictor_for, image):
        require_weights(FAMILY, "small")
        model = predictor_for("small")
        assert model.unit is None
        # The vendor's output is depth; the adapter's is its reciprocal, resized.
        from mozo.vendors.depth_anything_v3_deploy import resize

        depth = model._predictor.predict(image)
        expected = (1.0 / resize(depth, image.shape[:2])).numpy()
        assert np.array_equal(model.predict(image), expected)

    @pytest.mark.parametrize("variant", ALL)
    def test_the_onnx_graph_agrees_with_torch(self, predictor_for, image, variant):
        require_weights(FAMILY, variant, "onnx-fp32")
        pytest.importorskip("onnxruntime")
        want = predictor_for(variant).predict(image)
        got = predictor_for(variant, "onnx-fp32").predict(image)
        assert np.abs(got - want).max() / np.abs(want).max() < 1e-3


def _choose(device, variant, runtime="auto"):
    """What ``auto`` (or *runtime*) resolves to for *variant* with the family's own tables."""
    from mozo.registry import DERIVES, PREFERS

    return select_runtime(device, artifacts(FAMILY, variant), runtime,
                          derives=DERIVES[FAMILY], prefers=PREFERS[FAMILY])


class TestAuto:
    """On CUDA ``auto`` takes the fp16 engine (``registry.PREFERS``); the CPU stays on torch."""

    @pytest.mark.parametrize("variant", ALL)
    def test_cuda_takes_the_fp16_engine_when_tensorrt_is_usable(self, monkeypatch, variant):
        import mozo.runtimes as runtimes

        require_weights(FAMILY, variant, "onnx-fp16")
        monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: True)
        assert _choose("cuda", variant) == "tensorrt-fp16"
        assert _choose("cuda", variant, "tensorrt-fp32") == "tensorrt-fp32"
        monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: False)
        assert _choose("cuda", variant) == "torch-fp32"

    def test_the_cpu_stays_on_torch(self):
        require_weights(FAMILY, "small", "onnx-fp32")
        assert _choose("cpu", "small") == "torch-fp32"


class TestAdapter:
    def test_variants_come_from_the_vendor(self):
        from mozo.adapters.depth_anything_v3 import DepthAnythingV3Predictor

        assert set(DepthAnythingV3Predictor.VARIANTS) == set(MODEL_SPECS)

    def test_unknown_variant_is_rejected_before_any_download(self):
        from mozo.adapters.depth_anything_v3 import DepthAnythingV3Predictor

        with pytest.raises(ValueError, match="Unsupported variant"):
            DepthAnythingV3Predictor("giant")

    def test_registry_agrees_with_the_adapter(self):
        from mozo.adapters.depth_anything_v3 import DepthAnythingV3Predictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY[FAMILY]
        assert entry["adapter_class"] == DepthAnythingV3Predictor.__name__
        assert entry["module"] == "mozo.adapters.depth_anything_v3"
        assert entry["task_type"] == "depth_estimation"
        assert tuple(entry["variants"]) == DepthAnythingV3Predictor.VARIANTS
