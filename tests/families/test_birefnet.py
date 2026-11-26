"""BiRefNet, from the variant table down to an alpha matte on a real photograph.

Tests that need weights skip when the artifact cannot be obtained. They run ``lite``, the variant
fast enough to run on a CPU in a test suite; the other three share its code and differ in geometry,
which the structural tests below hold without weights. Agreement with upstream is not tested here:
it needs the authors' remote code at the pinned commit, and lives in ``tools/verify/birefnet.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import FIXTURE, published
from mozo.registry import BOXED, ENCODES, MODEL_REGISTRY, PROMPTED
from mozo.vendors.birefnet_deploy import BACKBONES, MODEL_SPECS, BiRefNet
from mozo.vendors.birefnet_deploy.image import postprocess, preprocess, refine_foreground
from mozo.vendors.birefnet_deploy.network import image2patches
from mozo.weights import WeightsError

FAMILY = "birefnet"
ALL = ("general", "lite", "matting", "hr")


@pytest.fixture(scope="module")
def predictor():
    from mozo.adapters.birefnet import BiRefNetPredictor

    try:
        return BiRefNetPredictor("lite", device="cpu", runtime="torch-fp32")
    except WeightsError as error:
        pytest.skip(f"{FAMILY}/lite weights unavailable: {error}")


@pytest.fixture(scope="module")
def image():
    from mozo.image import load_image

    return load_image(str(FIXTURE))


@pytest.fixture(scope="module")
def alpha(predictor, image):
    """The matte, computed once: a forward is several seconds on a CPU."""
    return predictor.predict(image)


class TestRegistry:
    def test_registry_agrees_with_the_adapter(self):
        from mozo.adapters.birefnet import BiRefNetPredictor

        entry = MODEL_REGISTRY[FAMILY]
        assert tuple(entry["variants"]) == BiRefNetPredictor.VARIANTS == ALL
        assert entry["adapter_class"] == "BiRefNetPredictor"
        assert entry["module"] == "mozo.adapters.birefnet"

    def test_it_is_asked_nothing_and_told_nothing(self):
        task = MODEL_REGISTRY[FAMILY]["task_type"]
        assert task == "image_matting"
        assert task not in PROMPTED and task not in BOXED
        assert FAMILY not in ENCODES


class TestGeometry:
    """Written down, not inferred: the strict load holds these against the checkpoint."""

    def test_swin_l_and_swin_t(self):
        assert BACKBONES["swin_v1_l"].channels == (192, 384, 768, 1536)
        assert BACKBONES["swin_v1_t"].channels == (96, 192, 384, 768)
        assert BACKBONES["swin_v1_l"].window_size == 12 and BACKBONES["swin_v1_t"].window_size == 7

    def test_the_two_passes_double_every_lateral_width(self):
        assert BACKBONES["swin_v1_l"].lateral_channels == (3072, 1536, 768, 384)

    def test_resolution_per_variant(self):
        assert {v: s.resolution for v, s in MODEL_SPECS.items()} == {
            "general": 1024, "lite": 1024, "matting": 1024, "hr": 2048}
        assert {v: s.backbone for v, s in MODEL_SPECS.items()} == {
            "general": "swin_v1_l", "lite": "swin_v1_t", "matting": "swin_v1_l", "hr": "swin_v1_l"}

    def test_every_resolution_folds_a_32_grid_into_the_first_patch_block(self):
        # ipt_blk5 takes 2**10 * 3 channels: a 32x32 grid of the image at stride 32.
        for spec in MODEL_SPECS.values():
            assert spec.resolution % 32 == 0
        net = BiRefNet(MODEL_SPECS["lite"])
        assert net.decoder.ipt_blk5.conv1.in_channels == 32 * 32 * 3

    def test_training_only_heads_are_not_built(self):
        names = {n for n, _ in BiRefNet(MODEL_SPECS["lite"]).named_modules()}
        assert not any("conv_ms_spvn" in n or "gdt_convs_pred" in n for n in names)
        assert "decoder.gdt_convs_attn_4" in names


class TestImage2Patches:
    def test_channel_order_is_c_then_rows_then_columns(self):
        x = torch.arange(2 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4)
        patches = image2patches(x, torch.empty(1, 1, 2, 2))
        assert patches.shape == (1, 8, 2, 2)
        # channel 0, grid cell (0, 1) is the top-right 2x2 block of channel 0
        assert torch.equal(patches[0, 1], x[0, 0, :2, 2:])
        # channel 1 starts after all four cells of channel 0
        assert torch.equal(patches[0, 4], x[0, 1, :2, :2])

    def test_a_one_by_one_grid_is_the_image(self):
        x = torch.randn(1, 3, 8, 8)
        assert torch.equal(image2patches(x, x), x)


class TestPrePost:
    def test_preprocess_squashes_to_the_resolution(self):
        t = preprocess(np.zeros((10, 300, 3), np.uint8), 1024)
        assert t.shape == (1, 3, 1024, 1024) and t.dtype == torch.float32

    def test_postprocess_truncates(self):
        # sigmoid(0) = 0.5 -> 127.5 -> 127, not 128
        out = postprocess(torch.zeros(1, 1, 4, 4), (4, 4))
        assert out.dtype == np.uint8 and np.all(out == 127)

    def test_postprocess_keeps_float16_logits_in_float16(self):
        # Upstream's sigmoid runs in the logits' dtype: float16 here gives 19, float32 gives 18.
        out = postprocess(torch.full((1, 1, 4, 4), -2.51953125, dtype=torch.float16), (4, 4))
        assert np.all(out == 19)

    def test_postprocess_returns_the_requested_size(self):
        assert postprocess(torch.zeros(1, 1, 8, 8), (3, 5)).shape == (3, 5)
        assert postprocess(torch.zeros(1, 1, 8, 8), (1, 1)).shape == (1, 1)

    def test_refine_changes_colour_where_the_alpha_is_soft(self):
        rgb = np.zeros((64, 64, 3), np.uint8)
        rgb[:, 32:] = 255
        alpha = np.full((64, 64), 128, np.uint8)
        assert not np.array_equal(refine_foreground(rgb, alpha), rgb)

    @pytest.mark.parametrize("bad", [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.float32),
    ])
    def test_the_vendor_refuses_arrays_it_cannot_interpret(self, bad):
        with pytest.raises(ValueError, match="uint8 RGB"):
            preprocess(bad, 1024)


class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_it_publishes_torch(self, variant):
        keys = published(FAMILY, variant)
        if not keys:
            pytest.skip(f"{FAMILY}/{variant} is not published locally")
        assert "torch-fp32" in keys
        assert not ({"LICENSE", "NOTICE"} & set(keys))


class TestMattes:
    def test_output_matches_the_input_resolution(self, alpha, image):
        assert alpha.shape == image.shape[:2] and alpha.dtype == np.uint8

    def test_accepts_a_path_as_well_as_an_array(self, predictor, alpha):
        assert np.array_equal(predictor.predict(FIXTURE), alpha)

    def test_the_model_says_something(self, alpha):
        assert alpha.max() > 200 and alpha.min() < 55

    def test_cutout_carries_the_matte_and_refine_changes_only_colour(self, predictor, image, alpha):
        plain = predictor.cutout(image)
        assert plain.shape == image.shape[:2] + (4,)
        assert np.array_equal(plain[..., :3], image) and np.array_equal(plain[..., 3], alpha)
        refined = predictor.cutout(image, refine=True)
        assert np.array_equal(refined[..., 3], alpha)
        assert not np.array_equal(refined[..., :3], image)

    @pytest.mark.parametrize("shape", [(64, 256, 3), (1, 1, 3)])
    def test_odd_shapes_come_back_at_their_own_size(self, predictor, shape):
        assert predictor.predict(np.full(shape, 128, dtype=np.uint8)).shape == shape[:2]


class TestRefusals:
    def test_an_unknown_variant_names_the_alternatives(self):
        from mozo.adapters.birefnet import BiRefNetPredictor

        with pytest.raises(ValueError, match="general"):
            BiRefNetPredictor("portrait")


class TestGraphRuntimes:
    """The published graphs carry a decomposed deformable convolution: close to torch, not equal."""

    def test_the_decomposition_tracks_torchvision(self):
        from torchvision.ops import deform_conv2d as reference

        from mozo.vendors.birefnet_deploy.graph import deform_conv2d

        torch.manual_seed(0)
        for k in (1, 3, 7):
            x, w = torch.randn(1, 8, 20, 24), torch.randn(16, 8, k, k)
            offset, mask = torch.randn(1, 2 * k * k, 20, 24) * 3, torch.rand(1, k * k, 20, 24) * 2
            want = reference(x, offset, w, None, padding=k // 2, mask=mask)
            got = deform_conv2d(x, offset, w, None, k // 2, mask)
            assert (got - want).abs().max() <= 1e-5 * want.abs().max()

    def test_the_swap_is_undone(self):
        from mozo.vendors.birefnet_deploy.blocks import DeformableConv2d
        from mozo.vendors.birefnet_deploy.graph import decomposed

        original = DeformableConv2d.deform
        with pytest.raises(RuntimeError), decomposed():
            assert DeformableConv2d.deform is not original
            raise RuntimeError
        assert DeformableConv2d.deform is original

    def test_tensorrt_is_derived_from_the_published_graphs(self):
        from mozo.registry import DERIVES

        assert DERIVES[FAMILY] == {"torch-fp16": "torch-fp32", "tensorrt-fp32": "onnx-fp32",
                                   "tensorrt-fp16": "onnx-fp16"}

    def test_onnx_on_the_cpu_agrees_with_torch(self, alpha, image):
        pytest.importorskip("onnxruntime")
        from mozo.adapters.birefnet import BiRefNetPredictor

        if "onnx-fp32" not in published(FAMILY, "lite"):
            pytest.skip("lite publishes no onnx-fp32 here")
        try:
            graph = BiRefNetPredictor("lite", device="cpu", runtime="onnx-fp32")
        except WeightsError as error:
            pytest.skip(f"{FAMILY}/lite onnx-fp32 unavailable: {error}")
        got = graph.predict(image)
        assert got.shape == alpha.shape
        assert np.abs(got.astype(int) - alpha.astype(int)).max() <= 2
