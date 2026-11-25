"""HQ-SAM: geometry, the letterbox, the prompt contract, and what multimask output returns.

What HQ-SAM shares with every promptable family -- the adapter's output shape, the server's
prompt parsing -- is in ``test_promptable.py``, checked against all of them.

Bit-exactness against upstream lives in ``tools/verify/sam_hq.py``, which needs a checkout and
cannot run here. What this file pins is what would still be green if the model had quietly
changed: geometry the checkpoint's key names must match, flags that carry no weights, and the
shapes the predictor hands back.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mozo.registry import get_model_info
from mozo.vendors.sam_hq_deploy.config import SHARED, SPECS, VitSpec
from mozo.vendors.sam_hq_deploy.image import (
    preprocess_shape, resize_longest_side, to_model_coords, to_original)
from mozo.vendors.sam_hq_deploy.network import SamHq
from mozo.vendors.sam_hq_deploy.predictor import UNUSED

FAMILY = "sam_hq"


@pytest.fixture(scope="module")
def tiny():
    """The tiny network, built but unloaded. Geometry needs no weights."""
    return SamHq(SPECS["tiny"]).eval()


# --- geometry ----------------------------------------------------------------------------------

def test_every_published_variant_has_a_geometry():
    assert set(get_model_info(FAMILY)["variants"]) <= set(SPECS)


@pytest.mark.parametrize("name", [n for n, s in SPECS.items() if isinstance(s, VitSpec)])
def test_the_vit_global_blocks_end_the_trunk_and_are_evenly_spaced(name):
    """Upstream writes these four indices out per size. The *first* is what HQ-SAM's decoder
    fuses, so a list off by one feeds it a windowed block's output and still runs."""
    spec = SPECS[name]
    assert len(spec.global_attn_indexes) == 4
    assert spec.global_attn_indexes[-1] == spec.depth - 1
    step = spec.depth // 4
    assert spec.global_attn_indexes == tuple(step * (i + 1) - 1 for i in range(4))


def test_the_vit_layernorm_epsilon_is_upstreams_not_torchs():
    """``build_sam`` passes ``eps=1e-6``; torch's default is 1e-5, and neither raises."""
    network = SamHq(SPECS["base"])
    norms = [m for m in network.image_encoder.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert norms and all(m.eps == 1e-6 for m in norms)


def test_the_tiny_trunk_ends_on_sams_grid_and_hands_over_stage_ones_output(tiny):
    """MobileSAM's edit keeps the last merge at stride 1, so the encoder ends at 64x64 like SAM's
    ViT; the early feature is stage 1's output, 160 wide, which is what the decoder's fusion
    layer was trained to take."""
    with torch.no_grad():
        embedding, interm = tiny.image_encoder(torch.zeros(1, 3, 1024, 1024))
    assert embedding.shape == (1, 256, 64, 64)
    assert [t.shape for t in interm] == [(1, 64, 64, SPECS["tiny"].vit_dim)]
    assert tiny.mask_decoder.compress_vit_feat[0].in_channels == SPECS["tiny"].vit_dim == 160


def test_tinyvit_attention_reads_its_biases_at_call_time(tiny):
    """Upstream gathers the biases into a buffer when ``eval()`` runs, and its builder runs
    ``eval()`` before the load -- so a model used straight from the builder attends with zeros.
    Here the gather happens in ``forward``; a cached copy reappearing would bring the trap back."""
    attention = tiny.image_encoder.layers[1].blocks[0].attn
    assert not hasattr(attention, "ab")
    x = torch.randn(1, 49, 128)
    before = attention(x)
    with torch.no_grad():
        attention.attention_biases.add_(1.0)
    try:
        assert not torch.equal(attention(x), before)
    finally:
        with torch.no_grad():
            attention.attention_biases.sub_(1.0)


def test_the_classifier_head_is_neither_built_nor_silently_loaded(tiny):
    """TinyViT's ImageNet head is in the tiny checkpoint and never runs: it is dropped by name,
    so the load stays strict about everything else."""
    names = {name for name, _ in tiny.named_parameters()}
    assert not any(name.startswith(prefix) for prefix in UNUSED for name in names)
    assert UNUSED == ("image_encoder.norm_head.", "image_encoder.head.")


def test_the_pixel_statistics_are_on_the_0_255_scale():
    assert SHARED["pixel_mean"] == (123.675, 116.28, 103.53)
    assert SHARED["pixel_std"] == (58.395, 57.12, 57.375)


# --- the letterbox -----------------------------------------------------------------------------

@pytest.mark.parametrize("height,width,expected", [
    (1281, 1920, (683, 1024)), (1920, 1281, (1024, 683)), (640, 640, (1024, 1024)),
    (1025, 2048, (513, 1024)),  # 512.5 rounds up, as upstream's int(x + 0.5) does
    (100, 3000, (34, 1024)),
])
def test_the_long_side_becomes_1024_and_the_short_side_keeps_the_aspect(height, width, expected):
    assert preprocess_shape(height, width, 1024) == expected
    resized = resize_longest_side(np.zeros((height, width, 3), np.uint8), 1024)
    assert resized.shape == (1, 3, *expected) and resized.dtype == torch.uint8


def test_the_encoder_pads_bottom_right_to_a_square(tiny):
    """HQ-SAM letterboxes where SAM 2 squashes; the pad is zero *after* normalising."""
    seen = {}
    hook = tiny.image_encoder.register_forward_pre_hook(lambda m, a: seen.setdefault("x", a[0]))
    try:
        with torch.no_grad():
            tiny.encode(torch.full((1, 3, 683, 1024), 255, dtype=torch.uint8))
    finally:
        hook.remove()
    x = seen["x"]
    assert x.shape == (1, 3, 1024, 1024)
    assert torch.all(x[:, :, 683:] == 0) and torch.all(x[:, :, :683] > 0)


def test_preprocessing_refuses_anything_that_is_not_rgb_uint8():
    for bad in (np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8),
                np.zeros((10, 10, 3), np.float32)):
        with pytest.raises(ValueError):
            resize_longest_side(bad, 1024)


def test_prompts_scale_by_each_axis_rounded_ratio():
    """x by new_w / old_w and y by new_h / old_h, from the *rounded* sizes, in float64."""
    coords = to_model_coords(np.array([[[1920.0, 1281.0]]]), (1281, 1920), 1024)
    assert coords.dtype == torch.float32
    assert coords[0, 0].tolist() == [1024.0, 683.0]


def test_masks_come_back_at_the_source_size_with_the_padding_cropped():
    """A logit map that is positive only inside the padded band must vanish after the crop."""
    low = torch.full((1, 1, 256, 256), -10.0)
    low[..., 200:, :] = 10.0  # rows 800..1023 of the square: all padding for a 683-high image
    full = to_original(low, (683, 1024), (1281, 1920), 1024)
    assert full.shape == (1, 1, 1281, 1920)
    assert (full > 0).sum() == 0


# --- the decoder's answer ----------------------------------------------------------------------

def _features(tiny):
    with torch.no_grad():
        return tiny.encode(torch.zeros((1, 3, 683, 1024), dtype=torch.uint8))


@pytest.mark.parametrize("multimask, chosen", [(True, slice(1, 4)), (False, slice(0, 1))])
def test_each_mask_is_a_sam_token_plus_the_hq_token(tiny, multimask, chosen):
    """Upstream keeps the best of tokens 1..3 for multimask output, token 0 otherwise; mozo keeps
    all of 1..3. Each must be that token's SAM mask plus the HQ token's, so the best of them is
    upstream's answer exactly."""
    features = _features(tiny)
    box = torch.tensor([[100.0, 100.0, 400.0, 500.0]])
    with torch.no_grad():
        low, iou = tiny.decode(features, None, box, None, multimask_output=multimask)
        sparse, dense = tiny.prompt_encoder(points=None, boxes=box, masks=None)
        sam, hq, raw_iou = tiny.mask_decoder(features["image_embed"],
                                             tiny.prompt_encoder.get_dense_pe(), sparse, dense,
                                             features["hq_features"])
    assert low.shape == (1, chosen.stop - chosen.start, 256, 256)
    assert torch.equal(low, sam[:, chosen] + hq)
    assert torch.equal(iou, raw_iou[:, chosen])


def test_a_box_is_a_box_not_two_points(tiny):
    """Unlike SAM 2, SAM keeps a box input: a box alone is two tokens (no padding token), and
    clicks alone get the padding token appended."""
    encoder = tiny.prompt_encoder
    box, _ = encoder(points=None, boxes=torch.tensor([[10.0, 20.0, 300.0, 400.0]]), masks=None)
    assert box.shape[1] == 2
    clicks, _ = encoder(points=(torch.tensor([[[50.0, 60.0]]]), torch.tensor([[1]])),
                        boxes=None, masks=None)
    assert clicks.shape[1] == 2
    assert torch.equal(clicks[0, 1], encoder.not_a_point_embed.weight[0])


# --- the registry and the adapter --------------------------------------------------------------

def test_registry_agrees_with_the_adapter():
    """The variant list is written twice -- here and in the adapter -- so that answering "what
    exists" needs no torch import. This is what holds the two copies in step."""
    from mozo.adapters.sam_hq import SamHqPredictor

    entry = get_model_info(FAMILY)
    assert entry["adapter_class"] == "SamHqPredictor"
    assert entry["module"] == "mozo.adapters.sam_hq"
    assert entry["task_type"] == "promptable_segmentation"
    assert list(entry["variants"]) == list(SamHqPredictor.VARIANTS)

