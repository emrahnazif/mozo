"""Does DINOv2 actually work.

Unlike the module tests, these load a real checkpoint and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_dinov2.py -q

mozo's first image-only embedder, so the promises tested are the ones an index depends on: one
L2-normalised row per image, in order, and a vector that does not depend on which other images it
was encoded beside.

``tools/verify/dinov2.py`` is the bit-exact comparison against the authors' torch-hub code; it
needs the reference checkout and does not run here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import FIXTURE, published, require_weights
from mozo.registry import ENCODE_ONLY, ENCODES, MODEL_REGISTRY
from mozo.vendors.dinov2_deploy import SPECS, VARIANTS

FAMILY = "dinov2"


@pytest.fixture(scope="module")
def predictor():
    """The small variant on the CPU, built once for the file."""
    require_weights(FAMILY, "small")
    from mozo.adapters.dinov2 import Dinov2Predictor
    from mozo.weights import WeightsError

    try:
        return Dinov2Predictor("small", device="cpu", runtime="torch-fp32")
    except WeightsError as error:
        pytest.skip(str(error))


# --- the registry and the adapter must agree ---


def test_registry_agrees_with_the_adapter():
    from mozo.adapters.dinov2 import Dinov2Predictor

    assert MODEL_REGISTRY[FAMILY]["variants"] == Dinov2Predictor.VARIANTS


def test_it_embeds_images_and_nothing_else():
    """``/encode`` refuses text before the download, and ``/predict`` refuses the family."""
    assert ENCODES[FAMILY] == frozenset({"image"})
    assert MODEL_REGISTRY[FAMILY]["task_type"] == "image_embedding"
    assert "image_embedding" in ENCODE_ONLY


def test_the_adapter_has_no_predict_and_no_text():
    from mozo.adapters.dinov2 import Dinov2Predictor

    assert not hasattr(Dinov2Predictor, "predict")
    assert not hasattr(Dinov2Predictor, "encode_text")


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_published_variant_ships_its_terms(variant):
    from mozo.weights import companions

    if not published(FAMILY, variant):
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert "LICENSE" in companions(FAMILY, variant)
    assert "NOTICE" in companions(FAMILY, variant)


# --- the geometry, which needs no weights ---


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_head_is_64_wide(variant):
    spec = SPECS[variant]
    assert spec.width == spec.heads * 64


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_state_dict_has_the_shape_the_authors_publish(variant):
    """Key counts of the authors' ``*_pretrain.pth``: blocks, registers and ``mask_token``."""
    from mozo.vendors.dinov2_deploy.checkpoint import build

    with torch.device("meta"):
        state = build(SPECS[variant]).state_dict()
    blocks = SPECS[variant].depth
    assert len(state) == 7 + 14 * blocks + (1 if SPECS[variant].registers else 0)
    assert state["pos_embed"].shape == (1, 1 + 37 * 37, SPECS[variant].width)
    assert "mask_token" in state


def test_the_giant_runs_swiglu_at_upstreams_width():
    """Two thirds of 4 x 1536, rounded up to a multiple of 8: 4096, and w12 holds both halves."""
    from mozo.vendors.dinov2_deploy.checkpoint import build

    with torch.device("meta"):
        mlp = build(SPECS["giant"]).blocks[0].mlp
    assert tuple(mlp.w12.weight.shape) == (2 * 4096, 1536)
    assert tuple(mlp.w3.weight.shape) == (1536, 4096)


def test_registers_change_the_position_interpolation_too():
    """Upstream's ``*_reg`` entries switch to an explicit size with antialiasing. Not derivable."""
    for variant in ("small", "base", "large", "giant"):
        assert SPECS[variant].interpolate_offset == 0.1
        assert not SPECS[variant].interpolate_antialias
        assert SPECS[variant].registers == 0
    for variant in ("small-reg", "base-reg"):
        assert SPECS[variant].interpolate_offset == 0.0
        assert SPECS[variant].interpolate_antialias
        assert SPECS[variant].registers == 4


# --- the preprocessing, which needs no weights ---


def test_the_crop_is_centred_the_way_torchvision_rounds():
    """A 343x256 image keeps 343 wide, and the 119-pixel margin starts the crop at column 60.

    ``(343 - 224) // 2`` would be 59 -- what ``transformers``' processor does, a pixel to the left.
    Each column is painted its own index, so the first column read back says where the crop began.
    """
    from mozo.vendors.dinov2_deploy.config import MEAN, STD
    from mozo.vendors.dinov2_deploy.image import preprocess

    columns = np.tile(np.arange(343, dtype=np.uint8)[None, :, None], (256, 1, 3))
    pixels = preprocess(columns)
    first = float(pixels[0, 100, 0]) * STD[0] + MEAN[0]
    assert pixels.shape == (3, 224, 224)
    assert round(first * 255) == 60


def test_a_tiny_image_is_scaled_up_rather_than_refused():
    from mozo.vendors.dinov2_deploy.image import preprocess

    assert preprocess(np.zeros((1, 1, 3), np.uint8)).shape == (3, 224, 224)


# --- the model ---


def test_a_single_image_comes_back_two_dimensional(predictor):
    assert predictor.encode_image(str(FIXTURE)).shape == (1, 384)


def test_vectors_leave_normalised(predictor):
    vectors = predictor.encode_image([str(FIXTURE), np.zeros((50, 80, 3), np.uint8)])
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


def test_order_is_kept_and_a_vector_does_not_depend_on_its_neighbours(predictor):
    rng = np.random.default_rng(0)
    images = [str(FIXTURE), rng.integers(0, 256, (300, 200, 3), dtype=np.uint8), str(FIXTURE)]
    together = predictor.encode_image(images)
    alone = np.concatenate([predictor.encode_image(image) for image in images])
    assert np.array_equal(together, alone)
    assert np.array_equal(together[0], together[2])


def test_a_mirrored_photograph_lands_closer_than_noise(predictor):
    from mozo.image import load_image

    photo = load_image(str(FIXTURE))
    noise = np.random.default_rng(0).integers(0, 256, photo.shape, dtype=np.uint8)
    original, mirrored, random = predictor.encode_image(
        [photo, np.ascontiguousarray(photo[:, ::-1]), noise])
    assert original @ mirrored > 0.8 > original @ random


def test_it_refuses_nothing(predictor):
    with pytest.raises(ValueError, match="at least one"):
        predictor.encode_image([])


def test_it_says_which_revision_produced_the_vectors(predictor):
    assert predictor.revision


def test_it_refuses_an_unknown_variant():
    from mozo.adapters.dinov2 import Dinov2Predictor

    with pytest.raises(ValueError, match="Unsupported variant"):
        Dinov2Predictor("huge", device="cpu")


# --- the published graphs ---


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_but_giant_publishes_its_graphs(variant):
    """``giant`` is past ONNX's 2 GB limit as one file. Another graph means ``EXECUTES`` changes."""
    from mozo.adapters.dinov2 import Dinov2Predictor
    from mozo.registry import DERIVES

    keys = published(FAMILY, variant)
    if not keys:
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    graphs = [] if variant == "giant" else ["onnx-fp16", "onnx-fp32"]
    assert sorted(keys) == graphs + ["torch-fp32"]
    assert Dinov2Predictor.EXECUTES == ("torch", "onnx", "tensorrt")
    assert DERIVES[FAMILY] == {"tensorrt-fp32": "onnx-fp32", "tensorrt-fp16": "onnx-fp16"}


def test_the_onnx_graph_agrees_with_torch(predictor):
    """Not bit-identical -- no graph runtime is -- but 5e-07 at worst when measured."""
    pytest.importorskip("onnxruntime")
    require_weights(FAMILY, "small", "onnx-fp32")
    from mozo.adapters.dinov2 import Dinov2Predictor

    graph = Dinov2Predictor("small", device="cpu", runtime="onnx-fp32")
    images = [str(FIXTURE), np.zeros((300, 200, 3), np.uint8)]
    np.testing.assert_allclose(graph.encode_image(images), predictor.encode_image(images),
                               atol=1e-5)


def test_tensorrt_fp32_agrees_with_torch(predictor):
    """The engine ``auto`` picks on CUDA: within 2e-05 of torch on 500 COCO images when measured."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("tensorrt")
    require_weights(FAMILY, "small", "onnx-fp32")
    from mozo.adapters.dinov2 import Dinov2Predictor

    engine = Dinov2Predictor("small", device="cuda")
    assert engine.runtime == "tensorrt-fp32"
    images = [str(FIXTURE), np.zeros((300, 200, 3), np.uint8)]
    np.testing.assert_allclose(engine.encode_image(images), predictor.encode_image(images),
                               atol=1e-4)
