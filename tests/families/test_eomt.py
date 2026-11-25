"""Does EoMT actually work.

Unlike the module tests, these load a real checkpoint and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_eomt.py -q

mozo's first scene segmentation, so the promises tested are the ones its result shape makes: one
detection per segment, masks at the picture's size and never overlapping, every pixel covered for a
semantic variant, and a box that is the mask's own.

``tools/verify/eomt.py`` is the bit-exact comparison against the authors' code; it needs the
reference checkout and does not run here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import FIXTURE, published, require_weights
from mozo.registry import MODEL_REGISTRY, NEEDS
from mozo.vendors.eomt_deploy import SPECS, VARIANTS

FAMILY = "eomt"


def _model(variant: str):
    require_weights(FAMILY, variant)
    from mozo.adapters.eomt import EomtPredictor
    from mozo.weights import WeightsError

    try:
        return EomtPredictor(variant, device="cpu", runtime="torch-fp32")
    except WeightsError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def panoptic():
    """The small panoptic variant on the CPU, built once for the file."""
    return _model("coco-panoptic-small")


@pytest.fixture(scope="module")
def semantic():
    """The ADE20K semantic variant on the CPU, built once for the file."""
    return _model("ade20k-semantic-large")


@pytest.fixture(scope="module")
def picture():
    from mozo.image import load_image

    return load_image(str(FIXTURE))


# --- the registry and the adapter must agree ---


def test_registry_agrees_with_the_adapter():
    from mozo.adapters.eomt import EomtPredictor

    assert MODEL_REGISTRY[FAMILY]["variants"] == EomtPredictor.VARIANTS


def test_the_task_is_semantic_segmentation():
    assert MODEL_REGISTRY[FAMILY]["task_type"] == "semantic_segmentation"


def test_every_runtime_reads_the_labels():
    """The checkpoint carries no names; the published labels file does."""
    assert NEEDS[FAMILY] == {"*": ("labels",)}


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
def test_the_state_dict_has_the_shape_upstream_builds(variant):
    """Upstream's network keys: 14 per block, 7 around the blocks, the two pixel statistics, the
    masked-attention buffer, the queries, 8 for the two heads and 5 per upscale block; patch 16,
    position embedding on patches only. 197 for small and base, 365 for large."""
    from mozo.vendors.eomt_deploy.network import build

    spec = SPECS[variant]
    with torch.device("meta"):
        state = build(spec).state_dict()
    assert len(state) == 14 * spec.depth + 7 + 2 + 1 + 1 + 8 + 2 * 5
    assert state["encoder.backbone.pos_embed"].shape == (1, spec.grid ** 2, spec.width)
    assert state["encoder.backbone.patch_embed.proj.weight"].shape == (spec.width, 3, 16, 16)
    assert state["encoder.backbone.reg_token"].shape == (1, 4, spec.width)
    assert state["q.weight"].shape == (spec.queries, spec.width)
    assert state["class_head.weight"].shape == (spec.classes + 1, spec.width)
    assert state["attn_mask_probs"].shape == (spec.query_blocks,)


def test_only_coco_panoptic_merges_stuff():
    for variant, spec in SPECS.items():
        assert spec.stuff == (tuple(range(80, 133)) if spec.task == "panoptic" else ())


# --- the pre-processing, which needs no weights ---


def test_panoptic_fits_the_long_side_and_pads_bottom_right():
    from mozo.vendors.eomt_deploy.image import resize_and_pad

    pixels = resize_and_pad(np.full((300, 600, 3), 7, np.uint8), 640)
    assert pixels.shape == (1, 3, 640, 640) and pixels.dtype == torch.uint8
    assert (pixels[0, :, :320] == 7).all() and (pixels[0, :, 320:] == 0).all()


def test_semantic_fits_the_short_side_and_slides_windows():
    """1000x512 after scaling: two 512 windows, their 24-pixel overlap split evenly."""
    from mozo.vendors.eomt_deploy.image import windows

    crops, origins = windows(np.zeros((256, 500, 3), np.uint8), 512)
    assert crops.shape == (2, 3, 512, 512)
    assert origins == [(0, 512), (488, 1000)]


def test_the_resized_size_is_rounded_not_floored():
    """701 x 640/1000 = 448.64: upstream rounds to 449."""
    from mozo.vendors.eomt_deploy.image import pad_size

    assert pad_size((701, 1000), 640) == [449, 640]


# --- the model ---


def test_panoptic_masks_partition_without_overlap(panoptic, picture):
    found = panoptic.predict(picture)
    masks = np.stack([d.masks[0] for d in found])
    assert masks.shape[1:] == picture.shape[:2] and masks.dtype == bool
    assert masks.sum(0).max() == 1
    assert all(0 < d.confidence <= 1 for d in found)


def test_panoptic_finds_the_people_one_by_one_and_stuff_once(panoptic, picture):
    found = panoptic.predict(picture)
    names = [d.class_name for d in found]
    assert names.count("person") >= 2
    stuff = [d.class_id for d in found if d.class_id >= 80]
    assert len(stuff) == len(set(stuff))


def test_the_box_is_the_masks_own(panoptic, picture):
    for d in panoptic.predict(picture):
        rows = np.flatnonzero(d.masks[0].any(axis=1))
        cols = np.flatnonzero(d.masks[0].any(axis=0))
        assert d.bbox == [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


def test_a_higher_threshold_leaves_pixels_void(panoptic, picture):
    loose = np.stack([d.masks[0] for d in panoptic.predict(picture)]).any(0).sum()
    strict = panoptic.predict(picture, threshold=0.99)
    covered = np.stack([d.masks[0] for d in strict]).any(0).sum() if len(strict) else 0
    assert covered < loose


def test_a_black_picture_is_an_answer_not_an_error(panoptic):
    assert len(panoptic.predict(np.zeros((240, 320, 3), np.uint8))) >= 0


def test_semantic_covers_every_pixel_once_with_no_score(semantic, picture):
    found = semantic.predict(picture)
    masks = np.stack([d.masks[0] for d in found])
    assert (masks.sum(0) == 1).all()
    assert all(d.confidence is None for d in found)
    assert [d.class_id for d in found] == sorted({d.class_id for d in found})
    assert "wall" in [d.class_name for d in found]


def test_semantic_refuses_a_threshold(semantic, picture):
    with pytest.raises(ValueError, match="no threshold"):
        semantic.predict(picture, threshold=0.5)


def test_the_answer_serialises(panoptic, picture):
    payload = panoptic.predict(picture).to_dict()
    assert payload and payload[0]["masks"][0]["format"] == "png"


def test_the_same_picture_gives_the_same_answer(panoptic, picture):
    a, b = panoptic.predict(picture), panoptic.predict(picture)
    assert [d.class_id for d in a] == [d.class_id for d in b]
    assert all(np.array_equal(x.masks[0], y.masks[0]) for x, y in zip(a, b))
