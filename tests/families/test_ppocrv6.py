"""PP-OCRv6: the networks' shapes, the pipeline's traps, the polygon offset and the CTC alphabet.

Bit-exactness against PaddleOCR lives in ``tools/verify/ppocrv6.py``, which needs the reference
installed. What this file pins is what would stay green if the model had quietly changed: the
alphabet layout, the preprocessing arithmetic, and the Clipper offset mozo carries instead of
depending on pyclipper.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from conftest import TEXT_FIXTURES, imported, require_weights
from mozo.registry import get_model_info
from mozo.vendors.ppocrv6_deploy import boxes, image
from mozo.vendors.ppocrv6_deploy.config import SPECS, VARIANTS
from mozo.vendors.ppocrv6_deploy.det import TextDetector
from mozo.vendors.ppocrv6_deploy.rec import TextRecogniser
from mozo.vendors.ppocrv6_deploy.text import Alphabet

FAMILY = "ppocrv6"


# --- the alphabet -------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_the_alphabet_is_blank_then_the_dictionary_then_a_space(variant):
    """The head is exactly this wide; the space is appended, not in the published dictionary."""
    spec = SPECS[variant].recogniser
    alphabet = Alphabet(spec.charset)
    assert len(alphabet) == spec.classes
    assert alphabet.symbols[0] == "blank" and alphabet.symbols[-1] == " "
    assert len(set(alphabet.symbols)) == len(alphabet.symbols)


def test_repeats_collapse_but_a_blank_between_them_does_not():
    alphabet = Alphabet("ppocrv6-tiny.json")
    a = alphabet.symbols.index("a")
    probs = np.full((2, 3, len(alphabet)), 1e-6, dtype=np.float32)
    probs[0, :, a] = 0.9                       # a a a
    probs[1, 0, a] = probs[1, 2, a] = 0.9      # a _ a
    probs[1, 1, 0] = 0.9
    assert [text for text, _ in alphabet.decode(probs)] == ["a", "aa"]


def test_confidence_is_the_mean_of_the_kept_steps_and_zero_for_nothing():
    alphabet = Alphabet("ppocrv6-tiny.json")
    probs = np.full((2, 4, len(alphabet)), 1e-6, dtype=np.float32)
    probs[0, 0, 5], probs[0, 1, 5], probs[0, 2, 0], probs[0, 3, 6] = 0.8, 0.6, 0.99, 0.5
    probs[1, :, 0] = 0.99
    (_, first), (empty, second) = alphabet.decode(probs)
    # The repeated 0.6 step is dropped with its character; the blank step never counts.
    assert first == pytest.approx((0.8 + 0.5) / 2)
    assert (empty, second) == ("", 0.0)


# --- preprocessing ------------------------------------------------------------------------------

def test_the_page_is_scaled_up_only_when_its_short_side_is_under_64():
    small, record = image.resize_for_detector(np.zeros((40, 100, 3), np.uint8))
    assert small.shape[:2] == (64, 160) and record.tolist()[:2] == [40, 100]
    page, record = image.resize_for_detector(np.zeros((736, 1280, 3), np.uint8))
    assert page.shape[:2] == (736, 1280) and record.tolist()[2:] == [1.0, 1.0]


def test_both_sides_round_to_a_multiple_of_32():
    page, _ = image.resize_for_detector(np.zeros((300, 500, 3), np.uint8))
    assert page.shape[:2] == (288, 512)


def test_normalisation_applies_the_imagenet_red_statistics_to_the_blue_channel():
    """Upstream normalises its BGR array with the RGB-ordered ImageNet mean: channel 0 gets
    0.485. That is what the networks were trained on, so it is kept."""
    out = image.normalize_for_detector(np.zeros((32, 32, 3), np.uint8))
    assert out.shape == (3, 32, 32)
    assert out[0, 0, 0] == np.float32(-0.485 / 0.229)


def test_a_line_keeps_its_aspect_and_pads_with_zeros_to_at_least_320():
    out = image.resize_for_recogniser(np.full((24, 40, 3), 255, np.uint8))
    assert out.shape == (3, 48, 320)
    assert (out[:, :, :80] == 1.0).all() and (out[:, :, 80:] == 0.0).all()


def test_a_batch_pads_every_line_to_the_widest():
    batch = image.batch_for_recogniser([np.ones((3, 48, 320), np.float32),
                                        np.ones((3, 48, 400), np.float32)])
    assert batch.shape == (2, 3, 48, 400) and (batch[0, :, :, 320:] == 0).all()


def test_a_tall_crop_is_turned_to_read_left_to_right():
    page = np.zeros((200, 200, 3), np.uint8)
    crop = image.crop_line(page, np.array([[50, 10], [70, 10], [70, 150], [50, 150]]))
    assert crop.shape[1] > crop.shape[0]


# --- postprocessing -----------------------------------------------------------------------------

def test_the_offset_matches_clippers_integer_output_on_a_known_box():
    """pyclipper's answer for this box at distance 2, recorded: round joins of three points,
    collinear points dropped, ending at the lowest-y vertex with the largest x."""
    box = np.array([[0.9, 0.9], [10.7, -0.6], [10.5, 5.5], [-0.5, 5.5]], dtype=np.float32)
    assert boxes._union(boxes._offset(box, 2.0)) == [
        (12, -1), (12, 5), (11, 7), (0, 7), (-2, 6), (-2, 0), (-1, -2), (10, -2)]


def test_unclip_returns_one_int64_polygon_as_pyclipper_does():
    box = np.array([[10, 10], [110, 10], [110, 40], [10, 40]], dtype=np.float32)
    grown = boxes.unclip(box, boxes.UNCLIP_RATIO)
    assert grown.shape[0] == 1 and grown.dtype == np.int64
    x, y = grown[0][:, 0], grown[0][:, 1]
    assert x.min() < 10 and x.max() > 110 and y.min() < 10 and y.max() > 40


def test_no_pyclipper_or_shapely_import_anywhere_in_the_vendor():
    """Upstream's DB post-processing needs pyclipper; this one carries the offset instead."""
    from pathlib import Path

    import mozo.vendors.ppocrv6_deploy as package

    root = Path(package.__file__).parent.parent
    for source in Path(package.__file__).parent.glob("*.py"):
        names = imported(source, root, "mozo.vendors")
        assert not any(n.split(".")[0] in ("pyclipper", "shapely") for n in names), source.name


def test_a_blob_on_the_map_becomes_one_quad_in_page_pixels():
    prob = np.zeros((64, 128), np.float32)
    prob[20:40, 10:100] = 0.9
    quads, scores = boxes.quads(prob, np.array([32, 64, 2.0, 2.0]))
    assert len(quads) == 1 and quads.dtype == np.int16
    assert scores[0] == pytest.approx(0.9)
    assert quads[0][:, 0].max() <= 64 and quads[0][:, 1].max() <= 32


def test_lines_sort_top_to_bottom_then_left_to_right_within_ten_pixels():
    a = np.array([[100, 12], [150, 12], [150, 30], [100, 30]])
    b = np.array([[10, 15], [50, 15], [50, 30], [10, 30]])
    c = np.array([[10, 60], [50, 60], [50, 80], [10, 80]])
    assert [q[0][0] for q in boxes.sort_quads(np.array([c, a, b]))] == [10, 100, 10]


# --- the networks -------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_the_detector_answers_at_the_input_resolution(variant):
    with torch.no_grad():
        out = TextDetector(SPECS[variant].detector).eval()(torch.zeros(1, 3, 64, 96))
    assert out.shape == (1, 1, 64, 96)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_recogniser_reads_one_step_per_eight_pixels(variant):
    spec = SPECS[variant].recogniser
    with torch.no_grad():
        out = TextRecogniser(spec).eval()(torch.zeros(2, 3, 48, 320))
    assert out.shape == (2, 40, spec.classes)
    assert torch.allclose(out.sum(-1), torch.ones(2, 40))


# --- the registry and the adapter ---------------------------------------------------------------

def test_registry_agrees_with_the_adapter():
    from mozo.adapters.ppocrv6 import PPOCRv6Predictor

    entry = get_model_info(FAMILY)
    assert entry["adapter_class"] == "PPOCRv6Predictor"
    assert entry["module"] == "mozo.adapters.ppocrv6"
    assert entry["task_type"] == "text_recognition"
    assert set(entry["variants"]) == set(PPOCRv6Predictor.VARIANTS) == set(SPECS)


def test_a_variant_that_does_not_exist_is_refused_before_any_download():
    from mozo.adapters.ppocrv6 import PPOCRv6Predictor

    with pytest.raises(ValueError, match="Unsupported variant"):
        PPOCRv6Predictor("huge")


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_ships_its_terms(variant):
    from mozo.weights import companions, license_of

    assert {"LICENSE", "NOTICE"} <= set(companions(FAMILY, variant))
    assert license_of(FAMILY, variant) == "Apache-2.0"


# --- with weights ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny():
    """The smallest checkpoint (6.4 MB); every variant goes through the same pipeline code."""
    require_weights(FAMILY, "tiny")
    from mozo.adapters.ppocrv6 import PPOCRv6Predictor

    return PPOCRv6Predictor("tiny", device="cpu", runtime="torch-fp32")


def test_a_read_carries_text_and_no_class_name(tiny):
    found = tiny.predict(TEXT_FIXTURES / "printed.png")
    assert [d.text for d in found][:2] == ["Hello World", "OCR 12345"]
    assert all(d.class_name is None and d.class_id is None for d in found)


def test_the_quad_survives_and_the_hull_contains_it(tiny):
    found = tiny.predict(TEXT_FIXTURES / "rotated.png")
    assert len(found) >= 1
    corners = np.asarray(found[0].segments, dtype=float)
    assert corners.shape == (4, 2) and corners[0][1] != corners[1][1]
    x1, y1, x2, y2 = found[0].bbox
    assert x1 <= corners[:, 0].min() and corners[:, 0].max() <= x2


def test_an_image_with_no_text_returns_nothing(tiny):
    assert len(tiny.predict(TEXT_FIXTURES / "blank.png")) == 0


def test_text_and_segments_survive_to_dict(tiny):
    payload = tiny.predict(TEXT_FIXTURES / "sign.png").to_dict()
    entry = payload[0] if isinstance(payload, list) else payload
    assert entry["text"] == "EXIT 42"
    assert len(entry["segments"]) == 4 and entry["class_name"] is None


def test_reading_the_same_page_twice_gives_the_same_answer(tiny):
    first = tiny.predict(TEXT_FIXTURES / "wide.png")
    second = tiny.predict(TEXT_FIXTURES / "wide.png")
    assert [(d.text, d.confidence) for d in first] == [(d.text, d.confidence) for d in second]


def test_rgb_in_matches_a_path_in(tiny):
    """The adapter takes RGB and the pipeline works in BGR; the flip happens once."""
    rgb = cv2.cvtColor(cv2.imread(str(TEXT_FIXTURES / "mixed.png")), cv2.COLOR_BGR2RGB)
    assert ([d.text for d in tiny.predict(rgb)]
            == [d.text for d in tiny.predict(TEXT_FIXTURES / "mixed.png")])
