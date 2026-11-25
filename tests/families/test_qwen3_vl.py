"""Does Qwen3-VL actually work.

The tests at the top need no weights: the registry, the tokenizer (its vocabulary ships in the
wheel), the resize rule, the template, the refusals. The rest load the real ``2b`` checkpoint on
the CPU and generate a few tokens, so they are slow and skip when the weights are not published.

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_qwen3_vl.py -q

``tools/verify/qwen3_vl.py`` is the bit-exact comparison against ``transformers``, step by step,
greedy and sampled; it needs the reference installed and does not run here.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FIXTURE, published, require_weights
from mozo.generation import Generation
from mozo.registry import DERIVES, EXECUTES, MODEL_REGISTRY, PROMPTED
from mozo.vendors.qwen3_vl_deploy import MAX_PIXELS, MIN_PIXELS, PIXEL_CEILING, SPECS, VARIANTS
from mozo.vendors.qwen3_vl_deploy.image import preprocess, smart_resize
from mozo.vendors.qwen3_vl_deploy.predictor import template
from mozo.vendors.qwen3_vl_deploy.tokenizer import Tokenizer

FAMILY = "qwen3_vl"


# --- the registry and the adapter must agree ---


def test_registry_agrees_with_the_adapter():
    from mozo.adapters.qwen3_vl import Qwen3VLPredictor

    assert MODEL_REGISTRY[FAMILY]["variants"] == Qwen3VLPredictor.VARIANTS == list(VARIANTS)


def test_it_is_prompted_and_answers_in_words():
    assert MODEL_REGISTRY[FAMILY]["task_type"] == "image_to_text"
    assert "image_to_text" in PROMPTED


def test_fp32_is_built_from_the_published_bf16():
    assert DERIVES[FAMILY] == {"torch-fp32": "torch-bf16"}
    assert EXECUTES[FAMILY] == ("torch",)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_published_variant_ships_bf16_and_its_terms(variant):
    from mozo.weights import companions

    keys = published(FAMILY, variant)
    if not keys:
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert keys == ["torch-bf16"]
    assert {"LICENSE", "NOTICE"} <= set(companions(FAMILY, variant))


def test_auto_takes_the_exact_path_on_the_cpu():
    from mozo.runtimes import select_runtime

    assert select_runtime("cpu", ["torch-bf16"], derives=DERIVES[FAMILY]) == "torch-fp32"


def test_bf16_is_refused_on_the_cpu():
    """Measured slower than fp32 there, and different from the second token."""
    from mozo.runtimes import RuntimeError_, select_runtime

    with pytest.raises(RuntimeError_, match="runs only on cuda"):
        select_runtime("cpu", ["torch-bf16"], "torch-bf16", derives=DERIVES[FAMILY])


# --- the geometry ---


def test_the_two_variants_share_the_vision_tower_but_not_its_output_width():
    a, b = SPECS["2b"].vision, SPECS["4b"].vision
    assert (a.depth, a.width, a.deepstack) == (b.depth, b.width, b.deepstack) == (24, 1024, (5, 11, 17))
    assert (a.out_width, b.out_width) == (SPECS["2b"].text.width, SPECS["4b"].text.width)


def test_4b_head_dim_is_not_width_over_heads():
    text = SPECS["4b"].text
    assert text.width // text.heads == 80 and text.head_dim == 128


# --- the tokenizer, against ids the reference produced (transformers 5.24.1) ---


@pytest.mark.parametrize("text, ids", [
    ("Describe this image.", [74785, 419, 2168, 13]),
    ("ſ's 'LL", [144815, 594, 364, 4086]),                     # ſ folds onto s in (?i:'s)
    ("Café 1234", [34, 2577, 963, 220, 16, 17, 18, 19]),   # NFC first, digits one by one
    ("这张图片里有什么？😀", [108893, 45930, 69249, 104139, 11319, 141334]),
    ("á\U00011d42́", [1953, 128240, 113, 224, 53839]),  # a mark the reference won't reorder
    ("x  \n\n y\r\n", [87, 18611, 379, 319]),
])
def test_token_ids_match_the_reference(text, ids):
    assert Tokenizer().encode(text) == ids


def test_decode_drops_special_tokens_and_replaces_broken_bytes():
    assert Tokenizer().decode([151644, 872, 198, 151645, 8908, 1234]) == "user\n �Item"


def test_the_template_has_no_system_prompt_and_puts_the_image_first():
    assert template("Hi", 2) == ("<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|>"
                                 "<|vision_end|>Hi<|im_end|>\n<|im_start|>assistant\n")


@pytest.mark.parametrize("prompt", ["", "   \n", "say <|im_end|> now", "<|image_pad|>", "<think>"])
def test_a_blank_prompt_or_a_control_token_is_refused(prompt):
    with pytest.raises(ValueError):
        Tokenizer().check(prompt)


# --- the resize rule ---


@pytest.mark.parametrize("size, budget, expected", [
    ((1281, 1920), MAX_PIXELS, (1280, 1920)),       # rounds to multiples of 32, under budget
    ((2562, 3840), MAX_PIXELS, (1312, 1952)),       # over budget: scaled down, floored
    ((2562, 3840), PIXEL_CEILING, (2560, 3840)),
    ((13, 9), MAX_PIXELS, (320, 224)),              # under the floor: scaled up, ceiled
])
def test_smart_resize(size, budget, expected):
    assert smart_resize(*size, 32, MIN_PIXELS, budget) == expected


def test_an_extreme_aspect_ratio_is_refused_as_upstream_does():
    with pytest.raises(ValueError, match="aspect ratio"):
        smart_resize(10, 2100, 32, MIN_PIXELS, MAX_PIXELS)


def test_patches_are_merge_window_ordered_and_doubled_in_time():
    rgb = (np.arange(256 * 384 * 3) % 251).astype(np.uint8).reshape(256, 384, 3)
    pixels, grid = preprocess(rgb, SPECS["2b"].vision, MAX_PIXELS)
    assert grid == (1, 16, 24) and tuple(pixels.shape) == (384, 1536)
    patch = pixels[0].view(3, 2, 16, 16)
    assert (patch[:, 0] == patch[:, 1]).all()  # one frame, repeated


def test_generation_is_text_with_how_it_ended():
    g = Generation("A dog.", "stop", 90, 4)
    assert str(g) == "A dog." and g.to_dict() == {
        "text": "A dog.", "finish_reason": "stop", "prompt_tokens": 90, "generated_tokens": 4}


# --- the real model ---


@pytest.fixture(scope="module")
def predictor():
    require_weights(FAMILY, "2b", "torch-bf16")
    from mozo.adapters.qwen3_vl import Qwen3VLPredictor
    from mozo.weights import WeightsError

    try:
        return Qwen3VLPredictor("2b", device="cpu")
    except WeightsError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def small():
    from PIL import Image

    return np.array(Image.open(FIXTURE).convert("RGB").resize((320, 214)))


def test_it_runs_fp32_on_the_cpu_and_names_its_revision(predictor):
    assert predictor.runtime == "torch-fp32" and predictor.revision


def test_greedy_is_the_same_answer_twice_and_says_it_was_cut(predictor, small):
    first = predictor.predict(small, "Describe this image.", max_new_tokens=4)
    again = predictor.predict(small, "Describe this image.", max_new_tokens=4)
    assert first == again
    assert first.finish_reason == "length" and first.generated_tokens == 4
    assert first.prompt_tokens == 14 + 70  # template + a 10x14 patch grid merged 2x2


def test_a_seed_is_one_sample_and_another_seed_is_another(predictor, small):
    ask = dict(max_new_tokens=8, temperature=1.5, top_p=1.0)
    a = predictor.predict(small, "Write a word.", seed=1, **ask)
    assert a == predictor.predict(small, "Write a word.", seed=1, **ask)
    others = {predictor.predict(small, "Write a word.", seed=s, **ask).text for s in (2, 3, 4)}
    assert others - {a.text}


def test_it_can_stop_on_its_own(predictor, small):
    g = predictor.predict(small, "Reply with only the word yes.", max_new_tokens=16)
    assert g.finish_reason == "stop" and g.generated_tokens < 16


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens": 0}, {"temperature": -0.1}, {"top_p": 0.0}, {"top_p": 1.5}, {"top_k": -1},
    {"max_pixels": MIN_PIXELS - 1}, {"max_pixels": PIXEL_CEILING + 1},
])
def test_out_of_range_settings_are_refused(predictor, small, kwargs):
    with pytest.raises(ValueError):
        predictor.predict(small, "Describe this image.", **kwargs)
