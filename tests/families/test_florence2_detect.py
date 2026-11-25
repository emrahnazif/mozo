"""Florence-2 detection: the vocabulary, the prompts, the decoding rules and the box arithmetic.

Bit-exactness against Microsoft's code lives in ``tools/verify/florence2_detect.py``, which needs
the reference installed. What this file pins is what would stay green if the model had quietly
changed: that the task prompts are upstream's sentences, that location tokens become pixel
coordinates the way upstream's quantiser does it, that the decoding rules are the ones the
checkpoints' configs turn on, and what the adapter returns and refuses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from conftest import FIXTURE, require_present
from mozo.registry import DERIVES, EXECUTES, get_model_info
from mozo.vendors.florence2_deploy.config import DECODING, SPECS
from mozo.vendors.florence2_deploy.decoding import _no_repeat_ngram, _process
from mozo.vendors.florence2_deploy.network import Florence2
from mozo.vendors.florence2_deploy.parse import parse_ocr, parse_od
from mozo.vendors.florence2_deploy.tokenizer import PROMPTS, Vocabulary

FAMILY = "florence2_detect"


@pytest.fixture(scope="module")
def vocabulary():
    return Vocabulary()


# --- the vocabulary and the prompts ---------------------------------------------------------------

def test_the_vocabulary_is_barts_plus_the_processors_tokens(vocabulary):
    """50,265 BPE pieces, then the 1,024 tokens ``Florence2Processor`` adds -- ``<loc_0>`` at
    50,269 -- every one of them special, so decoding keeps them verbatim."""
    assert len(vocabulary.pieces) == 51289 == SPECS["base"].language.vocab
    assert vocabulary.pieces[50269] == "<loc_0>" and vocabulary.pieces[51268] == "<loc_999>"
    assert vocabulary.pieces[:4] == ["<s>", "<pad>", "</s>", "<unk>"]
    assert set(range(50265, 51289)) <= vocabulary.special
    assert 1029 == len(vocabulary.special)


def test_the_prompts_are_upstreams_sentences(vocabulary):
    """``<OD>`` is not sent as itself: the processor swaps it for an English instruction."""
    text, _ = vocabulary.decode_with_spans(list(PROMPTS["<OD>"]))
    assert text == "<s>Locate the objects with category name in the image.</s>"
    text, _ = vocabulary.decode_with_spans(list(PROMPTS["<OCR_WITH_REGION>"]))
    assert text == "<s>What is the text in the image, with regions?</s>"


def test_each_token_is_decoded_on_its_own(vocabulary):
    """``decode_with_spans`` decodes token by token, as upstream: a character split across two
    byte-level tokens becomes replacement characters rather than being rejoined. The spans tile
    the text."""
    text, spans = vocabulary.decode_with_spans([0, 50269, 1437, 2])
    assert text.startswith("<s><loc_0>")
    assert [end - start for start, end in spans] == [3, 7, len(text) - 14, 4]
    assert spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


# --- the boxes ----------------------------------------------------------------------------------

def test_a_location_is_the_centre_of_its_bin():
    """``floor`` quantisation: bin k of 1,000 maps back to ``(k + 0.5) * size / 1000``."""
    [box] = parse_od("<s>cat<loc_0><loc_0><loc_999><loc_999></s>", None, None, (1000, 500))
    assert box.box == [0.5, 0.25, 999.5, 499.75]
    assert box.label == "cat" and box.score is None


def test_a_label_loses_its_non_ascii_characters_as_upstream():
    [box] = parse_od("<s>café<loc_1><loc_2><loc_3><loc_4>", None, None, (1000, 1000))
    assert box.label == "caf"


def test_one_label_can_own_several_boxes():
    found = parse_od("<s>face<loc_1><loc_1><loc_2><loc_2><loc_3><loc_3><loc_4><loc_4>"
                     "cup<loc_5><loc_5><loc_6><loc_6>", None, None, (1000, 1000))
    assert [b.label for b in found] == ["face", "face", "cup"]


def test_the_score_is_the_mean_log_probability_of_the_four_location_tokens():
    """Not of the label: upstream's ``avg_loc_scores``. ``exp`` of a mean, in Python floats."""
    text = "<s>dog<loc_1><loc_2><loc_3><loc_4></s>"
    pieces = ["<s>", "dog", "<loc_1>", "<loc_2>", "<loc_3>", "<loc_4>", "</s>"]
    spans, at = [], 0
    for piece in pieces:
        spans.append((at, at + len(piece)))
        at += len(piece)
    scores = [0.0, -5.0, -0.1, -0.2, -0.3, -0.4, 0.0]
    [box] = parse_od(text, spans, scores, (1000, 1000))
    assert box.score == math.exp((-0.1 - 0.2 - 0.3 - 0.4) / 4)


def test_an_ocr_line_is_four_corners_and_its_text():
    [line] = parse_ocr("<s>EXIT<loc_10><loc_20><loc_30><loc_20><loc_30><loc_40><loc_10><loc_40></s>",
                       (1000, 1000))
    assert line.text == "EXIT" and line.quad == [10.5, 20.5, 30.5, 20.5, 30.5, 40.5, 10.5, 40.5]


# --- the decoding rules ---------------------------------------------------------------------------

def test_the_decoding_is_the_checkpoints_default():
    """What every published ``text_config`` sets and ``generate()`` therefore does with no
    arguments: three beams, no repeated 3-grams, early stopping, length penalty one."""
    assert (DECODING.num_beams, DECODING.no_repeat_ngram, DECODING.early_stopping,
            DECODING.length_penalty) == (3, 3, True, 1.0)
    assert (DECODING.decoder_start, DECODING.bos, DECODING.eos) == (2, 0, 2)


def test_a_repeated_trigram_is_banned():
    ids = torch.tensor([[2, 0, 7, 8, 9, 7, 8]])
    out = _no_repeat_ngram(ids, torch.zeros(1, 20), 3)
    assert out[0, 9] == -math.inf and torch.isfinite(out[0, :9]).all()


def test_the_first_token_is_forced_to_bos_and_the_last_to_eos():
    first = _process(torch.tensor([[2]]), torch.zeros(1, 10), 30, DECODING)
    assert first[0, 0] == 0 and torch.isinf(first[0, 1:]).all()
    last = _process(torch.tensor([[2] * 29]), torch.zeros(1, 10), 30, DECODING)
    assert last[0, 2] == 0 and torch.isinf(last[0, [0, 1, 3]]).all()


# --- the geometry ---------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(SPECS))
def test_the_image_becomes_577_tokens(variant):
    """768 / (4 * 2 * 2 * 2) = 24, so a 24 x 24 grid plus its mean. Shapes only, on meta."""
    with torch.device("meta"):
        network = Florence2(SPECS[variant])
        tokens = network.encode_image(torch.empty(1, 3, 768, 768))
    assert tokens.shape == (1, 577, SPECS[variant].vision.projection)


def test_large_has_four_thousand_positions_and_large_ft_one():
    """Microsoft replaced ``Florence-2-large`` with a 4k-context checkpoint; ``-ft`` kept 1,024."""
    assert SPECS["large"].language.positions == 4096
    assert SPECS["large-ft"].language.positions == SPECS["base"].language.positions == 1024


# --- the registry and the adapter -----------------------------------------------------------------

def test_registry_agrees_with_the_adapter():
    from mozo.adapters.florence2_detect import Florence2DetectPredictor

    entry = get_model_info(FAMILY)
    assert entry["adapter_class"] == "Florence2DetectPredictor"
    assert entry["module"] == "mozo.adapters.florence2_detect"
    assert entry["task_type"] == "object_detection"
    assert entry["variants"] == Florence2DetectPredictor.VARIANTS == list(SPECS)


def test_fp32_is_built_from_the_published_fp16():
    assert EXECUTES[FAMILY] == ("torch",)
    assert DERIVES[FAMILY] == {"torch-fp32": "torch-fp16"}


def test_auto_takes_fp16_on_cuda_and_the_exact_upcast_on_the_cpu():
    """Measured on 500 COCO images, fp16 moved AP by at most 0.28 on any variant at about twice
    fp32's speed; it is also the precision Microsoft publishes."""
    from mozo.runtimes import select_runtime

    published = ["LICENSE", "NOTICE", "torch-fp16"]
    pick = lambda device: select_runtime(device, published, executes=EXECUTES[FAMILY],  # noqa: E731
                                         derives=DERIVES[FAMILY])
    assert pick("cuda") == "torch-fp16"
    assert pick("cpu") == "torch-fp32"


def test_a_variant_that_does_not_exist_is_refused_before_any_download():
    from mozo.adapters.florence2_detect import Florence2DetectPredictor

    with pytest.raises(ValueError, match="Unsupported variant"):
        Florence2DetectPredictor("huge")


# --- with weights ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base():
    require_present(FAMILY, "base", "torch-fp16")
    from mozo.adapters.florence2_detect import Florence2DetectPredictor

    return Florence2DetectPredictor("base", device="cpu")


@pytest.fixture(scope="module")
def found(base):
    """One decode of the photograph, shared: each is a full 3-beam search on the CPU."""
    return base.predict(FIXTURE)


def test_the_cpu_runs_the_exact_upcast(base):
    assert base.runtime == "torch-fp32"


def test_boxes_are_named_in_the_models_words_with_no_class_id(found):
    names = {d.class_name for d in found}
    assert {"laptop", "human face"} <= names
    assert all(d.class_id is None for d in found)
    assert all(0.0 < d.confidence <= 1.0 for d in found)


def test_the_same_image_gives_the_same_boxes(base, found):
    second = base.predict(FIXTURE)
    assert [(d.bbox, d.class_name, d.confidence) for d in found] == [
        (d.bbox, d.class_name, d.confidence) for d in second]


def test_threshold_keeps_boxes_at_or_above_it(base, found):
    kept = base.predict(FIXTURE, threshold=0.4)
    assert len(kept) == sum(d.confidence >= 0.4 for d in found) < len(found)


def test_labels_are_refused_because_there_is_no_class_list(base):
    with pytest.raises(ValueError, match="no class list"):
        base.predict(FIXTURE, labels=["person"])


def test_detections_serialise_with_their_names(base, found):
    payload = base.predict(np.full((64, 64, 3), 255, np.uint8)).to_dict()
    assert isinstance(payload, list)
    entry = found.to_dict()[0]
    assert entry["class_name"] and entry["class_id"] is None and len(entry["bbox"]) == 4
