"""Does Granite Embedding R2 actually work.

Unlike the module tests, these load a real checkpoint and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_granite_embedding_r2.py -q

mozo's first text-only family, so two promises are tested that no other family makes: nothing
about it needs an image, and a vector must not depend on which other texts it was encoded beside
-- batching would break that, so every text runs alone. The tokenizer tests pin ids the reference
produced for the inputs where an interpreter's own Unicode tables would give a different answer,
so they fail on any Python where the carried tables stop doing their job. Two variants, English
``small`` and ``multilingual-small``, with different tokenizers and the same network.

``tools/verify/granite_embedding_r2.py`` is the bit-exact comparison against ``transformers``; it
needs the reference installed and does not run here.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import published, require_weights
from mozo.registry import ENCODE_ONLY, ENCODES, MODEL_REGISTRY
from mozo.vendors.granite_embedding_r2_deploy import SPECS, VARIANTS

FAMILY = "granite_embedding_r2"

#: What each variant publishes, besides its LICENSE and NOTICE.
PUBLISHED = {"small": ["coreml-fp32", "torch-fp32"], "multilingual-small": ["torch-fp32"]}


@pytest.fixture(scope="module", params=VARIANTS)
def predictor(request):
    """Each variant on the CPU, built once for the file."""
    require_weights(FAMILY, request.param)
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor
    from mozo.weights import WeightsError

    try:
        return GraniteEmbeddingR2Predictor(request.param, device="cpu")
    except WeightsError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def small():
    """The English variant alone, for what only it has."""
    require_weights(FAMILY, "small")
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor
    from mozo.weights import WeightsError

    try:
        return GraniteEmbeddingR2Predictor("small", device="cpu")
    except WeightsError as error:
        pytest.skip(str(error))


# --- the registry and the adapter must agree ---


def test_registry_agrees_with_the_adapter():
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor

    assert set(MODEL_REGISTRY[FAMILY]["variants"]) == set(GraniteEmbeddingR2Predictor.VARIANTS)


def test_it_embeds_text_and_nothing_else():
    """``/encode`` refuses an image before the download, and ``/predict`` refuses the family."""
    assert ENCODES[FAMILY] == frozenset({"text"})
    assert MODEL_REGISTRY[FAMILY]["task_type"] in ENCODE_ONLY


def test_the_adapter_has_no_predict():
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor

    assert not hasattr(GraniteEmbeddingR2Predictor, "predict")


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_published_variant_ships_its_terms(variant):
    from mozo.weights import companions

    if not published(FAMILY, variant):
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert "LICENSE" in companions(FAMILY, variant)
    assert "NOTICE" in companions(FAMILY, variant)


@pytest.mark.parametrize("variant", VARIANTS)
def test_it_publishes_what_the_adapter_executes(variant):
    """If another graph is ever added, this is the reminder that ``EXECUTES`` changes with it.
    CoreML is ``small``'s alone: it was built on a Mac, which this round did not have."""
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor

    keys = published(FAMILY, variant)
    if not keys:
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert sorted(keys) == PUBLISHED[variant]
    assert GraniteEmbeddingR2Predictor.EXECUTES == ("torch", "coreml")


# --- the geometry ---


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_layer_pattern_starts_global_and_repeats_every_third(variant):
    spec = SPECS[variant]
    assert [spec.is_global(i) for i in range(spec.layers)] == [True, False, False] * 4


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_window_is_half_the_published_local_attention(variant):
    """``local_attention: 128`` is the whole window; a token sees 64 either side."""
    assert SPECS[variant].window == 64


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_head_dimension(variant):
    assert SPECS[variant].head_dim == 32


def test_the_variants_differ_where_the_published_configs_do():
    """Everything a config.json says differently, and nothing else."""
    english, multilingual = SPECS["small"], SPECS["multilingual-small"]
    assert (english.activation, multilingual.activation) == ("gelu", "silu")
    assert (english.context, multilingual.context) == (8192, 32768)
    assert (english.vocab, multilingual.vocab) == (50368, 180000)
    assert (multilingual.global_theta, multilingual.local_theta) == (150000.0, 160000.0)
    assert english.vocabulary != multilingual.vocabulary


# --- the bucketed graph, which needs no weights ---


@pytest.mark.parametrize("length", [2, 17, 64, 65, 66, 130, 255, 256])
def test_padding_to_a_bucket_does_not_move_the_vector(length):
    """What a CoreML graph runs: padded to 256 and masked, against the network on the text alone.

    Six layers keep it quick and still cover both kinds; the lengths straddle the 64-token window.
    """
    import dataclasses

    import torch
    import torch.nn.functional as F

    from mozo.vendors.granite_embedding_r2_deploy.graph import Bucketed, padded
    from mozo.vendors.granite_embedding_r2_deploy.network import Network

    torch.manual_seed(0)
    network = Network(dataclasses.replace(SPECS["small"], layers=6)).eval()
    ids = torch.randint(5, 50000, (1, length))
    with torch.inference_mode():
        want = F.normalize(network(ids)[:, 0], p=2, dim=1)
        got = Bucketed(network)(torch.from_numpy(padded(ids[0].tolist(), 256)).long(),
                                torch.tensor([length]))
    assert (got - want).abs().max() < 1e-6


# --- the tokenizer, which needs no weights ---


@pytest.fixture(scope="module")
def tokenizer():
    from mozo.vendors.granite_embedding_r2_deploy.tokenizer import Tokenizer

    return Tokenizer(SPECS["small"].context)


@pytest.fixture(scope="module")
def multilingual_tokenizer():
    from mozo.vendors.granite_embedding_r2_deploy.tokenizer import Tokenizer

    spec = SPECS["multilingual-small"]
    return Tokenizer(spec.context, spec.vocabulary)


@pytest.mark.parametrize("text, ids", [
    # A run of spaces is an added token, and the next word starts without one of its own.
    ("Hello  world", [50281, 12092, 50276, 10186, 50282]),
    # [MASK] strips the whitespace to its left.
    ("a   [MASK]", [50281, 66, 50284, 50282]),
    # Special tokens written out become those tokens.
    ("<|endoftext|> [SEP] x", [50281, 50279, 209, 50282, 1269, 50282]),
    # NFC: decomposed and precomposed are the same token.
    ("cafe\u0301", [50281, 68, 2320, 860, 50282]),
    ("caf\u00e9", [50281, 68, 2320, 860, 50282]),
    # A Gondi nukta the reference treats as class 0: e and the accent do NOT compose across it,
    # though every Python's own NFC composes them.
    ("e\U00011d42\u0301", [50281, 70, 172, 228, 115, 213, 136, 212, 50282]),
    # Codepoints newer than the reference's tables: punctuation to it, letters to `regex`, and
    # one that Python 3.14 decomposes.
    ("\U000105c9 \u088f", [50281, 172, 227, 234, 220, 209, 156, 97, 226, 50282]),
    ("2024 12.5", [50281, 938, 1348, 1249, 15, 22, 50282]),
])
def test_ids_match_the_reference_whatever_the_interpreter(tokenizer, text, ids):
    assert tokenizer.encode(text) == ids


@pytest.mark.parametrize("text, ids", [
    # No normalisation: decomposed and precomposed are different tokens.
    ("cafe\u0301 caf\u00e9", [179934, 66, 6790, 13189, 29642, 179938]),
    # o200k splits camel case at the capital and digits into threes.
    ("camelCase 1234567", [179934, 161439, 6084, 180, 7505, 18943, 22, 179938]),
    # A piece the vocabulary has whole is taken whole: merging would give 920, 5562, 7833, 2637.
    ("INCLUDING", [179934, 42164, 179938]),
    # Control bytes have no piece and vanish; a, b and c stay apart.
    ("a\x07b\x0cc", [179934, 64, 65, 66, 179938]),
    # A lead byte with no piece (0xF1) vanishes; its three continuation bytes do not.
    ("\U00040000", [179934, 182, 182, 182, 179938]),
    # A single_word token counts standing alone, and not against a letter.
    ("a <|reserved_200020|> b", [179934, 64, 180, 179956, 247, 179938]),
    ("a<|reserved_200020|>",
     [179934, 64, 27, 91, 108398, 62, 1132, 18756, 91, 29, 179938]),
    # Special tokens written out become those tokens.
    ("x<|return|>y", [179934, 87, 179938, 88, 179938]),
    ("Москва ΟΔΟΣ 東京", [179934, 139692, 28241, 16280, 169403, 167538, 179938]),
])
def test_multilingual_ids_match_the_reference(multilingual_tokenizer, text, ids):
    assert multilingual_tokenizer.encode(text) == ids


def test_it_refuses_nothing_and_blanks(tokenizer):
    with pytest.raises(ValueError, match="at least one"):
        tokenizer([])
    with pytest.raises(ValueError, match="blank"):
        tokenizer(["fine", "  \n"])


def test_it_refuses_a_text_over_the_context(tokenizer, multilingual_tokenizer):
    for each in (tokenizer, multilingual_tokenizer):
        context = each.context
        assert len(each("hello" + " hello" * (context - 3))[0]) == context
        with pytest.raises(ValueError, match="does not truncate"):
            each("hello" + " hello" * (context - 2))


def test_it_refuses_an_unpaired_surrogate(tokenizer):
    with pytest.raises(ValueError, match="surrogate"):
        tokenizer.encode("a\ud800b")


# --- the model ---


def test_a_single_text_comes_back_two_dimensional(predictor):
    assert predictor.encode_text("a forklift").shape == (1, 384)


def test_vectors_leave_normalised(predictor):
    vectors = predictor.encode_text(["a forklift", "how do I run a workflow?"])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


def test_order_is_kept(predictor):
    texts = ["short", "a much longer text about warehouses and pallets", "short"]
    vectors = predictor.encode_text(texts)
    assert np.array_equal(vectors[0], vectors[2])
    assert np.array_equal(vectors[1], predictor.encode_text(texts[1])[0])


def test_a_vector_does_not_depend_on_what_it_was_encoded_beside(predictor):
    """Texts of exactly equal length too: batched, those moved by up to 1.6e-03 on a multi-threaded
    x86 CPU, which is why every text runs alone."""
    texts = ["one two three", "four five six", "the quick brown fox jumps",
             "seven eight nine", "ten eleven twelve"]
    together = predictor.encode_text(texts * 4)
    alone = np.concatenate([predictor.encode_text(text) for text in texts * 4])
    assert np.array_equal(together, alone)


def test_a_question_finds_its_answer_across_languages():
    """What the multilingual variant is for: IBM's own card example, a query in one language and
    passages in three, each query nearest its own passage."""
    require_weights(FAMILY, "multilingual-small")
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor
    from mozo.weights import WeightsError

    try:
        model = GraniteEmbeddingR2Predictor("multilingual-small", device="cpu")
    except WeightsError as error:
        pytest.skip(str(error))
    queries = model.encode_text([
        "What is the tallest mountain in Japan?",
        "Wer hat das Lied Achy Breaky Heart geschrieben?",
        "ドイツの首都はどこですか？",
    ])
    passages = model.encode_text([
        "富士山は、静岡県と山梨県にまたがる活火山で、標高3776.12 mで日本最高峰の独立峰である。",
        "Achy Breaky Heart is a country song written by Don Von Tress.",
        "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland.",
    ])
    assert list((queries @ passages.T).argmax(axis=1)) == [0, 1, 2]


def test_meaning_lands_closer_than_mere_words(predictor):
    query, near, far = predictor.encode_text([
        "How do I count people crossing a line in a video?",
        "Track each person across frames and record when their path crosses the line.",
        "The forklift battery should be charged overnight.",
    ])
    assert query @ near > query @ far


def test_it_says_which_revision_produced_the_vectors(predictor):
    assert predictor.revision


def test_it_refuses_a_text_over_the_context_before_running(predictor):
    context = SPECS[predictor.variant].context
    with pytest.raises(ValueError, match=f"over the {context}-token context"):
        predictor.encode_text("hello" + " hello" * context)


# --- the CoreML graph, on a Mac with coremltools ---


@pytest.fixture(scope="module")
def graph():
    """The small variant as its CoreML package, built once for the file."""
    pytest.importorskip("coremltools")
    require_weights(FAMILY, "small")
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor
    from mozo.weights import WeightsError

    try:
        return GraniteEmbeddingR2Predictor("small", device="mps", runtime="coreml-fp32")
    except WeightsError as error:
        pytest.skip(str(error))


def test_coreml_agrees_with_torch(graph, small):
    """Lengths that fill a bucket, spill into the next, and need the longest.

    Not bit-identical -- no graph runtime is -- but close enough that an index built on one runtime
    serves queries from the other: measured 1.8e-07 at worst.
    """
    texts = ["a forklift", "word " * 63, "word " * 64, "Granite embeds text " * 120, "x " * 8000]
    np.testing.assert_allclose(graph.encode_text(texts), small.encode_text(texts), atol=1e-6)


def test_coreml_loads_only_the_lengths_it_needs(graph):
    """A short query loads the shortest length and nothing else."""
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor

    fresh = GraniteEmbeddingR2Predictor("small", device="mps", runtime="coreml-fp32")
    fresh.encode_text("a forklift")
    assert sorted(fresh._encoder._runners) == [64]
