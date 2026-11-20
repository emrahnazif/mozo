"""Does Granite Embedding R2 actually work.

Unlike the module tests, these load a real checkpoint and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_granite_embedding_r2.py -q

mozo's first text-only family, so two promises are tested that no other family makes: nothing
about it needs an image, and a vector must not depend on which other texts it was encoded beside
-- padding would break that, so nothing pads. The tokenizer tests pin ids the reference produced
for the inputs where an interpreter's own Unicode tables would give a different answer, so they
fail on any Python where the carried tables stop doing their job.

``tools/verify/granite_embedding_r2.py`` is the bit-exact comparison against ``transformers``; it
needs the reference installed and does not run here.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import published, require_weights
from mozo.registry import ENCODE_ONLY, ENCODES, MODEL_REGISTRY
from mozo.vendors.granite_embedding_r2_deploy import CONTEXT, SPECS, VARIANTS
from mozo.vendors.granite_embedding_r2_deploy.predictor import batches

FAMILY = "granite_embedding_r2"


@pytest.fixture(scope="module")
def predictor():
    """The small variant on the CPU, built once for the file."""
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
def test_only_torch_is_published(variant):
    """If a graph artifact is ever added, this is the reminder that ``EXECUTES`` changes with it."""
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor

    keys = published(FAMILY, variant)
    if not keys:
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert keys == ["torch-fp32"]
    assert GraniteEmbeddingR2Predictor.EXECUTES == ("torch",)


# --- the geometry ---


def test_the_layer_pattern_starts_global_and_repeats_every_third():
    spec = SPECS["small"]
    assert [spec.is_global(i) for i in range(spec.layers)] == [True, False, False] * 4


def test_the_window_is_half_the_published_local_attention():
    """``local_attention: 128`` is the whole window; a token sees 64 either side."""
    assert SPECS["small"].window == 64


def test_the_head_dimension():
    assert SPECS["small"].head_dim == 32


# --- the tokenizer, which needs no weights ---


@pytest.fixture(scope="module")
def tokenizer():
    from mozo.vendors.granite_embedding_r2_deploy.tokenizer import Tokenizer

    return Tokenizer(CONTEXT)


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


def test_it_refuses_nothing_and_blanks(tokenizer):
    with pytest.raises(ValueError, match="at least one"):
        tokenizer([])
    with pytest.raises(ValueError, match="blank"):
        tokenizer(["fine", "  \n"])


def test_it_refuses_a_text_over_the_context(tokenizer):
    assert len(tokenizer("hello" + " hello" * (CONTEXT - 3))[0]) == CONTEXT
    with pytest.raises(ValueError, match="does not truncate"):
        tokenizer("hello" + " hello" * (CONTEXT - 2))


def test_it_refuses_an_unpaired_surrogate(tokenizer):
    with pytest.raises(ValueError, match="surrogate"):
        tokenizer.encode("a\ud800b")


# --- batching, which needs no weights either ---


def test_equal_lengths_batch_together_on_the_cpu():
    assert sorted(map(sorted, batches([5, 7, 5, 5, 7], "cpu"))) == [[0, 2, 3], [1, 4]]


def test_every_text_runs_alone_off_the_cpu():
    """Equal-length batches were not bit-identical on MPS, and CUDA is unmeasured."""
    assert batches([5, 5, 5], "mps") == [[0], [1], [2]]
    assert batches([5, 5], "cuda") == [[0], [1]]


def test_a_batch_stays_within_the_token_budget():
    groups = batches([CONTEXT // 2] * 5, "cpu")
    assert all(len(group) * CONTEXT // 2 <= CONTEXT for group in groups)
    assert sorted(i for group in groups for i in group) == list(range(5))


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
    texts = ["one two three", "four five six", "the quick brown fox jumps"]
    together = predictor.encode_text(texts)
    alone = np.concatenate([predictor.encode_text(text) for text in texts])
    assert np.array_equal(together, alone)


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
    with pytest.raises(ValueError, match=f"over the {CONTEXT}-token context"):
        predictor.encode_text("hello" + " hello" * CONTEXT)
