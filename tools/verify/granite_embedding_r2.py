#!/usr/bin/env python3
"""Check that mozo's Granite Embedding R2 returns exactly what the published models do.

Three paths reach the same weights and this compares all three: the vendored package
(``mozo/vendors/granite_embedding_r2_deploy``), the reference (``transformers``'
``ModernBertModel`` and the published ``tokenizer.json``), and mozo itself (registry, weights,
adapter). Both variants, English ``small`` and ``multilingual-small``, which share the network and
nothing of their tokenizers.

**The reference is ``transformers``, and that is not a choice.** IBM publishes no inference code
of its own; the checkpoints are ``ModernBertModel`` and the model cards load them through
``sentence-transformers``, which is that model plus two steps -- take the first position, and,
with ``normalize_embeddings=True``, ``F.normalize`` it. Both are written out here, so the gate
needs no ``sentence-transformers``. (It also strips each text's leading and trailing whitespace,
which ``transformers`` does not: see the PROVENANCE.)

**Every stage, compared exactly.** Token ids, the embeddings, each of the twelve layers, the final
norm (whose first row is the raw vector) and the normalised vector, with ``torch.equal`` and no
tolerance -- read off ``encode_text`` itself, the path a caller runs.

**Four things about the comparison are pinned.**

*The device is the CPU* and the thread count one, so what is compared is arithmetic. On this
machine's x86 kernels the thread count does move the last bit (1.6e-07 between one and eight
threads, measured), on both sides alike.

*The reference runs in float32.* The checkpoints are stored in bfloat16 and ``transformers`` 5
loads in the stored dtype by default, so a reference loaded the obvious way runs in bfloat16 and
every comparison fails by 3e-03 to 7e-03 -- a failure that looks like the extraction's.

*Its attention is SDPA*, the call the vendored layers make. Eager attention is the same arithmetic
in a different order: 1.5e-07 to 1.8e-07 apart.

*Its parameters are re-allocated before comparing*, so ``from_pretrained``'s storage offsets cannot
steer BLAS onto a different path. Compare arithmetic, not addresses.

The reference's version does not move the answer: the transformers that saved each checkpoint
(4.51.3, 4.56.2) and 5.x agreed on every stage when each variant was built, though 5.x rewrote
this model. See ``PROVENANCE.md``.

**The tokenizers' answers must not depend on the Python running them.** ``--fuzz`` adds every
codepoint (and, in English, runs of mixed combining marks) to each tokenizer corpus; the
multilingual corpus always includes Belebele's passages and questions in 122 languages, fetched
from the Hub. ``--corpus PATH`` writes a corpus with the reference's ids, and
``granite_embedding_r2_corpus.py`` -- standard library only -- checks it under any other Python.

**``--falsify`` proves the gate can fail.** It perturbs one thing at a time -- a rotary base, the
window, the activation, an epsilon, the pooled position, three tokenizer rules -- reruns the
comparison against the reference stages already recorded, and requires each perturbation to be
caught, first at the stage it should be.

Run it with the checkpoints in place::

    python tools/verify/granite_embedding_r2.py                       # everything
    python tools/verify/granite_embedding_r2.py multilingual-small --falsify
    python tools/verify/granite_embedding_r2.py --fuzz                # plus every codepoint
    python tools/verify/granite_embedding_r2.py multilingual-small --long   # plus 32,768 tokens
    python tools/verify/granite_embedding_r2.py small --fuzz --corpus corpus.json
    python3.14 tools/verify/granite_embedding_r2_corpus.py corpus.json
    python tools/verify/granite_embedding_r2.py --mozo-only
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import random
import unicodedata
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# This checks a tree you just built, and reaching for the published bucket would verify the wrong
# bytes. Same line, same reason, as the other gates.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from fetch.granite_embedding_r2 import COMMITS, repository  # noqa: E402

from mozo.vendors.granite_embedding_r2_deploy import SPECS, Encoder  # noqa: E402
from mozo.vendors.granite_embedding_r2_deploy import layers as vendor_layers  # noqa: E402
from mozo.vendors.granite_embedding_r2_deploy import tokenizer as vendor_tokenizer  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: The versions this family's parity was established against. Printed, not enforced. ``small``
#: was first established on torch 2.11.0 / transformers 5.8.0 and passes unchanged here.
PINNED = {"torch": "2.13.0", "transformers": "5.9.0"}

#: Texts that exercise the English tokenizer's awkward corners: whitespace runs that are added
#: tokens, ``[MASK]``'s left strip, special tokens written out, NFC, a mark the reference will not
#: compose across, a codepoint newer than the reference's tables, mixed scripts.
AWKWARD = [
    "forklift",
    "How do I run a workflow on a video at five frames a second?",
    "a  b   c" + " " * 24 + "d" + " " * 25 + "e",
    "def f(x):\n\treturn x\n\n\n    # indented\r\n",
    "   leading and trailing   ",
    "cafe\u0301 vs caf\u00e9, e\U00011d42\u0301",
    "東京の倉庫 مستودع 🚜🤖 👩🏽\u200d🔧 \u088f \U000105c9",
    "<|endoftext|> [SEP] [CLS] [MASK] [PAD] |||IP_ADDRESS||| <|padding|> x",
    "a   [MASK] b\t[MASK][unused10] [UNUSED1]",
    "I'll don't O'Neil 'S 2024 12.5 a\xa0b a\u2028b",
]

#: And the multilingual one's: no normalisation, o200k's split (camel case, digits in threes,
#: case-insensitive contractions including the long s), whole pieces before merges, control bytes
#: and UTF-8 lead bytes with no piece, special tokens, and ``single_word`` tokens against a word
#: character or not.
AWKWARD_MULTILINGUAL = [
    "forklift",
    "What is the tallest mountain in Japan?",
    "富士山は、静岡県と山梨県にまたがる活火山で、標高3776.12 mで日本最高峰の独立峰である。",
    "Москва — столица России. مرحبا بالعالم. नमस्ते दुनिया. 안녕하세요 세계. ΟΔΟΣ ς",
    "cafe\u0301 vs caf\u00e9; HTTPServer camelCase 1234567 x/\n\ny INCLUDING",
    "I'LL don't O'Neil O'Shannessy 'S we'RE x'\u017f a'\u017fd",
    "a\x00b\x07c\x0bd\x0ce\x1ff \U00040000 \U00080000 \U00100000 \U000e0001 \ufeff\u200b",
    "<|startoftext|> <|return|> <|endoftext|> [MASK] <|endofprompt|> x",
    "<|reserved_200020|> standing, x<|reserved_200021|> against, <|reserved_200063|>_ too",
    "   leading and trailing   \r\n\t",
    "def f(x):\n\treturn x  # 🚜🤖 👩🏽\u200d🔧",
]

#: A paragraph in eleven scripts, repeated to make the long multilingual fixtures.
MULTILINGUAL_PARAGRAPH = (
    "Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland. "
    "Москва — столица России. مرحبا بالعالم. नमस्ते दुनिया. 안녕하세요 세계. "
    "Γειά σου Κόσμε. שלום עולם. สวัสดีชาวโลก. Xin chào thế giới. 東京は日本の首都です。 "
)


def versions() -> str:
    """What is actually installed, next to what parity was established against."""
    import transformers

    actual = {"torch": torch.__version__.split("+")[0], "transformers": transformers.__version__}
    return " ".join(f"{name} {actual[name]}" + ("" if actual[name] == want else f" (pinned {want})")
                    for name, want in PINNED.items())


def docs_text() -> str:
    """Every page of mozo's docs, joined: the real English the gate, the bench and the export
    check share, read in one place so a reorganised ``docs/`` moves all three together."""
    return "\n".join(path.read_text(encoding="utf-8")
                     for path in sorted((ROOT / "docs").rglob("*.md")))


def prefix_within(tokenizer, words: list[str], budget: int) -> str:
    """The longest run of *words*, from the start, that tokenizes to at most *budget* ids.

    Every word is at least one token, so no more than *budget* words can fit -- which bounds the
    search by the budget rather than by the length of the document.
    """
    low, high = 1, min(len(words), budget)
    while low < high:
        middle = (low + high + 1) // 2
        if len(tokenizer.encode(" ".join(words[:middle]))) <= budget:
            low = middle
        else:
            high = middle - 1
    return " ".join(words[:low])


def texts(variant: str, tokenizer, long: bool) -> dict[str, str]:
    """The fixtures, by name: the awkward ones, lengths either side of the attention window and
    of 128, a long text, and one exactly at the context -- English's 8,192 always, the
    multilingual 32,768 only with *long* (minutes on one thread)."""
    multilingual = variant.startswith("multilingual")
    awkward = AWKWARD_MULTILINGUAL if multilingual else AWKWARD
    out = {f"awkward {i}": text for i, text in enumerate(awkward)}
    for length in (63, 64, 65, 66, 127, 128, 129, 130):
        out[f"{length} tokens"] = "hello" + " hello" * (length - 3)
    if multilingual:
        words = (MULTILINGUAL_PARAGRAPH * 3000).split(" ")
        out["multilingual 1500"] = prefix_within(tokenizer, words, 1500)
        out["multilingual 8192"] = prefix_within(tokenizer, words, 8192)
        if long:
            out["multilingual at the context"] = prefix_within(tokenizer, words,
                                                               SPECS[variant].context)
    else:
        words = docs_text().split(" ")
        out["docs page"] = prefix_within(tokenizer, words, 1500)
        out["docs at the context"] = prefix_within(tokenizer, words, SPECS[variant].context)
    return out


def belebele() -> list[str]:
    """Belebele's passages, questions and answers in all 122 of its languages (CC-BY-SA-4.0),
    from the Hub's cache: real text in every script the multilingual model claims."""
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(name for name in HfApi().list_repo_files("facebook/belebele", repo_type="dataset")
                   if name.startswith("data/") and name.endswith(".jsonl"))
    out: dict[str, None] = {}
    for name in files:
        path = hf_hub_download("facebook/belebele", name, repo_type="dataset")
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for key in ("flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3",
                        "mc_answer4"):
                out[row[key]] = None
    return list(out)


def tokenizer_corpus(variant: str, sweep: bool) -> list[str]:
    """Texts for the tokenizer alone: every line of the docs, random mixtures of the awkward parts
    of Unicode and of the variant's own tokens, Belebele for the multilingual variant, and with
    *sweep*, every codepoint in contexts that cross each class boundary -- plus, in English, bases
    followed by runs of combining marks, the shape that exercises NFC, where interpreters
    disagree."""
    multilingual = variant.startswith("multilingual")
    corpus = docs_text().split("\n")
    rng = random.Random(0)
    pool = (list(" \t\n\r\x0b\x0c\x85\xa0\u1680\u2000\u2028\u2029\u3000aA\u00e9\u00df0٠\u00bd'!?[]<>|")
            + ["[MASK]", "\u0301", "\U00011d42", "\u088f", "\U000105c9", "가", "'ll"]
            + [chr(rng.randrange(0x20, 0x30000)) for _ in range(2000)])
    if multilingual:
        pool += (list("\x00\x07\x1f_/zZ189\u017f\u0308")
                 + ["<|startoftext|>", "<|return|>", "<|endoftext|>", "<|reserved_200020|>",
                    "<|reserved_200063|>", "'s", "'S", "'LL", "'re", "'VE", "'d", "'M", "'t",
                    "camelCase", "HTTPServer", "12345", "東京", "مرحبا", "Москва", "ΟΔΟΣ", "ς",
                    "🚜", "👩🏽\u200d🔧", "\U00040000", "\U00080000", "\U00100000", "\U000e0001",
                    "\ufeff", "\u200b", "\r\n", "x/\n"])
    else:
        pool += ["[SEP]", "<|endoftext|>", "[unused7]", "  ", " " * 23]
    for _ in range(20000):
        text = "".join(rng.choice(pool) for _ in range(rng.randrange(1, 30)))
        if not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
            corpus.append(text)
    every = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    if multilingual:
        corpus += belebele()
        corpus += ["".join(chr(rng.choice(every)) for _ in range(rng.randrange(1, 20)))
                   for _ in range(20000)]
        if sweep:
            corpus += [f"a{chr(c)}b {chr(c)}A1{chr(c)}{chr(c)}' {chr(c)}\nx'{chr(c)} "
                       f"<|reserved_200020|>{chr(c)}" for c in every]
        return corpus
    if sweep:
        corpus += [f"a{chr(c)}b {chr(c)}{chr(c)} 1{chr(c)} !{chr(c)} e{chr(c)}\u0301 {chr(c)}[MASK]"
                   for c in every]
        # Every mark with a combining class, plus characters newer than Unicode 13 that later
        # Pythons decompose, after a handful of bases.
        marks = [chr(c) for c in every if unicodedata.combining(chr(c))]
        marks += [chr(c) for c in (0x105C9, 0x105E4, 0x11383, 0x11385, 0x1138E, 0x11391, 0x113C5,
                                   0x113C7, 0x113C8, 0x16121, 0x16122, 0x16123)]
        bases = list("aeiouAEOUnc") + ["ᄀ", "ᅡ", "ᆨ", "가", "क", "ا"]
        corpus += ["".join(rng.choice(bases) + "".join(rng.choice(marks)
                                                       for _ in range(rng.randrange(0, 5)))
                           for _ in range(rng.randrange(1, 8)))
                   for _ in range(20000)]
    return corpus


def _record(network, stages: dict) -> list:
    """Hook every stage of *network* -- either side's -- into *stages*, by one set of names.

    Returns the hook handles, so the recording can stop when nothing reads it any more.
    """
    def keep(name):
        return lambda module, inputs, output: stages.__setitem__(name, output.detach().clone())
    modules = [("embeddings", network.embeddings), ("final_norm", network.final_norm)]
    modules += [(f"layer {index:02d}", layer) for index, layer in enumerate(network.layers)]
    return [module.register_forward_hook(keep(name)) for name, module in modules]


#: The order stages run in, so a perturbation's first failure can be named.
def stage_order(spec) -> list[str]:
    return (["ids", "embeddings"] + [f"layer {i:02d}" for i in range(spec.layers)]
            + ["final_norm", "unit"])


def reference_stages(variant: str, fixtures: dict[str, str]) -> tuple[object, dict]:
    """Run the reference over *fixtures*, recording every stage; also return its tokenizer."""
    from tokenizers import Tokenizer as Reference
    from transformers import AutoModel
    from huggingface_hub import hf_hub_download

    repo, commit = repository(variant), COMMITS[variant]
    theirs = AutoModel.from_pretrained(repo, revision=commit, dtype=torch.float32,
                                       attn_implementation="sdpa").eval()
    for parameter in theirs.parameters():
        parameter.data = parameter.data.clone()
    reference = Reference.from_file(hf_hub_download(repo, "tokenizer.json", revision=commit))
    # The multilingual tokenizer.json asks for padding to the longest text of a batch, and for
    # truncation at 32,768: ``encode_batch`` would pad every corpus slice. transformers sets both
    # per call, and mozo neither pads nor truncates, so the comparison turns them off.
    reference.no_padding()
    reference.no_truncation()

    stages: dict = {}
    hooks = _record(theirs, stages)
    recorded = {}
    for name, text in fixtures.items():
        ids = reference.encode(text).ids
        stages.clear()
        with torch.inference_mode():
            cls = theirs(input_ids=torch.tensor([ids])).last_hidden_state[:, 0]
        recorded[name] = {**stages, "ids": ids, "unit": F.normalize(cls, p=2, dim=1)}
    for hook in hooks:
        hook.remove()
    return reference, recorded


def compare(ours: Encoder, fixtures: dict[str, str], recorded: dict) -> tuple[int, list, dict]:
    """Every stage of *ours* against *recorded*: comparisons made, failures, and per fixture the
    first stage that differed (absent when none did)."""
    order = stage_order(ours.spec)
    stages: dict = {}
    hooks = _record(ours.network(), stages)
    checked, failures, first = 0, [], {}
    for name, text in fixtures.items():
        want = recorded[name]
        stages.clear()
        checked += 1
        if ours.tokenizer.encode(text) != want["ids"]:
            failures.append(f"{name}: token ids differ")
            first[name] = "ids"
            continue
        stages["unit"] = ours.encode_text(text)  # fills the rest through the hooks
        for stage in order[1:]:
            checked += 1
            if not torch.equal(stages[stage], want[stage]):
                failures.append(f"{name}: {stage} differs by "
                                f"{(stages[stage] - want[stage]).abs().max():.3e}")
                first.setdefault(name, stage)
    for hook in hooks:
        hook.remove()
    return checked, failures, first


def against_upstream(variant: str, ours: Encoder, sweep: bool, long: bool,
                     corpus_out: Path | None) -> tuple[int, list[str], dict, dict]:
    """Compare every stage of the vendored model against ``transformers``."""
    fixtures = texts(variant, ours.tokenizer, long)
    reference, recorded = reference_stages(variant, fixtures)

    checked, failures = 0, []
    corpus = tokenizer_corpus(variant, sweep)
    # Compared slice by slice, and the reference's ids kept only when they are to be written: the
    # multilingual corpus with every codepoint is 1.7 million texts, and a run holding every
    # Encoding (offsets, strings and masks besides the ids) was killed for memory.
    expected = [] if corpus_out is not None else None
    for start in range(0, len(corpus), 10000):
        chunk = corpus[start:start + 10000]
        for text, encoding in zip(chunk, reference.encode_batch(chunk)):
            checked += 1
            if ours.tokenizer.encode(text) != encoding.ids:
                failures.append(f"token ids differ for {text[:50]!r}")
            if expected is not None:
                expected.append(encoding.ids)
    print(f"  tokenizer: {len(corpus):,} texts")
    if corpus_out is not None:
        corpus_out.write_text(json.dumps({"variant": variant, "texts": corpus, "ids": expected},
                                         ensure_ascii=False))
        print(f"  wrote the corpus and the reference's ids to {corpus_out}")

    count, problems, _ = compare(ours, fixtures, recorded)
    checked += count
    failures += problems
    print(f"  model: {len(fixtures)} texts, every stage")

    # Encoding them together must not move any vector. Repeats force equal-length batches.
    vectors = {text: recorded[name]["unit"] for name, text in fixtures.items()}
    batch = list(vectors)
    batch += [text for text in batch if len(text) < 2000] * 3
    checked += 1
    if not torch.equal(ours.encode_text(batch), torch.cat([vectors[text] for text in batch])):
        failures.append("encoding texts together moved a vector")
    return checked, failures, fixtures, recorded


def against_mozo(variant: str, vendor: Encoder) -> tuple[int, list[str]]:
    """Compare mozo's adapter against the vendored encoder it wraps: it should change nothing."""
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor

    adapter = GraniteEmbeddingR2Predictor(variant, device="cpu")
    checked, failures = 0, []
    awkward = AWKWARD_MULTILINGUAL if variant.startswith("multilingual") else AWKWARD
    batch = awkward + ["one more, to make a batch"]
    checked += 1
    if not torch.equal(torch.from_numpy(adapter.encode_text(batch)), vendor.encode_text(batch)):
        failures.append("the adapter's vectors differ from the vendor's")
    checked += 1
    if adapter.revision is None:
        failures.append("the adapter does not say which revision produced its vectors")
    return checked, failures


# --- falsification ---


@contextlib.contextmanager
def _tables(vocabulary: str, **changes):
    """The tokenizer tables of *vocabulary*, with *changes*, for the duration of the block."""
    original = vendor_tokenizer._vocabulary
    perturbed = original(vocabulary)._replace(**changes)
    vendor_tokenizer._vocabulary = lambda name: perturbed if name == vocabulary else original(name)
    vendor_tokenizer._word.cache_clear()
    try:
        yield
    finally:
        vendor_tokenizer._vocabulary = original
        vendor_tokenizer._word.cache_clear()


@contextlib.contextmanager
def _activation(name: str, function):
    """The MLP activation called *name*, replaced by *function*, for the duration of the block."""
    original = vendor_layers.ACTIVATIONS[name]
    vendor_layers.ACTIVATIONS[name] = function
    try:
        yield
    finally:
        vendor_layers.ACTIVATIONS[name] = original


class _PoolSecond(Encoder):
    """Pools position 1 instead of 0: the unit vector, and only it, must fail."""

    def encode_text(self, texts):
        rows = self.tokenizer(texts)
        with torch.inference_mode():
            return torch.cat([F.normalize(self.network()(torch.tensor([row]))[:, 1], p=2, dim=1)
                              for row in rows])


def falsifications(variant: str) -> list[tuple[str, object, str, set[str] | None]]:
    """``(what, how, the stage it must first fail at, the fixtures it must fail on)``.

    *how* is either a replacement ``Spec`` or a zero-argument context manager factory. A fixture
    set of ``None`` means every fixture that exercises the change must fail and at least one must.
    """
    spec = SPECS[variant]
    multilingual = variant.startswith("multilingual")
    replace = dataclasses.replace
    vocabulary = spec.vocabulary
    cases = [
        ("local rotary base +1", replace(spec, local_theta=spec.local_theta + 1), "layer 01",
         None),
        ("global rotary base +1", replace(spec, global_theta=spec.global_theta + 1), "layer 00",
         None),
        # Only an input longer than the window can see the difference.
        ("window 64 -> 63", replace(spec, window=63), "layer 01", None),
        ("LayerNorm eps 1e-5 -> 1e-6", replace(spec, norm_eps=1e-6), "embeddings", None),
        ("pool position 1", _PoolSecond, "unit", None),
    ]
    if multilingual:
        cases += [
            ("SiLU -> exact GELU", lambda: _activation("silu", F.gelu), "layer 00", None),
            # ``INCLUDING`` is one piece whole and four merged.
            ("merge every piece (no ignore_merges)", lambda: _tables(vocabulary, whole={}),
             "ids", {"awkward 4"}),
            ("single_word ignored", lambda: _tables(vocabulary, single_word=frozenset()), "ids",
             {"awkward 8"}),
            # ``O'Shannessy`` takes ``'S`` as a contraction; most joins give the same ids anyway.
            ("contractions case-sensitive", lambda: _tables(vocabulary, split=_case_sensitive()),
             "ids", {"awkward 5"}),
        ]
    else:
        cases += [
            ("exact GELU -> tanh GELU",
             lambda: _activation("gelu", lambda x: F.gelu(x, approximate="tanh")), "layer 00",
             None),
            ("no NFC", lambda: _tables(vocabulary, boundaries=None), "ids", {"awkward 5"}),
            ("[MASK] keeps its whitespace", lambda: _tables(vocabulary, lstrip=frozenset()),
             "ids", {"awkward 8"}),
        ]
    return cases


def _case_sensitive():
    """The multilingual split pattern with its contractions in lower case only."""
    import gzip
    import re

    data = json.loads(gzip.decompress(vendor_tokenizer.vocab_path(
        SPECS["multilingual-small"].vocabulary).read_bytes()))
    data["contractions"] = [["".join(c for c in characters if c.islower() and c.isascii())
                             for characters in alternative] for alternative in data["contractions"]]
    pattern = vendor_tokenizer._o200k(data)
    assert isinstance(pattern, re.Pattern)
    return pattern


def falsify(variant: str, checkpoint: Path, fixtures: dict[str, str],
            recorded: dict) -> list[str]:
    """Run every perturbation; return what was *not* caught as expected."""
    problems = []
    for what, how, stage, must in falsifications(variant):
        if isinstance(how, type) and issubclass(how, Encoder):
            encoder, context = how(checkpoint, SPECS[variant], device="cpu"), contextlib.nullcontext()
        elif dataclasses.is_dataclass(how):
            encoder, context = Encoder(checkpoint, how, device="cpu"), contextlib.nullcontext()
        else:
            encoder, context = Encoder(checkpoint, SPECS[variant], device="cpu"), how()
        with context:
            _, failures, first = compare(encoder, fixtures, recorded)
        stages = sorted(set(first.values()), key=stage_order(SPECS[variant]).index)
        caught = bool(first) and stages[0] == stage and (must is None or must <= set(first))
        print(f"  {'caught' if caught else 'MISSED':6s}  {what:40s} {len(first):2d} of "
              f"{len(fixtures)} fixtures, first at {stages[0] if stages else '-'}"
              f" (expected {stage})")
        if not caught:
            problems.append(f"{what}: first failures {stages[:3]} on {sorted(first)[:4]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=list(SPECS),
                        help="variants to check (default: all)")
    parser.add_argument("--mozo-only", action="store_true", help="skip the reference comparison")
    parser.add_argument("--fuzz", action="store_true",
                        help="also put every codepoint through the tokenizer (minutes)")
    parser.add_argument("--long", action="store_true",
                        help="add a fixture at the multilingual 32,768-token context (minutes)")
    parser.add_argument("--falsify", action="store_true",
                        help="perturb the vendor one thing at a time and require each be caught")
    parser.add_argument("--corpus", type=Path,
                        help="write the tokenizer corpus and the reference's ids here, for "
                             "granite_embedding_r2_corpus.py to check under another Python "
                             "(one variant at a time)")
    args = parser.parse_args()
    if args.corpus and len(args.variants) != 1:
        parser.error("--corpus writes one variant's corpus; name exactly one variant")
    torch.set_num_threads(1)

    print(f"reference: {versions()}\n")
    total, problems = 0, []
    for variant in args.variants:
        try:
            checkpoint = resolve("granite_embedding_r2", variant, "torch-fp32")
        except WeightsError as error:
            print(f"{variant}: {error}")
            return 1
        vendor = Encoder(checkpoint, SPECS[variant], device="cpu")
        checked, failures = against_mozo(variant, vendor)
        total += checked
        problems.extend(f"[mozo/{variant}] {f}" for f in failures)
        print(f"{variant}: {checked} mozo-vs-vendor comparisons, {len(failures)} failed")
        if args.mozo_only:
            continue
        checked, failures, fixtures, recorded = against_upstream(
            variant, vendor, args.fuzz, args.long, args.corpus)
        total += checked
        problems.extend(f"[upstream/{variant}] {f}" for f in failures)
        print(f"{variant}: {checked:,} upstream comparisons, {len(failures)} failed")
        if args.falsify:
            print(f"{variant}: falsifying")
            missed = falsify(variant, checkpoint, fixtures, recorded)
            problems.extend(f"[falsify/{variant}] {m}" for m in missed)

    if problems:
        print(f"\n{len(problems)} FAILURES:")
        for problem in problems[:40]:
            print(f"  {problem}")
        return 1
    print(f"\n{total:,} comparisons, all identical" + (", every perturbation caught."
                                                       if args.falsify else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
