#!/usr/bin/env python3
"""Check that mozo's Granite Embedding R2 returns exactly what the published model does.

Three paths reach the same weights and this compares all three: the vendored package
(``mozo/vendors/granite_embedding_r2_deploy``), the reference (``transformers``'
``ModernBertModel`` and the published ``tokenizer.json``), and mozo itself (registry, weights,
adapter).

**The reference is ``transformers``, and that is not a choice.** IBM publishes no inference code
of its own; the checkpoint is a ``ModernBertModel`` and its model card loads it through
``sentence-transformers``, which is that model plus two steps -- take the ``[CLS]`` position, and,
with ``normalize_embeddings=True``, ``F.normalize`` it. Both are written out here, so the gate
needs no ``sentence-transformers``.

**Every stage, compared exactly.** Token ids, the embeddings, each of the twelve layers, the final
norm (whose first row is the raw ``[CLS]`` vector) and the normalised vector, with ``torch.equal``
and no tolerance -- read off ``encode_text`` itself, the path a caller runs.

**Four things about the comparison are pinned.**

*The device is the CPU* and the thread count one, so what is compared is arithmetic.

*The reference runs in float32.* The checkpoint is stored in bfloat16 and ``transformers`` 5 loads
in the stored dtype by default, so a reference loaded the obvious way runs in bfloat16 and every
comparison fails by 3e-03 -- a failure that looks like the extraction's.

*Its attention is SDPA*, the call the vendored layers make. Eager attention is the same arithmetic
in a different order: 1.8e-07 apart.

*Its parameters are re-allocated before comparing*, so ``from_pretrained``'s storage offsets cannot
steer BLAS onto a different path. Compare arithmetic, not addresses.

The reference's version does not move the answer: transformers 4.51.3, which saved the checkpoint,
and 5.8.0 agreed on every stage when the family was built, though 5.x rewrote this model. See
``PROVENANCE.md``.

**The tokenizer's answer must not depend on the Python running it.** ``--fuzz`` adds every
codepoint and runs of mixed combining marks to the tokenizer corpus; ``--corpus PATH`` writes that
corpus with the reference's ids, and ``granite_embedding_r2_corpus.py`` -- standard library only --
checks it under any other Python, where neither ``transformers`` nor torch need be installed.

Run it with the checkpoint in place::

    python tools/verify/granite_embedding_r2.py            # everything
    python tools/verify/granite_embedding_r2.py --fuzz     # plus 1.1M codepoints through the tokenizer
    python tools/verify/granite_embedding_r2.py --fuzz --corpus corpus.json
    python3.14 tools/verify/granite_embedding_r2_corpus.py corpus.json
    python tools/verify/granite_embedding_r2.py --mozo-only
"""

from __future__ import annotations

import argparse
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

from mozo.vendors.granite_embedding_r2_deploy import CONTEXT, SPECS, Encoder  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: The versions this family's parity was established against. Printed, not enforced.
PINNED = {"torch": "2.11.0", "transformers": "5.8.0"}

#: Texts that exercise the tokenizer's awkward corners: whitespace runs that are added tokens,
#: ``[MASK]``'s left strip, special tokens written out, NFC, a mark the reference will not compose
#: across, a codepoint newer than the reference's tables, mixed scripts.
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


def versions() -> str:
    """What is actually installed, next to what parity was established against."""
    import transformers

    actual = {"torch": torch.__version__, "transformers": transformers.__version__}
    return " ".join(f"{name} {actual[name]}" + ("" if actual[name] == want else f" (pinned {want})")
                    for name, want in PINNED.items())


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


def texts(tokenizer) -> dict[str, str]:
    """The fixtures, by name: the awkward ones, lengths either side of the attention window and
    of 128, a documentation page, and one exactly at the 8,192-token context."""
    out = {f"awkward {i}": text for i, text in enumerate(AWKWARD)}
    for length in (63, 64, 65, 66, 127, 128, 129, 130):
        out[f"{length} tokens"] = "hello" + " hello" * (length - 3)
    words = (ROOT / "docs" / "models.md").read_text(encoding="utf-8").split(" ")
    out["docs page"] = prefix_within(tokenizer, words, 1500)
    out["docs at the context"] = prefix_within(tokenizer, words, CONTEXT)
    return out


def tokenizer_corpus(sweep: bool) -> list[str]:
    """Texts for the tokenizer alone: every line of the docs, random mixtures of the awkward parts
    of Unicode, and with *sweep*, every codepoint in contexts that cross each class boundary plus
    bases followed by runs of combining marks -- the shape that exercises NFC's reordering and
    composition, where interpreters disagree."""
    corpus = [line for path in sorted((ROOT / "docs").glob("*.md"))
              for line in path.read_text(encoding="utf-8").split("\n")]
    rng = random.Random(0)
    pool = (list(" \t\n\r\x0b\x0c\x85\xa0\u1680\u2000\u2028\u2029\u3000aA\u00e9\u00df0٠\u00bd'!?[]<>|")
            + ["[MASK]", "[SEP]", "<|endoftext|>", "[unused7]", "  ", " " * 23, "'ll", "\u0301",
               "\U00011d42", "\u088f", "\U000105c9", "\u1100\u1161"]
            + [chr(rng.randrange(0x20, 0x30000)) for _ in range(2000)])
    for _ in range(20000):
        text = "".join(rng.choice(pool) for _ in range(rng.randrange(1, 30)))
        if not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
            corpus.append(text)
    if sweep:
        corpus += [f"a{chr(c)}b {chr(c)}{chr(c)} 1{chr(c)} !{chr(c)} e{chr(c)}\u0301 {chr(c)}[MASK]"
                   for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
        # Every mark with a combining class, plus characters newer than Unicode 13 that later
        # Pythons decompose, after a handful of bases.
        marks = [chr(c) for c in range(0x110000)
                 if not 0xD800 <= c <= 0xDFFF and unicodedata.combining(chr(c))]
        marks += [chr(c) for c in (0x105C9, 0x105E4, 0x11383, 0x11385, 0x1138E, 0x11391, 0x113C5,
                                   0x113C7, 0x113C8, 0x16121, 0x16122, 0x16123)]
        bases = list("aeiouAEOUnc") + ["\u1100", "\u1161", "\u11a8", "\uac00", "\u0915", "\u0627"]
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


def against_upstream(variant: str, ours: Encoder, sweep: bool,
                     corpus_out: Path | None) -> tuple[int, list[str]]:
    """Compare every stage of the vendored model against ``transformers``."""
    from tokenizers import Tokenizer as Reference
    from transformers import AutoModel
    from huggingface_hub import hf_hub_download

    repo, commit = repository(variant), COMMITS[variant]
    theirs = AutoModel.from_pretrained(repo, revision=commit, dtype=torch.float32,
                                       attn_implementation="sdpa").eval()
    for parameter in theirs.parameters():
        parameter.data = parameter.data.clone()
    reference = Reference.from_file(hf_hub_download(repo, "tokenizer.json", revision=commit))

    checked, failures = 0, []
    corpus = tokenizer_corpus(sweep)
    expected = [encoding.ids for encoding in reference.encode_batch(corpus)]
    for text, want in zip(corpus, expected):
        checked += 1
        if ours.tokenizer.encode(text) != want:
            failures.append(f"token ids differ for {text[:50]!r}")
    print(f"  tokenizer: {len(corpus):,} texts")
    if corpus_out is not None:
        corpus_out.write_text(json.dumps({"texts": corpus, "ids": expected}, ensure_ascii=False))
        print(f"  wrote the corpus and the reference's ids to {corpus_out}")

    theirs_stages, our_stages = {}, {}
    hooks = _record(theirs, theirs_stages) + _record(ours.network(), our_stages)
    fixtures = texts(ours.tokenizer)
    vectors = {}
    for name, text in fixtures.items():
        ids = reference.encode(text).ids
        checked += 1
        if ours.tokenizer.encode(text) != ids:
            failures.append(f"{name}: token ids differ")
            continue
        with torch.inference_mode():
            want_cls = theirs(input_ids=torch.tensor([ids])).last_hidden_state[:, 0]
        vectors[text] = ours.encode_text(text)  # fills our_stages through the hooks
        for stage, want in theirs_stages.items():
            checked += 1
            if not torch.equal(our_stages[stage], want):
                failures.append(f"{name}: {stage} differs by "
                                f"{(our_stages[stage] - want).abs().max():.3e}")
        checked += 1
        if not torch.equal(vectors[text], F.normalize(want_cls, p=2, dim=1)):
            failures.append(f"{name}: the unit vector differs")
    for hook in hooks:
        hook.remove()
    print(f"  model: {len(fixtures)} texts, every stage")

    # Encoding them together must not move any vector. Repeats force equal-length batches.
    batch = list(vectors)
    batch += [text for text in batch if len(text) < 2000] * 3
    checked += 1
    if not torch.equal(ours.encode_text(batch), torch.cat([vectors[text] for text in batch])):
        failures.append("encoding texts together moved a vector")
    return checked, failures


def against_mozo(variant: str, vendor: Encoder) -> tuple[int, list[str]]:
    """Compare mozo's adapter against the vendored encoder it wraps: it should change nothing."""
    from mozo.adapters.granite_embedding_r2 import GraniteEmbeddingR2Predictor

    adapter = GraniteEmbeddingR2Predictor(variant, device="cpu")
    checked, failures = 0, []
    batch = AWKWARD + ["one more, to make a batch of eleven"]
    checked += 1
    if not torch.equal(torch.from_numpy(adapter.encode_text(batch)), vendor.encode_text(batch)):
        failures.append("the adapter's vectors differ from the vendor's")
    checked += 1
    if adapter.revision is None:
        failures.append("the adapter does not say which revision produced its vectors")
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=list(SPECS),
                        help="variants to check (default: all)")
    parser.add_argument("--mozo-only", action="store_true", help="skip the reference comparison")
    parser.add_argument("--fuzz", action="store_true",
                        help="also put every codepoint through the tokenizer (about a minute)")
    parser.add_argument("--corpus", type=Path,
                        help="write the tokenizer corpus and the reference's ids here, for "
                             "granite_embedding_r2_corpus.py to check under another Python")
    args = parser.parse_args()
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
        if not args.mozo_only:
            checked, failures = against_upstream(variant, vendor, args.fuzz, args.corpus)
            total += checked
            problems.extend(f"[upstream/{variant}] {f}" for f in failures)
            print(f"{variant}: {checked:,} upstream comparisons, {len(failures)} failed")

    if problems:
        print(f"\n{len(problems)} FAILURES:")
        for problem in problems[:40]:
            print(f"  {problem}")
        return 1
    print(f"\n{total:,} comparisons, all identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
