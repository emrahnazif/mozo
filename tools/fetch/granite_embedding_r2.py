#!/usr/bin/env python3
"""Fetch Granite Embedding R2's published checkpoint into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/granite_embedding_r2.py                  # every variant
    python tools/fetch/granite_embedding_r2.py small
    python tools/fetch/granite_embedding_r2.py --derive-vocab small               # Python 3.10
    python tools/fetch/granite_embedding_r2.py --derive-vocab multilingual-small  # any Python

**Pinned to a commit, and verified against the Hub's own hash.** Files are read from
``resolve/<commit>/``, not ``resolve/main/``, so a re-release upstream cannot change what this
fetches. The safetensors file's sha256 is read from ``/api/models/<repo>/revision/<commit>`` at
fetch time, so a mismatch means the bytes changed under us rather than that someone transcribed a
digest wrongly.

**IBM stores the weights in bfloat16, and mozo publishes them that way.** The artifact key is
``torch-fp32`` because the key names what runs: the encoder loads the file into float32 and runs
in float32. Storing it in bfloat16 costs nothing -- every bfloat16 value is a float32 value, so
the upcast at load is exact and the vectors are bit-identical to a float32 file's -- and halves
the download, to 95 MB. Revision ``2026-11-20`` published the same tensors cast to float32 (191
MB); ``2026-11-24`` is those tensors as IBM stores them, and its vectors are identical.

**``pytorch_model.bin`` is not used.** The repository publishes the same weights twice; the
safetensors file is read, because a pickle is not something to load from the internet. That
means ``safetensors`` is a dependency of *this script*, and nothing in ``mozo/`` imports it.

The output is ``weights/granite_embedding_r2/<variant>/<revision>/torch-fp32.pth`` and a NOTICE.
``multilingual-small`` is published under its own revision, ``2026-11-25``; ``small`` stays at
``2026-11-24``.
The LICENSE beside them is not written by this script: a licence is part of what is published, so
it lives in the weights tree like any other artifact.
"""

from __future__ import annotations

import gzip
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    digest, download_verified, hub_lfs_sha256s, require_licence, variant_parser)

from mozo.vendors.granite_embedding_r2_deploy.config import SPECS  # noqa: E402
from mozo.vendors.granite_embedding_r2_deploy.tokenizer import vocab_path  # noqa: E402

_HF = "https://huggingface.co"

#: The revision each variant is published under: the newest, which a fetch writes by default.
REVISIONS = {"small": "2026-11-24", "multilingual-small": "2026-11-25"}

#: The upstream commit every file is read from, per variant. A commit rather than ``main``, so
#: what is fetched cannot move.
COMMITS = {"small": "2ab6fa8ea2d674564defd37171ae19079b864b33",
           "multilingual-small": "835ad14087e140460703cf0fae09f97d469d65c2"}

#: Each variant's paper, for its NOTICE.
PAPERS = {"small": "Granite Embedding R2 Models (2025), arXiv:2508.21085",
          "multilingual-small": "Granite Embedding Multilingual R2 Models (2026), arXiv:2605.13521"}

#: What each variant embeds, for its NOTICE.
LANGUAGES = {"small": "an English", "multilingual-small": "a multilingual (200+ languages)"}

#: Where Apache-2.0's canonical text lives. IBM ships no LICENSE file in the repository; the
#: licence is stated in the model card's metadata.
LICENCE_SOURCE = "https://www.apache.org/licenses/LICENSE-2.0.txt"


def repository(variant: str) -> str:
    """Where IBM publishes *variant*."""
    return f"ibm-granite/{SPECS[variant].upstream}"


def resolve(variant: str, name: str) -> str:
    """The URL of one file of *variant*, at its pinned commit."""
    return f"{_HF}/{repository(variant)}/resolve/{COMMITS[variant]}/{name}"


def _notice(variant: str, sha256: str) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    repo = repository(variant)
    return (
        f"Granite Embedding R2 -- {variant} ({SPECS[variant].upstream})\n"
        f"{LANGUAGES[variant]} text embedding model: text in, one {SPECS[variant].width}-dimensional "
        "vector out\n\n"
        "Copyright IBM Corp.\n\n"
        f"Source:  {_HF}/{repo} at commit {COMMITS[variant]}\n"
        f"Paper:   {PAPERS[variant]}\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one), as stated in the\n"
        "         model card's metadata. The repository is not gated.\n\n"
        "This file is not the byte stream IBM serves. tools/fetch/granite_embedding_r2.py\n"
        "verifies model.safetensors against the sha256 the Hub records, then writes the same\n"
        "tensors back out as an ordinary checkpoint, still in bfloat16. mozo upcasts them to\n"
        "float32 when loading, which is exact, and no tensor is renamed or dropped.\n"
        f"The source file is:\n           model.safetensors  sha256 {sha256}\n\n"
        "The tokenizer vocabulary shipped inside mozo derives from tokenizer.json in this same\n"
        "repository, under the same licence.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, and place it with its notice."""
    target = weights_dir / "granite_embedding_r2" / variant / revision / "torch-fp32.pth"
    staged = target.with_name("model.safetensors")

    sha256 = hub_lfs_sha256s(repository(variant), COMMITS[variant])["model.safetensors"]
    if not target.is_file():
        download_verified(resolve(variant, "model.safetensors"), staged, sha256,
                          label=variant, width=6, detail="Apache-2.0")
        from safetensors.torch import load_file

        state = load_file(staged)
        dtypes = {str(tensor.dtype) for tensor in state.values()}
        if dtypes != {"torch.bfloat16"}:
            raise SystemExit(f"expected every tensor in bfloat16, found {sorted(dtypes)}; the "
                             "upcast at load is only known to be exact from bfloat16")
        print(f"  {variant:6s} writing {len(state)} tensors, bfloat16 as published")
        torch.save(state, target)
        staged.unlink()

    (target.parent / "NOTICE").write_text(_notice(variant, sha256))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def _ranges(codepoints: list[int]) -> list[list[int]]:
    """Sorted codepoints as inclusive ``[low, high]`` runs."""
    runs: list[list[int]] = []
    for codepoint in codepoints:
        if runs and runs[-1][1] == codepoint - 1:
            runs[-1][1] = codepoint
        else:
            runs.append([codepoint, codepoint])
    return runs


def character_classes(published: str, lstrip: list[str]) -> dict[str, list]:
    """Read every Unicode table the tokenizer needs out of the reference itself.

    The split pattern asks whether a character is a letter (``\\p{L}``), a digit (``\\p{N}``) or
    whitespace (``\\s``), and the reference answers from the regular-expression engine compiled
    into ``tokenizers`` -- whose Unicode version is neither Python's nor the ``regex`` package's.
    Measured: ``regex`` 2026.4 calls 4,657 codepoints letters or digits that the reference treats
    as punctuation. So each codepoint is put to the reference's own pre-tokenizer and its class is
    read off how the reference splits it, which makes the answer a property of the package rather
    than of whatever is installed.

    An added token declared ``lstrip`` -- only ``[MASK]``, here -- strips whitespace to its left by
    Rust's ``char::is_whitespace``, which is read the same way: whatever the reference strips is
    whitespace. Every such token is probed, and they must agree, since the asset carries one set.

    And NFC is measured against Unicode 13's, which is what the tokenizer applies (see its
    ``_nfc``). The reference's decompositions and compositions agree with Unicode 13's on every
    codepoint. Its *combining classes* do not quite: 58 marks that Unicode 10 to 13 gave a nonzero
    class -- Sogdian, Dogra, Dives Akuru, Gondi viramas and nuktas among them -- it treats as class
    0, so it never reorders them and they block composition across themselves. Those are found by
    probing every mark against marks of every other class, and carried as *boundaries*: places
    the tokenizer's NFC does not reach across, which is exactly how a class-0 character with no
    decomposition behaves. Codepoints Unicode 13 had not assigned are boundaries for the same
    reason. Anything the reference does that this cannot express -- a different nonzero class, a
    mark it reorders that Unicode 13 does not -- refuses the derivation rather than writing an
    asset that is quietly wrong.

    Needs ``tokenizers`` installed and Python 3.10 running, both bootstrap-only.
    """
    from tokenizers import Tokenizer, normalizers, pre_tokenizers

    if unicodedata.unidata_version != "13.0.0":
        raise SystemExit("derive the vocabulary under Python 3.10, whose Unicode 13 tables the "
                         f"asset is defined against; this is Unicode {unicodedata.unidata_version}")

    reference = Tokenizer.from_str(published)
    pre = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    nfc, nfd = normalizers.NFC(), normalizers.NFD()
    a = reference.encode("a", add_special_tokens=False).ids
    stripping = {token: a + [reference.token_to_id(token)] for token in lstrip}

    def reordered(sequence: str) -> bool:
        return nfd.normalize_str(sequence) != sequence

    # One mark per combining class, to reorder each candidate against -- but only a mark the
    # reference itself reorders against U+0301 (class 230) the way its Unicode 13 class says.
    # Class 6, for one, has no such mark: its only two are among the marks the reference treats
    # as class 0.
    probes = {230: "\u0301"}
    for codepoint in range(0x110000):
        character = chr(codepoint)
        klass = unicodedata.combining(character)
        if not klass or klass in probes or nfd.normalize_str(character) != character:
            continue
        if reordered(f"a\u0301{character}" if klass < 230 else f"a{character}\u0301"):
            probes[klass] = character

    # Every character that appears in some canonical decomposition, and so could be composed.
    canonical_parts = {
        chr(int(part, 16))
        for codepoint in range(0x110000)
        if (decomposition := unicodedata.decomposition(chr(codepoint))) and decomposition[0] != "<"
        for part in decomposition.split()}

    letters, numbers, spaces, boundaries = [], [], [], []
    strip: dict[str, list[int]] = {token: [] for token in lstrip}
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:  # lone surrogates are not encodable text
            continue
        character = chr(codepoint)
        if len(pre.pre_tokenize_str("a" + character)) == 1:
            letters.append(codepoint)
        elif len(pre.pre_tokenize_str("1" + character)) == 1:
            numbers.append(codepoint)
        elif len(pre.pre_tokenize_str("!" + character)) != 1:
            spaces.append(codepoint)
        for token, stripped in stripping.items():
            if reference.encode(f"a{character}{token}", add_special_tokens=False).ids == stripped:
                strip[token].append(codepoint)
        decomposed = nfd.normalize_str(character)
        if (nfc.normalize_str(character) != unicodedata.normalize("NFC", character)
                or nfc.normalize_str(decomposed) != unicodedata.normalize("NFC", decomposed)):
            raise SystemExit(f"the reference's NFC disagrees with Unicode 13's at U+{codepoint:04X}; "
                             "the tokenizer's NFC would be wrong there")

        if unicodedata.category(character) == "Cn":
            boundaries.append(codepoint)
            continue
        # The reference's combining class for this character, as far as reordering shows it:
        # against each probe, does it reorder exactly where Unicode 13's class says it should?
        # A character that decomposes is covered by the NFC check above, and what it decomposes
        # to is probed in its own right.
        klass = unicodedata.combining(character)
        if decomposed != character or (
                not klass and not unicodedata.category(character).startswith("M")):
            continue
        agrees = starter = True
        for other, probe in probes.items():
            for sequence, unicode13 in ((f"a{probe}{character}", 0 < klass < other),
                                        (f"a{character}{probe}", 0 < other < klass)):
                ref = reordered(sequence)
                agrees &= ref == unicode13
                starter &= not ref
        if agrees:
            continue
        if starter and klass:
            # A wall only if nothing composes with it either: it must not appear in any
            # canonical decomposition, or NFC could join it to a neighbour across the split.
            if character in canonical_parts:
                raise SystemExit(f"U+{codepoint:04X} takes part in a composition, so splitting "
                                 "NFC at it would not be exact")
            boundaries.append(codepoint)
            continue
        raise SystemExit(f"the reference reorders U+{codepoint:04X} (Unicode 13 class {klass}) in "
                         "a way the tokenizer cannot express")

    if len({tuple(codepoints) for codepoints in strip.values()}) > 1:
        raise SystemExit(f"the lstrip tokens {lstrip} strip different whitespace; the asset "
                         "carries one set")
    return {"letters": _ranges(letters), "numbers": _ranges(numbers), "spaces": _ranges(spaces),
            "strip": _ranges(next(iter(strip.values()), [])), "boundaries": _ranges(boundaries)}


def _published_tokenizer(variant: str) -> str:
    """*variant*'s ``tokenizer.json`` at its pinned commit, as text: the reference is built from
    this exact string."""
    with urllib.request.urlopen(resolve(variant, "tokenizer.json"), timeout=300) as response:
        return response.read().decode("utf-8")


def _template(spec: dict) -> dict[str, list[int]]:
    """The template as ids: the special tokens before the text, and those after it."""
    processor = spec["post_processor"]
    names = [next(iter(part.values()))["id"] for part in processor["single"]]
    if names.count("A") != 1 or sum("Sequence" in part for part in processor["single"]) != 1:
        raise SystemExit(f"the template is {names}; the tokenizer handles one sequence between "
                         "special tokens")
    middle = names.index("A")
    ids = {name: processor["special_tokens"][name]["ids"] for name in names if name != "A"}
    return {"before": [i for name in names[:middle] for i in ids[name]],
            "after": [i for name in names[middle + 1:] for i in ids[name]]}


def _pieces(model: dict) -> tuple[list[str], list[list[int]]]:
    """The vocabulary in id order, and the merges as pairs of ids."""
    vocab = model["vocab"]
    pieces = [piece for piece, _ in sorted(vocab.items(), key=lambda item: item[1])]
    if [vocab[piece] for piece in pieces] != list(range(len(pieces))):
        raise SystemExit("the vocabulary's ids are not contiguous from 0")
    merges = [merge.split(" ") if isinstance(merge, str) else merge for merge in model["merges"]]
    return pieces, [[vocab[left], vocab[right]] for left, right in merges]


def _write_asset(variant: str, payload: dict) -> None:
    path = vocab_path(SPECS[variant].vocabulary)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0: gzip stamps the current time into its header by default, which would make the
    # committed asset differ on every rebuild and its recorded sha256 meaningless.
    path.write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False).encode(), 9, mtime=0))
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size:,} bytes")
    print(f"  {len(payload['pieces']):,} pieces, {len(payload['merges']):,} merges, "
          f"{len(payload['added'])} added tokens")
    print("  " + ", ".join(f"{len(payload[key])} {key} runs"
                           for key in ("letters", "numbers", "spaces", "strip", "upper", "lower",
                                       "word", "boundaries") if key in payload))
    print(f"  sha256 {digest(path)}")


def derive_english() -> None:
    """Rebuild the English asset from ``small``'s published ``tokenizer.json``.

    Merges are stored as pairs of ids rather than pairs of pieces: it is what the merge loop wants,
    and it avoids escaping questions in an asset whose pieces include raw spaces. Every added
    token is carried with the two flags that change matching -- whether it is matched before or
    after normalising, and whether it strips whitespace to its left. A flag the tokenizer does not
    implement (``rstrip``, ``single_word``) refuses the derivation instead of being dropped.
    """
    published = _published_tokenizer("small")
    spec = json.loads(published)

    expected = {
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                          "use_regex": True},
    }
    for stage, value in expected.items():
        if spec[stage] != value:
            raise SystemExit(f"tokenizer.json's {stage} is {spec[stage]}, not {value}; the "
                             "tokenizer implements only the latter")
    model = spec["model"]
    if model["type"] != "BPE" or model["dropout"] or model["ignore_merges"] or model["byte_fallback"]:
        settings = {k: v for k, v in model.items() if k not in ("vocab", "merges")}
        raise SystemExit(f"unexpected BPE settings: {settings}")

    pieces, merges = _pieces(model)
    added = []
    for token in spec["added_tokens"]:
        if token["rstrip"] or token["single_word"]:
            raise SystemExit(f"{token['content']!r} wants rstrip or single_word, which the "
                             "tokenizer does not implement")
        added.append({"id": token["id"], "content": token["content"],
                      "normalized": token["normalized"], "lstrip": token["lstrip"]})

    lstrip = [token["content"] for token in added if token["lstrip"]]
    classes = character_classes(published, lstrip)
    # ``lstrip`` backs up over whitespace *before* a match. A raw token that itself ended in
    # whitespace could match differently from the reference, which strips inside the match.
    strip = {chr(c) for low, high in classes["strip"] for c in range(low, high + 1)}
    if any(token["content"][-1:] in strip for token in added if not token["normalized"]):
        raise SystemExit("a raw added token ends in whitespace; the lstrip back-off assumes not")
    _write_asset("small", {"pieces": pieces, "merges": merges, "added": added,
                           "template": _template(spec), **classes})


#: o200k's split pattern, exactly as the multilingual ``tokenizer.json`` writes it. The tokenizer
#: rebuilds this pattern from probed classes, so a published pattern that differs by one character
#: refuses the derivation rather than producing an asset for a pattern it does not implement.
O200K = ("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
         "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
         "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+"
         "[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")


def o200k_classes(published: str, single_word: list[str]) -> dict:
    """Read every class o200k's split pattern names out of the reference, by probing it.

    The pattern asks seven Unicode questions -- ``\\p{L}``, ``\\p{N}``, ``\\s``, and two unions of
    letter categories with ``\\p{M}`` -- and case-insensitive matching asks an eighth, case folding.
    Each codepoint is put to the reference's own pre-tokenizer in a context where only one answer
    can split it differently, so the asset holds the reference's Unicode rather than Python's:

    - *lower* (``Ll Lm Lo M``): ``"a" + c`` is one piece.
    - *not whitespace, letter or digit*: ``"!!" + c`` is one piece (punctuation runs on).
    - *digit*: ``"1" + c`` is one piece.
    - *whitespace*: ``"!" + c`` is two pieces and ``c`` is not a digit; plus CR and LF, which
      the punctuation branch's ``[\\r\\n/]*`` would otherwise swallow.
    - *letter*: everything else. *Mark* is lower and not a letter.
    - *upper* (``Lu Lt Lm Lo M``): a letter for which ``c + "Aa"`` is one piece, and every mark.
    - *contractions*: for each letter of ``'s 't 're 've 'm 'll 'd``, the characters ``c`` that
      join ``"a'" + c`` (with the rest of the suffix) into one piece. ``ſ`` joins ``'s``.

    And the characters that stop a ``single_word`` token from counting -- the regular-expression
    engine's word characters -- read off which neighbours, before and after, make the reference
    ignore the first such token. Before and after must agree, and the other tokens are checked on a
    sample, since the asset carries one set.
    """
    from tokenizers import Tokenizer

    reference = Tokenizer.from_str(published)
    pre = reference.pre_tokenizer

    def pieces(text: str) -> int:
        return len(pre.pre_tokenize_str(text))

    every = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    lower = {cp for cp in every if pieces("a" + chr(cp)) == 1}
    other = {cp for cp in every if pieces("!!" + chr(cp)) == 1}
    numbers = {cp for cp in every if pieces("1" + chr(cp)) == 1}
    spaces = ({cp for cp in every if pieces("!" + chr(cp)) == 2} - numbers) | {0x0A, 0x0D}
    letters = set(every) - other - numbers - spaces
    marks = other & lower
    upper = {cp for cp in letters if pieces(chr(cp) + "Aa") == 1} | marks
    if not lower - marks <= letters:
        raise SystemExit("a character the lower-case class takes is neither letter nor mark")

    # Each contraction is a sequence of classes. Which letter a single-character suffix folds to
    # is not observable -- ``'s 't 'm 'd`` are alternatives of one group -- so they are one class,
    # and so are ``r`` and ``v``, both followed by ``e``.
    def joining(before: str, after: str) -> str:
        return "".join(chr(cp) for cp in every if pieces(before + chr(cp) + after) == 1)

    contractions = [[joining("a'", "")], [joining("a'", "e"), joining("a'r", "")],
                    [joining("a'", "l"), joining("a'l", "")]]
    if joining("a'v", "") != contractions[1][1]:
        raise SystemExit("'re and 've end in different classes; the asset holds them as one")
    firsts = [set(alternative[0]) for alternative in contractions]
    if any(firsts[i] & firsts[j] for i in range(3) for j in range(i + 1, 3)):
        raise SystemExit("two contractions start with the same character; their order would "
                         "matter, and the asset does not keep it")

    word: list[int] = []
    if single_word:
        def stops(token: str, before: str, after: str) -> bool:
            ids = reference.encode(before + token + after, add_special_tokens=False).ids
            return reference.token_to_id(token) not in ids

        first = single_word[0]
        word = [cp for cp in every if stops(first, chr(cp), "")]
        if word != [cp for cp in every if stops(first, "", chr(cp))]:
            raise SystemExit(f"{first!r} is stopped by different characters before and after")
        sample = list(range(0x250)) + word[::997] + [0x200C, 0x200D, 0x3000, 0x10FFFF]
        known = set(word)
        for token in single_word[1:]:
            if any(stops(token, chr(cp), "") != (cp in known) for cp in sample):
                raise SystemExit(f"{token!r} is stopped by different characters than {first!r}")
    return {"letters": _ranges(sorted(letters)), "numbers": _ranges(sorted(numbers)),
            "spaces": _ranges(sorted(spaces)), "upper": _ranges(sorted(upper)),
            "lower": _ranges(sorted(lower)), "contractions": contractions,
            "word": _ranges(word)}


def derive_multilingual() -> None:
    """Rebuild the multilingual asset from ``multilingual-small``'s published ``tokenizer.json``.

    Checks every stage against what the tokenizer implements first -- no normaliser, o200k's split
    then the byte mapping without its own regex, BPE taking known pieces whole, no added token
    matched after normalising, none with ``lstrip`` or ``rstrip`` -- and refuses anything else.
    Needs no particular Python: there is no normaliser, so no NFC to pin to a Unicode version.
    """
    published = _published_tokenizer("multilingual-small")
    spec = json.loads(published)
    expected = {
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": O200K}, "behavior": "Isolated",
             "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
             "use_regex": False}]},
    }
    for stage, value in expected.items():
        if spec[stage] != value:
            raise SystemExit(f"tokenizer.json's {stage} is {spec[stage]}, not {value}; the "
                             "tokenizer implements only the latter")
    model = spec["model"]
    settings = {k: v for k, v in model.items() if k not in ("vocab", "merges")}
    if settings != {"type": "BPE", "dropout": None, "unk_token": None,
                    "continuing_subword_prefix": None, "end_of_word_suffix": None,
                    "fuse_unk": False, "byte_fallback": False, "ignore_merges": True}:
        raise SystemExit(f"unexpected BPE settings: {settings}")

    pieces, merges = _pieces(model)
    added = []
    for token in spec["added_tokens"]:
        if token["normalized"] or token["lstrip"] or token["rstrip"]:
            raise SystemExit(f"{token['content']!r} is normalised, lstrip or rstrip, which this "
                             "pipeline does not implement")
        added.append({"id": token["id"], "content": token["content"], "normalized": False,
                      "lstrip": False, "single_word": token["single_word"]})
    single_word = [token["content"] for token in added if token["single_word"]]
    _write_asset("multilingual-small", {
        "pattern": "o200k", "normalizer": None, "ignore_merges": True,
        "pieces": pieces, "merges": merges, "added": added, "template": _template(spec),
        **o200k_classes(published, single_word)})


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", revision=None)
    parser.add_argument("--derive-vocab", action="store_true",
                        help="rebuild the named variants' tokenizer assets and exit")
    args = parser.parse_args()

    if args.derive_vocab:
        derivations = {"small": derive_english, "multilingual-small": derive_multilingual}
        if not args.variants or not set(args.variants) <= set(derivations):
            raise SystemExit(f"--derive-vocab takes one or more of {list(derivations)}")
        for variant in args.variants:
            derivations[variant]()
        return 0

    wanted = args.variants or list(SPECS)
    unknown = [name for name in wanted if name not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(SPECS)}")

    for variant in wanted:
        fetch(variant, args.revision or REVISIONS[variant], args.weights_dir)

    print(f"\n{len(wanted)} checkpoint(s) in {args.weights_dir}, all Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
