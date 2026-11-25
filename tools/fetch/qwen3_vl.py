#!/usr/bin/env python3
"""Fetch Qwen3-VL's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling: runs on a machine you control, never ships, and is the only place the upstream
URLs live.

    python tools/fetch/qwen3_vl.py                  # every variant
    python tools/fetch/qwen3_vl.py 2b
    python tools/fetch/qwen3_vl.py --derive-vocab   # rebuild the tokenizer asset (Python 3.10)

**Pinned to a commit, verified against the Hub's own hash.** Files are read from
``resolve/<commit>/`` and each safetensors file's sha256 is read from the Hub's API at fetch time.

**Qwen stores the weights in bfloat16, and mozo publishes them that way, as ``torch-bf16``.** The
``torch-fp32`` runtime is built from that file at load (``mozo.registry.DERIVES``): every bfloat16
value is a float32 value, so the upcast is exact and is what ``from_pretrained(dtype=float32)``
does. The 4B checkpoint's two shards are joined into one file; no tensor is renamed, cast or
dropped. ``safetensors`` is a dependency of *this script* only.

The output is ``weights/qwen3_vl/<variant>/<revision>/torch-bf16.pth`` and a NOTICE. The LICENSE
beside them is not written here (see ``require_licence``).
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

from mozo.vendors.qwen3_vl_deploy.config import SPECS  # noqa: E402
from mozo.vendors.qwen3_vl_deploy.tokenizer import VOCAB_PATH  # noqa: E402

_HF = "https://huggingface.co"

#: The revision this family is published under.
REVISION = "2026-11-25"

#: The upstream commit every file is read from, per variant.
COMMITS = {
    "2b": "89644892e4d85e24eaac8bacfd4f463576704203",
    "4b": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
}

#: The weight files of each repository at its commit.
SHARDS = {
    "2b": ["model.safetensors"],
    "4b": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
}

#: Qwen ships no LICENSE file in these repositories; the licence is the model card's metadata.
LICENCE_SOURCE = "https://www.apache.org/licenses/LICENSE-2.0.txt"


def repository(variant: str) -> str:
    return f"Qwen/{SPECS[variant].upstream}"


def resolve(variant: str, name: str) -> str:
    return f"{_HF}/{repository(variant)}/resolve/{COMMITS[variant]}/{name}"


def _notice(variant: str, hashes: dict[str, str]) -> str:
    repo = repository(variant)
    files = "\n".join(f"           {name}  sha256 {sha}" for name, sha in hashes.items())
    return (
        f"Qwen3-VL -- {variant} ({SPECS[variant].upstream})\n"
        "a vision-language model: an image and a question in, text out\n\n"
        "Copyright Alibaba Cloud (the Qwen team).\n\n"
        f"Source:  {_HF}/{repo} at commit {COMMITS[variant]}\n"
        "Paper:   Qwen3-VL Technical Report (2025); Qwen2.5-VL, arXiv:2502.13923\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one), as stated in the\n"
        "         model card's metadata (`license: apache-2.0`). The repository is not gated and\n"
        "         carries no LICENSE file of its own; the authors' code repository,\n"
        "         github.com/QwenLM/Qwen3-VL, is Apache-2.0.\n\n"
        "This file is not the byte stream Qwen serves. tools/fetch/qwen3_vl.py verifies each\n"
        "safetensors file against the sha256 the Hub records, then writes the same tensors out\n"
        "as one ordinary checkpoint, still in bfloat16. Nothing is renamed, cast or dropped.\n"
        f"The source files are:\n{files}\n\n"
        "The tokenizer vocabulary shipped inside mozo derives from tokenizer.json in this same\n"
        "repository, under the same licence.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, and place it with its notice."""
    target = weights_dir / "qwen3_vl" / variant / revision / "torch-bf16.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    recorded = hub_lfs_sha256s(repository(variant), COMMITS[variant])
    hashes = {name: recorded[name] for name in SHARDS[variant]}
    if not target.is_file():
        from safetensors.torch import load_file

        state: dict[str, torch.Tensor] = {}
        staged = []
        for name in SHARDS[variant]:
            path = target.with_name(name)
            if not path.is_file() or digest(path) != hashes[name]:
                download_verified(resolve(variant, name), path, hashes[name],
                                  label=variant, width=3, detail="Apache-2.0")
            part = load_file(path)
            if set(part) & set(state):
                raise SystemExit(f"{name} repeats tensors of an earlier shard")
            state.update(part)
            staged.append(path)
        dtypes = {str(t.dtype) for t in state.values()}
        if dtypes != {"torch.bfloat16"}:
            raise SystemExit(f"expected every tensor in bfloat16, found {sorted(dtypes)}")
        print(f"  {variant:3s} writing {len(state)} tensors, bfloat16 as published")
        torch.save(state, target)
        for path in staged:
            path.unlink()
    (target.parent / "NOTICE").write_text(_notice(variant, hashes))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def _ranges(codepoints: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for codepoint in codepoints:
        if runs and runs[-1][1] == codepoint - 1:
            runs[-1][1] = codepoint
        else:
            runs.append([codepoint, codepoint])
    return runs


def character_classes(published: str) -> dict[str, object]:
    """Read every Unicode table the tokenizer needs out of the reference itself.

    The split pattern asks whether a character is a letter (``\\p{L}``), a digit (``\\p{N}``) or
    whitespace (``\\s``), and which characters ``(?i:...)`` folds onto the contraction letters --
    all answered by the Oniguruma engine inside ``tokenizers``, whose Unicode version is neither
    Python's nor ``regex``'s. So each codepoint is put to the reference's own Split pre-tokenizer
    and its class read off how it is split:

    - letter: ``"a" + c`` is one piece (``\\p{L}+`` continues);
    - digit: not a letter, not CR/LF, and ``c + "a"`` is two pieces (``\\p{N}`` takes one
      character and nothing in front of a letter run may be a digit);
    - whitespace: CR, LF, or neither of the above and ``"!" + c`` is two pieces;
    - the contractions ``(?i:'s|'t|'re|'ve|'m|'ll|'d)``: a letter ``c`` completes a one-letter
      contraction if ``"'" + c + "a"`` is two pieces (the branch matched ahead of the letter run);
      for ``'re``, ``'ve`` and ``'ll`` each position is probed with the other held at its plain
      letter. Stored as the set of one-letter completions and a pair of sets per two-letter suffix;
      the reference's folding is per character (``ſ`` folds onto ``s``), which the fuzz confirms.

    NFC is measured against Unicode 13's, exactly as for Granite Embedding R2: decompositions and
    compositions must agree everywhere; marks whose combining class the reference treats
    differently, and every codepoint Unicode 13 left unassigned, become *boundaries* the
    tokenizer's NFC does not reach across. Anything else refuses the derivation.

    Needs ``tokenizers`` installed and Python 3.10 running, both bootstrap-only.
    """
    from tokenizers import Regex, normalizers, pre_tokenizers

    if unicodedata.unidata_version != "13.0.0":
        raise SystemExit("derive the vocabulary under Python 3.10, whose Unicode 13 tables the "
                         f"asset is defined against; this is Unicode {unicodedata.unidata_version}")

    spec = json.loads(published)
    pattern = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    split = pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)

    def count(text: str) -> int:
        return len(split.pre_tokenize_str(text))

    nfc, nfd = normalizers.NFC(), normalizers.NFD()

    def reordered(sequence: str) -> bool:
        return nfd.normalize_str(sequence) != sequence

    probes = {230: "́"}
    for codepoint in range(0x110000):
        character = chr(codepoint)
        klass = unicodedata.combining(character)
        if not klass or klass in probes or nfd.normalize_str(character) != character:
            continue
        if reordered(f"á{character}" if klass < 230 else f"a{character}́"):
            probes[klass] = character
    canonical_parts = {
        chr(int(part, 16))
        for codepoint in range(0x110000)
        if (decomposition := unicodedata.decomposition(chr(codepoint))) and decomposition[0] != "<"
        for part in decomposition.split()}

    single: list[int] = []
    pairs: dict[str, tuple[list[int], list[int]]] = {s: ([], []) for s in ("re", "ve", "ll")}
    letters, numbers, spaces, boundaries = [], [], [], []
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        character = chr(codepoint)
        if count("a" + character) == 1:
            letters.append(codepoint)
        elif character in "\r\n":
            spaces.append(codepoint)
        elif count(character + "a") == 2:
            numbers.append(codepoint)
        elif count("!" + character) == 2:
            spaces.append(codepoint)
        # Only a letter can fold onto one; a non-letter would split off as punctuation instead and
        # look the same. Every occurrence of the letter in its suffix is replaced: "ll" probes "'cca".
        if letters and letters[-1] == codepoint:
            if count("'" + character + "a") == 2:
                single.append(codepoint)
            else:
                for suffix, (first, second) in pairs.items():
                    if count("'" + character + suffix[1] + "a") == 2:
                        first.append(codepoint)
            for suffix, (first, second) in pairs.items():
                if count("'" + suffix[0] + character + "a") == 2:
                    second.append(codepoint)

        decomposed = nfd.normalize_str(character)
        if (nfc.normalize_str(character) != unicodedata.normalize("NFC", character)
                or nfc.normalize_str(decomposed) != unicodedata.normalize("NFC", decomposed)):
            raise SystemExit(f"the reference's NFC disagrees with Unicode 13's at U+{codepoint:04X}")
        if unicodedata.category(character) == "Cn":
            boundaries.append(codepoint)
            continue
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
            if character in canonical_parts:
                raise SystemExit(f"U+{codepoint:04X} takes part in a composition")
            boundaries.append(codepoint)
            continue
        raise SystemExit(f"the reference reorders U+{codepoint:04X} in a way the tokenizer "
                         "cannot express")

    if any(c > 0xFFFF for c in spaces):
        raise SystemExit("whitespace above the BMP; the tokenizer's single class assumes none")
    return {"letters": _ranges(letters), "numbers": _ranges(numbers), "spaces": _ranges(spaces),
            "boundaries": _ranges(boundaries),
            "contractions": {"'": _ranges(single),
                             **{s: [_ranges(a), _ranges(b)] for s, (a, b) in pairs.items()}}}


def derive_vocab() -> None:
    """Rebuild the tokenizer asset from the published ``tokenizer.json``.

    Committed rather than fetched, because it has to be in the wheel before a user runs anything.
    Merges are stored as pairs of ids. Every stage of ``tokenizer.json`` is checked against what
    the tokenizer implements, and anything else refuses the derivation.
    """
    with urllib.request.urlopen(resolve("2b", "tokenizer.json"), timeout=300) as response:
        published = response.read().decode("utf-8")
    spec = json.loads(published)
    for variant in SPECS:
        with urllib.request.urlopen(resolve(variant, "tokenizer.json"), timeout=300) as response:
            if response.read().decode("utf-8") != published:
                raise SystemExit(f"{variant}'s tokenizer.json differs from 2b's; one asset cannot "
                                 "serve both")

    expected = {
        "normalizer": {"type": "NFC"},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
                    "use_regex": False},
    }
    for stage, value in expected.items():
        if spec[stage] != value:
            raise SystemExit(f"tokenizer.json's {stage} is {spec[stage]}, not {value}")
    pre = spec["pre_tokenizer"]
    if (pre["type"] != "Sequence" or len(pre["pretokenizers"]) != 2
            or pre["pretokenizers"][0]["type"] != "Split"
            or pre["pretokenizers"][0]["behavior"] != "Isolated" or pre["pretokenizers"][0]["invert"]
            or pre["pretokenizers"][1] != {"type": "ByteLevel", "add_prefix_space": False,
                                           "trim_offsets": False, "use_regex": False}):
        raise SystemExit(f"unexpected pre-tokenizer: {pre}")
    pattern = pre["pretokenizers"][0]["pattern"]["Regex"]
    if pattern != (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+"
                   r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"):
        raise SystemExit(f"unexpected split pattern {pattern!r}")
    model = spec["model"]
    if (model["type"] != "BPE" or model["dropout"] or model.get("ignore_merges")
            or model["byte_fallback"] or model["continuing_subword_prefix"]
            or model["end_of_word_suffix"]):
        raise SystemExit(f"unexpected BPE settings: {[k for k in model if k not in ('vocab', 'merges')]}")

    vocab = model["vocab"]
    pieces = [piece for piece, _ in sorted(vocab.items(), key=lambda item: item[1])]
    if [vocab[piece] for piece in pieces] != list(range(len(pieces))):
        raise SystemExit("the vocabulary's ids are not contiguous from 0")
    merges = [merge.split(" ") if isinstance(merge, str) else merge for merge in model["merges"]]

    added = []
    for token in spec["added_tokens"]:
        if token["rstrip"] or token["lstrip"] or token["single_word"] or token["normalized"]:
            raise SystemExit(f"{token['content']!r} has a matching flag the tokenizer does not "
                             "implement")
        if token["id"] < len(pieces):
            raise SystemExit(f"{token['content']!r} overlaps the BPE vocabulary")
        added.append({"id": token["id"], "content": token["content"], "special": token["special"]})

    classes = character_classes(published)
    payload = json.dumps({
        "pieces": pieces,
        "merges": [[vocab[left], vocab[right]] for left, right in merges],
        "added": added,
        **classes,
    }, ensure_ascii=False).encode()
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_PATH.write_bytes(gzip.compress(payload, 9, mtime=0))
    print(f"{VOCAB_PATH.relative_to(ROOT)}  {VOCAB_PATH.stat().st_size:,} bytes")
    print(f"  {len(pieces):,} pieces, {len(merges):,} merges, {len(added)} added tokens")
    print(f"  {len(classes['letters'])} letter runs, {len(classes['numbers'])} digit runs, "
          f"{len(classes['spaces'])} space runs, {len(classes['boundaries'])} NFC boundary runs, "
          f"contractions {classes['contractions']}")
    print(f"  sha256 {digest(VOCAB_PATH)}")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", revision=REVISION)
    parser.add_argument("--derive-vocab", action="store_true",
                        help="rebuild the tokenizer asset and exit")
    args = parser.parse_args()
    if args.derive_vocab:
        if args.variants:
            raise SystemExit("--derive-vocab takes no variants")
        derive_vocab()
        return 0
    wanted = args.variants or list(SPECS)
    unknown = [name for name in wanted if name not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(SPECS)}")
    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)
    print(f"\n{len(wanted)} checkpoint(s) in {args.weights_dir}, all Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
