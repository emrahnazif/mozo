# SPDX-License-Identifier: Apache-2.0
"""Qwen's tokenizer: a byte-level BPE, rebuilt from its parts, and its decoder.

The published ``tokenizer.json`` (and ``transformers``' ``Qwen2Tokenizer``, which builds the same
pipeline in code) has four stages, each with a way to give plausible wrong ids without raising:

1. **Added tokens are cut out first, on the raw text.** All 26 (``<|im_start|>``,
   ``<|image_pad|>``, ``<think>`` ...) are declared ``normalized: false``, matched leftmost and,
   there, longest. mozo refuses a *prompt* that contains one (see :meth:`Tokenizer.check`), because
   upstream would read it as a control token; the template's own control tokens are the only ones
   that reach here.
2. **NFC** on what lies between them.
3. **Qwen's split pattern**::

       (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

   Not GPT-2's: digits split one by one, a letter run takes one non-letter in front of it, and
   punctuation keeps trailing newlines. Then each piece's UTF-8 bytes are mapped to printable
   characters (GPT-2's table).
4. **Byte-pair merges within each piece**, lowest rank first.

**Unicode tables are carried, not borrowed from the interpreter.** ``\\p{L}``, ``\\p{N}``, ``\\s``,
the case folding of ``(?i:...)`` and NFC's composition all depend on a Unicode version, and the
reference's (Oniguruma and Rust crates inside ``tokenizers``) is neither Python's nor the ``regex``
package's. So every class is read out of the reference when the vocabulary is derived
(``tools/fetch/qwen3_vl.py --derive-vocab``) and NFC is applied as Granite Embedding R2's tokenizer
applies it: split at the characters where the reference's NFC and Unicode 13's could differ, and
normalise the runs between. This module imports nothing outside the standard library.

**Decoding** is the inverse byte map over the generated ids, special tokens skipped, invalid UTF-8
replaced -- ``tokenizer.decode(ids, skip_special_tokens=True)``.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path
from typing import NamedTuple

__all__ = ["Tokenizer"]

VOCAB_PATH = Path(__file__).resolve().parent / "assets" / "qwen_bpe.json.gz"

_ASTRAL = "\U00010000-\U0010ffff"

#: Pieces whose ids are remembered. A miss costs a Python merge loop.
WORD_CACHE = 1 << 16


@lru_cache(maxsize=1)
def byte_encoder() -> dict[int, str]:
    """Map every byte to a printable character, GPT-2's way (a space becomes ``Ġ``)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapped = printable[:]
    spare = 0
    for byte in range(2**8):
        if byte not in printable:
            printable.append(byte)
            mapped.append(2**8 + spare)
            spare += 1
    return dict(zip(printable, (chr(code) for code in mapped)))


def _class(ranges: list[list[int]]) -> str:
    """A regular-expression character class body for inclusive codepoint *ranges*."""
    return "".join(
        re.escape(chr(low)) if low == high else f"{re.escape(chr(low))}-{re.escape(chr(high))}"
        for low, high in ranges)


def _planes(ranges: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    bmp, astral = [], []
    for low, high in ranges:
        if low <= 0xFFFF:
            bmp.append([low, min(high, 0xFFFF)])
        if high > 0xFFFF:
            astral.append([max(low, 0x10000), high])
    return bmp, astral


def _member(ranges: list[list[int]], negate: bool = False) -> str:
    """One character in *ranges* (or not in them), as a fast pattern.

    The BMP ranges keep ``re``'s bitmap; the astral ones sit behind a lookahead only an astral
    character passes, so plain text never walks hundreds of astral ranges (Granite's measurement:
    16x faster, identical over every codepoint).
    """
    bmp, astral = _planes(ranges)
    if negate:
        return (f"(?:[^{_class(bmp)}{_ASTRAL}]"
                f"|(?=[{_ASTRAL}])[^{_class(astral) or chr(0)}])")
    branches = ([f"[{_class(bmp)}]"] if bmp else []) + (
        [f"(?=[{_ASTRAL}])[{_class(astral)}]"] if astral else [])
    return f"(?:{'|'.join(branches)})" if branches else "(?!)"


class Tables(NamedTuple):
    byte_ids: list[int]
    ranks: dict[tuple[int, int], int]
    merged: list[int]
    pieces: list[str]
    added: dict[str, int]
    added_by_id: dict[int, str]
    special: frozenset[int]
    raw: "re.Pattern[str]"
    split: "re.Pattern[str]"
    boundaries: "re.Pattern[str]"


@lru_cache(maxsize=1)
def _vocabulary() -> Tables:
    data = json.loads(gzip.decompress(VOCAB_PATH.read_bytes()))
    pieces: list[str] = data["pieces"]
    ids = {piece: index for index, piece in enumerate(pieces)}
    ranks, merged = {}, []
    for rank, (left, right) in enumerate(data["merges"]):
        ranks[(left, right)] = rank
        merged.append(ids[pieces[left] + pieces[right]])
    byte_ids = [ids.get(character, -1) for _, character in sorted(byte_encoder().items())]

    added = {token["content"]: token["id"] for token in data["added"]}
    special = frozenset(token["id"] for token in data["added"] if token["special"])
    raw = re.compile("|".join(re.escape(t) for t in sorted(added, key=len, reverse=True)))

    letters, numbers = data["letters"], data["numbers"]
    spaces = data["spaces"]
    L = _member(letters)
    N = _member(numbers)
    not_crlf_L_N = _member(sorted(letters + numbers + [[10, 10], [13, 13]]), negate=True)
    not_s_L_N = _member(sorted(spaces + letters + numbers), negate=True)
    S = f"[{_class(spaces)}]"  # whitespace is all in the BMP
    not_S = f"(?:[^{_class(spaces)}])"
    # ``(?i:'s|'t|'re|'ve|'m|'ll|'d)`` with the reference's own case folding: the letters that
    # complete a one-letter contraction, then each two-letter one as a pair of classes. No match of
    # one is a prefix of another's, so the order of the branches cannot matter.
    folds = data["contractions"]
    contractions = "|".join(
        [f"'[{_class(folds[chr(39)])}]"]
        + [f"'[{_class(pair[0])}][{_class(pair[1])}]" for key, pair in folds.items() if key != "'"])
    split = re.compile(
        rf"{contractions}|{not_crlf_L_N}?{L}+|{N}| ?{not_s_L_N}+[\r\n]*|{S}*[\r\n]+"
        rf"|{S}+(?!{not_S})|{S}+")
    return Tables(byte_ids, ranks, merged, pieces, added, {v: k for k, v in added.items()},
                  special, raw, split, re.compile(_member(data["boundaries"])))


def _nfc(text: str, boundaries: "re.Pattern[str]") -> str:
    """NFC as the reference applies it, on whichever Python is running.

    Split at characters NFC cannot reach across in the reference (unassigned in Unicode 13, or
    marks the reference treats as class 0), normalise each run on its own, keep the walls as given.
    Each run holds only characters Unicode 13 assigned, so Unicode's stability policy makes its NFC
    the same on every Python from 3.10 up.
    """
    out, position = [], 0
    for match in boundaries.finditer(text):
        out.append(unicodedata.normalize("NFC", text[position:match.start()]))
        out.append(match.group())
        position = match.end()
    if not out:
        return unicodedata.normalize("NFC", text)
    out.append(unicodedata.normalize("NFC", text[position:]))
    return "".join(out)


def _merge(symbols: list[int], ranks: dict[tuple[int, int], int], merged: list[int]) -> list[int]:
    """Apply merges in priority order until none applies: lowest rank, then leftmost."""
    if len(symbols) < 2:
        return symbols
    following = list(range(1, len(symbols) + 1))
    preceding = list(range(-1, len(symbols) - 1))
    alive = [True] * len(symbols)
    heap: list[tuple[int, int, int]] = []
    for position in range(len(symbols) - 1):
        rank = ranks.get((symbols[position], symbols[position + 1]))
        if rank is not None:
            heappush(heap, (rank, position, position + 1))
    while heap:
        rank, left, right = heappop(heap)
        if not alive[left] or not alive[right] or following[left] != right:
            continue
        if ranks.get((symbols[left], symbols[right])) != rank:
            continue
        symbols[left] = merged[rank]
        alive[right] = False
        after = following[right]
        following[left] = after
        if after < len(symbols):
            preceding[after] = left
        before = preceding[left]
        if before >= 0:
            rank = ranks.get((symbols[before], symbols[left]))
            if rank is not None:
                heappush(heap, (rank, before, left))
        if after < len(symbols):
            rank = ranks.get((symbols[left], symbols[after]))
            if rank is not None:
                heappush(heap, (rank, left, after))
    return [symbol for symbol, kept in zip(symbols, alive) if kept]


@lru_cache(maxsize=WORD_CACHE)
def _word(word: str) -> tuple[int, ...]:
    tables = _vocabulary()
    return tuple(_merge([tables.byte_ids[b] for b in word.encode("utf-8")], tables.ranks, tables.merged))


class Tokenizer:
    """Text to ids and back, as ``Qwen2Tokenizer`` does."""

    def __init__(self) -> None:
        self._tables = _vocabulary()

    @property
    def added(self) -> tuple[str, ...]:
        """The 26 added tokens' text, which a prompt may not contain."""
        return tuple(self._tables.added)

    def check(self, prompt: str) -> None:
        """Refuse a prompt upstream would not read as text.

        Raises:
            ValueError: If it is blank, holds an unpaired surrogate, or contains an added token's
                text -- ``<|im_end|>`` would end the user's turn, ``<|image_pad|>`` would claim to be
                part of the image.
        """
        if not prompt.strip():
            raise ValueError("the prompt is blank; ask something about the image")
        try:
            prompt.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"the prompt contains an unpaired surrogate at position {error.start} "
                             "and is not encodable text") from None
        found = self._tables.raw.search(prompt)
        if found:
            raise ValueError(
                f"the prompt contains {found.group()!r}, one of the model's control tokens; upstream "
                "would act on it rather than read it, so mozo refuses it. Rephrase without it.")

    def encode(self, text: str) -> list[int]:
        """*text* to ids. Added tokens written in it become those tokens, as upstream."""
        tables = self._tables
        out: list[int] = []
        position = 0
        for match in tables.raw.finditer(text):
            self._ordinary(text[position:match.start()], out)
            out.append(tables.added[match.group()])
            position = match.end()
        self._ordinary(text[position:], out)
        return out

    def _ordinary(self, text: str, out: list[int]) -> None:
        if not text:
            return
        for word in self._tables.split.findall(_nfc(text, self._tables.boundaries)):
            out.extend(_word(word))

    def decode(self, ids: list[int]) -> str:
        """``decode(ids, skip_special_tokens=True)``: special tokens dropped, bytes back to text."""
        tables = self._tables
        decoder = _byte_decoder()
        text = "".join(tables.added_by_id[i] if i in tables.added_by_id else tables.pieces[i]
                       for i in ids if i not in tables.special)
        return bytes(decoder[c] for c in text).decode("utf-8", errors="replace")


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    return {character: byte for byte, character in byte_encoder().items()}
