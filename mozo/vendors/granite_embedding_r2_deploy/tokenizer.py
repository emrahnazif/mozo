# SPDX-License-Identifier: Apache-2.0
"""Granite Embedding R2's tokenizers: two byte-level BPEs, rebuilt from their parts.

Each published ``tokenizer.json`` is a pipeline, and each stage has a way to give plausible wrong
ids without raising. Two pipelines are carried, one per vocabulary asset, and they share every
stage but three.

**The English one** (``granite_bpe``, ModernBERT's):

1. **Added tokens are cut out first, in two passes.** The seven special tokens (``[CLS]``,
   ``[SEP]``, ``<|endoftext|>`` ...) are matched on the raw text. The other 109 -- ``[unused0]`` to
   ``[unused82]``, three ``|||..._ADDRESS|||`` placeholders, and **23 runs of 2 to 24 spaces** --
   are matched after normalising. Both passes take the leftmost match and, there, the longest. So
   an indented line loses its indentation to a space-run token before byte-pair encoding ever
   sees it, and the word after it starts with no space of its own: ``"Hello  world"`` is
   ``Hello``, a two-space token, then ``world`` -- not ``Ġworld``.
2. **``[MASK]`` eats the whitespace to its left.** It is the one added token declared ``lstrip``.
3. **NFC.** Unicode's composition, so ``e`` + U+0301 and ``é`` are the same token.
4. **GPT-2's split pattern**, then each piece's UTF-8 bytes mapped to printable characters.
5. **Byte-pair merges within each piece**, lowest rank first.

Then ``[CLS]`` in front and ``[SEP]`` behind.

**The multilingual one** (``granite_multilingual_bpe``, 180K pieces) differs in three stages:

- **No normalisation at all.** ``e`` + U+0301 and ``é`` are different tokens, as upstream.
- **o200k's split pattern**, which splits ``camelCase`` at the capital, keeps digits in runs of at
  most three, and matches contractions case-insensitively -- including ``'ſ``, because the long s
  case-folds to ``s``.
- **A piece that is already in the vocabulary is taken whole** (``ignore_merges``), before any
  merge is tried. And 40 control bytes and three UTF-8 lead bytes have no piece: upstream drops
  them without a word, and the bytes either side of the gap may then merge.

Its 44 reserved non-special tokens are ``single_word``: written out against a letter, digit or
underscore on either side, they are not that token but ordinary text. Its template is
``<|startoftext|> ... <|return|>``.

**Unicode tables are carried, not borrowed from the interpreter.** Each split pattern asks Unicode
questions -- is this a letter, upper or lower case, a mark, a digit, whitespace -- and the English
pipeline also composes. The answers change between Unicode versions. Python 3.10 ships Unicode 13
and 3.14 ships 16; the ``regex`` package ships newer still; the reference is none of them.
Measured against it: ``regex`` 2026.4 calls 4,657 codepoints letters or digits that the reference
treats as punctuation, Python 3.14 decomposes 20 characters the reference leaves alone, and even
Python 3.10 composes across 58 marks the reference will not. Any of those would make a document's
vector depend on which Python wrote it, silently, and an index written from two machines would not
agree with itself. So each split pattern is built from the reference's own classes, read out of it
when the vocabulary was derived, and NFC is applied the way the reference applies it. See
:func:`_nfc`.

The vocabularies ship beside this module, derived from the published ``tokenizer.json`` files by
``tools/fetch/granite_embedding_r2.py --derive-vocab``. See ``PROVENANCE.md`` for their hashes.

This module imports nothing outside the standard library, so it can be checked under any Python
on its own.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

__all__ = ["Tokenizer"]

ASSETS = Path(__file__).resolve().parent / "assets"


def vocab_path(vocabulary: str) -> Path:
    """Where the asset named *vocabulary* lives."""
    return ASSETS / f"{vocabulary}.json.gz"

#: Every codepoint above the Basic Multilingual Plane, as a class body.
_ASTRAL = "\U00010000-\U0010ffff"

#: Pieces whose ids are remembered. The reference keeps 10,000; this keeps more because a miss
#: here costs a Python merge loop rather than a Rust one. About 16 MB when full.
WORD_CACHE = 1 << 16


@lru_cache(maxsize=1)
def byte_encoder() -> dict[int, str]:
    """Map every byte to a printable character, GPT-2's way.

    Byte-pair merges operate on text, and a raw byte may be whitespace or a control character the
    merge rules cannot carry. So printable ASCII and Latin-1 stand for themselves and everything
    else is lifted into a range above 255, reversibly. A space becomes ``Ġ`` (U+0120).
    """
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
    """*ranges* split into those within the Basic Multilingual Plane and those above it."""
    bmp, astral = [], []
    for low, high in ranges:
        if low <= 0xFFFF:
            bmp.append([low, min(high, 0xFFFF)])
        if high > 0xFFFF:
            astral.append([max(low, 0x10000), high])
    return bmp, astral


def _member(ranges: list[list[int]], negate: bool = False) -> str:
    """One character in *ranges* (or, *negate*, not in them), as a fast regular expression.

    Not one class, though one class would match the same characters. Once a class reaches above
    U+FFFF, ``re`` can no longer test it with a bitmap and tries its ranges one by one at every
    position -- 344 astral ranges for the NFC boundaries alone, which made that one scan half of
    every encode, even of plain English. So the BMP ranges keep their bitmap, and the astral ones
    sit behind a single-range lookahead that only an astral character passes. Measured 16x faster
    on the boundary scan, and identical over every codepoint.
    """
    bmp, astral = _planes(ranges)
    if negate:
        return (f"(?:[^{_class(bmp)}{_ASTRAL}]"
                f"|(?=[{_ASTRAL}])[^{_class(astral) or chr(0)}])")
    branches = ([f"[{_class(bmp)}]"] if bmp else []) + (
        [f"(?=[{_ASTRAL}])[{_class(astral)}]"] if astral else [])
    return f"(?:{'|'.join(branches)})" if branches else "(?!)"


def _alternation(tokens: list[str]) -> "re.Pattern[str]":
    """Leftmost-longest matching over literal *tokens*; a pattern that never matches if none.

    Python's alternation takes the first branch that matches at a position, not the longest, so
    sorting longest first is what makes it longest: ``[unused10]`` before the ``[unused1]`` that
    prefixes it, and a 24-space run before the 23-space one.
    """
    return re.compile(
        "|".join(re.escape(token) for token in sorted(tokens, key=len, reverse=True)) or "(?!)")


class Tables(NamedTuple):
    """Everything read out of one vocabulary asset, named rather than positional."""

    #: Byte -> the id of the piece that stands for it, or -1 for a byte with no piece.
    byte_ids: list[int]
    #: Byte-mapped piece -> id, for a piece taken whole (``ignore_merges``); empty otherwise.
    whole: dict[str, int]
    #: Adjacent pair of ids -> the priority of merging them. Lower wins.
    ranks: dict[tuple[int, int], int]
    #: The id a merge produces, indexed by that merge's rank.
    merged: list[int]
    #: Added token -> id.
    added: dict[str, int]
    #: The added tokens matched on raw text, and those matched after normalising.
    raw: "re.Pattern[str]"
    normalised: "re.Pattern[str]"
    #: The added tokens that absorb whitespace to their left.
    lstrip: frozenset[str]
    #: What ``lstrip`` counts as whitespace: Rust's ``char::is_whitespace``.
    strip: frozenset[str]
    #: The added tokens that count only standing alone, and the characters that stop them: the
    #: reference's word characters (``\\w``), letters, marks, digits and connectors.
    single_word: frozenset[str]
    word: "re.Pattern[str]"
    #: The split pattern, over the reference's own character classes.
    split: "re.Pattern[str]"
    #: Where NFC must not reach across (see :func:`_nfc`), or ``None`` for no normalisation.
    boundaries: "re.Pattern[str] | None"
    #: The ids the template puts before and after every input.
    before: tuple[int, ...]
    after: tuple[int, ...]


def _gpt2(data: dict) -> "re.Pattern[str]":
    """GPT-2's pattern, as the reference writes it::

        's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

    with each class replaced by the reference's own membership. ``re`` rather than ``regex``,
    because once the classes are explicit nothing here needs Unicode properties at all.
    """
    letters, numbers = _member(data["letters"]), _member(data["numbers"])
    other = _member(sorted(data["spaces"] + data["letters"] + data["numbers"]), negate=True)
    spaces = _class(data["spaces"])  # all within the BMP, so one class is already fast
    return re.compile(
        rf"'s|'t|'re|'ve|'m|'ll|'d| ?{letters}+| ?{numbers}+| ?{other}+"
        rf"|[{spaces}]+(?![^{spaces}])|[{spaces}]+")


def _o200k(data: dict) -> "re.Pattern[str]":
    """o200k's pattern, as the reference writes it::

        [^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
        |[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
        |\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

    with each class replaced by the reference's own membership, and the case-insensitive
    contractions by the characters the reference matches for each letter -- read off it, because
    case folding is a Unicode table too: ``(?i:'s)`` matches ``'ſ``.
    """
    letters, numbers, spaces = data["letters"], data["numbers"], data["spaces"]
    upper, lower = _member(data["upper"]), _member(data["lower"])
    lead = _member(sorted(letters + numbers + [[10, 10], [13, 13]]), negate=True)
    other = _member(sorted(spaces + letters + numbers), negate=True)
    space = _class(spaces)  # all within the BMP
    # ``'s 't 'm 'd`` are one class of first character, ``'re 've`` two classes, ``'ll`` two.
    contraction = "'(?:" + "|".join(
        "".join(f"[{re.escape(characters)}]" for characters in alternative)
        for alternative in data["contractions"]) + ")"
    return re.compile(
        rf"{lead}?{upper}*{lower}+(?:{contraction})?"
        rf"|{lead}?{upper}+{lower}*(?:{contraction})?"
        rf"|{_member(numbers)}{{1,3}}| ?{other}+[\r\n/]*|[{space}]*[\r\n]+"
        rf"|[{space}]+(?![^{space}])|[{space}]+")


#: The split patterns an asset may name.
PATTERNS: dict[str, Callable[[dict], "re.Pattern[str]"]] = {"gpt2": _gpt2, "o200k": _o200k}


@lru_cache(maxsize=None)
def _vocabulary(vocabulary: str) -> Tables:
    """Read the asset named *vocabulary* once and build the lookups tokenizing needs.

    Cached at module level rather than per instance, so two encoders in one process share one copy.
    An asset says which split pattern it uses, whether it normalises and whether it takes known
    pieces whole; the English asset predates those keys, and its values are their defaults.
    """
    data = json.loads(gzip.decompress(vocab_path(vocabulary).read_bytes()))
    pieces: list[str] = data["pieces"]
    ids = {piece: index for index, piece in enumerate(pieces)}

    # ``merged`` indexed by rank rather than a second dict keyed like ``ranks``: the merge loop has
    # the rank in hand when it needs the result.
    ranks, merged = {}, []
    for rank, (left, right) in enumerate(data["merges"]):
        ranks[(left, right)] = rank
        merged.append(ids[pieces[left] + pieces[right]])

    # Bytes with no piece are -1 and dropped, as upstream drops a character it has no piece for
    # when there is no unknown token. English: five bytes that never occur in UTF-8 (0xC0, 0xC1,
    # 0xF5-0xFF). Multilingual: also 40 control bytes and the lead bytes 0xF1, 0xF2 and 0xF4.
    byte_ids = [ids.get(character, -1) for _, character in sorted(byte_encoder().items())]

    added = {token["content"]: token["id"] for token in data["added"]}
    raw = _alternation([t["content"] for t in data["added"] if not t["normalized"]])
    normalised = _alternation([t["content"] for t in data["added"] if t["normalized"]])
    lstrip = frozenset(t["content"] for t in data["added"] if t["lstrip"])
    strip = frozenset(chr(c) for low, high in data.get("strip", []) for c in range(low, high + 1))
    single_word = frozenset(t["content"] for t in data["added"] if t.get("single_word"))
    nfc = data.get("normalizer", "NFC") == "NFC"

    return Tables(byte_ids, ids if data.get("ignore_merges") else {}, ranks, merged, added, raw,
                  normalised, lstrip, strip, single_word, re.compile(_member(data.get("word", []))),
                  PATTERNS[data.get("pattern", "gpt2")](data),
                  re.compile(_member(data["boundaries"])) if nfc else None,
                  tuple(data["template"]["before"]), tuple(data["template"]["after"]))


def _nfc(text: str, boundaries: "re.Pattern[str]") -> str:
    """NFC as the reference applies it, on whichever Python is running.

    The reference's NFC is Unicode 13's with one difference, and both halves of that were measured
    on every codepoint when the vocabulary was derived:

    - **Its decompositions and compositions are Unicode 13's.** Newer Pythons know more: 3.14
      decomposes 20 characters the reference leaves alone, every one of them unassigned in
      Unicode 13.
    - **Its combining classes are older.** 58 marks that Unicode 10 to 13 gave a nonzero class, it
      treats as class 0: it never reorders them, and composition does not reach across them. So
      ``e`` + U+11D42 (a Gondi nukta) + U+0301 stays three characters, where Python 3.10 composes
      the ``é``.

    Both are handled the same way. A character of class 0 with no decomposition, which composes
    with nothing, is a wall: NFC on either side of it is independent of the other. So the text is
    split at those characters -- the unassigned ones and the 58 -- each run between them is
    normalised on its own, and they are kept as given. Unicode's stability policy guarantees each
    run normalises identically on every Python from 3.10 up, because it contains only characters
    Unicode 13 assigned.
    """
    if text.isascii():
        return text  # ASCII is NFC already, and no boundary is ASCII: skips the scan for most text
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
    """Apply merge rules in priority order until none applies.

    A heap over adjacent pairs, the way the reference does it: lowest rank first, and between two
    occurrences of one pair, the leftmost. Entries are left stale rather than removed, and skipped
    when popped.
    """
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
def _word(vocabulary: str, word: str) -> tuple[int, ...]:
    """One pre-tokenized piece to ids.

    Cached because prose repeats its words, bounded because a corpus's numbers, hashes and
    identifiers do not, and at module level because the answer depends only on the shared tables:
    every tokenizer in the process can use one cache.
    """
    tables = _vocabulary(vocabulary)
    raw = word.encode("utf-8")
    if tables.whole:
        # Upstream looks the whole byte-mapped piece up before merging anything. Not a shortcut:
        # merging can reach a different split of a piece the vocabulary has whole.
        encoder = byte_encoder()
        whole = tables.whole.get("".join(encoder[byte] for byte in raw))
        if whole is not None:
            return (whole,)
    symbols = [i for i in (tables.byte_ids[byte] for byte in raw) if i >= 0]
    return tuple(_merge(symbols, tables.ranks, tables.merged))


class Tokenizer:
    """Turn text into the token ids the encoder expects, the template's special tokens included.

    Examples:
        >>> Tokenizer(8192, "granite_bpe").encode("Hello  world")   # doctest: +SKIP
        [50281, 12092, 50276, 10186, 50282]

    Args:
        context: The most ids one text may become, ``[CLS]`` and ``[SEP]`` included. Checked by
            :meth:`__call__`, not by :meth:`encode`, which reproduces upstream and so enforces
            nothing. A parameter rather than an import, so this module stays free of the rest of
            the package and can be checked under any Python on its own.
        vocabulary: Which asset: ``"granite_bpe"`` (English) or ``"granite_multilingual_bpe"``.
            A variant's ``Spec.vocabulary`` names it. No default: the wrong one gives plausible
            ids and no error.
    """

    def __init__(self, context: int | None, vocabulary: str) -> None:
        self.context = context
        self.vocabulary = vocabulary
        self._tables = _vocabulary(vocabulary)

    def __call__(self, texts: str | Sequence[str]) -> list[list[int]]:
        """Encode one text or many, refusing what the encoder cannot take.

        Returns:
            One row of ids per text, unpadded: rows are as long as their texts.

        Raises:
            ValueError: If nothing is given, if a text is blank, if one is over :attr:`context`,
                or if one contains an unpaired surrogate.

        Note:
            Upstream truncates instead, or -- without ``truncation=True`` -- runs past the model's
            length on positions it never saw. Not carried: a chunk shortened without saying so is
            a different chunk, and only the caller can decide where to cut.
        """
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            raise ValueError("give at least one text to encode")
        rows = []
        for index, text in enumerate(batch):
            if not text.strip():
                raise ValueError(f"text {index} of {len(batch)} is blank; every text must say "
                                 "something")
            ids = self.encode(text)
            if self.context is not None and len(ids) > self.context:
                raise ValueError(
                    f"text {index} of {len(batch)} is {len(ids)} tokens, over the "
                    f"{self.context}-token context. Split it into chunks; mozo does not truncate.")
            rows.append(ids)
        return rows

    def _cut(self, pattern: "re.Pattern[str]", text: str, out: list[int],
             between: Callable[[str, list[int]], None]) -> None:
        """Emit the added tokens *pattern* finds in *text*, and hand what lies between to *between*.

        Both passes -- the raw one and the normalised one -- go through here, so an added token's
        ``lstrip`` and ``single_word`` are honoured in whichever pass matches it.
        """
        tables = self._tables
        position = 0
        for match in pattern.finditer(text):
            token, start = match.group(), match.start()
            if token in tables.single_word and (
                    (start and tables.word.match(text, start - 1))
                    or tables.word.match(text, match.end())):
                # Against a word character it is not the token. Upstream drops the match and does
                # not look again inside it, which is what moving on to the next match does here.
                continue
            if token in tables.lstrip:
                # Back over the whitespace before it, but never into the previous match.
                while start > position and text[start - 1] in tables.strip:
                    start -= 1
            between(text[position:start], out)
            out.append(tables.added[token])
            position = match.end()
        between(text[position:], out)

    def _ordinary(self, text: str, out: list[int]) -> None:
        """Split normalised text without added tokens into pieces, and encode each."""
        for word in self._tables.split.findall(text):
            out.extend(_word(self.vocabulary, word))

    def _normalised(self, text: str, out: list[int]) -> None:
        """Normalise a stretch of raw text, then cut out the added tokens matched after that."""
        boundaries = self._tables.boundaries
        self._cut(self._tables.normalised, text if boundaries is None else _nfc(text, boundaries),
                  out, self._ordinary)

    def encode(self, text: str) -> list[int]:
        """*text* to ids, between the template's tokens. No truncation and no padding.

        Special tokens written out in the text -- ``[SEP]``, ``<|return|>`` -- become that token,
        exactly as they do upstream, so a document that *mentions* ``[SEP]`` carries a separator
        in the middle. The published weights were evaluated that way; this reproduces it rather
        than escaping it.

        Raises:
            ValueError: If *text* contains an unpaired surrogate, which is a valid Python ``str``
                and not encodable text.
        """
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(
                f"the text contains an unpaired surrogate at position {error.start} and is not "
                "encodable; decode the input properly before tokenizing it") from None

        out = list(self._tables.before)
        self._cut(self._tables.raw, text, out, self._normalised)
        out.extend(self._tables.after)
        return out
