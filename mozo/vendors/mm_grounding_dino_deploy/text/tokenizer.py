# SPDX-License-Identifier: Apache-2.0
"""BERT's WordPiece tokenizer, over the vocabulary and character tables that ship beside this file.

MM-Grounding-DINO tokenizes prompts with `bert-base-uncased`'s vocabulary through ``transformers``'
``BertTokenizerFast`` -- a Rust ``BertNormalizer`` (drop control characters, space out CJK
ideographs, strip accents, lowercase), a ``BertPreTokenizer`` (split on whitespace, isolate
punctuation), then greedy longest-match WordPiece. This package carries the vocabulary instead of
downloading it, so a prompt never needs a network round trip to become token ids.

**What counts as a control character, an accent, a CJK ideograph or punctuation is read from a
table, not from Python's ``unicodedata``.** Those are properties of a Unicode version, and the
reference's Rust crates and the running interpreter ship different ones: written against
``unicodedata`` on Python 3.12, this tokenizer disagreed with the reference on **826 assigned
codepoints** (marks added in Unicode 14-15 that one side strips and the other keeps, punctuation
the Rust tables predate, CJK Extension I) and on every one of the 825,345 codepoints Python calls
unassigned, which it dropped as control characters where the reference emits ``[UNK]``. The table
(``assets/bert_chars.json.gz``) is the reference's own answer for every codepoint, derived by
``tools/fetch/mm_grounding_dino.py --derive-tokenizer``; with it the two agree on all 1,112,064.

The caption itself is still lowercased with ``str.lower`` first (``predictor.caption_for``),
because upstream's processor does exactly that before the tokenizer sees it -- so that one step is
the interpreter's on both sides.

The vocabulary is Google's ``vocab.txt`` for `bert-base-uncased` (Apache-2.0), byte-identical to
the ``vocab.txt`` every published LLMDet checkpoint carries (sha256 ``07eced37...``). See
``NOTICE``.
"""

from __future__ import annotations

import gzip
import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

__all__ = ["Tokenizer", "vocabulary"]

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_VOCAB = _ASSETS / "vocab.txt.gz"
_CHARS = _ASSETS / "bert_chars.json.gz"

_UNK = "[UNK]"
_CLS = "[CLS]"
_SEP = "[SEP]"

#: Longest WordPiece BERT will attempt before giving up on a word and emitting ``[UNK]``.
_MAX_CHARS_PER_WORD = 100


@lru_cache(maxsize=1)
def vocabulary() -> dict[str, int]:
    """Return token -> id, read once and kept for the life of the process.

    Read as a list and enumerated rather than split on newlines and filtered: a vocabulary entry
    may be an empty string, and dropping it would shift every id after it by one -- which loads,
    runs, and reads the wrong word.
    """
    with gzip.open(_VOCAB, "rt", encoding="utf-8") as handle:
        return {line.rstrip("\n"): index for index, line in enumerate(handle)}


class _Ranges:
    """Membership in a sorted list of inclusive ``[start, end]`` codepoint ranges."""

    def __init__(self, ranges: list[list[int]]) -> None:
        self.starts = [start for start, _ in ranges]
        self.ends = [end for _, end in ranges]

    def __contains__(self, code: int) -> bool:
        index = bisect_right(self.starts, code) - 1
        return index >= 0 and code <= self.ends[index]


@lru_cache(maxsize=1)
def _tables() -> tuple[dict[int, str], _Ranges, _Ranges, _Ranges, _Ranges, _Ranges]:
    """The reference's per-codepoint answers, read once.

    ``mapped``: what the normaliser turns a codepoint into, where that is neither itself nor one
    of the range classes. ``drop``/``space``/``spaced``: ranges it removes, turns into a space, or
    surrounds with spaces (CJK ideographs). ``whitespace``/``punctuation``: ranges the
    pre-tokenizer splits on, or isolates.
    """
    with gzip.open(_CHARS, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    mapped = {int(code): text for code, text in data["mapped"].items()}
    return (mapped, _Ranges(data["drop"]), _Ranges(data["space"]), _Ranges(data["spaced"]),
            _Ranges(data["whitespace"]), _Ranges(data["punctuation"]))


class Tokenizer:
    """`bert-base-uncased` WordPiece, as ``BertTokenizerFast`` runs it: text in, token ids out.

    Examples:
        >>> Tokenizer().encode("person. laptop. cup.")[0][:4]
        [101, 2711, 1012, 12191]
    """

    def __init__(self) -> None:
        self.vocab = vocabulary()
        self.cls_id = self.vocab[_CLS]
        self.sep_id = self.vocab[_SEP]
        self.unk_id = self.vocab[_UNK]
        (self._mapped, self._drop, self._space, self._spaced, self._whitespace,
         self._punctuation) = _tables()

    def _normalise(self, text: str) -> str:
        """``BertNormalizer``: one codepoint at a time, each by the reference's table."""
        out = []
        for char in text:
            code = ord(char)
            if code in self._mapped:
                out.append(self._mapped[code])
            elif code in self._drop:
                continue
            elif code in self._space:
                out.append(" ")
            elif code in self._spaced:
                out.extend((" ", char, " "))
            else:
                out.append(char)
        return "".join(out)

    def _pre_tokenize(self, text: str) -> list[str]:
        """``BertPreTokenizer``: split on whitespace, and stand every punctuation mark alone."""
        words: list[str] = []
        current: list[str] = []
        for char in text:
            code = ord(char)
            if code in self._whitespace:
                if current:
                    words.append("".join(current))
                    current = []
            elif code in self._punctuation:
                if current:
                    words.append("".join(current))
                    current = []
                words.append(char)
            else:
                current.append(char)
        if current:
            words.append("".join(current))
        return words

    def _wordpiece(self, word: str) -> list[str]:
        """Greedy longest-match-first, with ``##`` on every piece after the first."""
        if len(word) > _MAX_CHARS_PER_WORD:
            return [_UNK]

        pieces: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    found = piece
                    break
                end -= 1
            if found is None:
                # One unmatchable piece makes the *whole word* unknown, not just that piece.
                return [_UNK]
            pieces.append(found)
            start = end
        return pieces

    def tokenize(self, text: str) -> list[str]:
        """Return the WordPiece tokens for *text*, without the ``[CLS]``/``[SEP]`` wrapper."""
        return [piece for word in self._pre_tokenize(self._normalise(text))
                for piece in self._wordpiece(word)]

    def encode(self, text: str) -> tuple[list[int], list[int], list[int]]:
        """Return ``(input_ids, token_type_ids, attention_mask)`` for one caption.

        Wrapped in ``[CLS]`` and ``[SEP]``. One sequence, so the token types are all zero, and
        nothing is padded -- a single caption is its own longest.
        """
        ids = [self.cls_id]
        ids.extend(self.vocab.get(piece, self.unk_id) for piece in self.tokenize(text))
        ids.append(self.sep_id)
        return ids, [0] * len(ids), [1] * len(ids)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Look up ids for tokens already split, e.g. the separator set."""
        return [self.vocab.get(token, self.unk_id) for token in tokens]
