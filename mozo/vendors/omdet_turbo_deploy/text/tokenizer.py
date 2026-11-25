# SPDX-License-Identifier: MIT
"""The CLIP byte-pair tokenizer OmDet-Turbo's prompts go through.

``assets/bpe_simple_vocab_16e6.txt.gz`` is byte-identical to the file published with
``openai/CLIP`` (sha256 ``924691ac...48d6804a``), so both it and the algorithm below are MIT. Its
vocabulary is identical, id for id, to the ``vocab.json`` and ``merges.txt`` Om AI Lab publishes
beside the weights (checked by ``tools/verify/omdet_turbo.py``).

**This reproduces ``transformers``' ``CLIPTokenizer``, not OpenAI's ``SimpleTokenizer``**, because
that is what the published checkpoint's ``tokenizer_config.json`` names and what the reference
runs: NFC, collapse whitespace, lowercase, and nothing else -- no ``ftfy``, no HTML unescaping.
The authors' own repository tokenizes with OpenAI's, which repairs mis-decoded text first; on
ordinary prompts the two give the same ids.

**Padding is ``<|endoftext|>``, id 49407**, as the published config sets it -- not zero, as
OpenAI's ``clip.tokenize`` and OWLv2's config do. So only the two markers are added tokens here,
and ``!`` is an ordinary symbol. Pooling still finds the end of a prompt by ``argmax`` over the ids,
which lands on the *first* 49407 because ``argmax`` returns the first maximum.

**Truncation is the caller's choice.** The task sentence is built by mozo from every prompt, and
upstream truncates it at 77 tokens, keeping ``<|endoftext|>`` in the last slot; a class prompt
that does not fit is refused instead, naming its length.
"""

from __future__ import annotations

import gzip
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import regex
import torch

__all__ = ["Tokenizer", "VOCAB_PATH"]

#: The vocabulary shipped with this package.
VOCAB_PATH = Path(__file__).resolve().parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

#: Codepoint -> its lowercase, as the reference's Rust normaliser maps it. Built by
#: ``tools/fetch/omdet_turbo.py --derive-lowercase``; see :meth:`Tokenizer.clean` for why.
LOWERCASE_PATH = VOCAB_PATH.with_name("lowercase.json.gz")

#: Where the merge list stops. CLIP's file carries a header line and trailing entries that are
#: not merges; upstream slices exactly this range and the vocabulary ids depend on it.
MERGE_LIMIT = 49152 - 256 - 2 + 1

START_TOKEN = "<|startoftext|>"
END_TOKEN = "<|endoftext|>"

#: Split out before byte-pair encoding: the published tokenizer config's added-token table.
ADDED = ((START_TOKEN, 49406), (END_TOKEN, 49407))


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map every byte to a printable unicode character.

    BPE runs on text, but prompts are bytes. Mapping the unprintable bytes up into unused
    codepoints keeps the merge table reversible without the algorithm ever meeting a control
    character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = printable[:]
    spare = 0
    for byte in range(2**8):
        if byte not in printable:
            printable.append(byte)
            mapped.append(2**8 + spare)
            spare += 1
    return dict(zip(printable, (chr(code) for code in mapped)))


def pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Every adjacent symbol pair in ``word``."""
    return {(first, second) for first, second in zip(word, word[1:])}


class Tokenizer:
    """CLIP byte-pair encoding, as OmDet-Turbo's published config sets it up.

    Args:
        vocab_path: The gzipped merge list. Defaults to the vendored copy.
        context_length: Fixed width of the returned id matrix.
    """

    def __init__(self, vocab_path: Path = VOCAB_PATH, context_length: int = 77):
        self.context_length = context_length

        self.byte_encoder = bytes_to_unicode()
        with gzip.open(vocab_path) as handle:
            merges = handle.read().decode("utf-8").split("\n")
        merges = [tuple(merge.split()) for merge in merges[1:MERGE_LIMIT]]

        vocabulary = list(self.byte_encoder.values())
        vocabulary += [token + "</w>" for token in vocabulary]
        vocabulary += ["".join(merge) for merge in merges]
        vocabulary += [START_TOKEN, END_TOKEN]

        self.encoder = {token: index for index, token in enumerate(vocabulary)}
        self.ranks = {merge: rank for rank, merge in enumerate(merges)}
        self._cache: dict[str, str] = {}

        self.start_id = self.encoder[START_TOKEN]
        self.end_id = self.encoder[END_TOKEN]

        # ``\p{L}`` and ``\p{N}`` are Unicode categories, which the standard library's ``re``
        # cannot express -- hence ``regex``. Substituting ``\w`` here silently changes how
        # accented and non-Latin prompts split. This is the pre-tokenizer pattern verbatim, minus
        # the two markers: those are handled by ``ADDED`` before this runs.
        self.pattern = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
        )

        # One alternation over the added tokens, capturing so ``split`` keeps them.
        self.added = regex.compile("(" + "|".join(regex.escape(t) for t, _ in ADDED) + ")")
        self.added_ids = dict(ADDED)

        with gzip.open(LOWERCASE_PATH) as handle:
            self.lowercase = {int(code): folded for code, folded in json.load(handle).items()}

    def clean(self, text: str) -> str:
        """Normalise, collapse whitespace, lowercase.

        The three steps ``CLIPTokenizer`` applies and no others. Anything more -- repairing
        mojibake, unescaping entities -- moves the ids away from what the weights were trained on.

        Lowercasing is a table lookup per codepoint rather than ``str.lower()``. The reference
        lowercases in Rust, one codepoint at a time; Python applies the Greek final-sigma rule
        (``"ΑΣ"`` becomes ``"ας"``, the reference gives ``"ασ"``) and uses the interpreter's own
        Unicode tables, which differ from the reference's on 55 codepoints under Python 3.12.
        """
        text = unicodedata.normalize("NFC", text)
        return regex.sub(r"\s+", " ", text).translate(self.lowercase)

    def merge(self, token: str) -> str:
        """Apply the merge table to one whitespace-free token, returning space-joined symbols."""
        if token in self._cache:
            return self._cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        current = pairs(word)
        while current:
            # Always merge the highest-ranked (earliest-learned) pair available. That ordering is
            # the encoding -- taking any other pair first produces different, valid-looking ids.
            best = min(current, key=lambda pair: self.ranks.get(pair, float("inf")))
            if best not in self.ranks:
                break
            first, second = best
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    found = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:found])
                index = found
                if word[index] == first and index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            current = pairs(word) if len(word) > 1 else set()

        joined = " ".join(word)
        self._cache[token] = joined
        return joined

    def encode(self, text: str) -> list[int]:
        """Turn one prompt into token ids, without the start and end markers."""
        ids: list[int] = []
        for span in self.added.split(self.clean(text)):
            if span in self.added_ids:
                ids.append(self.added_ids[span])
                continue
            for token in self.pattern.findall(span):
                token = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
                ids.extend(self.encoder[symbol] for symbol in self.merge(token).split(" "))
        return ids

    def __call__(self, texts: list[str], *, truncate: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch into a fixed-width id matrix and its mask.

        Args:
            texts: The prompts.
            truncate: Cut a prompt that does not fit to ``context_length``, keeping the end marker
                in the last slot, as upstream does. Otherwise a prompt that does not fit raises.

        Returns:
            ``ids`` and ``mask``, both ``(len(texts), context_length)`` int64. Rows are padded on
            the right with the end-of-text id; ``mask`` is 1 where the row carries a real token.

        Raises:
            ValueError: If ``truncate`` is false and a prompt does not fit.
        """
        ids = torch.full((len(texts), self.context_length), self.end_id, dtype=torch.long)
        mask = torch.zeros(len(texts), self.context_length, dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = [self.start_id, *self.encode(text), self.end_id]
            if len(tokens) > self.context_length:
                if not truncate:
                    raise ValueError(
                        f"prompt {text!r} is {len(tokens) - 2} tokens; at most "
                        f"{self.context_length - 2} fit"
                    )
                tokens = tokens[: self.context_length]
                # Keep the sequence terminated even when the prompt did not fit.
                tokens[-1] = self.end_id
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, : len(tokens)] = 1
        return ids, mask
