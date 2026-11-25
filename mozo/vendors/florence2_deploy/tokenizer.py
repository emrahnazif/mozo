# SPDX-License-Identifier: MIT
"""Florence-2's vocabulary, for reading what the decoder wrote -- and its fixed task prompts.

The tokenizer is BART's byte-level BPE (50,265 tokens) plus the 1,024 tokens Florence-2's processor
adds at load (``<od>``, ``<loc_0>`` ... ``<loc_999>``, ``<cap>`` ...), all of them *special*. This
module carries the 51,289 pieces (derived from the published ``tokenizer.json`` by
``tools/fetch/florence2_detect.py --derive-vocab``) and does what upstream's post-processor does
with them:

* :meth:`Vocabulary.decode_with_spans` is ``Florence2PostProcesser.decode_with_spans``: special
  tokens kept verbatim -- the ``<loc_*>`` tokens *are* the boxes -- and every other token decoded
  *on its own* (``convert_tokens_to_string([token])``), so a character split across two byte-level
  tokens becomes two replacement characters, as upstream; plus the character span each token
  covers, which is how the OD score finds a box's four location tokens.

No encoder: the families here only send upstream's fixed task sentences, whose ids are constants
below (held against the reference processor by the gate). Encoding free text -- what
``<OPEN_VOCABULARY_DETECTION>`` would need for a phrase -- is not built yet. Standard library only.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

__all__ = ["PROMPTS", "Vocabulary"]

VOCAB_PATH = Path(__file__).resolve().parent / "assets" / "florence2_vocab.json.gz"

#: Task token -> the ids of the sentence upstream's processor sends instead (``<s> ... </s>``).
#: ``<OD>`` is "Locate the objects with category name in the image.", ``<OCR_WITH_REGION>`` is
#: "What is the text in the image, with regions?" (``Florence2Processor._construct_prompts``).
PROMPTS: dict[str, tuple[int, ...]] = {
    "<OD>": (0, 574, 22486, 5, 8720, 19, 4120, 766, 11, 5, 2274, 4, 2),
    "<OCR_WITH_REGION>": (0, 2264, 16, 5, 2788, 11, 5, 2274, 6, 19, 3806, 116, 2),
}


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """GPT-2's printable-character-per-byte table, inverted (``Ġ`` back to a space)."""
    printable = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    mapped = printable[:]
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return {chr(code): byte for byte, code in zip(printable, mapped)}


class Vocabulary:
    """The 51,289 pieces, and which are special."""

    def __init__(self, path: str | Path = VOCAB_PATH) -> None:
        data = json.loads(gzip.decompress(Path(path).read_bytes()))
        self.pieces: list[str] = data["pieces"]
        self.special: frozenset[int] = frozenset(data["special"])

    def _text(self, ids) -> str:
        decoder = _byte_decoder()
        return bytes(decoder[c] for c in "".join(self.pieces[i] for i in ids)).decode(
            "utf-8", errors="replace")

    def decode_with_spans(self, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
        """Each token decoded alone, concatenated, with the ``[start, end)`` span of each."""
        text, spans = "", []
        for token in ids:
            piece = self.pieces[token] if token in self.special else self._text([token])
            spans.append((len(text), len(text) + len(piece)))
            text += piece
        return text, spans
