# SPDX-License-Identifier: Apache-2.0
# Extracted from PaddleX 3.7.2, paddlex/inference/models/text_recognition/processors.py
# (BaseRecLabelDecode, CTCLabelDecode). Copyright (c) 2024-2026 PaddlePaddle Authors. Apache-2.0.
"""CTC decoding: per-step probabilities to a string and a confidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ASSETS = Path(__file__).resolve().parent / "assets"


class Alphabet:
    """Index 0 is CTC's blank; then the published ``character_dict``; then a space.

    Upstream appends the space (``use_space_char=True``) after the dictionary rather than
    listing it there, so the dictionary file alone is one symbol short of the head's width.
    """

    def __init__(self, charset: str) -> None:
        characters = json.loads((_ASSETS / charset).read_text(encoding="utf-8"))
        self.symbols = ["blank", *characters, " "]

    def __len__(self) -> int:
        return len(self.symbols)

    def decode(self, probs: np.ndarray) -> list[tuple[str, float]]:
        """Greedy CTC over ``(N, T, classes)``: argmax, collapse repeats, drop blanks.

        The confidence is the mean of the kept steps' maxima, in float32 as numpy computes it,
        and 0.0 for an empty read.
        """
        indices = probs.argmax(axis=-1)
        maxima = probs.max(axis=-1)
        out = []
        for index, prob in zip(indices, maxima):
            keep = np.ones(len(index), dtype=bool)
            keep[1:] = index[1:] != index[:-1]
            keep &= index != 0
            kept = prob[keep]
            text = "".join(self.symbols[i] for i in index[keep])
            out.append((text, np.mean(kept if len(kept) else [0]).tolist()))
        return out
