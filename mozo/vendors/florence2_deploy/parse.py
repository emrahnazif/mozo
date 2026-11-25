# SPDX-License-Identifier: MIT
"""Turning what the decoder wrote into boxes: upstream's post-processor, for two tasks.

Derived from ``Florence2PostProcesser`` in Microsoft's ``processing_florence2.py``:
``parse_description_with_bboxes_from_text_and_spans`` with ``SCORE_MODE: avg_loc_scores`` (what
``<OD>`` returns, scores included) and ``parse_ocr_from_text_and_spans`` (``<OCR_WITH_REGION>``).

A location token ``<loc_k>`` is bin ``k`` of 1,000 across the image's width or height, and its
coordinate is the bin's *centre*: ``(k + 0.5) * size / 1000`` ("floor" quantisation). The
arithmetic runs in torch float32, as upstream's ``BoxQuantizer.dequantize`` does, so the floats
agree to the bit.

Upstream behaviour kept, not fixed: an OD label loses every non-ASCII character
(``encode('ascii', errors='ignore')``); a phrase with no box after it is skipped; an ``<OD>`` box's
score is ``exp`` of the mean log-probability of its four location tokens on the beam that wrote
them (``math.exp`` of a Python mean), which says how sure the decoder was of the coordinates once
it had written the label -- not how likely the object is.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["Box", "Line", "parse_od", "parse_ocr"]

_BINS = 1000
_BOX = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
_OD_PHRASES = r"([^<]+(?:<loc_\d+>){4,})"
_OD_LABEL = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
_OCR_LINE = r"(.+?)" + r"<loc_(\d+)>" * 8


@dataclass(frozen=True)
class Box:
    """One ``<OD>`` detection: ``xyxy`` in the image's pixels, the label, upstream's score."""

    box: list[float]
    label: str
    score: float | None


@dataclass(frozen=True)
class Line:
    """One ``<OCR_WITH_REGION>`` line: four corners ``x1, y1, ..., x4, y4`` and the text read."""

    quad: list[float]
    text: str


def _boxes(bins: list[list[int]], size: tuple[int, int]) -> list[list[float]]:
    """``BoxQuantizer.dequantize``: split into xmin, ymin, xmax, ymax, shift, scale, rejoin."""
    xmin, ymin, xmax, ymax = torch.tensor(bins).split(1, dim=-1)
    per_x, per_y = size[0] / _BINS, size[1] / _BINS
    return torch.cat(((xmin + 0.5) * per_x, (ymin + 0.5) * per_y,
                      (xmax + 0.5) * per_x, (ymax + 0.5) * per_y), dim=-1).tolist()


def _overlapping(span: tuple[int, int], spans: list[tuple[int, int]]) -> list[int]:
    return [i for i, (start, end) in enumerate(spans) if not (end <= span[0] or start >= span[1])]


def parse_od(text: str, spans: list[tuple[int, int]] | None, scores: list[float] | None,
             size: tuple[int, int]) -> list[Box]:
    """Upstream's ``description_with_bboxes`` parse of *text* (from ``decode_with_spans``).

    Args:
        text: The decoded sequence, decoder start token removed.
        spans: Each token's character span in *text*; with *scores*, enables the OD score.
        scores: Each token's transition score (log-probability on its beam).
        size: The source image's ``(width, height)``.
    """
    cursor = 0
    if text.startswith("<s>"):
        cursor += 3
    text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")

    out = []
    for phrase_text in re.findall(_OD_PHRASES, text):
        # Upstream's two replaces both start from ``phrase_text``: only the second one counts.
        stripped = phrase_text.replace("<obj>", "", 1)
        if stripped == "":
            cursor += len(phrase_text)
            continue
        phrase = re.search(_OD_LABEL, stripped)
        if phrase is None:
            cursor += len(phrase_text)
            continue
        label = phrase.group().strip()
        found = list(re.finditer(_BOX, phrase_text))
        if not found:
            cursor += len(phrase_text)
            continue
        bins = [[int(match.group(j)) for j in range(1, 5)] for match in found]
        boxes = _boxes(bins, size)

        box_scores = None
        if spans is not None and scores is not None:
            box_scores = []
            for match in found:
                start, end = match.span(0)
                tokens = _overlapping((start + cursor, end + cursor), spans)
                values = [scores[i] for i in tokens]
                box_scores.append(sum(values) / len(values))

        label = label.encode("ascii", errors="ignore").decode("ascii")
        for index, box in enumerate(boxes):
            out.append(Box(box, label,
                           None if box_scores is None else math.exp(box_scores[index])))
        cursor += len(phrase_text)
    return out


def parse_ocr(text: str, size: tuple[int, int]) -> list[Line]:
    """Upstream's ``ocr`` parse of *text* (from :meth:`~.tokenizer.Vocabulary.decode_with_spans`)."""
    text = text.replace("<s>", "")
    out = []
    for line in re.findall(_OCR_LINE, text):
        corners = torch.tensor(np.array([int(i) for i in line[1:]]).reshape(-1, 2))
        x, y = corners.split(1, dim=-1)
        per_x, per_y = size[0] / _BINS, size[1] / _BINS
        quad = torch.cat(((x + 0.5) * per_x, (y + 0.5) * per_y), dim=-1).reshape(-1).tolist()
        out.append(Line(quad, line[0]))
    return out
