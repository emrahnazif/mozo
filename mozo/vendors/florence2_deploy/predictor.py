# SPDX-License-Identifier: MIT
"""Florence-2 end to end: an image and a task token in, parsed boxes out.

One network, several tasks. :meth:`Florence2Predictor.detect` is ``<OD>``; :meth:`read` is
``<OCR_WITH_REGION>`` (vendored and gated, not yet served by mozo -- its results carry no
confidence, which PixelFlow's array entry point cannot hold yet). Each is upstream's
``generate()`` with no arguments (3-beam search) and upstream's
``post_process_generation(sequence=..., transition_beam_score=...)``.

The *sequence* path, not the text path the model card shows: the card decodes to a string and
parses that, which gives no scores and leaves the decoder's leading ``</s>`` in the first OCR line
(``'</s>Hello World'``); the sequence path drops the start token first and is the one that returns
``<OD>``'s scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model
from .config import DECODING, SPECS
from .decoding import Beams, beam_search, transition_scores
from .image import preprocess
from .parse import Box, Line, parse_ocr, parse_od
from .tokenizer import PROMPTS, Vocabulary

__all__ = ["Florence2Predictor", "Output"]


@dataclass
class Output:
    """One decode, before parsing: the beams, and the source image's ``(width, height)``."""

    beams: Beams
    size: tuple[int, int]


class Florence2Predictor:
    """Florence-2 on one device and dtype.

    Args:
        weights: The published checkpoint (``torch-fp16.pth``: Microsoft's tensors, re-saved).
        variant: ``base``, ``large``, ``base-ft`` or ``large-ft``.
        device: Where to run.
        dtype: ``torch.float32`` (exact) or ``torch.float16``.
        max_new_tokens: The decoding length limit (upstream's examples use 1024).
    """

    def __init__(self, weights: str | Path, variant: str, device: str = "cpu",
                 dtype: torch.dtype = torch.float32, max_new_tokens: int = 1024) -> None:
        self.spec = SPECS[variant]
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.model = load_model(self.spec, weights, device, dtype)
        self.vocabulary = Vocabulary()

    @torch.inference_mode()
    def run(self, image: np.ndarray, task: str, hooks: dict | None = None,
            max_new_tokens: int | None = None) -> Output:
        """Decode *task* (a key of :data:`~.tokenizer.PROMPTS`) on an ``HxWx3`` RGB array."""
        pixels = preprocess(image)
        if hooks is not None:
            hooks["pixel_values"] = pixels
        pixels = pixels.to(self.device, self.dtype)
        prompt = torch.tensor([PROMPTS[task]], device=self.device)
        tokens = self.model.encode_image(pixels, hooks)
        encoded = self.model.encode(tokens, prompt, hooks)
        beams = beam_search(self.model.language_model, encoded,
                            max_new_tokens or self.max_new_tokens, DECODING)
        return Output(beams, (image.shape[1], image.shape[0]))

    def parse_od(self, output: Output) -> list[Box]:
        sequence = output.beams.sequence.tolist()[1:]
        text, spans = self.vocabulary.decode_with_spans(sequence)
        scores = transition_scores(output.beams, self.spec.language.vocab).tolist()
        if len(scores) != len(sequence):
            raise RuntimeError(f"{len(sequence)} tokens but {len(scores)} transition scores")
        return parse_od(text, spans, scores, output.size)

    def parse_ocr(self, output: Output) -> list[Line]:
        text, _ = self.vocabulary.decode_with_spans(output.beams.sequence.tolist()[1:])
        return parse_ocr(text, output.size)

    def detect(self, image: np.ndarray) -> list[Box]:
        """``<OD>``: every object the model names, with upstream's score."""
        return self.parse_od(self.run(image, "<OD>"))

    def read(self, image: np.ndarray) -> list[Line]:
        """``<OCR_WITH_REGION>``: every line of text, with its four corners."""
        return self.parse_ocr(self.run(image, "<OCR_WITH_REGION>"))
