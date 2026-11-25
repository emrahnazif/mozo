# SPDX-License-Identifier: Apache-2.0
"""One image and one prompt to text: preprocessing, the prompt template, the model, the decoder."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from .checkpoint import load_model
from .config import CONTEXT, Spec
from .image import MAX_PIXELS, MIN_PIXELS, PIXEL_CEILING, preprocess
from .model import Sampling
from .tokenizer import Tokenizer

__all__ = ["Answer", "Generator"]


class Answer(NamedTuple):
    """What one call produced."""

    text: str
    finish_reason: str  # "stop" or "length"
    prompt_tokens: int
    generated_tokens: int
    ids: list[int]


def template(prompt: str, image_tokens: int) -> str:
    """The published chat template for one user turn holding one image and one text, rendered.

    ``chat_template.json`` adds no system prompt when none is given, and puts the image before the
    text when the image comes first in the message -- which is how mozo always sends them.
    """
    return ("<|im_start|>user\n<|vision_start|>" + "<|image_pad|>" * image_tokens
            + "<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n")


class Generator:
    """The model, loaded, with everything around it.

    Args:
        weights: The ``torch-bf16`` checkpoint.
        spec: The variant's geometry.
        device: Where to run.
        dtype: ``torch.float32`` (the exact path) or ``torch.bfloat16``.
    """

    def __init__(self, weights: str | Path, spec: Spec, device: str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.spec = spec
        self.tokenizer = Tokenizer()
        self.model = load_model(spec, weights, device, dtype)

    def prompt_ids(self, prompt: str, image_tokens: int) -> torch.Tensor:
        return torch.tensor([self.tokenizer.encode(template(prompt, image_tokens))])

    def generate(self, rgb: np.ndarray, prompt: str, max_new_tokens: int = 128,
                 sampling: Sampling = Sampling(), max_pixels: int = MAX_PIXELS,
                 record: dict | None = None) -> Answer:
        """Answer *prompt* about *rgb* (``HxWx3`` uint8).

        Raises:
            ValueError: For a blank prompt or one holding a control token, a ``max_new_tokens``
                below 1, a ``max_pixels`` outside ``[65536, 16777216]``, or an image more than 200
                times wider than tall (or the reverse). :class:`~.model.Sampling` refuses its own
                settings out of range when it is built.
        """
        self.tokenizer.check(prompt)
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be a whole number of at least 1, got {max_new_tokens!r}")
        if not MIN_PIXELS <= max_pixels <= PIXEL_CEILING:
            raise ValueError(f"max_pixels must be between {MIN_PIXELS:,} and {PIXEL_CEILING:,} "
                             f"(upstream's own ceiling), got {max_pixels:,}")
        pixels, grid = preprocess(rgb, self.spec.vision, max_pixels)
        input_ids = self.prompt_ids(prompt, grid[1] * grid[2] // self.spec.vision.merge ** 2)
        if input_ids.shape[1] + max_new_tokens > CONTEXT:
            raise ValueError(f"{input_ids.shape[1]} prompt tokens and {max_new_tokens} new ones "
                             f"exceed the {CONTEXT:,}-token context")
        if record is not None:
            record["pixel_values"], record["grid"], record["input_ids"] = pixels, grid, input_ids
        result = self.model.generate(input_ids, pixels, grid, max_new_tokens, sampling, record)
        return Answer(self.tokenizer.decode(result.ids), "stop" if result.stopped else "length",
                      result.prompt_tokens, len(result.ids), result.ids)
