# SPDX-License-Identifier: Apache-2.0
"""mozo's Qwen3-VL adapter: an image and a question in, an answer in words out.

mozo's first family that *writes*. Every other one answers in one forward pass with boxes, masks,
scores, vectors or a picture; this one answers a token at a time, each fed back in, until it stops
or reaches ``max_new_tokens``. So the answer is a :class:`~mozo.generation.Generation`, which says
which of the two happened.

**Greedy by default, and then the same answer every time.** ``temperature=0`` takes the most likely
token at every step, which is deterministic and what mozo's gate holds bit-exact against
``transformers``. A positive temperature samples -- the same ``seed`` gives the same answer on the
same device, and the gate holds that too on the CPU. Qwen's own recommendation for this model
samples (``temperature=0.7, top_p=0.8, top_k=20``): greedy answers from a small model can repeat
themselves on long outputs.

**The image is the cost.** It becomes one token per 32 x 32 pixels after resizing, up to
``max_pixels`` (2.6 MP by default, 2,560 tokens), and every one of them is read before the first
word is written. Halving ``max_pixels`` roughly halves that; fine print and small objects suffer
first.

**On CUDA it runs in bfloat16** when the device is CUDA and ``runtime`` is ``auto``: the weights'
own precision, at about a third of fp32's time and half its memory. bfloat16 is not bit-identical
to fp32 and can change the words; ``runtime="torch-fp32"`` is the exact path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .. import registry
from ..generation import Generation
from ..image import load_image
from ..runtimes import get_default_device, select_runtime
from ..vendors.qwen3_vl_deploy import MAX_PIXELS, SPECS, VARIANTS, Generator, Sampling
from ..weights import artifacts, resolve, revision_of

__all__ = ["Qwen3VLPredictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "qwen3_vl"

_log = logging.getLogger(__name__)

_DTYPES = {"torch-fp32": torch.float32, "torch-bf16": torch.bfloat16}


class Qwen3VLPredictor:
    """Qwen3-VL Instruct: describe an image, answer a question about it, read the text in it.

    Args:
        variant: ``2b`` or ``4b``.
        device: Where to run. Defaults to the best available.
        runtime: ``auto``, ``torch-fp32`` (exact) or ``torch-bf16`` (CUDA only).
        checkpoint_path: Your own checkpoint in the published layout, instead of the published
            weights. Run in float32.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which runtime is executing.
        revision: The published revision in use, or ``None`` for your own checkpoint.

    Examples:
        >>> model = Qwen3VLPredictor("2b")                                   # doctest: +SKIP
        >>> model.predict("photo.jpg", "How many people are there?").text   # doctest: +SKIP
        'There are five people in the image.'
    """

    #: The variants carried. ``config.py`` in the vendored package has their geometry.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute: torch only.
    EXECUTES = registry.EXECUTES[FAMILY]

    def __init__(
        self,
        variant: str = "2b",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")
        self.variant = variant
        self.device = device or get_default_device()
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(self.device, artifacts(FAMILY, variant, revision=revision), runtime,
                                executes=self.EXECUTES, derives=registry.DERIVES.get(FAMILY))
        )
        # One file either way: the published bfloat16 checkpoint, which torch-fp32 upcasts at load.
        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve(FAMILY, variant, "torch-bf16", revision=revision))
        self.revision = None if checkpoint_path else revision_of(FAMILY, variant, revision=revision)
        self._model = Generator(weights, SPECS[variant], self.device, _DTYPES[self.runtime])
        _log.info("Qwen3-VL %s ready on %s via %s.", variant, self.device, self.runtime)

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int = 0,
        max_pixels: int = MAX_PIXELS,
    ) -> Generation:
        """Answer *prompt* about *image*.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            prompt: The question or instruction, in any language the model reads. It may not
                contain the model's control tokens (``<|im_start|>``, ``<|image_pad|>`` ...).
            max_new_tokens: The most tokens to write. An answer that reaches it is cut off, and
                says so in ``finish_reason``.
            temperature: 0 (default) takes the most likely token every time: deterministic.
                Above 0, tokens are sampled from the model's distribution sharpened (below 1) or
                flattened (above 1) by it.
            top_p: When sampling, keep only the most likely tokens whose probabilities add up to
                this. 1.0 keeps all.
            top_k: When sampling, keep only this many most likely tokens. 0 keeps all.
            seed: When sampling, which sample to draw: the same seed gives the same answer on the
                same device. Ignored when greedy.
            max_pixels: The image is resized to at most this many pixels (and at least 65,536),
                sides multiples of 32, aspect kept. Between 65,536 and 16,777,216 (upstream's own
                ceiling). More pixels read finer detail and cost time and memory in proportion.

        Returns:
            A :class:`~mozo.generation.Generation`.

        Raises:
            ValueError: For a blank prompt or one holding a control token, ``max_new_tokens`` below
                1, ``max_pixels`` out of range, a negative ``temperature`` or ``top_k``, a
                ``top_p`` outside (0, 1], or an image more than 200 times wider than tall.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"prompt must be one string, got {type(prompt).__name__}")
        if not temperature >= 0:
            raise ValueError(f"temperature must be 0 (greedy) or more, got {temperature}")
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be above 0 and at most 1, got {top_p}")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError(f"top_k must be a whole number, 0 to keep every token, got {top_k!r}")
        if isinstance(max_pixels, bool) or not isinstance(max_pixels, int):
            raise ValueError(f"max_pixels must be a whole number, got {max_pixels!r}")
        answer = self._model.generate(
            load_image(image), prompt, max_new_tokens,
            Sampling(float(temperature), float(top_p), int(top_k), int(seed)), max_pixels)
        return Generation(answer.text, answer.finish_reason, answer.prompt_tokens,
                          answer.generated_tokens)
