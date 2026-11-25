# SPDX-License-Identifier: Apache-2.0
"""A loaded OmDet-Turbo: an image and a vocabulary in, detections out.

**The text is encoded once per vocabulary.** A vocabulary's class embeddings and task-sentence
states depend on the words alone; they are computed together, as one batch, and kept keyed on the
whole tuple of prompts. Upstream caches each class prompt separately across calls, encoding only
the ones it has not seen -- which makes its numbers depend on what was asked before, because a
batched text forward is not bit-identical to the same rows run in another batch. Keying the whole
vocabulary gives exactly what upstream gives from an empty cache, every time.

**The task sentence is upstream's default**, ``"Detect cat, remote."``: the prompts joined by
``", "`` inside ``"Detect "`` and ``"."``. Upstream truncates it at 77 tokens and so does this --
with the 80 COCO names it is cut after about twenty of them, which is how the published model
behaves; the class embeddings, which do the scoring, are never truncated. A class prompt that does
not fit in 77 tokens is refused.

**The image path can be handed in** as ``forward``, so a graph runtime replaces Swin, the encoder
and the decoder while the text tower stays torch.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from torch import Tensor

from .checkpoint import build
from .config import Spec
from .postprocess import postprocess
from .preprocess import preprocess
from .text.tokenizer import Tokenizer

__all__ = ["Detector", "PROMPT_CACHE", "THRESHOLD", "task_sentence"]

#: Upstream's default confidence floor: transformers' post-processing and the authors' demo.
THRESHOLD = 0.3

#: How many encoded vocabularies to keep. A few KB per phrase; the bound exists only so a server
#: fed unbounded distinct vocabularies does not grow without limit.
PROMPT_CACHE = 64

#: ``(pixels, classes, task) -> (boxes, logits)``, the image path's contract.
Forward = Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]


def task_sentence(prompts: Sequence[str]) -> str:
    """Upstream's default task: ``"Detect {', '.join(prompts)}."``."""
    return f"Detect {', '.join(prompts)}."


class Detector:
    """OmDet-Turbo, ready to run.

    Args:
        weights: The published ``torch-fp32.pth``.
        spec: Geometry.
        device: Where the text tower (and the torch image path) runs.
        forward: Replaces the torch image path, for graph runtimes. The torch image path is then
            not kept.
    """

    def __init__(self, weights: str | Path, spec: Spec, device: str | torch.device = "cpu",
                 forward: Forward | None = None):
        self.spec = spec
        self.device = torch.device(device)
        self.image, self.text = build(spec, weights, device=self.device, image=forward is None)
        self.forward: Forward = forward or self.image
        self.tokenizer = Tokenizer(context_length=spec.text.context_length)
        self._prompts: OrderedDict[tuple[str, ...], tuple[Tensor, Tensor]] = OrderedDict()
        self._lock = Lock()

    def encode_text(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        """``(C, 1, 512)`` class embeddings and ``(T, 1, 512)`` task states for a vocabulary.

        Raises:
            ValueError: If a prompt does not fit in the text tower's 77 tokens.
        """
        key = tuple(prompts)
        with self._lock:
            if key in self._prompts:
                self._prompts.move_to_end(key)
                return self._prompts[key]
        class_ids, _ = self.tokenizer(list(key), truncate=False)
        task_ids, task_mask = self.tokenizer([task_sentence(key)], truncate=True)
        with torch.inference_mode():
            classes = self.text.classes(class_ids.to(self.device)).unsqueeze(1)
            task = self.text.task(task_ids.to(self.device), task_mask.to(self.device))
        with self._lock:
            self._prompts[key] = (classes, task)
            while len(self._prompts) > PROMPT_CACHE:
                self._prompts.popitem(last=False)
        return classes, task

    def with_forward(self, forward: Forward) -> "Detector":
        """A detector sharing this one's text tower, tokenizer and vocabulary cache, whose image
        path is *forward* -- to compare a graph against the torch path on identical text."""
        other = object.__new__(Detector)
        other.spec, other.device, other.text, other.tokenizer = (
            self.spec, self.device, self.text, self.tokenizer)
        other._prompts, other._lock = self._prompts, self._lock
        other.image, other.forward = None, forward
        return other

    def raw(self, image: np.ndarray, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        """The image path's ``(900, 4)`` boxes and ``(900, C)`` logits, before post-processing."""
        classes, task = self.encode_text(prompts)
        pixels = preprocess(image, self.spec.swin.image_size).to(self.device)
        with torch.inference_mode():
            boxes, logits = self.forward(pixels, classes, task)
        return boxes[0].float(), logits[0].float()

    def __call__(self, image: np.ndarray, prompts: Sequence[str],
                 threshold: float = THRESHOLD) -> tuple[Tensor, Tensor, Tensor]:
        """Detect every prompt in *image*.

        Args:
            image: ``HxWx3`` RGB uint8.
            prompts: The vocabulary; a detection's label is its index here.
            threshold: Keep scores strictly above this.

        Returns:
            ``(N, 4)`` xyxy pixel boxes, ``(N,)`` scores, ``(N,)`` prompt indices, best first,
            on the CPU.
        """
        boxes, logits = self.raw(image, prompts)
        found = postprocess(boxes, logits, image.shape[:2], threshold)
        return tuple(value.cpu() for value in found)
