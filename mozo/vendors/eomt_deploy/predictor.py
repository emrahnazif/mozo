# SPDX-License-Identifier: Apache-2.0
"""The seam: a picture in, its segments out.

Pre-processing (:mod:`.image`), the network, and post-processing (:mod:`.postprocess`) chained the
way upstream's ``eval_step`` and the authors' ``inference.ipynb`` chain them, last layer only. What
comes out is a partition: every segment's mask, class and score, masks never overlapping.

The network is built on first use, under a lock: ``mozo.server`` runs handlers in a threadpool,
and two cold requests should not both build it.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

import numpy as np
import torch

from . import postprocess
from .checkpoint import load_network
from .config import Spec
from .image import resize_and_pad, windows

__all__ = ["Segmenter", "Segments", "one_at_a_time"]

#: ``(B, 3, side, side)`` float pixels in ``[0, 1]`` to ``(mask_logits, class_logits)``.
Network = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def one_at_a_time(run: Network) -> Network:
    """A network built for one picture (a graph with a fixed batch of 1), applied to each picture
    of a batch in turn -- a semantic picture's windows."""
    def network(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = [run(pixels[i:i + 1].contiguous()) for i in range(len(pixels))]
        return torch.cat([o[0] for o in outputs]), torch.cat([o[1] for o in outputs])
    return network


def _to_host(masks: torch.Tensor) -> np.ndarray:
    """Bring ``(N, H, W)`` bool masks to the host; through pinned memory from a GPU, where a
    pageable copy of tens of megabytes is several times slower."""
    if masks.is_cuda:
        masks = torch.empty(masks.shape, dtype=torch.bool, pin_memory=True).copy_(masks)
    return masks.cpu().numpy()


@dataclass
class Segments:
    """One picture's segments, in upstream's order.

    Attributes:
        masks: ``(N, H, W)`` bool, at the picture's size; no two overlap.
        class_ids: ``(N,)`` int64.
        scores: ``(N,)`` float32 class probabilities for panoptic segments, or ``None`` for
            semantic ones, which have no score of their own.
    """

    masks: np.ndarray
    class_ids: np.ndarray
    scores: Optional[np.ndarray]


class Segmenter:
    """One loaded EoMT variant.

    Args:
        weights: Path to a published checkpoint.
        spec: Which variant it is.
        device: Where the network runs and where its output is post-processed.
        network: A callable that replaces the torch network, for a graph runtime. *weights* is
            then unused.
        half: Run the torch network under CUDA fp16 autocast (matrix products in half precision,
            norms and softmax in fp32); the logits come back as fp32 for the post-processing.
    """

    def __init__(self, weights: str | Path | None, spec: Spec, device: str = "cpu",
                 network: Network | None = None, half: bool = False) -> None:
        self.spec = spec
        self.device = device
        self._weights = weights
        self._network = network
        self._half = half
        self._lock = Lock()

    def network(self) -> Network:
        """The network, built on first use."""
        if self._network is None:
            with self._lock:
                if self._network is None:
                    self._network = load_network(self.spec, self._weights, self.device)
        return self._network

    def raw(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """The network's ``uint8`` input and its output for *image*:
        ``(pixels, mask_logits, class_logits, origins)``, where *origins* are a semantic picture's
        windows (``[]`` for panoptic)."""
        side = self.spec.image_size
        if self.spec.task == "semantic":
            pixels, origins = windows(image, side)
        else:
            pixels, origins = resize_and_pad(image, side), []
        pixels = pixels.to(self.device)
        with torch.autocast("cuda", torch.float16) if self._half else nullcontext():
            mask_logits, class_logits = self.network()(pixels / 255.0)
        return pixels, mask_logits.float(), class_logits.float(), origins

    def segment(self, image: np.ndarray, threshold: float = postprocess.MASK_THRESHOLD) -> Segments:
        """Segment one ``HxWx3`` RGB ``uint8`` picture.

        Args:
            image: The picture.
            threshold: Panoptic only: the class probability a query needs to become a segment.
        """
        size = tuple(image.shape[:2])
        with torch.inference_mode():
            _, mask_logits, class_logits, origins = self.raw(image)
            if self.spec.task == "semantic":
                labels = postprocess.semantic(mask_logits, class_logits, origins, size,
                                              self.spec.image_size)
                ids = torch.unique(labels)
                masks = labels[None] == ids[:, None, None]
                return Segments(_to_host(masks), ids.cpu().numpy(), None)
            segments, found = postprocess.panoptic(
                mask_logits[0], class_logits[0], size, self.spec.image_size,
                self.spec.stuff, threshold=threshold)
            ids = torch.tensor([s for s, _, _ in found], dtype=torch.long, device=segments.device)
            masks = segments[None] == ids[:, None, None]
            return Segments(_to_host(masks),
                            np.array([c for _, c, _ in found], dtype=np.int64),
                            np.array([p for _, _, p in found], dtype=np.float32))
