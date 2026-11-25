# SPDX-License-Identifier: Apache-2.0
"""The seam: images in, one L2-normalised vector per image out.

**The vector is the final-norm class token, normalised.** It is what upstream's own k-NN
evaluation embeds: ``dinov2/eval/knn.py`` wraps the backbone in ``ModelWithNormalize``,
``F.normalize(model(x), dim=1, p=2)``, and ``model(x)`` is ``x_norm_clstoken``. Normalised so that
a dot product between two vectors is their cosine, the contract every family that embeds keeps.

**A stored vector must not depend on what it was encoded beside.** One forward over several
images is not bit-identical to running each alone -- 1.5e-05 on base and 1.8e-05 on large on
the CPU, where MKL picks kernels by batch shape (with ``MKL_CBWR=AVX2,STRICT`` set they agree,
but nobody runs a server that way) -- so every image runs alone, on every device. The same image always gets the same vector,
whichever images it arrived with.

The network is built on first use, under a lock: ``mozo.server`` runs handlers in a threadpool,
and two cold requests should not both build it.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_network
from .config import Spec
from .image import preprocess

__all__ = ["Encoder"]


class Encoder:
    """One loaded DINOv2 variant.

    Args:
        weights: Path to a published checkpoint.
        spec: Which variant it is.
        device: Where the torch network runs, and where pixels are placed before a *backbone*.
        backbone: A callable that replaces the torch backbone: ``(1, 3, 224, 224)`` pixels to the
            ``(1, width)`` class token. How a graph runtime takes the network's place; the
            preprocessing and the normalisation stay here, so they are the same code on every
            runtime. *weights* is then unused.

    Attributes:
        spec: The variant's geometry. ``spec.width`` is the vector's dimension.
        device: The device in use.
    """

    def __init__(self, weights: str | Path | None, spec: Spec, device: str = "cpu",
                 backbone: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        self.spec = spec
        self.device = device
        self._weights = weights
        self._backbone = backbone
        self._lock = Lock()

    def network(self):
        """The backbone, built on first use."""
        if self._backbone is None:
            with self._lock:
                if self._backbone is None:
                    self._backbone = load_network(self.spec, self._weights, self.device)
        return self._backbone

    def encode_image(self, images: np.ndarray | Iterable[np.ndarray]) -> torch.Tensor:
        """Encode one image or many.

        Args:
            images: An ``HxWx3`` RGB ``uint8`` array, or an iterable of them, read one at a time.

        Returns:
            ``(N, width)`` float32 on the CPU, L2-normalised. Always two-dimensional, including
            for a single image.
        """
        batch = [images] if isinstance(images, np.ndarray) else images
        network = self.network()
        with torch.inference_mode():
            # Kept on the device until the end, so the next image's preprocessing overlaps this
            # one's forward instead of waiting on a copy back.
            vectors = [F.normalize(network(preprocess(image)[None].to(self.device)).float(),
                                   dim=1, p=2)
                       for image in batch]
        if not vectors:
            raise ValueError("give at least one image to encode")
        return torch.cat(vectors).cpu()
