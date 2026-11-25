# ------------------------------------------------------------------------
# BiRefNet -- Bilateral Reference for High-Resolution Dichotomous Image Segmentation
# Copyright (c) 2024 ZhengPeng. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""Load a checkpoint, run an image, return a matte or a cut-out.

Upstream has no predictor class: the model card and ``handler.py`` each inline the same dozen
lines. Two of their habits are not carried, because a library may not do them to its host:

* ``torch.set_float32_matmul_precision('high')`` at import (``handler.py``) -- a process-wide
  switch to TF32 matrix products on Ampere and later, which moves every number the host's other
  models compute. Here precision is the caller's.
* ``image.putalpha(mask)`` writes into the caller's own ``PIL.Image``. This returns new arrays.
"""

from __future__ import annotations

__all__ = ["Predictor"]

from pathlib import Path

import numpy as np
import torch

from .config import MODEL_SPECS
from .image import postprocess, preprocess, refine_foreground
from .network import BiRefNet


class Predictor:
    """One loaded BiRefNet, ready to matte.

    Attributes:
        model: The underlying :class:`~.network.BiRefNet`, in eval mode.
        resolution: The square side images are resized to.
        device: Where the weights live.
        dtype: The compute dtype. Parity is claimed for ``float32`` on CPU and nothing else.
        autocast: Run the network under torch's float16 autocast -- the authors' own CUDA
            examples run in half precision. Off by default; not covered by the parity claim.
    """

    def __init__(self, model: BiRefNet, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32, autocast: bool = False) -> None:
        self.model = model.eval().to(device=device, dtype=dtype)
        self.resolution = model.spec.resolution
        self.device = device
        self.dtype = dtype
        self.autocast = autocast

    @classmethod
    def from_pretrained(cls, variant: str, weights: str | Path, device: str | torch.device = "cpu",
                        dtype: torch.dtype = torch.float32, autocast: bool = False) -> "Predictor":
        """Build *variant*'s network and load *weights* strictly.

        The published files keep the publisher's dtype -- ``general`` and ``hr`` are stored in
        float16 -- and the load copies each tensor into a float32 parameter, which is exact.
        """
        # Built on the meta device and allocated empty on the target: every parameter and buffer
        # is overwritten by the strict load, so initialising 220M random values first (1.2 s for
        # Swin-L) and copying them to the device would be wasted. Not ``assign=True``, which
        # would keep the published float16 dtype instead of copying into float32.
        with torch.device("meta"):
            model = BiRefNet(MODEL_SPECS[variant])
        model = model.to_empty(device=device)
        model.load_state_dict(torch.load(Path(weights), map_location="cpu", weights_only=True),
                              strict=True)
        return cls(model, device=device, dtype=dtype, autocast=autocast)

    @torch.inference_mode()
    def logits(self, rgb: np.ndarray) -> torch.Tensor:
        """``(H, W, 3)`` uint8 RGB -> ``(1, 1, S, S)`` logits at the network's resolution."""
        pixels = preprocess(rgb, self.resolution, self.device, self.dtype)
        with torch.autocast(pixels.device.type, dtype=torch.float16, enabled=self.autocast):
            return self.model(pixels)

    def matte(self, rgb: np.ndarray) -> np.ndarray:
        """``(H, W, 3)`` uint8 RGB -> ``(H, W)`` uint8 alpha. 255 is foreground."""
        return postprocess(self.logits(rgb), rgb.shape[:2])

    def cutout(self, rgb: np.ndarray, *, refine: bool = False) -> np.ndarray:
        """``(H, W, 3)`` uint8 RGB -> ``(H, W, 4)`` uint8 RGBA, background transparent.

        Args:
            rgb: The image.
            refine: Run ``handler.py``'s blur-fusion foreground estimate on the colours first.
                The alpha is the same either way.
        """
        alpha = self.matte(rgb)
        return np.dstack([refine_foreground(rgb, alpha) if refine else rgb, alpha])
