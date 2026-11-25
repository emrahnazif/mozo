# SPDX-License-Identifier: Apache-2.0
"""OmDet-Turbo assembled: two halves that meet only in the decoder.

- :class:`TextTower` turns the prompts into class embeddings and the task sentence into token
  states. It depends on the words alone, so it runs once per vocabulary and is cached by the
  predictor.
- :class:`OmDetTurbo` is the image path: Swin, three layer norms, the hybrid encoder, the fusion
  decoder. It takes the cached text features as inputs, which is what lets a graph runtime cover
  all of it while the text tower stays in torch.
"""

from __future__ import annotations

from torch import Tensor, nn

from .config import Spec
from .decoder import Decoder
from .encoder import HybridEncoder
from .text.encoder import TextTower
from .vision.swin import Swin

__all__ = ["OmDetTurbo", "TextTower", "Vision"]


class Vision(nn.Module):
    """Swin, then a layer norm per level and channels-first (``OmDetTurboVisionBackbone``)."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.swin = Swin(spec.swin)
        self.norms = nn.ModuleList(
            nn.LayerNorm(width, eps=spec.layer_norm_eps) for width in spec.encoder_in_channels)

    def forward(self, pixels: Tensor) -> list[Tensor]:
        return [norm(level).permute(0, 3, 1, 2).contiguous()
                for norm, level in zip(self.norms, self.swin(pixels))]


class OmDetTurbo(nn.Module):
    """Pixels plus text features -> boxes and class logits.

    Args:
        spec: Geometry.
    """

    def __init__(self, spec: Spec):
        super().__init__()
        self.vision = Vision(spec)
        self.encoder = HybridEncoder(spec)
        self.decoder = Decoder(spec)

    def forward(self, pixels: Tensor, classes: Tensor, task: Tensor) -> tuple[Tensor, Tensor]:
        """Detect.

        Args:
            pixels: ``(B, 3, 640, 640)`` normalised.
            classes: ``(C, B, 512)`` projected class embeddings.
            task: ``(T, B, 512)`` task-sentence states.

        Returns:
            ``(B, 900, 4)`` boxes as (cx, cy, w, h) in [0, 1], and ``(B, 900, C)`` logits.
        """
        return self.decoder(self.encoder(self.vision(pixels)), classes, task)
