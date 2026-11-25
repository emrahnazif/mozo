# SPDX-License-Identifier: Apache-2.0
"""The fusion layer: image and text attend to each other through one shared score matrix.

One product of image queries against text keys serves both directions. Read along its rows it is
image-to-text attention; transposed, text-to-image. Each direction updates its own side through a
small learned layer scale (``vision_param``/``text_param``, initialised at 1e-4), so early in
training fusion is nearly a no-op.

Two details are upstream's and look like mistakes:

- **The residual is added to the normalised input**, not to the input:
  ``norm(x) + scale * f(norm(x))``. The usual pre-norm reading, ``x + scale * f(norm(x))``, is a
  different model.
- **The first softmax subtracts a global maximum** -- one scalar over the whole batch -- while the
  transposed direction subtracts a per-row one. The global one is not the stable-softmax idiom
  and changes the float result, so it is kept.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["FusionLayer"]


class _BiAttention(nn.Module):
    """Bidirectional multi-head attention between image tokens and text tokens."""

    def __init__(self, dim: int, embed_dim: int, heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim ** (-0.5)
        self.vision_proj = nn.Linear(dim, embed_dim)
        self.text_proj = nn.Linear(dim, embed_dim)
        self.values_vision_proj = nn.Linear(dim, embed_dim)
        self.values_text_proj = nn.Linear(dim, embed_dim)
        self.out_vision_proj = nn.Linear(embed_dim, dim)
        self.out_text_proj = nn.Linear(embed_dim, dim)

    def _shape(self, x: Tensor, length: int, batch: int) -> Tensor:
        return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, vision: Tensor, text: Tensor, vision_padding: Tensor,
                text_padding: Tensor) -> tuple[Tensor, Tensor]:
        batch, image_len, _ = vision.size()
        flat = (batch * self.heads, -1, self.head_dim)

        query = self._shape(self.vision_proj(vision) * self.scale, image_len, batch).view(*flat)
        key = self._shape(self.text_proj(text), -1, batch).view(*flat)
        vision_value = self._shape(self.values_vision_proj(vision), -1, batch).view(*flat)
        text_value = self._shape(self.values_text_proj(text), -1, batch).view(*flat)
        text_len = key.size(1)

        scores = torch.bmm(query, key.transpose(1, 2))
        # One scalar over the whole batch -- see the module docstring.
        scores = scores - scores.max()
        # fp16's range, not a tuning choice: upstream's comment says not to widen it.
        scores = torch.clamp(scores, min=-50000, max=50000)

        transposed = scores.transpose(1, 2)
        text_scores = transposed - torch.max(transposed, dim=-1, keepdim=True)[0]
        text_scores = torch.clamp(text_scores, min=-50000, max=50000)

        vision_mask = vision_padding[:, None, None, :].repeat(1, self.heads, 1, 1).flatten(0, 1)
        text_scores = text_scores.masked_fill(vision_mask, float("-inf")).softmax(dim=-1)

        text_mask = text_padding[:, None, None, :].repeat(1, self.heads, 1, 1).flatten(0, 1)
        vision_scores = scores.masked_fill(text_mask, float("-inf")).softmax(dim=-1)

        vision_out = torch.bmm(vision_scores, text_value)
        text_out = torch.bmm(text_scores, vision_value)

        vision_out = (vision_out.view(batch, self.heads, image_len, self.head_dim).transpose(1, 2)
                      .reshape(batch, image_len, self.embed_dim))
        text_out = (text_out.view(batch, self.heads, text_len, self.head_dim).transpose(1, 2)
                    .reshape(batch, text_len, self.embed_dim))
        return self.out_vision_proj(vision_out), self.out_text_proj(text_out)


class FusionLayer(nn.Module):
    """Normalise both sides, cross-attend, and add each update back with its layer scale.

    Args:
        dim: Model width on both sides.
        embed_dim: Width inside the attention -- half the encoder's feed-forward width.
        heads: Attention heads -- half the encoder's.
    """

    def __init__(self, dim: int, embed_dim: int, heads: int) -> None:
        super().__init__()
        self.layer_norm_vision = nn.LayerNorm(dim)
        self.layer_norm_text = nn.LayerNorm(dim)
        self.attn = _BiAttention(dim, embed_dim, heads)
        self.vision_param = nn.Parameter(torch.ones(dim))
        self.text_param = nn.Parameter(torch.ones(dim))

    def forward(self, vision: Tensor, text: Tensor, vision_padding: Tensor,
                text_padding: Tensor) -> tuple[Tensor, Tensor]:
        """Return the updated ``(vision, text)``. Paddings are True where a token is padding."""
        vision = self.layer_norm_vision(vision)
        text = self.layer_norm_text(text)
        delta_vision, delta_text = self.attn(vision, text, vision_padding, text_padding)
        return vision + self.vision_param * delta_vision, text + self.text_param * delta_text
