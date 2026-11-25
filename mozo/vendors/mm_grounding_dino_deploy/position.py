# SPDX-License-Identifier: Apache-2.0
"""Sine position encodings: for image cells, for text tokens, and for decoder boxes.

Three uses of one idea, and they do not share constants:

- **Image cells** (:func:`image_position`) use temperature **20**, the config's
  ``positional_embedding_temperature``. Every other DETR in this repository uses 10000.
- **Text tokens and decoder boxes** (:func:`sine_embed`) use 10000.
- **Decoder boxes swap x and y** (``exchange_xy``) and text tokens do not.

Written as ``transformers`` writes them, op for op. IDEA Research's Grounding DINO computes the same
encodings with a different order of operations, which is not the same float.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["image_position", "sine_embed"]


def image_position(valid: Tensor, num_pos_feats: int, temperature: int) -> Tensor:
    """Encode each cell's normalised position.

    Args:
        valid: ``(batch, height, width)``, True where the cell is picture rather than padding.
        num_pos_feats: Channels per axis; the result has twice this.
        temperature: The frequency base. 20 here, not 10000.

    Returns:
        ``(batch, 2 * num_pos_feats, height, width)``, y channels first.
    """
    scale = 2 * math.pi
    y_embed = valid.cumsum(1, dtype=torch.float32)
    x_embed = valid.cumsum(2, dtype=torch.float32)
    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=valid.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def sine_embed(positions: Tensor, num_pos_feats: int, exchange_xy: bool,
               temperature: int = 10000) -> Tensor:
    """Encode each coordinate of *positions* separately and concatenate.

    Args:
        positions: ``(..., n)``. Integer token positions or float box coordinates; integers are
            promoted by the first multiplication, as upstream's are.
        num_pos_feats: Channels per coordinate.
        exchange_xy: Swap the first two coordinates' encodings -- ``[y, x, ...]`` -- which the
            decoder's boxes need and text positions do not.
        temperature: The frequency base.

    Returns:
        ``(..., n * num_pos_feats)``.
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=positions.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def encode(x: Tensor) -> Tensor:
        x = x * scale / dim_t
        return torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=3).flatten(2)

    encoded = [encode(x) for x in positions.split([1] * positions.shape[-1], dim=-1)]
    if exchange_xy:
        encoded[0], encoded[1] = encoded[1], encoded[0]
    return torch.cat(encoded, dim=-1)
