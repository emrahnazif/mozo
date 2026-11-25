# SPDX-License-Identifier: MIT
"""The whole Florence-2 network: DaViT, the image projection, and the BART language model.

Derived from ``Florence2ForConditionalGeneration._encode_image`` and
``_merge_input_ids_with_image_features`` in Microsoft's ``modeling_florence2.py``. The image becomes
577 tokens -- one mean over the 24 x 24 grid, then the 576 grid tokens themselves -- projected to
the language model's width, and the task prompt's embeddings are appended after them. That
sequence is what the encoder reads.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import Spec
from .language import LanguageModel
from .vision import DaViT

__all__ = ["Florence2"]


class _PositionEmbedding2D(nn.Module):
    """``image_pos_embed``: learned row and column embeddings, concatenated (column first)."""

    def __init__(self, dim: int, rows: int) -> None:
        super().__init__()
        self.row_embeddings = nn.Embedding(rows, dim // 2)
        self.column_embeddings = nn.Embedding(rows, dim - dim // 2)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        height, width = grid.shape[1:3]
        x = self.column_embeddings(torch.arange(width, device=grid.device))
        y = self.row_embeddings(torch.arange(height, device=grid.device))
        pos = torch.cat([x.unsqueeze(0).repeat(height, 1, 1), y.unsqueeze(1).repeat(1, width, 1)],
                        dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(grid.shape[0], 1, 1, 1)
        return pos.permute(0, 2, 3, 1)


class _TemporalEmbedding(nn.Module):
    """``visual_temporal_embed``: a cosine table, *loaded* from the checkpoint (it is fp16 there).

    Upstream computes it at construction and then overwrites it from the state dict, so the values
    that run are the published, half-precision-rounded ones. Only row 0 is used: one image is one
    frame.
    """

    def __init__(self, dim: int, rows: int) -> None:
        super().__init__()
        self.register_buffer("pos_idx_to_embed", torch.zeros(rows, dim))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        pos = self.pos_idx_to_embed[0:frames.size(-2), :]
        return pos.view(1, pos.size(0), pos.size(1))


class Florence2(nn.Module):
    """The module tree of the published checkpoint, key for key."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.spec = spec
        vision = spec.vision
        self.vision_tower = DaViT(vision)
        self.image_projection = nn.Parameter(torch.empty(vision.dims[-1], vision.projection))
        self.image_proj_norm = nn.LayerNorm(vision.projection)
        self.image_pos_embed = _PositionEmbedding2D(vision.dims[-1], vision.pos_rows)
        self.visual_temporal_embed = _TemporalEmbedding(vision.dims[-1], vision.temporal_rows)
        self.language_model = LanguageModel(spec.language)

    def encode_image(self, pixels: torch.Tensor, hooks: dict | None = None) -> torch.Tensor:
        """``(B, 3, 768, 768)`` pixels to ``(B, 577, width)`` image tokens."""
        x = self.vision_tower(pixels, hooks)
        batch, tokens, channels = x.shape
        side = int(tokens ** 0.5)
        x = x.view(batch, side, side, channels)
        x = x + self.image_pos_embed(x)
        x = x.view(batch, side * side, channels)
        frames = x.view(batch, 1, -1, channels)
        x = frames + self.visual_temporal_embed(frames[:, :, 0]).view(1, 1, 1, channels)
        spatial = x.view(batch, 1, -1, channels).mean(dim=2)
        temporal = x.view(batch, 1, -1, channels).mean(dim=1)
        x = torch.cat([spatial, temporal], dim=1)
        x = self.image_proj_norm(x @ self.image_projection)
        if hooks is not None:
            hooks["image_features"] = x
        return x

    def encode(self, image_tokens: torch.Tensor, prompt_ids: torch.Tensor,
               hooks: dict | None = None) -> torch.Tensor:
        """Image tokens and the prompt's ids to the encoder's output, ``(B, 577 + n, width)``."""
        embeds = torch.cat([image_tokens, self.language_model.embed(prompt_ids)], dim=1)
        if hooks is not None:
            hooks["encoder_input"] = embeds
        return self.language_model.encode(embeds, hooks)
