# SPDX-License-Identifier: Apache-2.0
"""CLIP ViT-B/16's text tower, as OmDet-Turbo reads it twice.

The same transformer encodes two different things:

- **the task sentence** ("Detect cat, remote."): every token's final hidden state, cut to the
  sentence's length, goes into the decoder, where the queries attend to it;
- **each class prompt** ("cat"): the hidden state at the end-of-text marker, times
  ``text_projection``, is what a query is scored against.

**No attention mask, only causality.** Upstream calls the text model with the ids alone, so the
only mask is the causal one; padding comes after the end marker and a causal mask already hides it
from every token that is read. The reference runs this through ``scaled_dot_product_attention``
with ``is_causal=True`` and an explicit scale, and so does this: an explicit additive mask is the
same algebra and different bits.

**Pooling by ``argmax`` over the ids**, which finds the first ``<|endoftext|>`` because padding is
the same id and ``argmax`` returns the first maximum.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import TextSpec

__all__ = ["TextTower"]


class Attention(nn.Module):
    """Causal multi-head self-attention with separate q, k, v projections."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.q_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        shape = (batch, length, -1, self.head_dim)
        query = self.q_proj(x).view(shape).transpose(1, 2)
        key = self.k_proj(x).view(shape).transpose(1, 2)
        value = self.v_proj(x).view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                                  dropout_p=0.0, is_causal=True, scale=self.scale)
        return self.out_proj(attended.transpose(1, 2).contiguous().reshape(batch, length, -1))


class FeedForward(nn.Module):
    """Widen, ``quick_gelu``, narrow."""

    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(width, intermediate)
        self.fc2 = nn.Linear(intermediate, width)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        return self.fc2(x * torch.sigmoid(1.702 * x))


class Block(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, spec: TextSpec):
        super().__init__()
        self.self_attn = Attention(spec.width, spec.heads)
        self.layer_norm1 = nn.LayerNorm(spec.width, eps=spec.layer_norm_eps)
        self.mlp = FeedForward(spec.width, spec.intermediate)
        self.layer_norm2 = nn.LayerNorm(spec.width, eps=spec.layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Embeddings(nn.Module):
    def __init__(self, spec: TextSpec):
        super().__init__()
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.width)
        self.position_embedding = nn.Embedding(spec.context_length, spec.width)
        self.register_buffer("position_ids", torch.arange(spec.context_length).expand((1, -1)),
                             persistent=False)

    def forward(self, ids: Tensor) -> Tensor:
        return self.token_embedding(ids) + self.position_embedding(
            self.position_ids[:, : ids.shape[-1]])


class Layers(nn.Module):
    def __init__(self, spec: TextSpec):
        super().__init__()
        self.layers = nn.ModuleList(Block(spec) for _ in range(spec.layers))


class TextTower(nn.Module):
    """Token ids in; per-token states for a task, or one projected embedding per class.

    Args:
        spec: Text geometry.
    """

    def __init__(self, spec: TextSpec):
        super().__init__()
        self.embeddings = Embeddings(spec)
        self.encoder = Layers(spec)
        self.final_layer_norm = nn.LayerNorm(spec.width, eps=spec.layer_norm_eps)
        self.text_projection = nn.Parameter(torch.zeros(spec.width, spec.projection))

    def hidden(self, ids: Tensor) -> Tensor:
        """``(B, L)`` ids -> ``(B, L, width)`` final hidden states."""
        x = self.embeddings(ids)
        for block in self.encoder.layers:
            x = block(x)
        return self.final_layer_norm(x)

    def task(self, ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Encode task sentences, as upstream's ``encode_type="task"``.

        Returns:
            ``(length, B, width)`` states cut to the longest sentence, and ``(B, length)`` mask.
        """
        hidden = self.hidden(ids)
        length = (mask != 0).sum(1).max().item()
        return hidden[:, :length, :].transpose(0, 1), mask[:, :length]

    def classes(self, ids: Tensor) -> Tensor:
        """Encode class prompts, as upstream's ``encode_type="class"``: ``(B, projection)``."""
        hidden = self.hidden(ids)
        pooled = hidden[torch.arange(hidden.shape[0]), ids.argmax(dim=-1)]
        return pooled @ self.text_projection
