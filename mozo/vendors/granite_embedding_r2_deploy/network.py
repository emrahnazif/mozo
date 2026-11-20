# SPDX-License-Identifier: Apache-2.0
"""The whole encoder: token ids in, one hidden state per token out.

Module names are upstream's (``embeddings.tok_embeddings``, ``layers.N.attn.Wqkv``,
``final_norm`` ...), because the checkpoint's keys are, and a rename would be a second table to
keep in step with a file nobody can change. The strict load is then a straight
``load_state_dict``, and it is what holds :mod:`~mozo.vendors.granite_embedding_r2_deploy.config`
against the checkpoint: a written-down width, depth or vocabulary that disagreed would fail there.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import Spec
from .layers import EncoderLayer, Rotary, sliding_mask

__all__ = ["Network"]


class Embeddings(nn.Module):
    """Token lookup, then a LayerNorm. No positions: those are rotary, applied in attention."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(spec.vocab, spec.width)
        self.norm = nn.LayerNorm(spec.width, eps=spec.norm_eps, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(tokens))


class Network(nn.Module):
    """Granite Embedding R2's encoder. ``(N, L)`` int64 ids -> ``(N, L, width)``.

    Every row must be the same length and none may be padding: there is no attention mask for
    padding, because nothing here pads. A padded batch is not bit-identical to its rows run alone
    (measured 1.1e-07), so :class:`~mozo.vendors.granite_embedding_r2_deploy.predictor.Encoder`
    only ever batches rows of equal length.
    """

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.window = spec.window
        self.embeddings = Embeddings(spec)
        self.layers = nn.ModuleList(EncoderLayer(spec, layer) for layer in range(spec.layers))
        self.final_norm = nn.LayerNorm(spec.width, eps=spec.norm_eps, bias=False)
        self.rotary_global = Rotary(spec.head_dim, spec.global_theta)
        self.rotary_local = Rotary(spec.head_dim, spec.local_theta)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(tokens)
        length = tokens.shape[1]
        rotations = {True: self.rotary_global(length, hidden.dtype),
                     False: self.rotary_local(length, hidden.dtype)}
        masks = {True: None, False: sliding_mask(length, self.window, tokens.device)}
        for layer in self.layers:
            hidden = layer(hidden, *rotations[layer.is_global], masks[layer.is_global])
        return self.final_norm(hidden)
