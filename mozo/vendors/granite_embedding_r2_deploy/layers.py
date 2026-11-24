# SPDX-License-Identifier: Apache-2.0
"""ModernBERT's encoder block, as Granite Embedding R2 runs it.

A pre-norm transformer with four details that are not ordinary, each of which runs without
complaint if it is got wrong:

**Rotary positions, at two bases.** No position embedding is added anywhere. Queries and keys are
rotated instead, and the rotation's base depends on the layer: the full-attention layers turn at
``global_theta`` (80,000), the sliding ones at ``local_theta`` (10,000). The angles are computed in
float32 from an *integer* ``arange`` divided by the head dimension, the way upstream does it -- a
float ``arange`` rounds differently.

**Two kinds of attention.** Every third layer, from the first, attends to everything. The others
see ``abs(query - key) <= window`` -- 64 either side, 129 positions in all. Upstream's attention
module carries ``window + 1``, but only its FlashAttention path reads that; the mask the SDPA path
uses is built from ``window`` itself. Copying the 65 changes every input longer than 65 tokens.

**A gated MLP with exact GELU.** One matrix produces the value and the gate together, split in that
order: ``Wo(gelu(value) * gate)``. ``hidden_activation`` is ``"gelu"``, the erf form, not the tanh
approximation SigLIP 2 uses.

**No biases, and no norm before the first attention.** Not in any linear layer, not in any
LayerNorm. And layer 0 feeds its input straight to attention: upstream makes its ``attn_norm`` an
``nn.Identity``, so there is no weight for it in the checkpoint, and the strict load is what says so.

Attention is ``scaled_dot_product_attention``, the call upstream's SDPA path makes. The gate runs
the reference with ``attn_implementation="sdpa"``; eager attention is the same arithmetic in a
different order (1.8e-07 apart, measured), and at 8,192 tokens it would materialise a 3 GB score
matrix per layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import Spec

__all__ = ["EncoderLayer", "Rotary", "sliding_mask"]


class Rotary(nn.Module):
    """The cosines and sines one layer type's rotation uses, for every position in the input."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        # Integer arange, then float: what upstream's ``compute_default_rope_parameters`` does.
        exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
        self.register_buffer("inv_freq", 1.0 / (theta**exponents), persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)``, each ``(1, 1, L, head_dim)``: the shape attention broadcasts over heads.

        *positions* is ``(L,)`` float32, ``0 .. L-1``. In float32 whatever the model's dtype --
        upstream disables autocast here -- then cast.
        """
        freqs = (self.inv_freq[None, :, None] @ positions[None, None, :]).transpose(1, 2)
        angles = torch.cat((freqs, freqs), dim=-1)[:, None]
        return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the rotation in float32, as upstream does, and return in *x*'s dtype."""
    wide = x.float()
    return ((wide * cos) + (_rotate_half(wide) * sin)).to(x.dtype)


def sliding_mask(positions: torch.Tensor, window: int) -> torch.Tensor:
    """``(1, 1, L, L)`` bool: True where ``abs(query - key) <= window``, for ``(L,)`` *positions*.

    Always built, at every length. Upstream skips it below 64 tokens and passes nothing, which is
    the same as a mask that allows everything; measured bit-identical at every length from 2 to
    8,192, so there is no branch here for an exported graph to freeze.

    Two comparisons rather than ``abs(query - key) <= window``, which says the same thing but builds
    two ``(L, L)`` tensors on the way: 1.1 GB at 8,192 tokens for a 67 MB mask.
    """
    keys, queries = positions[None, :], positions[:, None]
    return ((keys >= queries - window) & (keys <= queries + window))[None, None]


class Attention(nn.Module):
    """Multi-head self-attention over rotated queries and keys, with an optional mask."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.heads, self.head_dim = spec.heads, spec.head_dim
        self.Wqkv = nn.Linear(spec.width, 3 * spec.width, bias=False)
        self.Wo = nn.Linear(spec.width, spec.width, bias=False)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
        batch, length, _ = hidden.shape
        qkv = self.Wqkv(hidden).view(batch, length, 3, self.heads, self.head_dim)
        queries, keys, values = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        queries, keys = _rotate(queries, cos, sin), _rotate(keys, cos, sin)
        attended = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=mask, scale=self.head_dim**-0.5)
        return self.Wo(attended.transpose(1, 2).reshape(batch, length, -1))


class MLP(nn.Module):
    """``Wo(gelu(value) * gate)``, with value and gate from one projection, in that order."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.Wi = nn.Linear(spec.width, 2 * spec.mlp, bias=False)
        self.Wo = nn.Linear(spec.mlp, spec.width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value, gate = self.Wi(hidden).chunk(2, dim=-1)
        return self.Wo(F.gelu(value) * gate)


class EncoderLayer(nn.Module):
    """One pre-norm block. Layer 0 has no norm before attention; every other layer does.

    Attributes:
        is_global: Whether this layer attends to the whole input, or only within the window.
            Decided here, once, so the network asks each layer rather than redoing the arithmetic.
    """

    def __init__(self, spec: Spec, layer: int) -> None:
        super().__init__()
        self.is_global = spec.is_global(layer)
        self.attn_norm = (nn.Identity() if layer == 0
                          else nn.LayerNorm(spec.width, eps=spec.norm_eps, bias=False))
        self.attn = Attention(spec)
        self.mlp_norm = nn.LayerNorm(spec.width, eps=spec.norm_eps, bias=False)
        self.mlp = MLP(spec)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
        hidden = hidden + self.attn(self.attn_norm(hidden), cos, sin, mask)
        return hidden + self.mlp(self.mlp_norm(hidden))
