# SPDX-License-Identifier: Apache-2.0
"""The text decoder: Qwen3's transformer with multimodal rotary positions and DeepStack.

Extracted from ``transformers/models/qwen3_vl/modeling_qwen3_vl.py`` v5.24.1
(``Qwen3VLTextModel`` and its layers) and from what ``generate`` hands it: the attention call
``integrations/sdpa_attention.py`` makes, and the cache ``cache_utils.DynamicLayer`` keeps.
Parameter names are upstream's, so ``model.language_model.*`` loads strictly.

What is particular to it:

- **Per-head RMSNorm on queries and keys** before the rotary embedding (Qwen3's q/k-norm).
- **M-RoPE, interleaved.** Each token has three positions -- time, row, column; text tokens have
  the same number in all three. The 64 rotary frequencies are dealt out T, H, W, T, H, W ...
  (``mrope_section`` 24/20/20), not in three blocks.
- **DeepStack.** The vision tower's three intermediate feature sets are *added* to the image
  positions' hidden states after decoder layers 0, 1 and 2. Only where image tokens are, so only
  at prefill.
- **The attention call is the one transformers makes on this hardware**: SDPA with
  ``enable_gqa=True`` (on CPU and CUDA, with no mask and head_dim <= 256), ``is_causal`` when more
  than one query and no mask -- which is always, here, since there is no padding -- and no mask at
  all for single-token decode steps. A hand-built causal mask would be different arithmetic.
- **The cache grows by ``torch.cat`` from an empty tensor**, as ``DynamicLayer`` does: even the
  prefill's keys are a contiguous copy. A preallocated cache attends over a padded length and is
  not the same reduction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import TextSpec

__all__ = ["Cache", "TextDecoder", "rope_index"]


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Cache:
    """One growing key/value pair per layer: ``DynamicLayer``'s ``torch.cat`` exactly."""

    def __init__(self, layers: int) -> None:
        self.keys: list[torch.Tensor | None] = [None] * layers
        self.values: list[torch.Tensor | None] = [None] * layers

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer] is None:
            self.keys[layer] = torch.tensor([], dtype=k.dtype, device=k.device)
            self.values[layer] = torch.tensor([], dtype=v.dtype, device=v.device)
        self.keys[layer] = torch.cat([self.keys[layer], k], dim=-2)
        self.values[layer] = torch.cat([self.values[layer], v], dim=-2)
        return self.keys[layer], self.values[layer]

    @property
    def length(self) -> int:
        return 0 if self.keys[0] is None else self.keys[0].shape[-2]


class Attention(nn.Module):
    def __init__(self, spec: TextSpec, index: int) -> None:
        super().__init__()
        self.index = index
        self.head_dim = spec.head_dim
        self.groups = spec.heads // spec.kv_heads
        self.scaling = spec.head_dim ** -0.5
        self.q_proj = nn.Linear(spec.width, spec.heads * spec.head_dim, bias=False)
        self.k_proj = nn.Linear(spec.width, spec.kv_heads * spec.head_dim, bias=False)
        self.v_proj = nn.Linear(spec.width, spec.kv_heads * spec.head_dim, bias=False)
        self.o_proj = nn.Linear(spec.heads * spec.head_dim, spec.width, bias=False)
        self.q_norm = RMSNorm(spec.head_dim, spec.norm_eps)
        self.k_norm = RMSNorm(spec.head_dim, spec.norm_eps)

    def forward(self, x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor], cache: Cache
                ) -> torch.Tensor:
        shape = (*x.shape[:-1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(shape)).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = (t.unsqueeze(1) for t in rotary)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        k, v = cache.update(self.index, k, v)
        queries = q.shape[2]
        # sdpa_attention_forward: no mask ever reaches here (no padding), so GQA goes to SDPA, and
        # is_causal is set exactly when there is more than one query.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                             scale=self.scaling, is_causal=queries > 1,
                                             enable_gqa=self.groups > 1)
        out = out.reshape(out.shape[0], -1, queries, v.shape[-1]).transpose(1, 2).contiguous()
        return self.o_proj(out.reshape(*x.shape[:-1], -1).contiguous())


class MLP(nn.Module):
    def __init__(self, spec: TextSpec) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(spec.width, spec.mlp, bias=False)
        self.up_proj = nn.Linear(spec.width, spec.mlp, bias=False)
        self.down_proj = nn.Linear(spec.mlp, spec.width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, spec: TextSpec, index: int) -> None:
        super().__init__()
        self.self_attn = Attention(spec, index)
        self.mlp = MLP(spec)
        self.input_layernorm = RMSNorm(spec.width, spec.norm_eps)
        self.post_attention_layernorm = RMSNorm(spec.width, spec.norm_eps)

    def forward(self, x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor], cache: Cache
                ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), rotary, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


def rope_index(input_ids: torch.Tensor, image_token: int, grid: tuple[int, int, int] | None,
               merge: int) -> tuple[torch.Tensor, int]:
    """``get_rope_index`` for one sequence holding at most one image: ``((3, 1, L), delta)``.

    Text runs count up in all three rows. An image's tokens get ``(t, row, column)`` offset by the
    position where it starts, and the text after it resumes at that start plus the longer side of
    the merged grid -- so positions are *not* token indices after an image, and ``delta`` (the
    largest position + 1 - length) carries the difference into decoding.
    """
    kinds = (input_ids[0] == image_token).tolist()
    pieces, current, start = [], 0, 0
    while start < len(kinds):
        end = start
        while end < len(kinds) and kinds[end] == kinds[start]:
            end += 1
        if not kinds[start]:
            pieces.append(torch.arange(end - start).view(1, -1).expand(3, -1) + current)
            current += end - start
        else:
            t, h, w = grid[0], grid[1] // merge, grid[2] // merge
            tt, hh, ww = torch.meshgrid(torch.arange(t).long(), torch.arange(h) + current,
                                        torch.arange(w) + current, indexing="ij")
            positions = torch.stack([tt, hh, ww], dim=0).reshape(3, -1)
            positions[0] += current
            pieces.append(positions)
            current += max(grid[1], grid[2]) // merge
        start = end
    positions = torch.cat(pieces, dim=1).reshape(3, -1)
    delta = int(positions.max()) + 1 - len(kinds)
    return positions.view(3, 1, -1).to(input_ids.dtype), delta


class TextDecoder(nn.Module):
    """``Qwen3VLTextModel`` plus the tied output projection."""

    def __init__(self, spec: TextSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embed_tokens = nn.Embedding(spec.vocab, spec.width)
        self.layers = nn.ModuleList(Layer(spec, i) for i in range(spec.layers))
        self.norm = RMSNorm(spec.width, spec.norm_eps)
        self.register_buffer(
            "inv_freq",
            1.0 / (spec.rope_theta ** (torch.arange(0, spec.head_dim, 2, dtype=torch.float) / spec.head_dim)),
            persistent=False)

    def _rotary(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``Qwen3VLTextRotaryEmbedding``: *positions* ``(3, 1, L)``."""
        positions = positions.expand(3, -1, -1)
        freqs = positions[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos, sin = freqs.cos() * 1.0, freqs.sin() * 1.0
        section = self.spec.mrope_section

        def recompose(f: torch.Tensor) -> torch.Tensor:
            thw = f[0]
            for dim, offset in enumerate((1, 2), start=1):
                idx = slice(offset, section[dim] * 3, 3)
                thw[..., idx] = f[dim, ..., idx]
            return torch.cat((thw, thw), dim=-1)

        sin, cos = recompose(sin), recompose(cos)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def forward(self, embeds: torch.Tensor, positions: torch.Tensor, cache: Cache,
                visual: torch.Tensor | None = None, deepstack: list[torch.Tensor] | None = None,
                record: dict | None = None) -> torch.Tensor:
        """Hidden states of the last position only, normed: ``(1, 1, width)``.

        *visual* marks image positions ``(1, L)`` and *deepstack* the features added there.
        """
        rotary = self._rotary(embeds, positions)
        x = embeds
        for number, layer in enumerate(self.layers):
            x = layer(x, rotary, cache)
            if deepstack is not None and number < len(deepstack):
                x = x.clone()
                x[visual, :] = x[visual, :] + deepstack[number].to(x.device, x.dtype)
            if record is not None:
                record[f"text_layer_{number}"] = x
        x = self.norm(x)
        if record is not None:
            record["final_norm"] = x
        return x[:, slice(-1, None), :]

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The tied output projection: ``lm_head`` is ``embed_tokens``' weight."""
        return F.linear(hidden, self.embed_tokens.weight)
