# SPDX-License-Identifier: MIT
"""Florence-2's language model: a BART encoder-decoder, SDPA attention, with a decoding cache.

Derived from ``Florence2Encoder``, ``Florence2Decoder``, their layers and ``Florence2Attention`` in
Microsoft's ``modeling_florence2.py`` (see ``PROVENANCE.md``). Post-norm layers, exact GELU,
learned positions offset by two (BART's quirk), no embedding scale (``scale_embedding`` is false),
and one embedding table shared by the encoder, the decoder and the output projection.

**The attention is SDPA, though the reference is loaded with ``attn_implementation="eager"``.** That
setting reaches the outer config only; the language model's own config keeps transformers'
default, so upstream builds ``Florence2SdpaAttention`` throughout the BART half (found by the gate:
eager arithmetic differs by 2e-07 per attention). With no padding anywhere, transformers turns every
mask into ``None`` (an all-ones mask is dropped for SDPA, and a one-token decoder step needs no
causal mask), so ``scaled_dot_product_attention`` runs unmasked and ``is_causal=False``, as here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import LanguageSpec

__all__ = ["LanguageModel"]

#: BART's learned positions start at row 2.
_OFFSET = 2


class _Attention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.q_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def _shape(self, tensor, length: int, batch: int):
        return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2).contiguous()

    def project(self, states):
        """Keys and values of *states* -- the cross-attention cache, computed once."""
        batch = states.shape[0]
        return self._shape(self.k_proj(states), -1, batch), self._shape(self.v_proj(states), -1, batch)

    def forward(self, hidden, keys=None, values=None, past=None):
        """Attend from *hidden*: to itself (``past`` grows it), or to given ``keys``/``values``.

        Returns:
            ``(output, (keys, values))`` -- the second item is the self-attention cache to keep.
        """
        batch, length, width = hidden.size()
        query = self.q_proj(hidden)
        if keys is None:
            keys = self._shape(self.k_proj(hidden), -1, batch)
            values = self._shape(self.v_proj(hidden), -1, batch)
            if past is not None:
                keys = torch.cat([past[0], keys], dim=2)
                values = torch.cat([past[1], values], dim=2)
        output = F.scaled_dot_product_attention(self._shape(query, length, batch), keys, values)
        output = output.transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(output), (keys, values)


class _EncoderLayer(nn.Module):
    def __init__(self, spec: LanguageSpec) -> None:
        super().__init__()
        self.self_attn = _Attention(spec.width, spec.heads)
        self.self_attn_layer_norm = nn.LayerNorm(spec.width)
        self.fc1 = nn.Linear(spec.width, spec.ffn)
        self.fc2 = nn.Linear(spec.ffn, spec.width)
        self.final_layer_norm = nn.LayerNorm(spec.width)

    def forward(self, hidden):
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(hidden)[0])
        hidden = self.final_layer_norm(hidden + self.fc2(F.gelu(self.fc1(hidden))))
        # Upstream's half-precision guard, kept for the fp16 runtime; a no-op in fp32.
        if hidden.dtype == torch.float16 and (torch.isinf(hidden).any() or torch.isnan(hidden).any()):
            limit = torch.finfo(hidden.dtype).max - 1000
            hidden = torch.clamp(hidden, min=-limit, max=limit)
        return hidden


class _DecoderLayer(nn.Module):
    def __init__(self, spec: LanguageSpec) -> None:
        super().__init__()
        self.self_attn = _Attention(spec.width, spec.heads)
        self.self_attn_layer_norm = nn.LayerNorm(spec.width)
        self.encoder_attn = _Attention(spec.width, spec.heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(spec.width)
        self.fc1 = nn.Linear(spec.width, spec.ffn)
        self.fc2 = nn.Linear(spec.ffn, spec.width)
        self.final_layer_norm = nn.LayerNorm(spec.width)

    def forward(self, hidden, cross, past):
        attended, cache = self.self_attn(hidden, past=past)
        hidden = self.self_attn_layer_norm(hidden + attended)
        hidden = self.encoder_attn_layer_norm(hidden + self.encoder_attn(hidden, *cross)[0])
        hidden = self.final_layer_norm(hidden + self.fc2(F.gelu(self.fc1(hidden))))
        return hidden, cache


class _Encoder(nn.Module):
    def __init__(self, spec: LanguageSpec) -> None:
        super().__init__()
        self.embed_positions = nn.Embedding(spec.positions + _OFFSET, spec.width)
        self.layers = nn.ModuleList(_EncoderLayer(spec) for _ in range(spec.layers))
        self.layernorm_embedding = nn.LayerNorm(spec.width)

    def forward(self, embeds, hooks: dict | None = None):
        positions = torch.arange(0, embeds.shape[1], dtype=torch.long,
                                 device=self.embed_positions.weight.device).expand(embeds.shape[0], -1)
        hidden = self.layernorm_embedding(embeds + self.embed_positions(positions + _OFFSET))
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if hooks is not None:
                hooks[f"encoder_layer_{index}"] = hidden
        return hidden


class _Decoder(nn.Module):
    def __init__(self, spec: LanguageSpec) -> None:
        super().__init__()
        self.embed_positions = nn.Embedding(spec.positions + _OFFSET, spec.width)
        self.layers = nn.ModuleList(_DecoderLayer(spec) for _ in range(spec.layers))
        self.layernorm_embedding = nn.LayerNorm(spec.width)


class _Model(nn.Module):
    def __init__(self, spec: LanguageSpec) -> None:
        super().__init__()
        self.shared = nn.Embedding(spec.vocab, spec.width, padding_idx=1)
        self.encoder = _Encoder(spec)
        self.decoder = _Decoder(spec)


class LanguageModel(nn.Module):
    """``language_model.*`` of the checkpoint: the BART model plus ``final_logits_bias``.

    ``lm_head`` is ``shared`` (tied, as published: the checkpoint stores the table once).
    """

    def __init__(self, spec: LanguageSpec) -> None:
        super().__init__()
        self.spec = spec
        self.model = _Model(spec)
        self.register_buffer("final_logits_bias", torch.zeros((1, spec.vocab)))

    @property
    def positions(self) -> int:
        return self.spec.positions

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.shared(ids)

    def encode(self, embeds: torch.Tensor, hooks: dict | None = None) -> torch.Tensor:
        return self.model.encoder(embeds, hooks)

    def cross_cache(self, encoded: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Each decoder layer's cross-attention keys and values -- computed once per decode."""
        return [layer.encoder_attn.project(encoded) for layer in self.model.decoder.layers]

    def step(self, ids: torch.Tensor, cross, past, position: int):
        """One decoding step: *ids* ``(beams, 1)`` at *position*, returning logits and the cache.

        Args:
            ids: The last token of each running hypothesis.
            cross: :meth:`cross_cache`'s result.
            past: Each layer's self-attention ``(keys, values)``, or ``None`` at the first step.
            position: How many tokens precede *ids* (the cache length).
        """
        decoder = self.model.decoder
        positions = torch.arange(position, position + ids.shape[1], dtype=torch.long,
                                 device=decoder.embed_positions.weight.device).expand(ids.shape[0], -1)
        hidden = decoder.layernorm_embedding(
            self.model.shared(ids) + decoder.embed_positions(positions + _OFFSET))
        cache = []
        for index, layer in enumerate(decoder.layers):
            hidden, kept = layer(hidden, cross[index], None if past is None else past[index])
            cache.append(kept)
        logits = F.linear(hidden, self.model.shared.weight) + self.final_logits_bias
        return logits, cache
