# SPDX-License-Identifier: Apache-2.0
"""The whole model, and the decoding loop ``generate`` runs it in.

``Qwen3VLForConditionalGeneration.forward`` and the parts of ``GenerationMixin`` that touch the
numbers (``_sample``, the logits warpers, ``DynamicCache``), reduced to one image, one prompt, batch
one. Upstream's loop is general -- beams, assistants, padding, stopping criteria by the dozen -- and
none of that changes a logit here, so it is not carried. What is carried is every operation that
does, in upstream's order:

1. **Prefill.** Embed the prompt, scatter the image tokens in (``masked_scatter``), compute the
   3-D positions (:func:`~.language.rope_index`), run the decoder with DeepStack, and project
   **only the last position** (``logits_to_keep=1``: a one-row ``lm_head`` is not bit-identical to
   the last row of a full one).
2. **Choose.** Logits to float32; greedy takes ``argmax``. Sampling applies upstream's warpers in
   upstream's order -- temperature, top-k, top-p -- then ``softmax`` and ``torch.multinomial`` on
   the caller's generator.
3. **Decode.** One token at a time through the cache, at position ``cache length + delta`` in all
   three rows, until a stop id or the budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import Spec, Tokens
from .language import Cache, TextDecoder, rope_index
from .vision import VisionTower

__all__ = ["Qwen3VL", "Sampling", "Result"]


@dataclass(frozen=True)
class Sampling:
    """How the next token is chosen. ``temperature == 0`` is greedy and ignores the rest.

    Attributes:
        temperature: Divides the logits before sampling. Upstream skips the division at exactly 1.0;
            dividing by 1.0 changes nothing, so neither does skipping it.
        top_p: Keep the smallest set of tokens whose probability reaches this; 1.0 keeps all.
        top_k: Keep this many most likely tokens; 0 keeps all.
        seed: Seeds the generator ``torch.multinomial`` draws from.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


@dataclass(frozen=True)
class Result:
    """What one generation produced, as ids. The adapter decodes them."""

    ids: list[int]
    stopped: bool
    prompt_tokens: int


def _warp(logits: torch.Tensor, sampling: Sampling) -> torch.Tensor:
    """``TemperatureLogitsWarper``, ``TopKLogitsWarper``, ``TopPLogitsWarper``, in that order."""
    scores = logits
    if sampling.temperature != 1.0:
        scores = scores / sampling.temperature
    if sampling.top_k != 0:
        top_k = min(sampling.top_k, scores.size(-1))
        remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(remove, -float("inf"))
    if sampling.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_remove = cumulative <= (1 - sampling.top_p)
        sorted_remove[..., -1:] = 0
        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
        scores = scores.masked_fill(remove, -float("inf"))
    return scores


class Qwen3VL(nn.Module):
    """The vision tower and the text decoder, named as upstream's ``model.*`` keys without ``model.``."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.spec = spec
        self.tokens = Tokens()
        self.visual = VisionTower(spec.vision)
        self.language_model = TextDecoder(spec.text)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, pixels: torch.Tensor, grid: tuple[int, int, int],
                 max_new_tokens: int, sampling: Sampling = Sampling(),
                 record: dict | None = None) -> Result:
        """Generate up to *max_new_tokens* after the prompt *input_ids* ``(1, L)``.

        *record*, when given, collects every stage and every step's logits for the gate.
        """
        device = self.language_model.embed_tokens.weight.device
        dtype = self.language_model.embed_tokens.weight.dtype
        input_ids = input_ids.to(device)
        generator = None
        if not sampling.greedy:
            generator = torch.Generator(device=device).manual_seed(sampling.seed)

        # Prefill.
        embeds = self.language_model.embed_tokens(input_ids)
        image, deepstack = self.visual(pixels.to(device, self.visual.patch_embed.proj.weight.dtype),
                                       grid, record)
        image = image.to(embeds.device, embeds.dtype)
        mask = (input_ids == self.tokens.image_pad)
        if int(mask.sum()) * embeds.shape[-1] != image.numel():
            raise ValueError(f"the prompt has {int(mask.sum())} image tokens and the image "
                             f"{image.shape[0]}")
        embeds = embeds.masked_scatter(mask.unsqueeze(-1).to(embeds.device), image)
        positions, delta = rope_index(input_ids, self.tokens.image_pad, grid, self.spec.vision.merge)
        if record is not None:
            record["embeds"] = embeds
            record["positions"] = positions
        cache = Cache(self.spec.text.layers)
        hidden = self.language_model(embeds, positions.to(device), cache, mask, deepstack, record)

        ids: list[int] = []
        stopped = False
        steps = record.setdefault("logits", []) if record is not None else None
        while True:
            logits = self.language_model.logits(hidden)[:, -1].to(copy=True, dtype=torch.float32)
            if steps is not None:
                steps.append(logits[0].cpu())
            if sampling.greedy:
                token = torch.argmax(logits, dim=-1)
            else:
                probs = nn.functional.softmax(_warp(logits, sampling), dim=-1)
                token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
            token_id = int(token)
            ids.append(token_id)
            if token_id in self.tokens.stop:
                stopped = True
                break
            if len(ids) >= max_new_tokens:
                break
            step = torch.tensor([[token_id]], device=device)
            length = cache.length
            pos = torch.arange(length, length + 1, device=device).view(1, 1, -1).expand(3, 1, -1) + delta
            hidden = self.language_model(self.language_model.embed_tokens(step).to(dtype), pos, cache)
        return Result(ids, stopped, int(input_ids.shape[1]))
