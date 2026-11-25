# SPDX-License-Identifier: Apache-2.0
"""Beam search, exactly as ``transformers`` 4.53.3 runs it for Florence-2's default ``generate()``.

Transcribed from ``GenerationMixin._beam_search`` and its helpers (``_get_top_k_continuations``,
``_get_running_beams_for_next_iteration``, ``_update_finished_beams``,
``_beam_search_has_unfinished_sequences``), the three logits processors Florence-2's config turns
on (``NoRepeatNGramLogitsProcessor``, ``ForcedBOSTokenLogitsProcessor``,
``ForcedEOSTokenLogitsProcessor``), the two stopping criteria (max length, end-of-sequence) and
``compute_transition_scores`` -- reduced to one image (batch 1). Apache-2.0, from
``huggingface/transformers`` v4.53.3; see ``NOTICE``.

What makes it exact rather than merely equivalent:

* the three beams run as one batch of three, because batch-1 and batch-3 matrix products are not
  bit-identical;
* the encoder runs once on one row and its output is repeated, as ``generate`` does, and the
  cross-attention keys and values are computed from the repeated rows;
* every step's log-probabilities are taken in float32 (``logits.to(dtype=torch.float32)``) whatever
  the model's precision, then processed, then offset by the running beam scores;
* ties among candidates are broken by ``torch.topk``, as upstream, not by a sort.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import DECODING, Decoding

__all__ = ["Beams", "beam_search", "transition_scores"]


@dataclass
class Beams:
    """What a beam search returns.

    Attributes:
        sequence: The best hypothesis, decoder start token first, as ``generate`` returns it.
        beam_indices: For each generated token, which of the flattened beams it came from.
        scores: Each step's processed log-probabilities, ``(beams, vocab)``, as ``output_scores``.
    """

    sequence: torch.Tensor
    beam_indices: torch.Tensor
    scores: list[torch.Tensor]


def _no_repeat_ngram(ids: torch.Tensor, scores: torch.Tensor, size: int) -> torch.Tensor:
    """``NoRepeatNGramLogitsProcessor``: ban any token that would repeat an n-gram already written.

    Upstream builds each hypothesis's n-grams from the *whole* decoder sequence, the start token
    included, and bans what followed every earlier occurrence of the last ``size - 1`` tokens. One
    linear scan per hypothesis finds the same set without building the n-gram table each step.
    """
    hypotheses, length = ids.shape
    out = scores.clone()
    if length + 1 < size:
        return out
    rows, banned = [], []
    for row, tokens in enumerate(ids.tolist()):
        prefix = tokens[length + 1 - size:length]
        for start in range(length - size + 1):
            if tokens[start:start + size - 1] == prefix:
                rows.append(row)
                banned.append(tokens[start + size - 1])
    if banned:
        out[rows, banned] = -float("inf")
    return out


def _process(ids: torch.Tensor, scores: torch.Tensor, max_length: int, rules: Decoding):
    scores = _no_repeat_ngram(ids, scores, rules.no_repeat_ngram)
    length = ids.shape[-1]
    if length == 1:  # ForcedBOSTokenLogitsProcessor
        scores = torch.full_like(scores, -math.inf)
        scores[:, rules.bos] = 0
    if length == max_length - 1:  # ForcedEOSTokenLogitsProcessor
        scores = torch.full_like(scores, -math.inf)
        scores[:, rules.eos] = 0
    return scores


def _gather(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    while indices.dim() < tensor.dim():
        indices = indices.unsqueeze(-1)
    return torch.take_along_dim(input=tensor, indices=indices, dim=1)


def beam_search(language, encoded: torch.Tensor, max_new_tokens: int,
                rules: Decoding = DECODING) -> Beams:
    """Decode from *encoded* (``(1, S, width)``, the encoder's output) with upstream's beam search.

    Args:
        language: The :class:`~.language.LanguageModel`.
        encoded: The encoder output for one image and prompt.
        max_new_tokens: The length limit; the last position is forced to ``</s>``.
        rules: The decoding settings (Florence-2's, from its ``text_config``).
    """
    device = encoded.device
    beams, vocab = rules.num_beams, language.spec.vocab
    cur_len = prompt_len = 1
    max_length = prompt_len + max_new_tokens
    keep = 2 * beams  # upstream: max(2, 1 + number of end-of-sequence ids) * beams
    top_mask = torch.cat((torch.ones(beams, dtype=torch.bool),
                          torch.zeros(keep - beams, dtype=torch.bool))).to(device)

    running = torch.full((1, beams, max_length), rules.pad, dtype=torch.int64, device=device)
    running[:, :, :cur_len] = rules.decoder_start
    sequences = running.clone()
    running_scores = torch.zeros((1, beams), dtype=torch.float, device=device)
    running_scores[:, 1:] = -1e9
    beam_scores = torch.full((1, beams), -1e9, dtype=torch.float, device=device)
    finished = torch.zeros((1, beams), dtype=torch.bool, device=device)
    running_indices = torch.full((1, beams, max_length - cur_len), -1, dtype=torch.int32,
                                 device=device)
    beam_indices = running_indices.clone()

    cross = language.cross_cache(encoded.repeat_interleave(beams, dim=0))
    past = None
    all_scores: list[torch.Tensor] = []
    while True:
        flat = running[:, :, :cur_len].reshape(beams, cur_len)
        logits, past = language.step(flat[:, -1:], cross, past, cur_len - 1)
        logits = logits[:, -1, :].to(copy=True, dtype=torch.float32)
        log_probs = _process(flat, F.log_softmax(logits, dim=-1), max_length, rules)
        all_scores.append(log_probs)  # rebound below, never mutated

        log_probs = log_probs.reshape(1, beams, vocab) + running_scores[:, :, None]
        log_probs = log_probs.reshape(1, beams * vocab)

        # _get_top_k_continuations
        topk_log_probs, topk_indices = torch.topk(log_probs, k=keep)
        topk_beams = topk_indices // vocab
        topk_running_indices = _gather(running_indices, topk_beams)
        topk_sequences = _gather(running, topk_beams)
        topk_sequences[:, :, cur_len] = topk_indices % vocab
        topk_running_indices[:, :, cur_len - prompt_len] = topk_beams

        # stopping criteria: max length, or the token just chosen is </s>
        hits = topk_sequences[:, :, cur_len] == rules.eos
        if cur_len + 1 >= max_length:
            hits = torch.ones_like(hits)

        # _get_running_beams_for_next_iteration
        topk_running_log_probs = topk_log_probs + hits.to(torch.float32) * -1.0e9
        next_indices = torch.topk(topk_running_log_probs, k=beams)[1]
        running = _gather(topk_sequences, next_indices)
        running_scores = _gather(topk_running_log_probs, next_indices)
        running_indices = _gather(topk_running_indices, next_indices)

        # _update_finished_beams
        just_finished = hits & top_mask[None, :]
        candidate = topk_log_probs / ((cur_len + 1 - prompt_len) ** rules.length_penalty)
        full = torch.all(finished, dim=-1, keepdim=True) & rules.early_stopping
        candidate += full.to(torch.float32) * -1.0e9
        candidate += (~just_finished) * -1.0e9
        merged_sequences = torch.cat((sequences, topk_sequences), dim=1)
        merged_scores = torch.cat((beam_scores, candidate), dim=1)
        merged_indices = torch.cat((beam_indices, topk_running_indices), dim=1)
        merged_finished = torch.cat((finished, just_finished), dim=1)
        best = torch.topk(merged_scores, k=beams)[1]
        sequences = _gather(merged_sequences, best)
        beam_scores = _gather(merged_scores, best)
        beam_indices = _gather(merged_indices, best)
        finished = _gather(merged_finished, best)

        reorder = running_indices[..., cur_len - prompt_len].reshape(beams)
        past = [(keys.index_select(0, reorder), values.index_select(0, reorder))
                for keys, values in past]
        cur_len += 1

        # _beam_search_has_unfinished_sequences (early_stopping is True)
        best_running = running_scores[:, :1] / ((cur_len - prompt_len) ** rules.length_penalty)
        worst_finished = torch.where(finished, torch.min(beam_scores, dim=1, keepdim=True)[0], -1.0e9)
        improvable = torch.any(best_running > worst_finished)
        open_beam = ~(torch.all(finished) & rules.early_stopping)
        continuable = ~torch.all(hits)
        if not bool(improvable & open_beam & continuable):
            break

    indices = beam_indices[0, :1, :]
    generated = int(((indices + 1).bool()).sum(dim=1).max())
    sequence = sequences[0, 0, :prompt_len + generated]
    return Beams(sequence, indices[:, :generated], all_scores)


def transition_scores(beams: Beams, vocab: int) -> torch.Tensor:
    """``compute_transition_scores(sequences, scores, beam_indices, normalize_logits=False)``.

    The log-probability (as processed) of each generated token on the beam it was chosen from.
    """
    scores = torch.stack(beams.scores).reshape(len(beams.scores), -1).transpose(0, 1)
    indices = beams.beam_indices
    mask = indices < 0
    length = (1 - mask.long()).sum(-1).max()
    indices = indices.clone()[:, :length]
    mask = mask[:, :length]
    indices[mask] = 0
    offsets = indices * vocab
    sequence = beams.sequence.unsqueeze(0)
    cut = sequence.shape[-1] - length
    out = scores.gather(0, sequence[:, cut:] + offsets)
    out[mask] = 0
    return out[0]
