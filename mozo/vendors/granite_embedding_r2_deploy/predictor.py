# SPDX-License-Identifier: Apache-2.0
"""The seam: text in, one L2-normalised vector per text out.

**The vector is the ``[CLS]`` position's final hidden state, normalised.** That is the published
pooling -- ``1_Pooling/config.json`` sets ``pooling_mode_cls_token`` and nothing else -- and the
normalisation is what ``sentence-transformers`` applies with ``normalize_embeddings=True``, the
setting IBM's model card uses for retrieval. Upstream returns the vector unnormalised; mozo
normalises so that a dot product between any two vectors is their cosine, the same contract as
every other family that embeds.

**A stored vector must not depend on what it was encoded beside.** A padded batch is not
bit-identical to its rows run alone (1.1e-07, measured), so nothing here pads. Texts of exactly
equal length are batched together on the CPU, where that was measured bit-identical to running
each alone at every length and batch size tried; on any other device each text runs alone,
because on MPS equal-length batches were *not* identical (1.8e-07) and CUDA has not been
measured. So the same text always gets the same vector, whichever texts it arrived with.

**Too long is refused, never truncated.** Upstream runs past 8,192 tokens without a word, on
rotary positions the model never saw. A chunk shortened without saying so is a different chunk,
and only the caller can decide where to cut.

The tokenizer and the network are each built on first use, under their own lock: ``mozo.server``
runs handlers in a threadpool, and two cold requests should not both build the same part.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Sequence

import torch
import torch.nn.functional as F

from .checkpoint import load_network
from .config import CONTEXT, Spec
from .network import Network
from .tokenizer import Tokenizer

__all__ = ["Encoder", "batches"]

#: Most tokens run in one forward pass on the CPU. Bounds the activations of a batch of short
#: texts to what a single full-length text needs anyway. Batch size does not change a vector.
BATCH_TOKENS = CONTEXT

#: Device types where a batch of equal-length texts was measured bit-identical to running each
#: alone. CPU only: on MPS it was not (1.8e-07), and CUDA has not been measured. Measuring it is
#: a change to this line.
BATCHES_ON = frozenset({"cpu"})


def batches(lengths: list[int], device: str) -> list[list[int]]:
    """Which texts run together, by index: equal lengths only, never padded.

    On a device in :data:`BATCHES_ON`, texts of the same length share a forward pass, up to
    :data:`BATCH_TOKENS` tokens each; anywhere else every text runs alone.
    """
    if torch.device(device).type not in BATCHES_ON:
        return [[index] for index in range(len(lengths))]
    by_length: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        by_length.setdefault(length, []).append(index)
    out = []
    for length, indices in by_length.items():
        size = BATCH_TOKENS // length  # at least 1: the tokenizer refuses anything longer
        out.extend(indices[start:start + size] for start in range(0, len(indices), size))
    return out


class Encoder:
    """One loaded Granite Embedding R2 variant.

    Args:
        weights: Path to a published checkpoint.
        spec: Which variant it is.
        device: Where to run.

    Attributes:
        spec: The variant's geometry. ``spec.width`` is the vector's dimension.
        device: The device in use.

    Examples:
        >>> encoder = Encoder(path, SPECS["small"])                          # doctest: +SKIP
        >>> encoder.encode_text(["how do I run a workflow?", "a forklift"])  # doctest: +SKIP
        tensor([[...], [...]])                                       # (2, 384), unit length
    """

    def __init__(self, weights: str | Path, spec: Spec, device: str = "cpu") -> None:
        self.spec = spec
        self.device = device
        self._weights = Path(weights)
        self._tokenizer: Tokenizer | None = None
        self._network: Network | None = None
        self._locks = {"tokenizer": Lock(), "network": Lock()}

    def _once(self, part: str, build):
        """Return ``self._<part>``, building it the first time and only once.

        Test, take the part's lock, test again: two threads can both pass the first test, and only
        the second decides which of them builds.
        """
        attribute = f"_{part}"
        if getattr(self, attribute) is None:
            with self._locks[part]:
                if getattr(self, attribute) is None:
                    setattr(self, attribute, build())
        return getattr(self, attribute)

    @property
    def tokenizer(self) -> Tokenizer:
        """The tokenizer, built on first use, refusing anything over the 8,192-token context."""
        return self._once("tokenizer", lambda: Tokenizer(CONTEXT))

    def network(self) -> Network:
        """The encoder network, built on first use."""
        return self._once("network", lambda: load_network(self.spec, self._weights, self.device))

    def encode_text(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Encode one text or many.

        Args:
            texts: A text, or a sequence of them. Each must be at most 8,192 tokens.

        Returns:
            ``(N, width)`` float32, L2-normalised, in the order given. Always two-dimensional,
            including for a single text.

        Raises:
            ValueError: If nothing is given, if a text is blank, if one does not fit, or if one
                contains an unpaired surrogate.
        """
        rows = self.tokenizer(texts)
        network = self.network()
        out = torch.empty(len(rows), self.spec.width)
        with torch.inference_mode():
            for indices in batches([len(row) for row in rows], self.device):
                tokens = torch.tensor([rows[i] for i in indices], device=self.device)
                out[indices] = F.normalize(network(tokens)[:, 0], p=2, dim=1).cpu()
        return out
