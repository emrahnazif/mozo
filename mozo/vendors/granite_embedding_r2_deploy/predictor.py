# SPDX-License-Identifier: Apache-2.0
"""The seam: text in, one L2-normalised vector per text out.

**The vector is the ``[CLS]`` position's final hidden state, normalised.** That is the published
pooling -- ``1_Pooling/config.json`` sets ``pooling_mode_cls_token`` and nothing else -- and the
normalisation is what ``sentence-transformers`` applies with ``normalize_embeddings=True``, the
setting IBM's model card uses for retrieval. Upstream returns the vector unnormalised; mozo
normalises so that a dot product between any two vectors is their cosine, the same contract as
every other family that embeds.

**A stored vector must not depend on what it was encoded beside**, so every text runs alone.
A padded batch is not bit-identical to its rows run alone (1.1e-07, measured), and on x86 with
more than one thread not even a batch of texts of exactly equal length is: 7e-06 to 1.6e-03 on
the raw vector at 16 to 64 tokens, measured with both variants at the default thread count.
(Equal-length batches looked safe on an M1 Max and on one thread, and were used until that was
measured.) So the same text always gets the same vector, whichever texts it arrived with.

**Too long is refused, never truncated.** Upstream runs past the context (8,192 tokens in
English, 32,768 multilingual) without a word, on rotary positions the model never saw. A chunk
shortened without saying so is a different chunk, and only the caller can decide where to cut.

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
from .config import Spec
from .network import Network
from .tokenizer import Tokenizer

__all__ = ["Encoder"]


class Encoder:
    """One loaded Granite Embedding R2 variant, English or multilingual.

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
        """The variant's tokenizer, built on first use, refusing anything over its context."""
        return self._once("tokenizer",
                          lambda: Tokenizer(self.spec.context, self.spec.vocabulary))

    def network(self) -> Network:
        """The encoder network, built on first use."""
        return self._once("network", lambda: load_network(self.spec, self._weights, self.device))

    def encode_text(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Encode one text or many.

        Args:
            texts: A text, or a sequence of them. Each must fit the variant's context:
                8,192 tokens in English, 32,768 multilingual.

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
            for index, row in enumerate(rows):
                tokens = torch.tensor([row], device=self.device)
                out[index] = F.normalize(network(tokens)[:, 0], p=2, dim=1)[0].cpu()
        return out
