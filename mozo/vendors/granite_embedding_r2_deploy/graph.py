# SPDX-License-Identifier: Apache-2.0
"""The encoder at a fixed length, for a graph runtime that cannot take any other.

CoreML executes a fixed shape on the GPU and a flexible one only on the CPU: a range of lengths
crashes it on load, and a list of lengths ran 3-29x slower than torch on MPS. So a published graph
holds the same network at a few fixed lengths, and a text is padded up to the smallest one that
fits it.

**Padding is masked out, and changes nothing that matters.** Padding goes on the end, and every
attention layer hides the padded keys, so a real token attends to exactly what it would unpadded:
the same neighbours, at the same rotary positions. Measured against :class:`Network` on the
unpadded text, 1.8e-07 or less at every length tried, and zero when the text fills the bucket.
That is arithmetic order, the size of the difference between two attention kernels in torch.

**Positions come from the input, never from ``torch.arange``.** A tensor built from a length the
trace knows is a constant to a converter, so every mask and rotation was folded into the package --
67 MB of mask for the longest bucket alone -- and loading it took 40 s. Derived from the ids, they
are computed on the device instead: 191 MB for every bucket together, and 1 s to load one.

The vector is the same as :class:`~mozo.vendors.granite_embedding_r2_deploy.predictor.Encoder`'s:
the ``[CLS]`` position's final hidden state, L2-normalised.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layers import sliding_mask
from .network import Network

__all__ = ["Bucketed", "padded"]


class Bucketed(nn.Module):
    """``input_ids (1, B)`` padded after ``length (1,)`` real tokens -> a unit vector ``(1, width)``.

    What follows ``length`` in ``input_ids`` is never read by a real token, so any id will do;
    :func:`padded` writes zeros.
    """

    def __init__(self, network: Network) -> None:
        super().__init__()
        self.network = network

    def forward(self, input_ids: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        network = self.network
        # 0, 1, ... B-1, from the ids rather than from B, so that nothing below is a constant.
        positions = torch.cumsum(torch.ones_like(input_ids, dtype=torch.float32), dim=1)[0] - 1
        real = (positions < length.to(torch.float32))[None, None, None, :]
        masks = {True: real, False: sliding_mask(positions, network.window) & real}
        hidden = network.run(network.embeddings(input_ids), positions, masks)
        return F.normalize(hidden[:, 0], p=2, dim=1)


def padded(row: Sequence[int], length: int) -> np.ndarray:
    """``(1, length)`` int32: *row*'s ids, then zeros -- what :class:`Bucketed` is fed."""
    out = np.zeros((1, length), dtype=np.int32)
    out[0, :len(row)] = row
    return out
