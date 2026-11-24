# SPDX-License-Identifier: Apache-2.0
"""Loading a published checkpoint into the encoder, strictly.

The checkpoint is IBM's ``ModernBertModel`` state dict, in bfloat16 as IBM stores it; the load
copies it into the float32 encoder, an exact upcast. Its keys are this package's module names
unchanged, so the load is one ``load_state_dict`` with ``strict=True`` -- and that strictness is
what holds :mod:`~mozo.vendors.granite_embedding_r2_deploy.config` against the file. A width,
depth or vocabulary written down wrong fails here, as does a norm the checkpoint has no weight for
(layer 0's, which upstream makes an ``nn.Identity``).

There is nothing to drop. The published repository is the bare encoder: no masked-language-model
head, no classifier, no pooler.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Spec
from .network import Network

__all__ = ["load_network"]


def load_network(spec: Spec, path: str | Path, device: str = "cpu") -> Network:
    """Build the encoder for *spec* and fill it from the checkpoint at *path*."""
    network = Network(spec)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    network.load_state_dict(state, strict=True)
    return network.eval().to(device)
