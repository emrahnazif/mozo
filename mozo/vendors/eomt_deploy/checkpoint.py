# SPDX-License-Identifier: Apache-2.0
"""Building the network for a variant and filling it from a published checkpoint, strictly.

The checkpoint is upstream's ``network`` state dict in its own key layout (``encoder.backbone.*``,
``q``, ``class_head``, ``mask_head``, ``upscale``, ``attn_mask_probs``), so the load is one
``load_state_dict(strict=True)``, and that strictness is what holds
:mod:`~mozo.vendors.eomt_deploy.config` against the file.

The network is built on the ``meta`` device and the checkpoint's tensors assigned into it, so a
large model is not first initialised at random and then overwritten.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Spec
from .network import EoMT, build

__all__ = ["load_network"]


def load_network(spec: Spec, path: str | Path, device: str = "cpu") -> EoMT:
    """Build the network for *spec* and fill it from the checkpoint at *path*."""
    with torch.device("meta"):
        network = build(spec)
    state = torch.load(path, map_location=device, weights_only=True)
    network.load_state_dict(state, strict=True, assign=True)
    return network.eval()
