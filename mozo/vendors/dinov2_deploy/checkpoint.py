# SPDX-License-Identifier: Apache-2.0
"""Building the backbone for a variant and filling it from a published checkpoint, strictly.

The checkpoint is the authors' state dict in their own key layout, so the load is one
``load_state_dict(strict=True)``, and that strictness is what holds
:mod:`~mozo.vendors.dinov2_deploy.config` against the file: a width, a depth, a missing register
or the giant's FFN width written down wrong all fail here. ``mask_token`` is in every checkpoint
and used only in training; the network declares it so nothing is silently skipped.

The network is built on the ``meta`` device and the checkpoint's tensors are assigned into it,
so a 1.1B-parameter giant is not first initialised at random and then overwritten. The tensors
come from a plain ``torch.load``, not ``mmap``, straight onto the device: each one gets an
allocation of its own, as it would in upstream's ``load_state_dict`` copy, and a GPU load never
holds the whole model in host memory first.
"""

from __future__ import annotations

from pathlib import Path

import torch

from . import vision_transformer as vits
from .config import Spec
from .vision_transformer import DinoVisionTransformer

__all__ = ["build", "load_network"]


def build(spec: Spec) -> DinoVisionTransformer:
    """The backbone for *spec*, with the keyword arguments upstream's hub passes, uninitialised."""
    return getattr(vits, spec.arch)(
        img_size=spec.image_size,
        patch_size=spec.patch,
        init_values=1.0,
        ffn_layer=spec.ffn,
        num_register_tokens=spec.registers,
        interpolate_antialias=spec.interpolate_antialias,
        interpolate_offset=spec.interpolate_offset,
    )


def load_network(spec: Spec, path: str | Path, device: str = "cpu") -> DinoVisionTransformer:
    """Build the backbone for *spec* and fill it from the checkpoint at *path*."""
    with torch.device("meta"):
        network = build(spec)
    state = torch.load(path, map_location=device, weights_only=True)
    network.load_state_dict(state, strict=True, assign=True)
    return network.eval()
