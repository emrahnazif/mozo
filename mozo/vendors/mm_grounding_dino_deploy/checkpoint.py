# SPDX-License-Identifier: Apache-2.0
"""Load a published checkpoint into :class:`~.network.MMGroundingDino`, strictly.

The published file holds the tensors of ``transformers``' ``model.safetensors``, unchanged, under
their own names (``tools/fetch/mm_grounding_dino.py`` repacks the container, not the contents).
This module maps those names onto this package's modules and loads with ``strict=True``.

Two things about the names are worth knowing:

- **Per-layer heads arrive under two prefixes.** ``transformers`` points the decoder at the same
  ``bbox_embed`` and ``class_embed`` lists the detection head holds, and its writer saves each
  shared tensor once -- so layer 0's box head is stored as ``bbox_embed.0`` and layers 1-5 as
  ``model.decoder.bbox_embed.1``-``5``. They are six *different* heads.
  ``transformers`` 5.x reads that layout as six copies of layer 0 (fixed upstream in October
  2026, after 5.12): it loads, runs, and moves every box. This mapping takes each layer's own.
- **Five scalars are dropped by name**: the score biases of decoder layers 0-4. Upstream computes
  a score at every layer for its training loss; inference reads only the last, so only that one
  is built.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from .config import Spec
from .network import MMGroundingDino

__all__ = ["build", "load_state_dict", "rename"]

#: Prefix -> replacement, first match wins. Everything else under ``model.`` loses that prefix.
_PREFIXES = (
    ("model.backbone.conv_encoder.model.", "backbone."),
    ("model.decoder.bbox_embed.", "bbox_embed."),
    ("model.decoder.class_embed.", "class_embed."),
    ("model.", ""),
)


def load_state_dict(path: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Read a published checkpoint onto *device*: a plain dict of tensors, so ``weights_only``."""
    return torch.load(path, map_location=device, weights_only=True)


def rename(state: dict[str, torch.Tensor], spec: Spec) -> dict[str, torch.Tensor]:
    """Map upstream's tensor names onto this package's, dropping the unused score biases.

    Raises:
        KeyError: If the last decoder layer's score bias is missing.
    """
    renamed: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix, replacement in _PREFIXES:
            if key.startswith(prefix):
                key = replacement + key[len(prefix):]
                break
        renamed[key] = value

    last = spec.dec_layers - 1
    renamed["class_embed.bias"] = renamed.pop(f"class_embed.{last}.bias")
    return {key: value for key, value in renamed.items()
            if not re.fullmatch(r"class_embed\.\d+\.bias", key)}


def build(spec: Spec, weights: str | Path, device: str = "cpu") -> MMGroundingDino:
    """Build the network *spec* describes and load *weights* into it.

    Args:
        spec: Which variant to build. See :data:`~.config.SPECS`.
        weights: Path to the checkpoint.
        device: Where to put the model.

    Raises:
        RuntimeError: If the checkpoint does not fit the architecture.
    """
    # Read straight onto the device and adopted there (``assign=True``), rather than held whole in
    # host memory and copied across.
    state = rename(load_state_dict(weights, device), spec)
    # Built on the meta device, so its parameters are described rather than allocated and
    # initialised, then adopt the loaded tensors (``assign=True``). The load stays strict, so a
    # tensor left on ``meta`` -- which is what a missing key would look like -- is an error.
    with torch.device("meta"):
        model = MMGroundingDino(spec)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().to(device)
