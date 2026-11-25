# SPDX-License-Identifier: Apache-2.0
"""Loading a published checkpoint into the model, strictly.

The checkpoint is Qwen's ``Qwen3VLForConditionalGeneration`` state dict in bfloat16, as published,
with keys ``model.visual.*`` and ``model.language_model.*`` -- this package's module names behind a
``model.`` prefix. There is no ``lm_head.weight``: the output projection is tied to the token
embedding (``tie_word_embeddings``), and :meth:`~.language.TextDecoder.logits` reads that one
tensor. The load is ``strict=True``, which holds :mod:`~.config` against the file.

Loading into float32 is an exact upcast (every bfloat16 value is a float32 value): the same
numbers ``from_pretrained(dtype=torch.float32)`` produces. Only parameters take the dtype; the
rotary frequency buffers stay float32, as upstream computes them in float32 whatever the model's
dtype.

The model is built on the meta device and given empty storage directly on *device*, so no random
initialisation runs (19.5 s for ``2b`` on 8 CPU threads, all of it overwritten by the load) and a
CUDA load never holds a second copy in host memory. The rotary buffers, which the checkpoint does
not carry, are then recomputed on the CPU in float32 -- the expression upstream uses -- and moved.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Spec
from .model import Qwen3VL

__all__ = ["load_model"]

_PREFIX = "model."


def load_model(spec: Spec, path: str | Path, device: str = "cpu",
               dtype: torch.dtype = torch.float32) -> Qwen3VL:
    """Build the model for *spec* in *dtype* and fill it from the checkpoint at *path*."""
    with torch.device("meta"):
        model = Qwen3VL(spec)
    model = model.to(dtype).to_empty(device=device)
    for tower in (model.visual, model.language_model):
        tower.inv_freq = tower.rotary_frequencies().to(device)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    foreign = [key for key in state if not key.startswith(_PREFIX)]
    if foreign:
        raise ValueError(f"unexpected keys outside {_PREFIX!r}: {foreign[:5]}")
    model.load_state_dict({key[len(_PREFIX):]: value for key, value in state.items()}, strict=True)
    del state  # load_state_dict copied every tensor; drop the whole-file mapping
    return model.eval()
