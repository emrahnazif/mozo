# SPDX-License-Identifier: MIT
"""Loading a published checkpoint into the model, strictly.

mozo publishes Microsoft's ``model.safetensors`` re-saved with ``torch.save`` (mozo does not depend
on ``safetensors``): the same 666 tensors, the same names, the same fp16 values. The names are this
package's module names, so the load is ``strict=True`` with no renaming -- which holds
:mod:`~.config` against the file. The embedding table is stored once (``language_model.model.shared``):
the encoder's and decoder's token embeddings and the output projection are that one tensor here,
as they are tied upstream.

Loading into float32 is an exact upcast -- what ``from_pretrained(torch_dtype=torch.float32)`` does
-- and loading into float16 keeps the published values. The model is built on the meta device and
given storage directly on *device*, so no random initialisation runs and nothing is copied twice.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Spec
from .network import Florence2

__all__ = ["load_model"]


def load_model(spec: Spec, path: str | Path, device: str = "cpu",
               dtype: torch.dtype = torch.float32) -> Florence2:
    """Build the network for *spec* in *dtype* on *device* and fill it from *path*."""
    with torch.device("meta"):
        model = Florence2(spec)
    model = model.to(dtype).to_empty(device=device)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state  # load_state_dict copied every tensor; drop the whole-file mapping
    return model.eval()
