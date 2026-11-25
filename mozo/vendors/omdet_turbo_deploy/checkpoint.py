# SPDX-License-Identifier: Apache-2.0
"""Loading the published checkpoint onto this package's two modules, strictly.

mozo publishes Om AI Lab's ``model.safetensors`` repacked as an ordinary ``.pth``: the same 868
tensors under the same names (``tools/fetch/omdet_turbo.py``). Those names are transformers' --
``vision_backbone.vision_backbone._backbone.layers_0...`` -- so loading is four prefix renames and
a split into the image path and the text tower. Nothing is dropped: every tensor in the file is
read by a module here, and the loads are strict, so a renamed or missing key fails at once.

``load_state_dict`` copies a tensor into a parameter of a different dtype without a word, so the
dtypes are compared first: a checkpoint that is not the published fp32 one is refused rather than
silently cast.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import Spec
from .network import OmDetTurbo, TextTower

__all__ = ["RULES", "build", "split"]

#: Published prefix -> (module, prefix inside it), first match wins.
RULES: tuple[tuple[str, str, str], ...] = (
    ("vision_backbone.vision_backbone._backbone.", "image", "vision.swin."),
    ("vision_backbone.layer_norms.", "image", "vision.norms."),
    ("language_backbone.model.text_model.", "text", ""),
    ("language_backbone.text_projection", "text", "text_projection"),
    ("encoder.", "image", "encoder."),
    ("decoder.", "image", "decoder."),
)


def split(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Rename a published state dict into ``{"image": ..., "text": ...}``.

    Raises:
        KeyError: For a key no rule covers.
    """
    parts: dict[str, dict[str, torch.Tensor]] = {"image": {}, "text": {}}
    for key, tensor in state.items():
        for published, module, ours in RULES:
            if key.startswith(published):
                parts[module][ours + key[len(published):]] = tensor
                break
        else:
            raise KeyError(f"{key!r} is not part of OmDet-Turbo's published checkpoint")
    return parts


def _load(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = module.state_dict()
    wrong = [f"{key}: {tensor.dtype} where {expected[key].dtype} is built"
             for key, tensor in state.items()
             if key in expected and tensor.dtype != expected[key].dtype]
    if wrong:
        raise ValueError("checkpoint dtypes differ from the model's: " + "; ".join(wrong[:5]))
    module.load_state_dict(state, strict=True)


def build(spec: Spec, weights: str | Path,
          device: str | torch.device = "cpu") -> tuple[OmDetTurbo, TextTower]:
    """Build both halves, load them strictly, and put them in eval mode on *device*."""
    parts = split(torch.load(weights, map_location="cpu", weights_only=True))
    image, text = OmDetTurbo(spec), TextTower(spec.text)
    _load(image, parts["image"])
    _load(text, parts["text"])
    return image.eval().to(device), text.eval().to(device)
