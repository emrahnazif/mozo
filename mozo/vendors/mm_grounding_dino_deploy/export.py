# SPDX-License-Identifier: Apache-2.0
"""Write the network as an ONNX graph, for a compiled runtime to build from.

Not upstream's: upstream ships no graph. This exists because a TensorRT engine is built from a
graph, and this model's input size is the user's choice (:func:`~.image.fit_square`), so the
graph cannot be published ahead of time -- it is written on the user's machine, once per size.

**One square input size, any caption length.** The image is fitted into a fixed square, so
height and width are fixed; the caption's token count is left dynamic, up to the model's
256-token budget, so one graph serves every prompt. The padding mask is an input, as
:meth:`~.network.MMGroundingDino.forward` takes it.

**Through ``torch.export``**, the exporter torch has defaulted to since 2.9. The TorchScript
exporter before it is deprecated and slated for removal, and it baked shape arithmetic into the
graph as constants -- correct at the traced size and wrong at any other. Exporting this way is
what required the level sizes to travel as Python integers (``transformer.py``) rather than be
read back off a tensor, which ``torch.export`` cannot follow.

**fp16 is in the graph, not in the runtime.** Exported under fp16 autocast, the graph carries the
casts torch inserts -- the same precision split as the ``torch-fp16`` runtime -- and a strongly
typed engine (TensorRT 11 builds no other kind) executes exactly those types.

Needs ``onnx`` and ``onnxscript`` installed, which ``torch.onnx`` imports to write the file.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import warnings
from pathlib import Path

import torch

from .network import MMGroundingDino

__all__ = ["INPUTS", "OUTPUTS", "example", "export", "profile", "require_exporter"]

#: Graph input names, in :meth:`~.network.MMGroundingDino.forward`'s order.
INPUTS = ("image", "input_ids", "token_type_ids", "token_mask", "attention", "position_ids",
          "pixel_mask")
OUTPUTS = ("logits", "boxes")

#: A caption to trace with. Its length does not matter -- the token axis is dynamic -- only that it
#: is real text, so the trace sees the masks it will see in use.
_EXAMPLE_TOKENS = 16


def example(size: int, device: str | torch.device = "cpu",
            tokens: int = _EXAMPLE_TOKENS) -> tuple[torch.Tensor, ...]:
    """Inputs of the graph's shapes and kinds, in :data:`INPUTS` order.

    A picture filling half a *size* square, so padding is present, and a real caption's tokens,
    so no mask hides everything. What :func:`export` traces with, and a fair probe of an engine
    built from the graph: all-zero masks would make every attention empty, and NaN correct.
    """
    image = torch.zeros((1, 3, size, size), device=device)
    pixel_mask = torch.zeros((1, size, size), dtype=torch.bool, device=device)
    pixel_mask[:, :, size // 2:] = True                                    # a portrait picture
    ids = torch.full((1, tokens), 1012, dtype=torch.long, device=device)  # "." throughout
    ids[0, 0], ids[0, -1] = 101, 102                                       # [CLS] ... [SEP]
    return (image, ids, torch.zeros_like(ids), torch.ones_like(ids, dtype=torch.bool),
            torch.ones((1, tokens, tokens), dtype=torch.bool, device=device),
            torch.arange(tokens, device=device)[None], pixel_mask)


def profile(preferred: int, longest: int) -> dict[str, tuple[tuple[int, ...], ...]]:
    """The ``(min, opt, max)`` shapes of the token inputs, for a TensorRT optimisation profile.

    Every input but the image and its mask is per token; ``attention`` is tokens by tokens.
    """
    ranges = {name: ((1, 2), (1, preferred), (1, longest)) for name in INPUTS[1:6]}
    ranges["attention"] = tuple((1, n, n) for n in (2, preferred, longest))
    return ranges


def require_exporter() -> None:
    """Raise, naming the pip line, if the packages ``torch.onnx`` writes the file with are missing.

    Cheap, so a caller can check before spending a minute loading the model it would export.
    """
    missing = [name for name in ("onnx", "onnxscript") if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(f"Exporting to ONNX needs {' and '.join(missing)}: "
                          f"pip install {' '.join(missing)}")


def export(model: MMGroundingDino, path: str | Path, size: int, *, fp16: bool) -> None:
    """Write *model* to *path* as an ONNX graph taking a *size* square and any caption.

    Args:
        model: The loaded network, on the device it should be traced on.
        path: Where to write the graph. The exporter may write its weights beside it, under the
            same name with a suffix.
        size: Side of the square.
        fp16: Trace under fp16 autocast, so the graph carries ``torch-fp16``'s precision.

    Quiet: the exporter narrates each step on stdout, raises warnings about its own internals, and
    its optimiser logs a warning for every op it declines to fold -- dozens of lines nobody can act
    on, printed even where logging was never configured. ``logging.disable`` is process-wide, so
    for the export's minute other threads' warnings are held back too; errors still pass.
    """
    require_exporter()
    device = next(model.parameters()).device
    text = torch.export.Dim("tokens", min=2, max=model.spec.max_text_len)
    dynamic = (None, {1: text}, {1: text}, {1: text}, {1: text, 2: text}, {1: text}, None)
    half = torch.autocast(device.type, torch.float16) if fp16 else contextlib.nullcontext()
    previous = logging.root.manager.disable
    try:
        logging.disable(logging.WARNING)
        with (torch.no_grad(), half, warnings.catch_warnings(),
              contextlib.redirect_stdout(io.StringIO())):
            warnings.simplefilter("ignore")
            # Optimised, which is not optional here: the optimiser is what lowers ``split`` from
            # the sequence ops ``torch.export`` emits, which TensorRT does not parse, to Split.
            torch.onnx.export(model.eval(), example(size, device), str(path), dynamo=True,
                              input_names=list(INPUTS), output_names=list(OUTPUTS),
                              dynamic_shapes=dynamic, opset_version=18)
    finally:
        logging.disable(previous)
