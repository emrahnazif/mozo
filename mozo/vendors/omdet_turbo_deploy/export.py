# SPDX-License-Identifier: Apache-2.0
"""Write the image path as an ONNX graph, for ONNX Runtime and TensorRT.

Not upstream's code, though upstream's idea: the authors' repository exports the same cut -- image
plus pre-computed label and task features in, boxes and logits out -- because the text tower runs
once per vocabulary and has no business in a per-image graph.

**Fixed 640x640 image, any vocabulary size, any task length.** Swin's shift masks are built for
640 (``vision/swin.py``), so the image axis is fixed; the class axis and the task-token axis are
dynamic, so one graph serves every vocabulary.

**Through ``torch.export``**, the exporter torch defaults to since 2.9. Deformable attention reads
its level sizes as Python ints (``decoder.py``) so the graph carries them as constants.

**fp16 is in the graph**, exported under fp16 autocast, so a strongly typed TensorRT engine runs
exactly autocast's precision split.

Needs ``onnx`` and ``onnxscript``, which ``torch.onnx`` imports to write the file.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import warnings
from pathlib import Path

import torch

from .network import OmDetTurbo

__all__ = ["INPUTS", "OUTPUTS", "MAX_CLASSES", "example", "export", "require_exporter"]

INPUTS = ("pixels", "classes", "task")
OUTPUTS = ("boxes", "logits")

#: The largest vocabulary a graph is exported for. ONNX Runtime does not care; a TensorRT
#: optimisation profile needs a ceiling.
MAX_CLASSES = 1024


def example(spec, device: str | torch.device = "cpu", classes: int = 3,
            tokens: int = 9) -> tuple[torch.Tensor, ...]:
    """Inputs of the graph's shapes, in :data:`INPUTS` order: random features, not zeros, so
    cosine scores are defined (a zero vector normalises to zero) and a probe sees real work."""
    generator = torch.Generator().manual_seed(0)
    side = spec.swin.image_size
    return (torch.randn((1, 3, side, side), generator=generator).to(device),
            torch.randn((classes, 1, spec.class_embed_dim), generator=generator).to(device),
            torch.randn((tokens, 1, spec.class_embed_dim), generator=generator).to(device))


def require_exporter() -> None:
    """Raise, naming the pip line, if the packages ``torch.onnx`` writes the file with are missing."""
    missing = [name for name in ("onnx", "onnxscript") if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(f"Exporting to ONNX needs {' and '.join(missing)}: "
                          f"pip install {' '.join(missing)}")


def export(model: OmDetTurbo, path: str | Path, *, fp16: bool) -> None:
    """Write *model*'s image path to *path* as one ONNX file.

    Args:
        model: The loaded image path, on the device to trace on (CUDA for ``fp16``).
        path: Where to write the graph.
        fp16: Trace under fp16 autocast.
    """
    require_exporter()
    device = next(model.parameters()).device
    spec = model.decoder.spec
    classes = torch.export.Dim("classes", min=1, max=MAX_CLASSES)
    tokens = torch.export.Dim("tokens", min=1, max=spec.text.context_length)
    dynamic = (None, {0: classes}, {0: tokens})
    half = torch.autocast(device.type, torch.float16) if fp16 else contextlib.nullcontext()
    previous = logging.root.manager.disable
    try:
        logging.disable(logging.WARNING)
        with (torch.no_grad(), half, warnings.catch_warnings(),
              contextlib.redirect_stdout(io.StringIO())):
            warnings.simplefilter("ignore")
            program = torch.onnx.export(model.eval(), example(spec, device), dynamo=True,
                                        input_names=list(INPUTS), output_names=list(OUTPUTS),
                                        dynamic_shapes=dynamic, opset_version=18)
            # One self-contained file: an artifact is a file, and the weights are ~200 MB.
            program.save(str(path), external_data=False)
    finally:
        logging.disable(previous)
