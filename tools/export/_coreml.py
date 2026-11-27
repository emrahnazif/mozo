"""Converter fixes coremltools 9 needs for mozo's traced graphs. Bootstrap tooling; never ships.

Importing this module registers them. Each replaces a coremltools converter that fails on a
pattern torch 2.11's tracer emits, with one that produces the same arithmetic:

- ``int`` on a shape value: coremltools hands the converter a one-element array where it expects
  a scalar (``TypeError: only 0-dimensional arrays can be converted to Python scalars``).
- ``linear`` with a 1-D weight: ``F.linear(x, w)`` is then a dot product over the last axis, which
  coremltools refuses (it asserts a 2-D weight). D-FINE's box integral is written this way.
"""

from __future__ import annotations

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    _TORCH_OPS_REGISTRY,
    register_torch_op,
)

_linear = _TORCH_OPS_REGISTRY.get_func("linear")


@register_torch_op(torch_alias=["int"], override=True)
def _int(context, node):
    x = _get_inputs(context, node, expected=1)[0]
    if x.val is not None:
        context.add(mb.const(val=np.int32(np.asarray(x.val).reshape(-1)[0]), name=node.name))
        return
    if x.rank > 0:
        x = mb.reshape(x=x, shape=[])
    context.add(mb.cast(x=x, dtype="int32", name=node.name))


@register_torch_op(torch_alias=["linear"], override=True)
def _linear_any_weight(context, node):
    inputs = _get_inputs(context, node, min_expected=2)
    x, weight = inputs[0], inputs[1]
    if weight.rank != 1:
        return _linear(context, node)
    y = mb.squeeze(x=mb.matmul(x=x, y=mb.reshape(x=weight, shape=[-1, 1])), axes=[-1])
    bias = inputs[2] if len(inputs) > 2 else None
    if bias is not None and bias.val is not None:
        y = mb.add(x=y, y=bias)
    context.add(y, torch_name=node.name)
