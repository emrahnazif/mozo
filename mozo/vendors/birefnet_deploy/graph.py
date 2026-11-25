# ------------------------------------------------------------------------
# BiRefNet -- Bilateral Reference for High-Resolution Dichotomous Image Segmentation
# Copyright (c) 2024 ZhengPeng. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""A deformable convolution an ONNX graph can carry, for exporting :class:`~.network.BiRefNet`.

Written for this package; upstream exports through a third-party converter. The torch runtime
never uses this module.

**Why it exists.** ``torchvision.ops.deform_conv2d`` has no ONNX symbolic, and the ONNX
``DeformConv`` operator it would map to (opset 19) is not implemented by ONNX Runtime (1.21, 1.23)
and reaches TensorRT only through its ``ModulatedDeformConv2d`` plugin -- which ``mozo``'s runner
does not load, and which segfaulted TensorRT 11.5's parser when it was loaded. So the graph spells
the operator out in ops both runtimes have: for each kernel tap, sample the input bilinearly at
the offset positions (``GridSample``), scale by the modulator, and accumulate a 1x1 convolution
with that tap's weights.

**It is not bit-identical to torchvision**, and is not meant to be: the sampling coordinates go
through ``grid_sample``'s normalisation and back, and the sum over taps is ordered differently
from torchvision's im2col matrix product. ``tools/export/birefnet.py`` measures the difference on
real decoder activations, and the graph runtimes are held to measured accuracy rather than
equality. The torch runtime is the exact one.

Per tap, not all taps at once: stacking the 49 taps of the 7x7 branch before one matrix product
would materialise ``C * 49 * H * W`` values -- 822 MB at the last decoder stage of a 1024 input,
3.3 GB at 2048 -- where the loop never holds more than one tap.
"""

from __future__ import annotations

__all__ = ["decomposed", "deform_conv2d"]

import contextlib

import torch
import torch.nn.functional as F

from .blocks import DeformableConv2d


def deform_conv2d(x: torch.Tensor, offset: torch.Tensor, weight: torch.Tensor,
                  bias: torch.Tensor | None, padding: int, mask: torch.Tensor) -> torch.Tensor:
    """Modulated deformable convolution, stride 1, dilation 1, one offset group, one group.

    Args:
        x: ``(N, C, H, W)``.
        offset: ``(N, 2 * kh * kw, H, W)``, torchvision's layout: ``(dy, dx)`` per tap, taps in
            row-major order.
        weight: ``(O, C, kh, kw)``.
        bias: ``(O,)`` or ``None``.
        padding: Symmetric zero padding, as torchvision's ``padding``.
        mask: ``(N, kh * kw, H, W)``.

    Returns:
        ``(N, O, H, W)``: the same shape torchvision's operator returns at stride 1.
    """
    n, _, height, width = x.shape
    kh, kw = (int(s) for s in weight.shape[-2:])  # Python ints under tracing too
    rows = torch.arange(height, dtype=x.dtype, device=x.device).view(1, height, 1)
    cols = torch.arange(width, dtype=x.dtype, device=x.device).view(1, 1, width)
    out = None
    for tap in range(kh * kw):
        a, b = divmod(tap, kw)
        y = rows + (a - padding) + offset[:, 2 * tap]
        x_ = cols + (b - padding) + offset[:, 2 * tap + 1]
        # Pixel coordinates to grid_sample's [-1, 1] under align_corners=True, where -1 and 1 are
        # the centres of the first and last pixels -- torchvision's convention. Out-of-range
        # corners read zero, as torchvision's bilinear_interpolate does.
        grid = torch.stack((x_ * (2.0 / (width - 1)) - 1.0, y * (2.0 / (height - 1)) - 1.0), dim=-1)
        sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        term = F.conv2d(sampled * mask[:, tap:tap + 1], weight[:, :, a:a + 1, b:b + 1])
        out = term if out is None else out + term
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def _deform(self: DeformableConv2d, x: torch.Tensor, offset: torch.Tensor,
            modulator: torch.Tensor) -> torch.Tensor:
    return deform_conv2d(x, offset, self.regular_conv.weight, self.regular_conv.bias, self.padding,
                         modulator)


@contextlib.contextmanager
def decomposed():
    """Within the block, every :class:`~.blocks.DeformableConv2d` uses :func:`deform_conv2d`.

    For an exporter: ``with decomposed(): torch.onnx.export(model, ...)``. The swap is of one
    method on the class and is undone on exit, exceptions included.
    """
    original = DeformableConv2d.deform
    DeformableConv2d.deform = _deform
    try:
        yield
    finally:
        DeformableConv2d.deform = original
