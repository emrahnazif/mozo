# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the mayaku authors. Extracted from mayaku 3.1.1 (mayaku/model/blocks.py), reduced
# to inference: quantisation-aware training, weight rescaling and the training-only helpers are gone.
"""Building blocks in their trained form, and the one fusion that turns them into the deploy graph.

The checkpoint stores each block as it trained: QARepVGG multi-branch 3x3s and a BatchNorm after
every convolution. :func:`fuse_tree` collapses them into one biased convolution per branch group,
exactly as upstream's ``Detector.fuse()`` does, so what runs afterwards is the five-op graph:

    Conv    1x1 and 3x3, stride 1 or 2, bias always
    Relu
    Add     same shape
    Resize  2x nearest
    MaxPool 3x3 stride 1

The fusion arithmetic is torch's ``fuse_conv_bn_weights``, called in upstream's order. Changing the
order (folding the post-addition BatchNorm before summing the branches, say) gives the same
function and different bits, and the gate would say so.

Upstream's ``ResidualQuant`` modules are absent. In the fp32 deploy graph they are the identity
(``strip_fake_quant``), and their only state is the int8 range buffers the loader drops by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights

#: The stem takes stride 4 and each of the three stages another factor of two.
STRIDES = (8, 16, 32)


def per_level(x, nl):
    """One width broadcast to every level, or one per level already, as a tuple of length *nl*."""
    t = (x,) * nl if isinstance(x, int) else tuple(x)
    assert len(t) == nl, t
    return t


def up2x(x):
    """The graph's one Resize: 2x nearest-neighbour upsample."""
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Fusible(nn.Module):
    """Owns weights that collapse at load. :meth:`fuse` returns the module that replaces it."""

    def fuse(self):
        raise NotImplementedError


def fuse_tree(module):
    """Replace every :class:`Fusible` beneath *module*, in place. Idempotent."""
    for name, child in list(module.named_children()):
        if isinstance(child, Fusible):
            setattr(module, name, child.fuse())
        else:
            fuse_tree(child)
    return module


def pad_1x1_to_3x3(w):
    return F.pad(w, [1, 1, 1, 1])


def identity_3x3(channels, device, dtype):
    """The 3x3 kernel that reproduces the identity: a 1 on the centre tap."""
    eye = torch.eye(channels, device=device, dtype=dtype)
    return pad_1x1_to_3x3(eye.view(channels, channels, 1, 1))


def conv_from(weight, bias, ref):
    """A copy of convolution *ref* carrying these weights and a bias."""
    c = nn.Conv2d(ref.in_channels, ref.out_channels, ref.kernel_size, ref.stride, ref.padding,
                  ref.dilation, ref.groups, bias=True, device=weight.device, dtype=weight.dtype)
    c.weight = nn.Parameter(weight.detach().clone())
    c.bias = nn.Parameter(bias.detach().clone())
    return c


def fuse_bn_chain(seq):
    """Collapse every conv-BN pair in a Sequential, keeping everything else."""
    out, i = [], 0
    mods = list(seq)
    while i < len(mods):
        if (i + 1 < len(mods) and isinstance(mods[i], nn.Conv2d)
                and isinstance(mods[i + 1], nn.BatchNorm2d)):
            out.append(fuse_conv_bn_eval(mods[i], mods[i + 1]))
            i += 2
        else:
            out.append(mods[i])
            i += 1
    return nn.Sequential(*out)


class ConvBN(Fusible):
    """conv -> BN, folding to conv+bias. No activation."""

    def __init__(self, cin, cout, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))

    def fuse(self):
        return fuse_conv_bn_eval(self.conv, self.bn)


class ConvBNReLU(ConvBN):
    def forward(self, x):
        return F.relu(super().forward(x))

    def fuse(self):
        return nn.Sequential(super().fuse(), nn.ReLU())


class RepConv3x3(Fusible):
    """QARepVGG block (Chu et al. 2022, arXiv 2212.01593): trains as three branches, deploys as one
    biased 3x3. BN on the 3x3 branch, a bare 1x1 branch, a bare identity branch, and one BatchNorm
    after the addition."""

    def __init__(self, cin, cout, s=1, act=True):
        super().__init__()
        self.act = act
        self.dense = ConvBN(cin, cout, 3, s)
        self.one = nn.Conv2d(cin, cout, 1, s, 0, bias=False)
        self.identity = (cin == cout and s == 1)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        y = self.dense(x) + self.one(x)
        if self.identity:
            y = y + x
        y = self.bn(y)
        return F.relu(y) if self.act else y

    def fuse(self):
        conv, bn = self.dense.conv, self.dense.bn
        w, b = fuse_conv_bn_weights(conv.weight, conv.bias, bn.running_mean,
                                    bn.running_var, bn.eps, bn.weight, bn.bias)
        w = w + pad_1x1_to_3x3(self.one.weight)
        if self.identity:
            w = w + identity_3x3(conv.out_channels, w.device, w.dtype)
        # the post-addition BN folds into the summed kernel
        w, b = fuse_conv_bn_weights(w, b, self.bn.running_mean,
                                    self.bn.running_var, self.bn.eps,
                                    self.bn.weight, self.bn.bias)
        fused = conv_from(w, b, conv)
        return nn.Sequential(fused, nn.ReLU()) if self.act else fused


class Bottleneck(nn.Module):
    """1x1 reduce -> 3x3 -> 1x1 expand -> add."""

    def __init__(self, c, expansion=4):
        super().__init__()
        mid = max(8, c // expansion)
        self.reduce = ConvBNReLU(c, mid, 1)
        self.spatial = RepConv3x3(mid, mid, 1, act=True)
        self.expand = ConvBN(mid, c, 1)

    def forward(self, x):
        return F.relu(self.expand(self.spatial(self.reduce(x))) + x)


class Stem(nn.Module):
    """3x3 convolutions to stride 4: two with *mid* (3 -> mid -> cout, both strided), else three."""

    def __init__(self, cout, mid=0):
        super().__init__()
        m = mid or max(8, cout // 2)
        self.c1 = RepConv3x3(3, m, 2)
        self.c2 = nn.Identity() if mid else RepConv3x3(m, m, 1)
        self.c3 = RepConv3x3(m, cout, 2)

    def forward(self, x):
        return self.c3(self.c2(self.c1(x)))


class SPPFAdd(nn.Module):
    """Spatial pyramid pooling with the concatenation written as a sum of per-input 1x1s, and an
    optional global-context term: the pool chain continues until every position holds the maximum
    over the whole map (``max(h, w) - 1`` pools in all), which joins the sum through a
    1x1 -> ReLU -> 1x1 bottleneck. The pool count follows the input, so an exported graph fixes it
    at its canvas."""

    def __init__(self, c, global_ctx=False):
        super().__init__()
        mid = c // 2
        self.pre = ConvBNReLU(c, mid, 1)
        self.pool = nn.MaxPool2d(3, 1, 1)
        self.proj = nn.ModuleList(ConvBN(mid, c, 1) for _ in range(4))
        self.ctx = self.ctx_out = None
        if global_ctx:
            hid = max(8, c // 4)
            self.ctx = nn.Conv2d(mid, hid, 1)
            self.ctx_out = nn.Conv2d(hid, c, 1)

    def forward(self, x):
        t = self.pre(x)
        p2 = self.pool(self.pool(t))
        p4 = self.pool(self.pool(p2))
        p6 = self.pool(self.pool(p4))
        y = self.proj[0](t)
        for proj, scale in zip(self.proj[1:], (p2, p4, p6)):
            y = proj(scale) + y
        if self.ctx is not None:
            # max(h, w) - 1 pools reach every position; p6 has had 6
            g = p6
            for _ in range(max(t.shape[-2:]) - 1 - 6):
                g = self.pool(g)
            y = self.ctx_out(F.relu(self.ctx(g))) + y
        return F.relu(y)


class DenseOut(nn.Module):
    """Dense aggregation of a level's fused tensor: y = ReLU(x + P1(m1) + P2(m2)) with
    m1 = 3x3(reduce(x)), m2 = 3x3(m1), summed in upstream's order."""

    def __init__(self, c):
        super().__init__()
        h = max(8, c // 2)
        self.reduce = ConvBNReLU(c, h, 1)
        self.s1 = RepConv3x3(h, h, 1, act=True)
        self.s2 = RepConv3x3(h, h, 1, act=True)
        self.p1 = ConvBN(h, c, 1)
        self.p2 = ConvBN(h, c, 1)

    def forward(self, x):
        m1 = self.s1(self.reduce(x))
        m2 = self.s2(m1)
        return F.relu(self.p2(m2) + (self.p1(m1) + x))
