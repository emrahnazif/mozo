# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the mayaku authors. Extracted from mayaku 3.1.1 (mayaku/model/{tiers,backbone,
# neck,head,aux,detector}.py), reduced to the street models' inference path: no keypoints, no
# oriented boxes, no "tower" auxiliary arm, no stride-8 mask path, no quantisation-aware training.
"""The street detector: backbone, Feature Pyramid Grid neck, decoupled head, and the mask branch.

``forward`` returns upstream's flat list of raw maps, in its order: ``cls0, box0, cls1, box1, cls2,
box2`` and, for a segmentation checkpoint, ``mask, ker0, ker1, ker2``. Nothing after the graph --
sigmoid, DFL expectation, NMS, the per-instance mask network -- is in here; that is
:mod:`.decode` and :mod:`.mask`, which every runtime shares.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (STRIDES, Bottleneck, ConvBN, ConvBNReLU, DenseOut, Fusible, RepConv3x3,
                     SPPFAdd, Stem, fuse_bn_chain, fuse_tree, per_level, up2x)

#: Channels of the mask feature, and the per-instance network the host runs over it: three 1x1
#: layers (8 + 2 coordinate channels) -> 8 -> 8 -> 1, whose 169 weights each anchor emits.
MASK_CH = 8
MASK4_CH = 16
KERNEL_LAYOUT = ((MASK_CH + 2, 8), (8, 8), (8, 1))
KERNEL_PARAMS = sum(i * o + o for i, o in KERNEL_LAYOUT)   # 169


@dataclasses.dataclass(frozen=True)
class Tier:
    """One architecture, written out rather than inferred: the strict load holds it against the
    checkpoint. Only the fields the street models set are kept; the rest are upstream's defaults
    and are checked against the checkpoint's own record (:func:`check_sidecar`)."""

    stem: int
    stem_mid: int
    s4_depth: int
    width: tuple
    depth: tuple
    neck: tuple
    head_width: int
    paths: int = 3
    head_conv: int = 2
    reg_max: int = 16
    expansion: int = 2
    global_ctx: bool = True
    aux_width: int = 64
    mask_stride: int = 4


#: Upstream's ``TIERS["n"]`` and ``TIERS["m"]``, the two sizes the street models come in.
TIERS = {
    "n": Tier(stem=32, stem_mid=16, s4_depth=2, width=(96, 192, 320), depth=(3, 3, 2),
              neck=(64, 128, 256), head_width=80),
    "m": Tier(stem=64, stem_mid=32, s4_depth=2, width=(288, 448, 768), depth=(4, 4, 3),
              neck=(192, 320, 576), head_width=192),
}


class Backbone(nn.Module):
    """A stride-4 stem and stage, then three bottleneck stages at strides 8 / 16 / 32; the coarsest
    passes through the spatial pyramid pooling. Returns the stride-4 feature too, which the
    stride-4 mask path reads."""

    def __init__(self, cfg: Tier):
        super().__init__()
        self.stem = Stem(cfg.stem, cfg.stem_mid)
        self.s4 = nn.Sequential(*[Bottleneck(cfg.stem, cfg.expansion)
                                  for _ in range(cfg.s4_depth)])
        chs = (cfg.stem, *tuple(cfg.width))
        self.down, self.stages = nn.ModuleList(), nn.ModuleList()
        for i, n in enumerate(cfg.depth):
            self.down.append(RepConv3x3(chs[i], chs[i + 1], 2))
            self.stages.append(nn.Sequential(
                *[Bottleneck(chs[i + 1], cfg.expansion) for _ in range(n)]))
        self.sppf = SPPFAdd(cfg.width[-1], cfg.global_ctx)

    def forward(self, x):
        x = stem = self.s4(self.stem(x))
        outs = []
        for down, stage in zip(self.down, self.stages):
            x = stage(down(x))
            outs.append(x)
        outs[-1] = self.sppf(outs[-1])
        return stem, outs


class FPGStage(nn.Module):
    """One neck pathway over levels fine to coarse. Each level's sum is built one term at a time
    in upstream's order -- same, then up, then down, then skip -- because float addition does not
    associate."""

    def __init__(self, nl, ch, skip, down):
        super().__init__()
        self.nl = nl
        c = per_level(ch, nl)
        self.same = nn.ModuleList(ConvBN(c[i], c[i], 1) for i in range(nl))
        self.up = nn.ModuleList(ConvBN(c[i + 1], c[i], 1) for i in range(nl - 1))
        self.down = nn.ModuleList(
            ConvBN(c[i], c[i + 1], 1, 2) for i in range(nl - 1)) if down else None
        self.skip = nn.ModuleList(ConvBN(c[i], c[i], 1) for i in range(nl)) if skip else None
        self.proc = nn.ModuleList(Bottleneck(c[i], 2) for i in range(nl))

    def forward(self, xs, prev=None):
        out = []
        for i in range(self.nl):
            y = self.same[i](xs[i])
            if i < self.nl - 1:
                y = y + up2x(self.up[i](xs[i + 1]))
            if self.down is not None and i > 0:
                y = self.down[i - 1](xs[i - 1]) + y
            if self.skip is not None and prev is not None:
                y = self.skip[i](prev[i]) + y
            out.append(F.relu(y))
        return [p(o) for p, o in zip(self.proc, out)]


class FPG(nn.Module):
    """Feature Pyramid Grid (Chen et al. 2020, arXiv 2004.03580), fused by add only: lateral 1x1s,
    ``n_path`` pathways (skips from the third on, bottom-up flow on the last), a dense output block
    per level."""

    def __init__(self, cin, ch, n_path):
        super().__init__()
        nl = len(STRIDES)
        c = per_level(ch, nl)
        self.lateral = nn.ModuleList(ConvBNReLU(a, b, 1) for a, b in zip(cin, c))
        self.paths = nn.ModuleList(
            FPGStage(nl, c, skip=(i >= 2), down=(i == n_path - 1)) for i in range(n_path))
        self.out = nn.ModuleList(DenseOut(c[i]) for i in range(nl))

    def forward(self, xs):
        xs = [lat(x) for lat, x in zip(self.lateral, xs)]
        history = [xs]
        for path in self.paths:
            prev = history[-2] if len(history) >= 2 else None
            xs = path(xs, prev)
            history.append(xs)
        return [o(x) for o, x in zip(self.out, xs)]


class Head(Fusible):
    """Three scales, a class tower and a box tower per scale, raw logits out: ``nc`` class logits
    and ``4 * reg_max`` DFL bin logits per cell."""

    def __init__(self, ch, nc, reg_max=16, n_conv=2, width=0):
        super().__init__()
        nl = len(STRIDES)
        chs = per_level(ch, nl)
        ws = tuple(w or c for w, c in zip(per_level(width, nl), chs))
        self.widths = ws

        def tower():
            return nn.ModuleList(
                nn.Sequential(*[m for i in range(n_conv)
                                for m in (nn.Conv2d(chs[lv] if i == 0 else ws[lv], ws[lv],
                                                    3, 1, 1, bias=False),
                                          nn.BatchNorm2d(ws[lv]), nn.ReLU())])
                for lv in range(nl))

        # upstream's name for the class tower, kept for the checkpoint keys
        self.tower = tower()
        self.box_tower = tower()
        self.cls = nn.ModuleList(nn.Conv2d(ws[lv], nc, 1) for lv in range(nl))
        self.box = nn.ModuleList(nn.Conv2d(ws[lv], 4 * reg_max, 1) for lv in range(nl))

    def towers(self, xs):
        """Per level, the (class, box) tower outputs; the mask branch reads the box tower's."""
        return [(self.tower[i](x), self.box_tower[i](x)) for i, x in enumerate(xs)]

    def outputs(self, cb):
        out = []
        for i, (c, b) in enumerate(cb):
            out.append(self.cls[i](c))
            out.append(self.box[i](b))
        return out

    def fuse(self):
        self.tower = nn.ModuleList(fuse_bn_chain(t) for t in self.tower)
        self.box_tower = nn.ModuleList(fuse_bn_chain(t) for t in self.box_tower)
        return self


class MaskBranch(nn.Module):
    """Upstream's ``AuxBranch`` with masks only, at stride 4, reading the box towers.

    A stride-8 trunk (P3 through a 3x3, P4 and P5 through 1x1s then 2x nearest, summed, one ReLU),
    then the stride-4 mask feature U-Net style: the trunk through a 1x1 and 2x nearest plus a 1x1 of
    the backbone's stride-4 output, one ReLU, a 3x3 readout to 8 channels. Per anchor, a 1x1 on the
    box tower emits the 169 weights of that instance's mask network. Kept under upstream's name
    ``aux`` on the detector, so the checkpoint keys load as they are."""

    def __init__(self, cin, head_in, width, stem):
        super().__init__()
        c3, c4, c5 = cin
        self.p3 = ConvBN(c3, width, 3)
        self.p4 = ConvBN(c4, width, 1)
        self.p5 = ConvBN(c5, width, 1)
        self.mask_up = ConvBN(width, MASK4_CH, 1)
        self.mask_skip = ConvBN(stem, MASK4_CH, 1)
        self.mask4 = nn.Conv2d(MASK4_CH, MASK_CH, 3, padding=1)
        self.ker = nn.ModuleList(nn.Conv2d(c, KERNEL_PARAMS, 1) for c in head_in)

    def forward(self, feats, boxes, stem):
        r = up2x(self.p4(feats[1])) + up2x(up2x(self.p5(feats[2])))
        f = F.relu(self.p3(feats[0]) + r)
        m = self.mask4(F.relu(self.mask_skip(stem) + up2x(self.mask_up(f))))
        return [m] + [k(x) for k, x in zip(self.ker, boxes)]


def output_names(seg: bool) -> list[str]:
    """The graph's outputs in order, upstream's names: what an exported graph calls them too."""
    names = [f"{kind}{i}" for i in range(len(STRIDES)) for kind in ("cls", "box")]
    return names + (["mask"] + [f"ker{i}" for i in range(len(STRIDES))] if seg else [])


class StreetNetwork(nn.Module):
    """The whole graph. Built in its trained form so the checkpoint loads strictly, then fused in
    place by :meth:`fuse`, after which it is the deploy graph upstream's ``for_deploy`` runs."""

    def __init__(self, cfg: Tier, nc: int, seg: bool):
        super().__init__()
        self.seg = seg
        self.backbone = Backbone(cfg)
        self.neck = FPG(cfg.width, cfg.neck, cfg.paths)
        self.head = Head(cfg.neck, nc, reg_max=cfg.reg_max, n_conv=cfg.head_conv,
                         width=cfg.head_width)
        self.aux = (MaskBranch(per_level(cfg.neck, len(STRIDES)), self.head.widths,
                               cfg.aux_width, cfg.stem) if seg else None)

    def forward(self, x):
        stem, feats = self.backbone(x)
        feats = self.neck(feats)
        cb = self.head.towers(feats)
        out = self.head.outputs(cb)
        if self.aux is not None:
            out = out + self.aux(feats, [b for _, b in cb], stem)
        return out

    def fuse(self):
        return fuse_tree(self.eval())


#: State-dict keys present in every street checkpoint and read by nothing here: the int8 range
#: observers of quantisation-aware training (``act_fq.mn`` / ``act_fq.mx`` on every convolution
#: and residual add). The fp32 deploy graph never reads them -- upstream's ``strip_fake_quant``
#: turns every observer off -- so they are dropped by name and everything else loads strictly.
QAT_RANGE_SUFFIXES = (".act_fq.mn", ".act_fq.mx")


def read_checkpoint(path: str | os.PathLike) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """``(state_dict, sidecar)`` from a Mayaku checkpoint. ``weights_only``: the sidecar is JSON."""
    obj = torch.load(os.fspath(path), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "model" not in obj or not isinstance(obj.get("mayaku"), dict):
        raise ValueError(f"{path} is not a Mayaku checkpoint: no 'model' state and 'mayaku' sidecar")
    return obj["model"], obj["mayaku"]


def build_network(state: dict[str, torch.Tensor], cfg: Tier, nc: int, seg: bool) -> StreetNetwork:
    """Build, load strictly (the QAT range buffers dropped by name), and fuse."""
    network = StreetNetwork(cfg, nc, seg)
    weights = {k: v for k, v in state.items() if not k.endswith(QAT_RANGE_SUFFIXES)}
    network.load_state_dict(weights, strict=True)
    return network.fuse()
