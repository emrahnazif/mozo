# SPDX-License-Identifier: Apache-2.0
"""RTMO's network: CSPDarknet, the hybrid encoder, the head convolutions and the DCC.

Extracted from MMPose v1.3.2 (``mmpose/models/backbones/csp_darknet.py``,
``models/utils/csp_layer.py``, ``models/necks/hybrid_encoder.py``,
``models/necks/channel_mapper.py``, ``models/utils/reparam_layers.py``,
``models/utils/transformer.py`` and ``models/heads/hybrid_heads/rtmo_head.py``) with the mmcv and
mmengine bricks they are built from -- ``ConvModule``, ``Swish``, ``MultiheadAttention``, ``FFN``,
``Scale`` -- written out in plain torch. Module and attribute names are upstream's, so a published
checkpoint loads with ``strict=True`` and no renaming.

Two activations, and they are not interchangeable. The backbone and the head use mmcv's
``Swish``, which is ``x * torch.sigmoid(x)``; the neck and the GAU use ``nn.SiLU``. They are the
same function and not the same arithmetic: ``F.silu`` is one fused kernel and the product is two
rounded operations.

Three batch-norm epsilons, for the same kind of reason: ``1e-3`` in the backbone and head (their
configs say so), ``1e-5`` in the neck, the projector and the DCC (those configs say nothing, so
torch's default applies).

Left out: the training branches -- ``out_kpt_reg`` (proxy keypoints for the assigner),
``sigma_fc`` (the MLE loss's learned deviation), the assigner, the losses -- and mmdeploy's
``switch_to_deploy`` rewrites, which are different arithmetic from the model the AP was measured on.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import KEYPOINTS, Spec, get_spec
from .postprocess import decode_boxes, flatten, grid_priors

__all__ = ["ConvModule", "DCC", "RTMO", "TRAINING_ONLY"]

#: Checkpoint prefixes no inference path reads, dropped by name before the strict load. Each is
#: checked to be present, so a checkpoint whose layout moved fails rather than loading by luck.
TRAINING_ONLY = ("head.loss_oks.", "head.dcc.sigma_fc.", "head.head_module.out_kpt_reg.")

#: The backbone's stage table (MMPose's ``CSPDarknet.arch_settings['P5']``): in, out, blocks,
#: identity shortcut, SPP.
_P5 = ((64, 128, 3, True, False), (128, 256, 9, True, False),
       (256, 512, 9, True, False), (512, 1024, 3, False, True))


class Swish(nn.Module):
    """mmcv's ``Swish``: ``x * sigmoid(x)``, two operations, deliberately not ``F.silu``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    """mmcv's ``ConvModule`` as RTMO uses it: convolution without bias, batch norm, activation.

    ``act`` is ``"swish"``, ``"silu"`` or ``None``. The attribute names (``conv``, ``bn``,
    ``activate``) are mmcv's, which is what the checkpoint keys are written against.
    """

    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, padding: int = 0,
                 groups: int = 1, *, eps: float, act: str | None):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=eps)
        self.activate = {"swish": Swish(), "silu": nn.SiLU(inplace=True), None: None}[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return x if self.activate is None else self.activate(x)


# --- backbone: CSPDarknet (P5), as YOLOX's ---------------------------------------------------

def _dark(cin: int, cout: int, kernel: int, stride: int = 1, padding: int = 0) -> ConvModule:
    """A backbone convolution: eps 1e-3, Swish."""
    return ConvModule(cin, cout, kernel, stride, padding, eps=1e-3, act="swish")


class Focus(nn.Module):
    """Space-to-depth, then a 3x3 convolution. The concatenation order is upstream's: TL, BL, TR, BR."""

    def __init__(self, cout: int):
        super().__init__()
        self.conv = _dark(3 * 4, cout, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]),
                      dim=1)
        return self.conv(x)


class SPPBottleneck(nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, ...] = (5, 9, 13)):
        super().__init__()
        mid = channels // 2
        self.conv1 = _dark(channels, mid, 1)
        self.poolings = nn.ModuleList(nn.MaxPool2d(k, stride=1, padding=k // 2) for k in kernels)
        self.conv2 = _dark(mid * (len(kernels) + 1), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        return self.conv2(x)


class DarknetBottleneck(nn.Module):
    def __init__(self, channels: int, add_identity: bool):
        super().__init__()
        # ``expansion`` is 1.0 inside a CSPLayer, so hidden == channels.
        self.conv1 = _dark(channels, channels, 1)
        self.conv2 = _dark(channels, channels, 3, 1, 1)
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    def __init__(self, channels: int, blocks: int, add_identity: bool):
        super().__init__()
        mid = int(channels * 0.5)
        self.main_conv = _dark(channels, mid, 1)
        self.short_conv = _dark(channels, mid, 1)
        self.final_conv = _dark(2 * mid, channels, 1)
        self.blocks = nn.Sequential(*[DarknetBottleneck(mid, add_identity) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.short_conv(x)
        main = self.blocks(self.main_conv(x))
        return self.final_conv(torch.cat((main, short), dim=1))


class CSPDarknet(nn.Module):
    """Returns the outputs of stages 2, 3 and 4 (strides 8, 16, 32)."""

    def __init__(self, widen: float, deepen: float):
        super().__init__()
        self.stem = Focus(int(_P5[0][0] * widen))
        for index, (cin, cout, blocks, add_identity, spp) in enumerate(_P5):
            cin, cout = int(cin * widen), int(cout * widen)
            blocks = max(round(blocks * deepen), 1)
            stage = [_dark(cin, cout, 3, 2, 1)]
            if spp:
                stage.append(SPPBottleneck(cout))
            stage.append(CSPLayer(cout, blocks, add_identity))
            self.add_module(f"stage{index + 1}", nn.Sequential(*stage))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stage1(self.stem(x))
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5


# --- neck: the hybrid encoder (RT-DETR's), then a channel mapper ------------------------------

def _neck(cin: int, cout: int, kernel: int, stride: int = 1, padding: int = 0,
          act: str | None = "silu") -> ConvModule:
    """A neck convolution: torch's default eps, ``nn.SiLU``."""
    return ConvModule(cin, cout, kernel, stride, padding, eps=1e-5, act=act)


class RepVGGBlock(nn.Module):
    """Training form, not re-parameterised: ``act(bn(conv3x3(x)) + bn(conv1x1(x)) + 0)``.

    The ``+ 0`` is upstream's: ``without_branch_norm`` is true, so the identity branch is the integer
    zero. It is kept because it is free and removing it is a change nobody has measured.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.branch_3x3 = _neck(channels, channels, 3, 1, 1, act=None)
        self.branch_1x1 = _neck(channels, channels, 1, act=None)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.branch_3x3(x) + self.branch_1x1(x) + 0)


class CSPRepLayer(nn.Module):
    def __init__(self, cin: int, cout: int, blocks: int, widen: float):
        super().__init__()
        hidden = int(cout * widen)
        self.conv1 = _neck(cin, hidden, 1)
        self.conv2 = _neck(cin, hidden, 1)
        self.bottlenecks = nn.Sequential(*[RepVGGBlock(hidden) for _ in range(blocks)])
        self.conv3 = _neck(hidden, cout, 1) if hidden != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class SinePositionalEncoding(nn.Module):
    """MMPose's sine encoding. ``dim_t`` is a checkpoint buffer, so the frequencies are loaded."""

    def __init__(self, channels: int, temperature: float):
        super().__init__()
        half = channels // 2
        self.register_buffer("dim_t", temperature ** (torch.arange(half, dtype=torch.float32) / half))

    def grid(self, h: int, w: int) -> torch.Tensor:
        """The 2-D encoding of an ``h x w`` grid: ``(h*w, channels, 2)``, row then column."""
        grid_h, grid_w = torch.meshgrid(
            torch.arange(int(h), dtype=torch.float32, device=self.dim_t.device),
            torch.arange(int(w), dtype=torch.float32, device=self.dim_t.device), indexing="ij")
        grid_h, grid_w = grid_h.flatten(), grid_w.flatten()
        dim_t = self.dim_t.reshape(1, -1)
        freq_h = grid_h.unsqueeze(-1) / dim_t
        freq_w = grid_w.unsqueeze(-1) / dim_t
        pos_h = torch.cat((freq_h.cos(), freq_h.sin()), dim=-1)
        pos_w = torch.cat((freq_w.cos(), freq_w.sin()), dim=-1)
        return torch.stack((pos_h, pos_w), dim=-1)

    def positions(self, position: torch.Tensor) -> torch.Tensor:
        """The 1-D encoding of arbitrary positions, appended as a last axis."""
        dim_t = self.dim_t.reshape(*((1,) * position.ndim), -1)
        freq = position.unsqueeze(-1) / dim_t
        return torch.cat((freq.cos(), freq.sin()), dim=-1)


class MultiheadAttention(nn.Module):
    """mmcv's wrapper: position added to query and key (not value), residual from the raw query.

    ``nn.MultiheadAttention`` is called the way mmcv calls it -- sequence-first, ``need_weights``
    left at its default -- because that decides which of torch's attention paths runs.
    """

    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dims, heads, 0.0)

    def forward(self, query: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Two tensors on purpose: with ``query is key`` torch takes its packed self-attention path,
        # which is different arithmetic from mmcv's two separate sums.
        q = (query + pos).transpose(0, 1)
        k = (query + pos).transpose(0, 1)
        v = query.transpose(0, 1)
        out = self.attn(query=q, key=k, value=v, attn_mask=None, key_padding_mask=None)[0]
        return query + out.transpose(0, 1)


class FFN(nn.Module):
    def __init__(self, dims: int, hidden: int):
        super().__init__()
        # mmcv's layout, which the checkpoint keys follow: layers.0 = (Linear, GELU, Dropout),
        # layers.1 = Linear, layers.2 = Dropout.
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(dims, hidden), nn.GELU(), nn.Dropout(0.0)),
            nn.Linear(hidden, dims), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class EncoderLayer(nn.Module):
    """``DetrTransformerEncoderLayer``: post-norm attention, then post-norm FFN."""

    def __init__(self, dims: int, heads: int, hidden: int):
        super().__init__()
        self.self_attn = MultiheadAttention(dims, heads)
        self.ffn = FFN(dims, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(dims), nn.LayerNorm(dims)])

    def forward(self, query: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        query = self.norms[0](self.self_attn(query, pos))
        return self.norms[1](self.ffn(query))


class Encoder(nn.Module):
    """``DetrTransformerEncoder`` with its one layer; a container so the key is ``layers.0``."""

    def __init__(self, dims: int, heads: int, hidden: int):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dims, heads, hidden)])

    def forward(self, query: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, pos)
        return query


class ChannelMapper(nn.Module):
    def __init__(self, cin: int, cout: int, count: int):
        super().__init__()
        self.convs = nn.ModuleList(_neck(cin, cout, 1, act=None) for _ in range(count))

    def forward(self, inputs: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(conv(x) for conv, x in zip(self.convs, inputs))


class HybridEncoder(nn.Module):
    """One transformer layer on the stride-32 map, then a top-down FPN and a bottom-up PAN.

    Upstream's defaults that RTMO's configs leave alone, and which are therefore fixed here: the
    encoder runs on input 2 only, with one layer, a 2-D sine position of temperature 10000, and the
    outputs kept are PAN levels 1 and 2 (strides 16 and 32).
    """

    def __init__(self, spec: Spec):
        super().__init__()
        hidden = spec.hidden
        levels = len(spec.neck_in)
        self.hidden = hidden
        self.input_proj = nn.ModuleList(_neck(c, hidden, 1, act=None) for c in spec.neck_in)
        self.encoder = nn.ModuleList([Encoder(hidden, spec.heads, spec.ffn)])
        self.sincos_pos_enc = SinePositionalEncoding(hidden // 2, 10000)
        blocks = round(3 * spec.deepen)
        self.lateral_convs = nn.ModuleList(_neck(hidden, hidden, 1) for _ in range(levels - 1))
        self.fpn_blocks = nn.ModuleList(
            CSPRepLayer(hidden * 2, hidden, blocks, spec.widen) for _ in range(levels - 1))
        self.downsample_convs = nn.ModuleList(
            _neck(hidden, hidden, 3, 2, 1) for _ in range(levels - 1))
        self.pan_blocks = nn.ModuleList(
            CSPRepLayer(hidden * 2, hidden, blocks, spec.widen) for _ in range(levels - 1))
        self.projector = ChannelMapper(hidden, spec.pose_channels, 2)

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        feats = [proj(x) for proj, x in zip(self.input_proj, inputs)]

        h, w = feats[2].shape[2:]
        flat = feats[2].flatten(2).permute(0, 2, 1).contiguous()
        pos = self.sincos_pos_enc.grid(h, w).transpose(-1, -2).reshape(1, h * w, -1)
        memory = self.encoder[0](flat, pos)
        feats[2] = memory.permute(0, 2, 1).contiguous().view([-1, self.hidden, h, w])

        levels = len(feats)
        inner = [feats[-1]]
        for idx in range(levels - 1, 0, -1):
            high = self.lateral_convs[levels - 1 - idx](inner[0])
            inner[0] = high
            up = F.interpolate(high, scale_factor=2., mode="nearest")
            inner.insert(0, self.fpn_blocks[levels - 1 - idx](torch.cat([up, feats[idx - 1]], 1)))

        outs = [inner[0]]
        for idx in range(levels - 1):
            down = self.downsample_convs[idx](outs[-1])
            outs.append(self.pan_blocks[idx](torch.cat([down, inner[idx + 1]], 1)))

        return self.projector([outs[1], outs[2]])


# --- head ----------------------------------------------------------------------------------

def _head(cin: int, cout: int, groups: int = 1) -> ConvModule:
    """A head convolution: 3x3, eps 1e-3, Swish."""
    return ConvModule(cin, cout, 3, 1, 1, groups=groups, eps=1e-3, act="swish")


class RTMOHeadModule(nn.Module):
    """Per level: split the projector's channels in half, a score branch and a pose branch."""

    def __init__(self, spec: Spec):
        super().__init__()
        # Each branch gets half the projector's channels, and the score branch keeps that width.
        cin = cls = int(256 * spec.widen)
        pose = spec.groups * int(spec.widen * spec.channels_per_group)
        levels = len(spec.strides)
        self.conv_cls = nn.ModuleList(
            nn.Sequential(_head(cin, cls), _head(cls, cls)) for _ in range(levels))
        self.out_cls = nn.ModuleList(nn.Conv2d(cls, 1, 1) for _ in range(levels))
        self.conv_pose = nn.ModuleList(
            nn.Sequential(_head(cin, pose), *[_head(pose, pose, spec.groups) for _ in range(3)])
            for _ in range(levels))
        self.out_bbox = nn.ModuleList(nn.Conv2d(pose, 4, 1) for _ in range(levels))
        self.out_kpt_vis = nn.ModuleList(nn.Conv2d(pose, KEYPOINTS, 1) for _ in range(levels))
        self.out_pose = nn.ModuleList(nn.Conv2d(pose, spec.pose_channels, 1) for _ in range(levels))

    def forward(self, feats: tuple[torch.Tensor, ...]) -> tuple[list[torch.Tensor], ...]:
        scores, boxes, visible, poses = [], [], [], []
        for i, x in enumerate(feats):
            cls_feat, reg_feat = x.split(x.size(1) // 2, 1)
            cls_feat = self.conv_cls[i](cls_feat)
            reg_feat = self.conv_pose[i](reg_feat)
            scores.append(self.out_cls[i](cls_feat))
            boxes.append(self.out_bbox[i](reg_feat))
            visible.append(self.out_kpt_vis[i](reg_feat))
            poses.append(self.out_pose[i](reg_feat))
        return scores, boxes, visible, poses


class ScaleNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.scale = dims ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``torch.norm``, as upstream outside ONNX export (it switches to ``linalg.norm`` there).
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class ChannelWiseScale(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class GAUEncoder(nn.Module):
    """A gated attention unit over the 17 joint tokens, additive position, with a scaled shortcut."""

    def __init__(self, dims: int, expansion: int, s: int):
        super().__init__()
        self.s = s
        self.e = int(dims * expansion)
        self.o = nn.Linear(self.e, dims, bias=False)
        self.uv = nn.Linear(dims, 2 * self.e + s, bias=False)
        self.gamma = nn.Parameter(torch.ones(2, s))
        self.beta = nn.Parameter(torch.zeros(2, s))
        self.ln = ScaleNorm(dims)
        self.act_fn = nn.SiLU(True)
        self.res_scale = ChannelWiseScale(dims)
        self.sqrt_s = math.sqrt(s)

    def forward(self, x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.ln(x)
        uv = self.act_fn(self.uv(x))
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)
        dim = base.ndim - self.gamma.ndim + 1
        gamma = self.gamma.view(*((1,) * dim), *self.gamma.size())
        beta = self.beta.view(*((1,) * dim), *self.beta.size())
        base = base.unsqueeze(-2) * gamma + beta
        q, k = torch.unbind(base, dim=-2)
        pos_enc = pos_enc.reshape(*((1,) * (q.ndim - 2)), q.size(-2), q.size(-1))
        q = q + pos_enc
        k = k + pos_enc
        qk = torch.matmul(q, k.transpose(-1, -2))
        kernel = torch.square(F.relu(qk / self.sqrt_s))
        x = u * torch.matmul(kernel, v)
        return self.res_scale(shortcut) + self.o(x)


class DCC(nn.Module):
    """Dynamic coordinate classifier: a pose vector and its box in, 17 joint positions out.

    Each joint gets two 1-D heatmaps whose bins are laid over the box (enlarged 1.25x), encoded by
    the bins' *positions* rather than their indices -- that is the "dynamic" part. The position is
    the softmax-weighted mean of the bin centres, so it is continuous, not an argmax.
    """

    def __init__(self, spec: Spec):
        super().__init__()
        feat = spec.dcc_channels
        self.feat_channels = feat
        self.spe = SinePositionalEncoding(spec.spe_channels, spec.spe_temperature)
        self.gau = GAUEncoder(feat, spec.gau_expansion, spec.gau_s)
        self.pos_enc = nn.Parameter(torch.zeros(KEYPOINTS, spec.gau_s))
        self.pose_to_kpts = nn.Sequential(
            nn.Linear(spec.pose_channels, feat * KEYPOINTS), nn.BatchNorm1d(feat * KEYPOINTS))
        self.x_fc = nn.Linear(spec.spe_channels, feat)
        self.y_fc = nn.Linear(spec.spe_channels, feat)
        self.register_buffer("y_bins", torch.linspace(-0.5, 0.5, spec.bins[1]))
        self.register_buffer("x_bins", torch.linspace(-0.5, 0.5, spec.bins[0]))

    @staticmethod
    def _softmax(hms: torch.Tensor) -> torch.Tensor:
        """Upstream's hand-written softmax: clamp, shift by the max, ``exp / (sum + 1e-8)``."""
        hms = hms.clamp(min=-5e4, max=5e4)
        hms = hms - hms.max(dim=-1, keepdims=True).values
        exp = hms.exp()
        return exp / (exp.sum(dim=-1, keepdims=True) + 1e-8)

    def forward(self, pose_feats: torch.Tensor, bbox_cs: torch.Tensor,
                grids: torch.Tensor) -> torch.Tensor:
        """``(N, C)`` pose vectors, ``(N, 4)`` box centre+size, ``(N, 2)`` cells -> ``(N, 17, 2)``."""
        center, scale = bbox_cs.split(2, dim=-1)
        # Encoded relative to the grid cell...
        local = center - grids
        x_bins = self.x_bins.view(1, -1) * scale[..., 0:1] + local[..., 0:1]
        y_bins = self.y_bins.view(1, -1) * scale[..., 1:2] + local[..., 1:2]
        x_enc = self.x_fc(self.spe.positions(x_bins))
        y_enc = self.y_fc(self.spe.positions(y_bins))

        kpt = self.pose_to_kpts(pose_feats)
        kpt = kpt.reshape(*kpt.shape[:-1], KEYPOINTS, self.feat_channels)
        kpt = self.gau(kpt, self.pos_enc)
        x_hms = self._softmax(torch.matmul(kpt, x_enc.transpose(-1, -2).contiguous()))
        y_hms = self._softmax(torch.matmul(kpt, y_enc.transpose(-1, -2).contiguous()))

        # ...and decoded in absolute input pixels: the same bins, laid over the box centre itself.
        x_bins = self.x_bins.view(1, -1) * scale[..., 0:1] + center[..., 0:1]
        y_bins = self.y_bins.view(1, -1) * scale[..., 1:2] + center[..., 1:2]
        x = (x_hms * x_bins.unsqueeze(1)).sum(dim=-1)
        y = (y_hms * y_bins.unsqueeze(1)).sum(dim=-1)
        return torch.stack((x, y), dim=-1)


class Head(nn.Module):
    """``RTMOHead``'s inference modules. The decode around them lives in :mod:`~.postprocess`."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.head_module = RTMOHeadModule(spec)
        self.dcc = DCC(spec)


class RTMO(nn.Module):
    """The dense half of RTMO: a BGR ``0..255`` float batch in, one row per grid cell out.

    ``forward`` stops before anything data-dependent -- the score filter, the sort, NMS -- so it is
    one static graph, which is what an exported runtime replaces. The DCC runs afterwards, on the
    survivors only (see :class:`~.predictor.Predictor`), as upstream runs it.
    """

    def __init__(self, variant: str):
        super().__init__()
        spec = get_spec(variant)
        self.spec = spec
        self.backbone = CSPDarknet(spec.widen, spec.deepen)
        self.neck = HybridEncoder(spec)
        self.head = Head(spec)

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``(B, 3, S, S)`` -> ``scores (B, N)``, ``boxes (B, N, 4)``, ``visibility (B, N, 17)``,
        ``pose (B, N, C)``, over the ``N`` cells of both levels, stride 16 first.

        Scores and visibilities are sigmoids; boxes are xyxy in input pixels; pose vectors are raw.
        """
        scores, boxes, visible, poses = self.head.head_module(self.neck(self.backbone(batch)))
        priors, strides = grid_priors([s.shape[2:] for s in scores], self.spec.strides,
                                      scores[0].dtype, scores[0].device)
        return (flatten(scores).sigmoid()[..., 0],
                decode_boxes(flatten(boxes), priors, strides),
                flatten(visible).sigmoid(),
                flatten(poses))
