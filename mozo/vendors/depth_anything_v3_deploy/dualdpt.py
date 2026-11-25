# flake8: noqa E501
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for depth_anything_v3_deploy (see PROVENANCE.md): the depth branch only. Upstream's
# DualDPT runs a second, fully independent fusion chain and head for camera rays
# (``refinenet*_aux``, ``output_conv1_aux``, ``output_conv2_aux``); no depth value reads from it,
# so it is not built and its tensors are not published. One view's forward (no chunk loop),
# plain dicts instead of ``addict``. The depth branch is upstream's, line for line.

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn as nn

from .dpt import _make_fusion_block, _make_scratch
from .head_utils import create_uv_grid, custom_interpolate, position_grid_to_embed

#: The scale of the UV embedding added to the head's features (upstream's ``ratio=0.1``).
UV_RATIO = 0.1


class DualDPT(nn.Module):
    """
    The depth branch of upstream's dual-head DPT: depth and its confidence.
    """

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        head_name: str = "depth",
    ) -> None:
        super().__init__()

        # -------------------- configuration --------------------
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio

        self.head_main = head_name
        self.out_channels = tuple(out_channels)
        self.features = features

        # Always expect 4 scales; enforce intermediate idx = (0, 1, 2, 3)
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        # -------------------- token pre-norm + per-stage projection --------------------
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # -------------------- spatial re-sizers (align to common scale before fusion) --------------------
        # design: stage strides (x4, x2, x1, /2) relative to patch grid to align to a common pivot scale
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        # -------------------- scratch: stage adapters + fusion --------------------
        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain (independent)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # Primary head neck + head (independent)
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

    # -------------------------------------------------------------------------
    # Public forward. Upstream chunks along the view axis; one view is one chunk.
    # -------------------------------------------------------------------------

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        uv: List[torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns ``{depth: [B, S, H, W], depth_conf: [B, S, H, W]}``. *uv* is
        :func:`uv_embeddings` for this ``H, W``, computed beforehand, or ``None`` to compute it.
        """
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        out_dict = self._forward_impl(feats, H, W, patch_start_idx, uv)
        return {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}

    # -------------------------------------------------------------------------
    # Internal forward (single chunk)
    # -------------------------------------------------------------------------

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        uv: List[torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if uv is None:
            uv = [None] * 5
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H, pe=uv[stage_idx])
            x = self.resize_layers[stage_idx](x)  # align scales
            resized_feats.append(x)

        # 2) Fuse pyramid
        fused_main = self._fuse(resized_feats)

        # 3) Upsample to target resolution and (optional) add pos-embed again
        # depth_anything_v3_deploy: upstream's int(ph * patch_size / down_ratio), as integer
        # division -- the same number for every down_ratio that divides the patch size (it is
        # 1), and an int(float) would freeze the output size into a traced graph.
        h_out = ph * self.patch_size // self.down_ratio
        w_out = pw * self.patch_size // self.down_ratio

        fused_main = custom_interpolate(
            fused_main, (h_out, w_out), mode="bilinear", align_corners=True
        )
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H, pe=uv[4])

        # Primary head: conv2 -> activate (output_conv1 was applied inside _fuse)
        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)
        return {
            self.head_main: main_pred.squeeze(-1),
            f"{self.head_main}_conf": main_conf,
        }

    # -------------------------------------------------------------------------
    # Subroutines
    # -------------------------------------------------------------------------

    def _fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Feature pyramid fusion, main chain. Returns the finest scale after ``output_conv1``.
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # level 4 -> 3
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        # level 3 -> 2
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        # level 2 -> 1
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        # level 1 (final)
        out = self.scratch.refinenet1(out, l1_rn)

        out = self.scratch.output_conv1(out)
        return out

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float | None = None,
                       pe: torch.Tensor | None = None) -> torch.Tensor:
        """Simple UV positional embedding added to feature maps."""
        # depth_anything_v3_deploy: the embedding depends on the shapes alone, so it can be
        # computed beforehand (:func:`uv_embedding`) and passed in, as a graph exported from this
        # module takes it -- its float arithmetic on the shape cannot be traced.
        if pe is None:
            pe = uv_embedding(x.shape[-1], x.shape[-2], x.shape[1], W, H, x.dtype, x.device, ratio)
        pe = pe.expand(x.shape[0], -1, -1, -1)
        return x + pe

    def _apply_activation_single(
        self, x: torch.Tensor, activation: str = "linear"
    ) -> torch.Tensor:
        """
        Apply activation to single channel output, maintaining semantic consistency with value branch in multi-channel case.
        Supports: exp / relu / sigmoid / softplus / tanh / linear / expp1
        """
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        # Default linear
        return x


def uv_embedding(pw: int, ph: int, channels: int, W: int, H: int, dtype: torch.dtype,
                 device: torch.device | str, ratio: float | None = None) -> torch.Tensor:
    """Upstream's UV embedding (``_add_pos_embed``) for a ``channels x ph x pw`` map of an
    ``H x W`` image, before it is broadcast over the batch. *ratio* defaults to :data:`UV_RATIO`."""
    ratio = UV_RATIO if ratio is None else ratio
    pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=dtype, device=device)
    pe = position_grid_to_embed(pe, channels) * ratio
    return pe.permute(2, 0, 1)[None]


def uv_embeddings(H: int, W: int, out_channels: Sequence[int], features: int,
                  patch_size: int = 14, down_ratio: int = 1, dtype: torch.dtype = torch.float32,
                  device: torch.device | str = "cpu") -> List[torch.Tensor]:
    """The five UV embeddings a :class:`DualDPT` adds for an ``H x W`` input, in order."""
    ph, pw = H // patch_size, W // patch_size
    h_out = ph * patch_size // down_ratio
    w_out = pw * patch_size // down_ratio
    return ([uv_embedding(pw, ph, oc, W, H, dtype, device) for oc in out_channels]
            + [uv_embedding(w_out, h_out, features // 2, W, H, dtype, device)])
