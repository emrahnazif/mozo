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

# Taken from upstream's model/da3.py (``DepthAnything3Net``) and utils/alignment.py
# (``compute_sky_mask``, ``set_sky_regions_to_max_depth``). Modified for
# depth_anything_v3_deploy (see PROVENANCE.md): single-image depth only. The camera
# encoder/decoder, ray pose, Gaussian-splat head, exported features and the nested metric model
# are not carried. The sky fill draws its 100,000-pixel sample from a generator it is handed
# instead of torch's global RNG; with a generator seeded as the global one was, the draw is the
# same.

from __future__ import annotations

import torch
import torch.nn as nn

from .config import PATCH_SIZE, ModelSpec
from .dinov2.dinov2 import DinoV2
from .dinov2.vision_transformer import interpolate_patch_pos_embed
from .dpt import DPT
from .dualdpt import DualDPT, uv_embeddings

#: Sky head output below which a pixel is not sky (upstream's threshold).
SKY_THRESHOLD = 0.3


def compute_sky_mask(sky_prediction: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Non-sky mask from the sky head: ``True`` where the pixel is *not* sky."""
    return sky_prediction < threshold


def set_sky_regions_to_max_depth(
    depth: torch.Tensor, non_sky_mask: torch.Tensor, max_depth: float | torch.Tensor
) -> torch.Tensor:
    """Fill sky pixels with *max_depth* (upstream's, without the confidence it has no use for)."""
    depth = depth.clone()
    depth[~non_sky_mask] = max_depth
    return depth


def fill_sky(depth: torch.Tensor, sky: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Set the sky to the 99th percentile of the rest of the depth map, as upstream does.

    Upstream's ``DepthAnything3Net._process_mono_sky_estimation``, step for step: nothing changes
    when 10 or fewer pixels are sky or not sky; otherwise the 0.99 quantile of the non-sky depth
    -- over a random sample of 100,000 of those pixels when there are more -- replaces every sky
    pixel. Upstream draws the sample from torch's global generator; here it comes from
    *generator*, which must live on *depth*'s device.

    Args:
        depth: ``[B, S, H, W]`` depth.
        sky: The sky head's ``[B, S, H, W]`` output.
        generator: Where the sample is drawn from.
    """
    non_sky_mask = compute_sky_mask(sky, threshold=SKY_THRESHOLD)
    if non_sky_mask.sum() <= 10:
        return depth
    if (~non_sky_mask).sum() <= 10:
        return depth

    non_sky_depth = depth[non_sky_mask]
    if non_sky_depth.numel() > 100000:
        idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device,
                            generator=generator)
        sampled_depth = non_sky_depth[idx]
    else:
        sampled_depth = non_sky_depth
    non_sky_max = torch.quantile(sampled_depth, 0.99)

    # Set sky regions to maximum depth
    return set_sky_regions_to_max_depth(depth, non_sky_mask, max_depth=non_sky_max)


def shape_inputs(spec: ModelSpec, pos_embed: torch.Tensor, height: int, width: int,
                 device: torch.device | str = "cpu") -> list[torch.Tensor]:
    """The tensors the network derives from an ``height x width`` input's shape alone, in fp32.

    The backbone's position grid resampled for this shape, then -- for the any-view models, whose
    head adds UV embeddings -- those five embeddings. Computed beforehand, they are what a graph
    exported from the network takes beside the pixels: their float arithmetic on the shape cannot
    be traced. Handed to :meth:`DepthAnything3Net.forward`, they give exactly the result of
    letting it compute them.

    Args:
        spec: The variant's specification.
        pos_embed: The learned ``backbone.pretrained.pos_embed``, ``1 x 1370 x C``.
        height, width: The network input's size, multiples of 14.
        device: Where to put the tensors.
    """
    # On *device*, as the network computes them on its own: sin, cos and bicubic kernels differ
    # in the last bit between devices.
    tensors = [interpolate_patch_pos_embed(pos_embed.to(device), height, width, PATCH_SIZE)]
    if not spec.sky:
        tensors += uv_embeddings(height, width, spec.out_channels, spec.features, PATCH_SIZE,
                                 dtype=torch.float32, device=device)
    # Contiguous, so each is a plain tensor rather than a view of a differently-shaped one,
    # which is what a graph input has to be. The values are the same either way.
    return [t.contiguous() for t in tensors]


class DepthAnything3Net(nn.Module):
    """Backbone and depth head of one Depth Anything 3 checkpoint, for one view.

    :meth:`forward` is the network: it returns the head's raw outputs, ``depth`` and, for a
    model with a sky head, ``sky``. That is the part a graph runtime can execute.
    :func:`fill_sky` is the post-process upstream applies after it.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.backbone = DinoV2(
            name=spec.encoder,
            out_layers=list(spec.out_layers),
            alt_start=spec.alt_start,
            qknorm_start=spec.qknorm_start,
            rope_start=spec.rope_start,
            cat_token=spec.cat_token,
        )
        if spec.sky:
            self.head = DPT(
                dim_in=spec.head_dim_in,
                output_dim=1,
                features=spec.features,
                out_channels=list(spec.out_channels),
            )
        else:
            self.head = DualDPT(
                dim_in=spec.head_dim_in,
                output_dim=2,
                features=spec.features,
                out_channels=list(spec.out_channels),
            )

    def shape_inputs(self, height: int, width: int,
                     device: torch.device | str = "cpu") -> list[torch.Tensor]:
        """:func:`shape_inputs` for this network's spec and learned position embedding."""
        return shape_inputs(self.spec, self.backbone.pretrained.pos_embed, height, width, device)

    def forward(self, x: torch.Tensor,
                shape_inputs: list[torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Run the network on ``x`` of shape ``[B, 1, 3, H, W]``.

        Args:
            x: The normalised image.
            shape_inputs: :meth:`shape_inputs` for this ``H, W``, or ``None`` to compute them.

        Returns:
            ``depth`` ``[B, 1, H, W]``; ``sky`` too for a model with a sky head; ``depth_conf``
            for one without.
        """
        if x.shape[1] != 1:
            # Upstream reorders three or more views around a reference view before the global
            # layers; that path is not carried, so more than one view is refused.
            raise ValueError(f"one view per image; got {x.shape[1]}")
        grid, uv = (None, None) if shape_inputs is None else (shape_inputs[0], shape_inputs[1:])
        feats = self.backbone(x, patch_pos_embed=grid)
        H, W = x.shape[-2], x.shape[-1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            if isinstance(self.head, DualDPT):
                return self.head(feats, H, W, patch_start_idx=0, uv=uv)
            return self.head(feats, H, W, patch_start_idx=0)
