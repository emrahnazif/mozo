# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

# Modified for depth_anything_v3_deploy: ``vitg`` (CC-BY-NC weights only) removed, and with it
# the SwiGLU feed-forward; the import is relative.

from typing import List
import torch.nn as nn

from .vision_transformer import (
    vit_base,
    vit_large,
    vit_small,
)


class DinoV2(nn.Module):
    def __init__(
        self,
        name: str,
        out_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl"}
        self.name = name
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        encoder_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
        }
        encoder_fn = encoder_map[self.name]
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=14,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
        )

    def forward(self, x, **kwargs):
        return self.pretrained.get_intermediate_layers(
            x,
            self.out_layers,
            **kwargs,
        )
