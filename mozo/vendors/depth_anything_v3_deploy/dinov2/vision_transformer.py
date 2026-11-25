# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

# Modified for depth_anything_v3_deploy (see PROVENANCE.md): reduced to one view's forward.
# Removed: reference-view selection and reordering (fires only for three or more views),
# exported auxiliary features, the mask token path, SwiGLU/identity feed-forwards and vit_giant2,
# the logger. ``einops.rearrange`` calls are spelled as the reshapes they perform. Every line of
# arithmetic that remains is upstream's, in upstream's order.

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn

from .layers import Block, Mlp, PatchEmbed, PositionGetter, RotaryPositionEmbedding2D


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1.0,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        cat_token=True,
    ):
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule
        ffn_layer = Mlp

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

    def interpolate_pos_encoding(self, x, w, h, patch_pos_embed=None):
        # Upstream calls this with (H, W) under the names (w, h); kept as upstream has it.
        # depth_anything_v3_deploy: split in two. The resampling of the patch grid, which
        # depends on the image's shape alone, is :meth:`interpolated_patch_pos_embed`; a caller
        # may compute it beforehand and pass it in -- which is how a graph exported from this
        # module takes it, because the resampling's float arithmetic on the shape cannot be
        # traced. Either way the same tensor reaches the same ops.
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        dim = x.shape[-1]
        if patch_pos_embed is None:
            patch_pos_embed = self.interpolated_patch_pos_embed(w, h)
        # upstream: .view(1, -1, dim); reshape is the same values whatever the strides.
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def interpolated_patch_pos_embed(self, w, h):
        """The learned patch position grid resampled for an ``(w, h) = (H, W)`` input."""
        return interpolate_patch_pos_embed(self.pos_embed, w, h, self.patch_size,
                                           self.interpolate_offset, self.interpolate_antialias)

    def prepare_cls_token(self, B, S):
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(self, x, patch_pos_embed=None):
        B, S, nc, w, h = x.shape
        x = x.reshape(B * S, nc, w, h)  # rearrange "b s c h w -> (b s) c h w"
        x = self.patch_embed(x)
        cls_token = self.prepare_cls_token(B, S)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h, patch_pos_embed)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        x = x.reshape(B, S, *x.shape[1:])  # rearrange "(b s) n c -> b s n c"
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=device
            )
            pos = pos.reshape(B, -1, *pos.shape[1:])  # rearrange "(b s) n c -> b s n c"
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
                pos_special = pos_special.reshape(B, -1, *pos_special.shape[1:])
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(self, x, n=1, patch_pos_embed=None):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x, patch_pos_embed)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            if self.alt_start != -1 and i == self.alt_start:
                # No camera condition: the learned reference token for the first view, the
                # learned source token for the rest (none, for one view).
                ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(x, blk, "global", pos=g_pos)
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
        return output

    def process_attention(self, x, block, attn_type="global", pos=None):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = x.reshape(b * s, *x.shape[2:])  # "b s n c -> (b s) n c"
            if pos is not None:
                pos = pos.reshape(b * s, *pos.shape[2:])
        elif attn_type == "global":
            x = x.reshape(b, s * n, *x.shape[3:])  # "b s n c -> b (s n) c"
            if pos is not None:
                pos = pos.reshape(b, s * n, *pos.shape[3:])
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")

        x = block(x, pos=pos)

        if attn_type == "local":
            x = x.reshape(b, s, *x.shape[1:])  # "(b s) n c -> b s n c"
        elif attn_type == "global":
            x = x.reshape(b, s, n, *x.shape[2:])  # "b (s n) c -> b s n c"
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        patch_pos_embed: torch.Tensor | None = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n, patch_pos_embed)
        camera_tokens = [out[0] for out in outputs]
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
        outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]
        return tuple(zip(outputs, camera_tokens))


def interpolate_patch_pos_embed(pos_embed, w, h, patch_size, interpolate_offset=0.1,
                                interpolate_antialias=False):
    """Upstream's resampling of the patch position grid, from ``interpolate_pos_encoding``.

    A function of the learned ``pos_embed`` and the input's ``(w, h) = (H, W)`` alone, so a graph
    runner can compute it from that one tensor without building the transformer.
    """
    pos_embed = pos_embed.float()
    patch_pos_embed = pos_embed[:, 1:]
    N = pos_embed.shape[1] - 1
    dim = pos_embed.shape[-1]
    w0 = w // patch_size
    h0 = h // patch_size
    M = int(math.sqrt(N))  # Recover the number of patches in each dimension
    assert N == M * M
    kwargs = {}
    if interpolate_offset:
        # Historical kludge: add a small number to avoid floating point error in the
        # interpolation, see https://github.com/facebookresearch/dino/issues/8
        # Note: still needed for backward-compatibility, the underlying operators are using
        # both output size and scale factors
        sx = float(w0 + interpolate_offset) / M
        sy = float(h0 + interpolate_offset) / M
        kwargs["scale_factor"] = (sx, sy)
    else:
        # Simply specify an output size instead of a scale factor
        kwargs["size"] = (w0, h0)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        mode="bicubic",
        antialias=interpolate_antialias,
        **kwargs,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    return patch_pos_embed


def vit_small(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, depth=24, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
