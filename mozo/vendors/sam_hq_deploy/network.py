# SPDX-License-Identifier: Apache-2.0
"""The HQ-SAM network: an image encoder, SAM's prompt encoder and HQ-SAM's mask decoder.

Upstream's ``Sam`` module carries a batched ``forward`` for training and evaluation, which the
predictor never calls. This builds the same three sub-networks under the same attribute names --
so the checkpoint loads strictly with no rename table -- and splits them along the seam the
predictor already uses: :meth:`SamHq.encode` is the expensive half and depends only on the image;
:meth:`SamHq.decode` is cheap and depends only on the prompt.

The one thing HQ-SAM adds to that seam is what crosses it. SAM's decoder needs only the final
image embedding; HQ-SAM's also fuses an *early* encoder feature -- the output of the ViT's first
global-attention block, or of TinyViT's stage 1 -- so :meth:`encode` returns both.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from .config import SHARED, TinySpec, VitSpec
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoderHQ
from .prompt_encoder import PromptEncoder
from .tiny_vit import TinyViT
from .transformer import TwoWayTransformer

__all__ = ["SamHq"]


def _image_encoder(spec: VitSpec | TinySpec) -> tuple[nn.Module, int]:
    """Build the trunk a variant names, and say how wide its early feature is."""
    size = SHARED["image_size"]
    if isinstance(spec, TinySpec):
        return TinyViT(
            img_size=size,
            embed_dims=list(spec.embed_dims),
            depths=list(spec.depths),
            num_heads=list(spec.num_heads),
            window_sizes=list(spec.window_sizes),
            mlp_ratio=spec.mlp_ratio,
            mbconv_expand_ratio=spec.mbconv_expand_ratio,
            local_conv_size=spec.local_conv_size,
        ), spec.vit_dim
    return ImageEncoderViT(
        depth=spec.depth,
        embed_dim=spec.embed_dim,
        img_size=size,
        mlp_ratio=SHARED["mlp_ratio"],
        # 1e-6, not torch's default 1e-5: upstream passes it explicitly.
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=spec.num_heads,
        patch_size=SHARED["patch_size"],
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=spec.global_attn_indexes,
        window_size=SHARED["window_size"],
        out_chans=SHARED["prompt_embed_dim"],
    ), spec.embed_dim


class SamHq(nn.Module):
    """HQ-SAM restricted to what single-image prompting runs.

    Args:
        spec: The variant's geometry, from :data:`.config.SPECS`.

    Attributes:
        image_size: Square side the encoder runs at. Prompts are scaled into this space.
    """

    def __init__(self, spec: VitSpec | TinySpec):
        super().__init__()
        self.image_size = SHARED["image_size"]
        dim = SHARED["prompt_embed_dim"]
        embedding = self.image_size // SHARED["patch_size"]

        self.image_encoder, vit_dim = _image_encoder(spec)
        self.prompt_encoder = PromptEncoder(
            embed_dim=dim,
            image_embedding_size=(embedding, embedding),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=dim, mlp_dim=2048, num_heads=8),
            transformer_dim=dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=vit_dim,
        )
        # Not in the checkpoint: upstream registers them non-persistent, from the builder's
        # arguments, which are the same for every variant.
        self.register_buffer("pixel_mean", torch.Tensor(SHARED["pixel_mean"]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(SHARED["pixel_std"]).view(-1, 1, 1), False)

    def encode(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the image half.

        Args:
            image: ``(1, 3, h, w)`` resized RGB, still on the 0..255 scale and unpadded -- the
                output of :func:`.image.resize_longest_side` as a tensor. Normalising and padding
                happen here because upstream does them in ``Sam.preprocess``, on the device.

        Returns:
            ``image_embed`` ``(1, 256, 64, 64)``, ``hq_features`` ``(1, 32, 256, 256)`` -- the
            early encoder feature already fused with the embedding, which depends only on the
            image -- and ``input_size``, the resized image's ``(h, w)`` before padding, which the
            mask resize needs to crop the padding back off.
        """
        h, w = image.shape[-2:]
        x = (image - self.pixel_mean) / self.pixel_std
        x = nn.functional.pad(x, (0, self.image_size - w, 0, self.image_size - h))
        features, interm = self.image_encoder(x)
        return {"image_embed": features,
                "hq_features": self.mask_decoder.hq_features(features, interm),
                "input_size": (h, w)}

    def decode(
        self,
        features: dict[str, torch.Tensor],
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        mask_input: torch.Tensor | None,
        multimask_output: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the prompt half.

        Returns:
            ``(B, C, 256, 256)`` low-resolution logits and ``(B, C)`` predicted IoU, ``C`` three
            with *multimask_output* and one without. Each mask is a SAM token's mask plus the
            HQ token's -- upstream's default ``hq_token_only=False``. With *multimask_output*,
            upstream keeps only the candidate with the highest predicted IoU; all three are kept
            here, and the best of them is exactly upstream's answer.
        """
        sparse, dense = self.prompt_encoder(points=points, boxes=boxes, masks=mask_input)
        masks_sam, masks_hq, iou = self.mask_decoder(
            image_embeddings=features["image_embed"],
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            hq_features=features["hq_features"],
        )
        chosen = slice(1, 4) if multimask_output else slice(0, 1)
        return masks_sam[:, chosen] + masks_hq, iou[:, chosen]
