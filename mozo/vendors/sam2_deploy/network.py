# SPDX-License-Identifier: Apache-2.0
"""The SAM 2 network: an image encoder, a prompt encoder, a mask decoder, and the video tracker.

Upstream's ``SAM2Base`` is one class for both jobs. Here the modules are built with the same names
and the same arguments (so the published checkpoint loads strictly, every tensor accounted for), the
image path is :meth:`Sam2.encode` / :meth:`Sam2.decode` below, and the tracking step -- memory
attention over earlier frames, the memory encoder, object pointers -- is
:class:`~.tracking.TrackingStep`, copied from ``SAM2Base``. The frame bookkeeping that drives it
lives in :mod:`.video`.

The split into :meth:`Sam2.encode` and :meth:`Sam2.decode` is the seam mozo builds on. The
encoder is the expensive half and depends only on the image; the decoder is cheap and depends on
the prompt. Keeping them apart is what lets one encode serve many prompts, and it is also what
lets each half be exported to a graph runtime on its own.
"""

from __future__ import annotations

import torch
from torch import nn

from .backbones.hieradet import Hiera
from .backbones.image_encoder import FpnNeck, ImageEncoder
from .config import DECODER, SHARED, TRACKER, Spec
from .memory_attention import MemoryAttention, MemoryAttentionLayer
from .memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
from .position_encoding import PositionEmbeddingSine
from .sam.mask_decoder import MaskDecoder
from .sam.prompt_encoder import PromptEncoder
from .sam.transformer import RoPEAttention, TwoWayTransformer
from .sam2_utils import MLP
from .tracking import TrackingStep

__all__ = ["Sam2"]


class Sam2(TrackingStep, nn.Module):
    """SAM 2: single images through :meth:`encode`/:meth:`decode`, video through the tracking step.

    Args:
        spec: The variant's geometry, from :data:`.config.SPECS`.

    Attributes:
        image_size: Square side the encoder runs at. Prompts are scaled into this space.
    """

    def __init__(self, spec: Spec):
        super().__init__()
        self.image_size = SHARED["image_size"]
        self.num_feature_levels = SHARED["num_feature_levels"]
        hidden = SHARED["hidden_dim"]

        self.image_encoder = ImageEncoder(
            trunk=Hiera(
                embed_dim=spec.embed_dim,
                num_heads=spec.num_heads,
                stages=spec.stages,
                global_att_blocks=spec.global_att_blocks,
                window_pos_embed_bkg_spatial_size=spec.window_pos_embed_bkg_spatial_size,
                window_spec=spec.window_spec,
            ),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=hidden, normalize=True, scale=None, temperature=10000
                ),
                d_model=hidden,
                backbone_channel_list=list(spec.backbone_channel_list),
                fpn_top_down_levels=list(SHARED["fpn_top_down_levels"]),
                fpn_interp_model=SHARED["fpn_interp_model"],
            ),
            scalp=SHARED["scalp"],
        )

        # Built with the same arguments upstream's ``_build_sam_heads`` passes. The flags that
        # look like tracker concerns -- object scores, the multimask output token -- still decide
        # how many tokens the decoder allocates, so they change the shape of the weights and have
        # to match the checkpoint even though nothing here reads their outputs.
        embedding = self.image_size // SHARED["backbone_stride"]
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=hidden,
            image_embedding_size=(embedding, embedding),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=hidden, mlp_dim=2048, num_heads=8),
            transformer_dim=hidden,
            iou_head_depth=3,
            iou_head_hidden_dim=hidden,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            **DECODER,
        )

        # Trained as "there is no memory to attend to", which is exactly the situation a single
        # image is in -- so despite belonging to the tracker by name, this is added on the image
        # path as well. Leaving it out costs 4.5e-02 on the embedding and a few hundred pixels per
        # mask: close enough to look correct. Found by comparing against upstream rather than by
        # reading, and recorded in PROVENANCE.md, because nothing in the suite pins it yet.
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, hidden))
        self._build_tracker(hidden)

    def _build_tracker(self, hidden: int) -> None:
        """The video tracker's modules, as ``sam2.1_hiera_*.yaml`` and ``SAM2Base.__init__`` build them.

        Built for every variant and every caller, image-only ones included, so that the published
        checkpoint loads with nothing left over: 7.5 M parameters, the same in all four sizes.
        """
        mem_dim = TRACKER["mem_dim"]
        self.hidden_dim = hidden
        self.mem_dim = mem_dim
        self.backbone_stride = SHARED["backbone_stride"]
        self.sam_prompt_embed_dim = hidden
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        for name in ("num_maskmem", "sigmoid_scale_for_mem_enc", "sigmoid_bias_for_mem_enc",
                     "binarize_mask_from_pts_for_mem_enc", "directly_add_no_mem_embed",
                     "multimask_output_in_sam", "multimask_output_for_tracking",
                     "multimask_min_pt_num", "multimask_max_pt_num", "max_obj_ptrs_in_encoder",
                     "max_cond_frames_in_attn", "memory_temporal_stride_for_eval",
                     "use_signed_tpos_enc_to_obj_ptrs", "only_obj_ptrs_in_the_past_for_eval"):
            setattr(self, name, TRACKER[name])

        def rope(**extra):
            return RoPEAttention(rope_theta=10000.0, feat_sizes=[64, 64], embedding_dim=hidden,
                                 num_heads=1, downsample_rate=1, dropout=0.1, **extra)

        self.memory_attention = MemoryAttention(
            d_model=hidden,
            pos_enc_at_input=True,
            layer=MemoryAttentionLayer(
                activation="relu",
                cross_attention=rope(rope_k_repeat=True, kv_in_dim=mem_dim),
                d_model=hidden,
                dim_feedforward=2048,
                dropout=0.1,
                pos_enc_at_attn=False,
                pos_enc_at_cross_attn_keys=True,
                pos_enc_at_cross_attn_queries=False,
                self_attention=rope(),
            ),
            num_layers=TRACKER["memory_attention_layers"],
        )
        self.memory_encoder = MemoryEncoder(
            out_dim=mem_dim,
            mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
            fuser=Fuser(
                layer=CXBlock(dim=hidden, kernel_size=7, padding=3, layer_scale_init_value=1e-6,
                              use_dwconv=True),
                num_layers=2,
            ),
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=mem_dim, normalize=True, scale=None, temperature=10000
            ),
        )
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(self.num_maskmem, 1, 1, mem_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, hidden))
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, hidden))
        self.no_obj_embed_spatial = nn.Parameter(torch.zeros(1, mem_dim))
        self.obj_ptr_proj = MLP(hidden, hidden, hidden, 3)
        self.obj_ptr_tpos_proj = nn.Linear(hidden, mem_dim)
        # Present in the checkpoint and read only when a mask is the prompt (upstream's
        # ``_use_mask_as_output``), which this package does not take. Built so the load is strict.
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

    @torch.no_grad()
    def encode(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the expensive half: an image in, the features every prompt will reuse out.

        Args:
            batch: ``(1, 3, image_size, image_size)`` normalised float tensor.

        Returns:
            ``image_embed`` at stride 16 and the two ``high_res_feats`` the decoder skips into.
            This dict is the whole of what a cache needs to hold; nothing else survives the call.
        """
        out = self.image_encoder(batch)
        # Upstream projects these two inside ``forward_image`` rather than in the decoder, so that
        # a second click on the same image does not repeat them. Same reason they are cached here.
        levels = out["backbone_fpn"]
        levels[0] = self.sam_mask_decoder.conv_s0(levels[0])
        levels[1] = self.sam_mask_decoder.conv_s1(levels[1])
        maps = levels[-self.num_feature_levels :]
        # Upstream adds this in the flattened HWxNxC layout, where it broadcasts over positions;
        # here the maps stay NCHW, so the same bias is applied per channel instead.
        embed = maps[-1] + self.no_mem_embed.view(1, -1, 1, 1)
        return {"image_embed": embed, "high_res_feats": maps[:-1]}

    @torch.no_grad()
    def decode(
        self,
        features: dict[str, torch.Tensor],
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the cheap half: cached features plus a prompt in, low-resolution masks out.

        Args:
            features: What :meth:`encode` returned for this image.
            point_coords: ``(B, N, 2)`` x, y already scaled into ``image_size`` space.
            point_labels: ``(B, N)``. 1 positive, 0 negative, 2 and 3 the corners of a box.
            mask_input: ``(B, 1, 256, 256)`` logits from an earlier call, to refine. One channel,
                not the three a multimask call returns -- pick the candidate first.
            multimask_output: Return three candidate masks rather than one.

        Returns:
            ``(B, C, 256, 256)`` mask logits and their ``(B, C)`` predicted IoU.
        """
        points = None if point_coords is None else (point_coords, point_labels)
        sparse, dense = self.sam_prompt_encoder(points=points, boxes=None, masks=mask_input)
        # One encode serves a batch of prompts, so the single image embedding is repeated across
        # them inside the decoder rather than being encoded once per prompt. The count is read off
        # the prompt encoder's own output, because a mask-only prompt carries a batch too and
        # deriving it from the points alone would miss that.
        batched = sparse.shape[0] != features["image_embed"].shape[0]
        low_res, iou, _, _ = self.sam_mask_decoder(
            image_embeddings=features["image_embed"],
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            repeat_image=batched,
            high_res_features=list(features["high_res_feats"]),
        )
        return low_res, iou
