# SPDX-License-Identifier: Apache-2.0
"""Per-variant model geometry, replacing upstream's Hydra configs.

Upstream builds a model by handing a YAML file to Hydra, which imports and instantiates each
``_target_`` class. That pulls in ``hydra-core`` and ``omegaconf`` to do what amounts to calling
four constructors with recorded numbers, so the numbers are written here directly and the
constructors are called in :mod:`.network`.

Every value below was read out of ``sam2/configs/sam2.1/sam2.1_hiera_*.yaml``. The four variants
differ only in the trunk and in the neck's input widths; everything else is identical across them,
which is why it lives in :data:`SHARED` rather than being repeated four times.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "Spec", "SHARED", "DECODER", "TRACKER"]


@dataclass(frozen=True)
class Spec:
    """The geometry of one published variant.

    Args:
        embed_dim: Width of the Hiera trunk's first stage.
        num_heads: Attention heads in the trunk's first stage.
        backbone_channel_list: Trunk output widths, coarsest first, as the neck receives them.
        stages: Blocks per trunk stage.
        global_att_blocks: Indices of the blocks that attend globally rather than within a window.
        window_pos_embed_bkg_spatial_size: Spatial size of the background window position embedding.
        window_spec: Window size per stage.
    """

    embed_dim: int
    num_heads: int
    backbone_channel_list: tuple[int, ...]
    stages: tuple[int, ...] = (2, 3, 16, 3)
    global_att_blocks: tuple[int, ...] = (12, 16, 20)
    window_pos_embed_bkg_spatial_size: tuple[int, int] = (14, 14)
    window_spec: tuple[int, ...] = (8, 4, 14, 7)


#: Settings every variant shares. Named rather than inlined so the two places that need the
#: resolution -- the preprocessing and the prompt scaling -- read the same number.
SHARED = {
    "image_size": 1024,
    "backbone_stride": 16,
    "hidden_dim": 256,
    "scalp": 1,
    "fpn_top_down_levels": (2, 3),
    "fpn_interp_model": "nearest",
    "num_feature_levels": 3,
}

#: The mask decoder's post-processing, which carries no weights and is not in the YAML: upstream's
#: ``build_sam2`` and ``build_sam2_video_predictor`` both add it as Hydra overrides when
#: ``apply_postprocessing`` is left at its default. It decides what a single-mask call returns when
#: that mask is unstable, so a strict load cannot check it -- only the gate can.
DECODER = {
    "dynamic_multimask_via_stability": True,
    "dynamic_multimask_stability_delta": 0.05,
    "dynamic_multimask_stability_thresh": 0.98,
}

#: The video tracker's settings. Identical in all four ``sam2.1_hiera_*.yaml`` files (the tracker
#: is the same network in every size; only the trunk differs), plus the two the video predictor's
#: builder adds: ``binarize_mask_from_pts_for_mem_enc`` and ``fill_hole_area``. The second is listed
#: so that the decision is visible: upstream fills holes only with a CUDA extension it builds
#: optionally, and without it skips the step, which is what this package reproduces (PROVENANCE).
TRACKER = {
    "num_maskmem": 7,
    "mem_dim": 64,
    "sigmoid_scale_for_mem_enc": 20.0,
    "sigmoid_bias_for_mem_enc": -10.0,
    "binarize_mask_from_pts_for_mem_enc": True,
    "directly_add_no_mem_embed": True,
    "multimask_output_in_sam": True,
    "multimask_output_for_tracking": True,
    "multimask_min_pt_num": 0,
    "multimask_max_pt_num": 1,
    "max_obj_ptrs_in_encoder": 16,
    "max_cond_frames_in_attn": -1,
    "memory_temporal_stride_for_eval": 1,
    "use_signed_tpos_enc_to_obj_ptrs": True,
    "only_obj_ptrs_in_the_past_for_eval": True,
    "memory_attention_layers": 4,
    "fill_hole_area": 0,
}

#: Variant name -> geometry. ``base_plus`` takes the trunk defaults; the other three override.
SPECS: dict[str, Spec] = {
    "tiny": Spec(
        embed_dim=96,
        num_heads=1,
        backbone_channel_list=(768, 384, 192, 96),
        stages=(1, 2, 7, 2),
        global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
    ),
    "small": Spec(
        embed_dim=96,
        num_heads=1,
        backbone_channel_list=(768, 384, 192, 96),
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    ),
    "base_plus": Spec(
        embed_dim=112,
        num_heads=2,
        backbone_channel_list=(896, 448, 224, 112),
    ),
    "large": Spec(
        embed_dim=144,
        num_heads=2,
        backbone_channel_list=(1152, 576, 288, 144),
        stages=(2, 6, 36, 4),
        global_att_blocks=(23, 33, 43),
        window_pos_embed_bkg_spatial_size=(7, 7),
        window_spec=(8, 4, 16, 8),
    ),
}
