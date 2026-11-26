# SPDX-License-Identifier: Apache-2.0
"""The geometry of SAM 3's image path, as plain data.

Upstream spreads these numbers across a Hydra-style builder (``facebookresearch/sam3``) and a
tree of ``PreTrainedConfig`` dataclasses (``transformers``). Both amount to calling constructors
with recorded constants, so the constants are written here and the constructors are called by the
modules that need them -- :mod:`.vision.vit`, :mod:`.vision.neck`, :mod:`.text.encoder` and the
three stages under :mod:`.grounding`.

Meta publishes exactly one SAM 3, so there is no variant table -- unlike every other family mozo
serves. :data:`SPEC` is the model.

One number is deliberately *not* taken from ``transformers``. Its ``Sam3VitConfig`` defaults
``layer_norm_eps`` to ``1e-6``; ``sam3/model/vitdet.py:719`` pins ``eps=1e-5``, and the weights
were trained under that. We follow the weights. See ``PROVENANCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "BlockSpec",
    "CLICK",
    "ClickSpec",
    "DECODER",
    "DecoderSpec",
    "FUSION",
    "GEOMETRY",
    "GeometrySpec",
    "MASK",
    "MaskSpec",
    "SCORING",
    "ScoringSpec",
    "SPEC",
    "Spec",
    "TEXT",
    "TextSpec",
    "TRACKER",
    "TrackerSpec",
    "TrunkSpec",
    "VIDEO",
    "VideoSpec",
]


@dataclass(frozen=True)
class TrunkSpec:
    """The RoPE ViT that does ~all of SAM 3's work (463M of the 849M image model).

    Args:
        hidden: Width of every block.
        intermediate: MLP width. ``4736`` is ``1024 * 4.625``, not a round multiple.
        layers: Number of blocks.
        heads: Attention heads. RoPE runs on ``hidden // heads`` = 64.
        patch: Patch side. 14, so a 1008 image becomes a 72x72 grid.
        image_size: Square side the trunk runs at.
        pretrain_image_size: Side the position embedding was trained at -- 336, i.e. 24x24
            patches, which is why it must be tiled up to 72x72.
        window: Attention window, in patches, for every block not in ``global_blocks``.
        global_blocks: Blocks attending over the whole grid rather than within a window.
        layer_norm_eps: See the module docstring -- Meta's value, not ``transformers``'.
    """

    hidden: int = 1024
    intermediate: int = 4736
    layers: int = 32
    heads: int = 16
    patch: int = 14
    image_size: int = 1008
    pretrain_image_size: int = 336
    window: int = 24
    global_blocks: tuple[int, ...] = (7, 15, 23, 31)
    layer_norm_eps: float = 1e-5

    @property
    def grid(self) -> int:
        """Side of the patch grid the trunk produces: 1008 // 14 = 72."""
        return self.image_size // self.patch

    @property
    def pretrain_grid(self) -> int:
        """Side of the grid the position embedding was trained at: 336 // 14 = 24."""
        return self.pretrain_image_size // self.patch

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads


@dataclass(frozen=True)
class TextSpec:
    """The CLIP-L text tower that turns a prompt into features (354M, 41% of the checkpoint).

    Args:
        width: Block width.
        layers: Number of blocks.
        heads: Attention heads.
        intermediate: MLP width -- a plain 4x here, unlike the trunk's 4.625x.
        vocab_size: CLIP's byte-pair vocabulary.
        context_length: Prompt length in tokens. **32, not CLIP's usual 77** -- the checkpoint's
            position embedding is ``(32, 1024)``. Longer prompts are truncated.
        fusion_width: What ``resizer`` projects to, matching the grounding stage's width.
        layer_norm_eps: PyTorch's default, which is what upstream's bare ``nn.LayerNorm`` uses.
    """

    width: int = 1024
    layers: int = 24
    heads: int = 16
    intermediate: int = 4096
    vocab_size: int = 49408
    context_length: int = 32
    fusion_width: int = 256
    layer_norm_eps: float = 1e-5


@dataclass(frozen=True)
class BlockSpec:
    """One transformer block's width. Shared by the geometry and fusion encoders, which the
    checkpoint shows are the same structure.

    Args:
        hidden: Block width, and the width everything downstream of the text tower runs at.
        heads: Attention heads.
        intermediate: Feed-forward width.
        layers: How many blocks the stack holds.
    """

    hidden: int = 256
    heads: int = 8
    intermediate: int = 2048
    layers: int = 6


@dataclass(frozen=True)
class GeometrySpec(BlockSpec):
    """The geometry encoder, which turns exemplar boxes into prompt tokens.

    It runs on **every** prompt, not only those carrying boxes: with no boxes it still emits a
    single CLS token that has cross-attended to the image, and that token is concatenated onto
    the text tokens. ``transformers`` skips this module entirely when no boxes are supplied,
    which makes its text-only prompt one token shorter than the weights were trained for.

    Args:
        roi_size: Side of the square each box is pooled to before projection.
    """

    layers: int = 3
    roi_size: int = 7


@dataclass(frozen=True)
class DecoderSpec(BlockSpec):
    """The DETR decoder, which turns learned queries into boxes.

    Args:
        queries: Object queries. Each may become one detected instance, so this is the hard
            ceiling on how many things a single prompt can find.
    """

    queries: int = 200


@dataclass(frozen=True)
class ScoringSpec(BlockSpec):
    """The head that scores each object query against the prompt.

    Args:
        limit: Symmetric clamp on the returned logit, so a confident query cannot saturate a
            later sigmoid into a hard 0 or 1.
    """

    limit: float = 12.0


@dataclass(frozen=True)
class MaskSpec(BlockSpec):
    """The head that turns queries and pixels into masks.

    Args:
        upsampling_stages: Conv/norm pairs the pixel decoder allocates. Upstream builds three but
            runs one per gap between pyramid levels, and there are three levels -- so the third
            pair is loaded and never used. Kept so a strict load has somewhere to put it.
        groups: Group count for the pixel decoder's ``GroupNorm``.
    """

    upsampling_stages: int = 3
    groups: int = 8


@dataclass(frozen=True)
class Spec:
    """The whole vision half: the trunk, plus the FPN that reshapes its output.

    Args:
        trunk: The ViT above.
        fpn_hidden: Channel width every FPN level is projected to.
        scale_factors: Resolution multipliers applied to the trunk's 72x72 grid, giving
            288/144/72/36. Coarsest last.
        scalp: How many of the lowest-resolution levels to discard before anything downstream
            sees them. Upstream sets 1, so the 36x36 level is built and then dropped.
        mean: Per-channel mean. SAM 3 normalises to [-1, 1], not to ImageNet statistics.
        std: Per-channel standard deviation.
    """

    trunk: TrunkSpec = field(default_factory=TrunkSpec)
    fpn_hidden: int = 256
    scale_factors: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)
    scalp: int = 1
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class ClickSpec(BlockSpec):
    """The click head's geometry.

    Every value is ``transformers/models/sam3_tracker``'s own config default, and every one is
    confirmed by loading the published checkpoint strict -- a shape that disagreed would not fit.

    ``image_size``, ``patch`` and ``hidden`` are not independent readings of that config: they
    are the trunk's square, the trunk's patch and the neck's width, which this head runs on.
    They are restated here so the head reads one spec rather than three, and
    ``tests/families/test_sam3.py`` pins them equal to :data:`SPEC` -- a hand-kept coincidence
    would put a click on the wrong feature without any shape mismatch to catch it.

    Args:
        image_size: The square an image is resized to, and prompts are scaled into.
        patch: Trunk patch size. ``grid`` is what the two of them make.
        point_embeddings: Exclude, include, and a box's two corners. There is no box input.
        mask_channels: Hidden channels of the mask-refinement downscaler.
        multimask_outputs: Candidate masks besides the single-mask token.
        downsample: How far cross-attention projects down before attending. Self-attention
            does not.
        iou_head_depth: Depth of the IoU head.
        stability_delta: How far either side of the threshold the stability check looks.
        stability_thresh: Below this, the single-mask token is replaced by the best candidate.
    """

    image_size: int = 1008
    patch: int = 14
    layers: int = 2
    point_embeddings: int = 4
    mask_channels: int = 16
    multimask_outputs: int = 3
    downsample: int = 2
    iou_head_depth: int = 3
    layer_norm_eps: float = 1e-6
    stability_delta: float = 0.05
    stability_thresh: float = 0.98

    @property
    def grid(self) -> int:
        """Side of the feature grid the click head attends over: 1008 // 14 = 72."""
        return self.image_size // self.patch


@dataclass(frozen=True)
class TrackerSpec:
    """The video tracker's geometry: memory attention, the memory encoder, object pointers.

    Every value is ``transformers/models/sam3_tracker_video``'s own config default and is
    confirmed by loading the published checkpoint strict, except where a note says otherwise.

    Args:
        hidden: Width of the current frame's features and of the memory attention.
        memory_dim: Width of a stored memory (the memory encoder projects down to it).
        num_maskmem: Memory slots: the conditioning frames' plus six previous frames'.
        max_obj_ptrs: Object pointers a frame reads, its own included.
        max_cond_frames: Conditioning frames a frame attends to, the temporally closest.
        attention_layers: Memory-attention blocks.
        attention_intermediate: Their feed-forward width.
        rope_theta: Rotary frequency base.
        mask_input_size: Side a mask is resized to before it is encoded: four times the decoder's
            288 logits, i.e. 1152 -- larger than the 1008 image, by design.
        downsampler_layers: Stride-2 convolutions taking 1152 to 72.
        fuser_layers: ConvNeXt blocks fusing a mask into the image features.
        fuser_kernel: Their depthwise kernel.
        sigmoid_scale: Scale applied to a mask probability before it is encoded.
        sigmoid_bias: Bias applied after the scale.
        memory_selection_threshold: A past frame is a memory candidate only above this score.
        no_object_score: The logit an absent object's mask is set to.
    """

    hidden: int = 256
    memory_dim: int = 64
    num_maskmem: int = 7
    max_obj_ptrs: int = 16
    max_cond_frames: int = 4
    attention_layers: int = 4
    attention_intermediate: int = 2048
    rope_theta: float = 10000.0
    mask_input_size: int = 1152
    downsampler_layers: int = 4
    fuser_layers: int = 2
    fuser_kernel: int = 7
    sigmoid_scale: float = 20.0
    sigmoid_bias: float = -10.0
    memory_selection_threshold: float = 0.01
    no_object_score: float = -1024.0


@dataclass(frozen=True)
class VideoSpec:
    """The concept tracker's decision thresholds: Meta's video defaults, temporal disambiguation on.

    These are ``build_sam3_video_model``'s arguments in ``facebookresearch/sam3`` @ ``5eb25fb``
    (read from the run, not the source: the reference is run as a black box), and they are part of
    the gated function -- changing one gives a tracker Meta never ran.

    Args:
        detection_threshold: A detection below this is not considered at all.
        nms_iou: Mask IoU above which the weaker of two detections is dropped.
        association_iou: Mask IoU at which a detection is "the same object" as a track.
        track_match_iou: Mask IoU at which a track counts as matched by some detection.
        new_object_threshold: An unmatched detection starts an object only above this.
        hotstart_delay: Frames a new object is on probation.
        hotstart_unmatched: Unmatched frames within probation that remove it.
        hotstart_duplicate: Frames overlapping an older object within probation that remove it.
        keep_alive_init: Initial keep-alive count of a new object.
        keep_alive_max: Ceiling of the keep-alive count.
        keep_alive_min: Floor of the keep-alive count.
        occlusion_suppression_iou: Mask IoU at which the more recently occluded of two overlapping
            objects is suppressed.
        hole_area: Largest hole filled, and largest speck removed, in the 288-pixel logits.
        recondition_every: Every this many frames, confident matches re-seed their track.
        high_confidence: Score a detection and a track need to re-seed (both).
        high_iou: Mask IoU a detection needs with its track to re-seed it.
        shrink_ratio: A mask keeping less than this share of its area after overlaps are resolved
            is suppressed before its memory is encoded.
        occluded_logit: The logit a suppressed mask is set to.
    """

    detection_threshold: float = 0.5
    nms_iou: float = 0.1
    association_iou: float = 0.1
    track_match_iou: float = 0.5
    new_object_threshold: float = 0.7
    hotstart_delay: int = 15
    hotstart_unmatched: int = 8
    hotstart_duplicate: int = 8
    keep_alive_init: int = 30
    keep_alive_max: int = 30
    keep_alive_min: int = -1
    occlusion_suppression_iou: float = 0.7
    hole_area: int = 16
    recondition_every: int = 16
    high_confidence: float = 0.8
    high_iou: float = 0.8
    shrink_ratio: float = 0.3
    occluded_logit: float = -10.0


#: The published model. Meta ships one.
SPEC = Spec()

#: The text tower's geometry.
TEXT = TextSpec()

#: The geometry encoder's geometry.
GEOMETRY = GeometrySpec()

#: The vision-language fusion encoder's.
FUSION = BlockSpec()

#: The DETR decoder's.
DECODER = DecoderSpec()

#: The scoring head's.
SCORING = ScoringSpec()

#: The mask head's.
MASK = MaskSpec()

#: The click head's geometry.
CLICK = ClickSpec()

#: The video tracker's geometry.
TRACKER = TrackerSpec()

#: The concept tracker's thresholds.
VIDEO = VideoSpec()
