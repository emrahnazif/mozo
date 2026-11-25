# SPDX-License-Identifier: Apache-2.0
"""The numbers that define an MM-Grounding-DINO variant, as frozen data.

Upstream keeps these in each checkpoint's ``config.json``, read through ``transformers``'
``MMGroundingDinoConfig`` alongside dozens of training-only fields (loss weights, matcher costs,
dropout rates). Written out here as frozen dataclasses: a config read at run time is a config
that can differ between two runs, and the training settings are not carried at all.

The three published LLMDet sizes share everything but the Swin backbone and, for ``large``, the
depth of the image pyramid: Swin-L feeds all four of its stages to the neck and a fifth level is
made by convolution, where Swin-T and Swin-B feed three and the neck makes a fourth. That one
difference is a different code path through the neck, the encoder and every deformable
attention, which is why ``large`` is carried and not merely a bigger ``base``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "VARIANTS", "Spec", "SwinSpec"]


@dataclass(frozen=True)
class SwinSpec:
    """Geometry of the Swin Transformer image backbone."""

    embed_dim: int
    depths: tuple[int, ...]
    num_heads: tuple[int, ...]
    window_size: int

    @property
    def num_features(self) -> tuple[int, ...]:
        """Channel width of each stage's output."""
        return tuple(self.embed_dim * 2**i for i in range(len(self.depths)))


#: The three backbones the published checkpoints use, as their ``backbone_config`` states them.
#: Swin-B and Swin-L are the 384-pretrained models, which is why their window is 12 and not 7.
_SWIN = {
    "swin-t": SwinSpec(96, (2, 2, 6, 2), (3, 6, 12, 24), 7),
    "swin-b": SwinSpec(128, (2, 2, 18, 2), (4, 8, 16, 32), 12),
    "swin-l": SwinSpec(192, (2, 2, 18, 2), (6, 12, 24, 48), 12),
}


@dataclass(frozen=True)
class Spec:
    """One published variant, in full.

    Attributes:
        variant: mozo's name for it.
        backbone_name: Which Swin geometry to build.
        return_stages: Which Swin stages (0-based) feed the neck. The remaining feature levels,
            up to ``num_feature_levels``, are stride-2 convolutions over the last of them.
        num_feature_levels: Depth of the image pyramid the transformer reads.
        hidden_dim: Width of the transformer, and of the projected text features.
        num_queries: Object queries the decoder carries. Every one produces a row of logits.
        max_text_len: Hard cap on prompt tokens. Upstream truncates past it; mozo refuses.
    """

    variant: str
    backbone_name: str
    return_stages: tuple[int, ...]
    num_feature_levels: int
    hidden_dim: int = 256
    nheads: int = 8
    num_queries: int = 900
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    enc_n_points: int = 4
    dec_n_points: int = 4
    max_text_len: int = 256
    text_hidden_dim: int = 768
    #: Not 10000. ``positional_embedding_temperature`` is 20 in every published config, and a
    #: sine position encoding copied from any other DETR brings 10000 with it -- which runs, and
    #: moves every box.
    pe_temperature: int = 20
    #: Shortest side, and the cap on the longest. Aspect ratio is preserved and nothing is
    #: padded, so the tensor a photograph becomes depends on the photograph.
    short_side: int = 800
    max_side: int = 1333
    #: The threshold the authors' own examples use -- the LLMDet README and every model card call
    #: ``post_process_grounded_object_detection(threshold=0.4)``. Upstream also carries a
    #: ``text_threshold``; it selects which tokens are decoded into a phrase, and this package
    #: reports the caller's prompt instead of decoding one, so it is not carried.
    box_threshold: float = 0.4

    @property
    def swin(self) -> SwinSpec:
        return _SWIN[self.backbone_name]

    @property
    def backbone_channels(self) -> tuple[int, ...]:
        """Channel width of each level handed to the neck, backbone stages only."""
        features = self.swin.num_features
        return tuple(features[i] for i in self.return_stages)


#: Published variants, smallest first. All three are LLMDet: the MM-Grounding-DINO architecture,
#: fine-tuned from MM-Grounding-DINO's own checkpoints with a language model's supervision.
SPECS: dict[str, Spec] = {
    "llmdet-tiny": Spec("llmdet-tiny", "swin-t", return_stages=(1, 2, 3), num_feature_levels=4),
    "llmdet-base": Spec("llmdet-base", "swin-b", return_stages=(1, 2, 3), num_feature_levels=4),
    "llmdet-large": Spec("llmdet-large", "swin-l", return_stages=(0, 1, 2, 3),
                         num_feature_levels=5),
}

VARIANTS = list(SPECS)
