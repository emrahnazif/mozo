# SPDX-License-Identifier: Apache-2.0
"""The encoder, the query selection between them, and the cross-modality decoder.

The encoder runs three things per layer, in this order: fusion (image <-> text), a text
self-attention over the phrase-isolating mask, and deformable self-attention over the image
pyramid. The decoder then runs 900 queries through self-attention, text cross-attention and
deformable image cross-attention, refining a box at every layer.

Between them sits language-guided query selection: every image token is scored against the text,
the best 900 are taken, and their positions become the decoder's initial boxes.

**What makes this MM-Grounding-DINO rather than Grounding DINO** is all in the heads:

- the contrastive score is scaled by ``1/sqrt(256)`` and carries a learned **bias**, one per head;
- every decoder layer has **its own** box head and its own score bias -- Grounding DINO shares one
  box head across all six (``decoder_bbox_embed_share``), and these checkpoints do not;
- boxes are refined through ``torch.special.logit(x, eps=1e-5)``, where Grounding DINO's
  ``inverse_sigmoid`` clamps at 1e-3.

Not carried: denoising training, auxiliary per-layer losses, the one-stage branch, gradient
checkpointing. All are training paths; none is reachable from a published checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .fuse import FusionLayer
from .ops import MultiheadAttention, MultiScaleDeformableAttention
from .position import sine_embed

__all__ = ["MLP", "ContrastiveEmbedding", "Decoder", "Encoder"]


class MLP(nn.Module):
    """A plain feed-forward stack with ReLU between layers and none at the end."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        widths = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + widths, widths + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class ContrastiveEmbedding(nn.Module):
    """Score every query against every text token: a scaled dot product plus a learned bias.

    The result is padded out to ``max_text_len`` with ``-inf`` so the logit tensor has a fixed
    width whatever the prompt length, and columns past the prompt are never selected.
    """

    def __init__(self, max_text_len: int) -> None:
        super().__init__()
        self.max_text_len = max_text_len
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, vision: Tensor, text: Tensor, text_valid: Tensor) -> Tensor:
        scores = vision @ text.transpose(-1, -2)
        scores = scores / math.sqrt(vision.shape[-1])
        scores = scores + self.bias
        scores = scores.masked_fill(~text_valid[:, None, :], float("-inf"))
        padded = torch.full((*scores.shape[:-1], self.max_text_len), float("-inf"),
                            device=scores.device)
        padded[..., : scores.shape[-1]] = scores
        return padded


class _TextEnhancerLayer(nn.Module):
    """Post-norm self-attention over the text tokens, masked to keep phrases apart."""

    def __init__(self, dim: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(dim, heads)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.layer_norm_before = nn.LayerNorm(dim)
        self.layer_norm_after = nn.LayerNorm(dim)
        self.heads = heads

    def forward(self, hidden: Tensor, attend: Tensor, position: Tensor) -> Tensor:
        # The boolean phrase mask, True where attention is allowed, becomes an additive one: the
        # dtype's most negative finite value where it is not.
        mask = attend[:, None, :, :].repeat(1, self.heads, 1, 1).to(dtype=hidden.dtype)
        mask = (1.0 - mask) * torch.finfo(hidden.dtype).min
        queries = hidden + position
        hidden = self.layer_norm_before(hidden + self.self_attn(queries, queries, hidden, mask))
        return self.layer_norm_after(hidden + self.fc2(F.relu(self.fc1(hidden))))


class _DeformableLayer(nn.Module):
    """Deformable self-attention over the image pyramid, then a feed-forward, both post-norm."""

    def __init__(self, dim: int, heads: int, levels: int, points: int, ffn: int) -> None:
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(dim, heads, levels, points)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, hidden: Tensor, position: Tensor, valid: Tensor, reference: Tensor,
                spatial_shapes: Tensor, level_shapes: list[tuple[int, int]]) -> Tensor:
        attended = self.self_attn(hidden, position, hidden, valid, reference, spatial_shapes,
                                  level_shapes)
        hidden = self.self_attn_layer_norm(hidden + attended)
        return self.final_layer_norm(hidden + self.fc2(F.relu(self.fc1(hidden))))


class _EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, levels: int, points: int, ffn: int) -> None:
        super().__init__()
        self.text_enhancer_layer = _TextEnhancerLayer(dim, heads // 2, ffn // 2)
        self.fusion_layer = FusionLayer(dim, ffn // 2, heads // 2)
        self.deformable_layer = _DeformableLayer(dim, heads, levels, points, ffn)


class Encoder(nn.Module):
    """Six layers of fuse, then text self-attention, then deformable image attention."""

    def __init__(self, dim: int, heads: int, levels: int, points: int, ffn: int,
                 num_layers: int) -> None:
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            _EncoderLayer(dim, heads, levels, points, ffn) for _ in range(num_layers))

    @staticmethod
    def reference_points(level_shapes: list[tuple[int, int]], valid_ratios: Tensor,
                         device) -> Tensor:
        """A normalised grid of cell centres per level, scaled by each image's valid fraction."""
        points = []
        for level, (height, width) in enumerate(level_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij")
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            points.append(torch.stack((ref_x, ref_y), -1))
        stacked = torch.cat(points, 1)
        return stacked[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        vision: Tensor,
        vision_valid: Tensor,
        vision_position: Tensor,
        spatial_shapes: Tensor,
        level_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        text: Tensor,
        text_valid: Tensor,
        text_attend: Tensor,
        text_position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the encoded ``(vision, text)``."""
        reference = self.reference_points(level_shapes, valid_ratios, vision.device)
        text_position = sine_embed(text_position_ids[..., None], self.dim, exchange_xy=False)
        vision_padding, text_padding = ~vision_valid, ~text_valid
        for layer in self.layers:
            vision, text = layer.fusion_layer(vision, text, vision_padding, text_padding)
            text = layer.text_enhancer_layer(text, text_attend, text_position)
            vision = layer.deformable_layer(vision, vision_position, vision_valid, reference,
                                            spatial_shapes, level_shapes)
        return vision, text


class _DecoderLayer(nn.Module):
    """Query self-attention, text cross-attention, deformable image cross-attention, FFN."""

    def __init__(self, dim: int, heads: int, levels: int, points: int, ffn: int) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.encoder_attn_text = MultiheadAttention(dim, heads)
        self.encoder_attn_text_layer_norm = nn.LayerNorm(dim)
        self.encoder_attn = MultiScaleDeformableAttention(dim, heads, levels, points)
        self.encoder_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, hidden: Tensor, position: Tensor, reference: Tensor, memory: Tensor,
                memory_valid: Tensor, spatial_shapes: Tensor,
                level_shapes: list[tuple[int, int]], text: Tensor, text_mask: Tensor) -> Tensor:
        queries = hidden + position
        hidden = self.self_attn_layer_norm(hidden + self.self_attn(queries, queries, hidden))
        attended = self.encoder_attn_text(hidden + position, text, text, text_mask)
        hidden = self.encoder_attn_text_layer_norm(hidden + attended)
        attended = self.encoder_attn(hidden, position, memory, memory_valid, reference,
                                     spatial_shapes, level_shapes)
        hidden = self.encoder_attn_layer_norm(hidden + attended)
        return self.final_layer_norm(hidden + self.fc2(F.relu(self.fc1(hidden))))


class Decoder(nn.Module):
    """Six decoder layers, refining a box after each with that layer's own box head."""

    def __init__(self, dim: int, heads: int, levels: int, points: int, ffn: int,
                 num_layers: int) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.layer_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            _DecoderLayer(dim, heads, levels, points, ffn) for _ in range(num_layers))
        self.reference_points_head = MLP(2 * dim, dim, dim, 2)

    def forward(
        self,
        queries: Tensor,
        reference: Tensor,
        memory: Tensor,
        memory_valid: Tensor,
        spatial_shapes: Tensor,
        level_shapes: list[tuple[int, int]],
        valid_ratios: Tensor,
        text: Tensor,
        text_valid: Tensor,
        box_heads: nn.ModuleList,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Return each layer's normalised output, and the reference box each layer started from.

        The second list has one more entry than the first: the boxes after the last refinement.
        """
        # Additive text mask: the dtype's most negative value on padding, zero elsewhere. mozo
        # never pads a caption, so it is all zeros -- and adding zero is exact, so it is kept
        # rather than special-cased.
        text_mask = (~text_valid)[:, None, None, :].repeat(1, self.heads, queries.shape[1], 1)
        text_mask = text_mask.to(dtype=text.dtype) * torch.finfo(text.dtype).min

        hidden = queries
        outputs, references = [], [reference]
        for index, layer in enumerate(self.layers):
            reference_input = reference[:, :, None] * torch.cat([valid_ratios, valid_ratios],
                                                                -1)[:, None]
            position = self.reference_points_head(
                sine_embed(reference_input[:, :, 0, :], self.dim // 2, exchange_xy=True))
            hidden = layer(hidden, position, reference_input, memory, memory_valid,
                           spatial_shapes, level_shapes, text, text_mask)
            # Refined from the layer's raw output, while the heads read the normalised one --
            # upstream's split, and the reason the two are kept apart.
            delta = box_heads[index](hidden)
            reference = (delta + torch.special.logit(reference, eps=1e-5)).sigmoid()
            outputs.append(self.layer_norm(hidden))
            references.append(reference)
        return outputs, references
