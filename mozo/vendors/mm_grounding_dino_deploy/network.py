# SPDX-License-Identifier: Apache-2.0
"""The whole model: backbone, neck, text tower, transformer, heads.

    image --> Swin --> input_proj_vision --+
                                           +--> encoder --> query selection --> decoder --> logits, boxes
    prompt -> BERT --> text_projection ----+

The two towers meet in the encoder's fusion layers and again in the decoder's text
cross-attention. The output is 900 queries, each carrying a box and a score against every text
token -- there is no class head, because there is no class list.

Parameter names follow ``transformers``' ``MMGroundingDinoForObjectDetection`` with its
``model.`` prefix and wrapper modules removed; ``checkpoint.py`` holds the mapping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import Spec
from .position import image_position
from .text.bert import BertEncoder
from .transformer import MLP, ContrastiveEmbedding, Decoder, Encoder
from .vision.swin import SwinBackbone

__all__ = ["MMGroundingDino", "phrase_masks"]


def phrase_masks(
    input_ids: Tensor, separators: Tensor
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """Build the phrase-isolating attention mask, the per-phrase positions, and the phrase map.

    A caption is several prompts joined by ``.``, and the separators are what tell them apart.
    Between two separators lies one phrase; the mask lets its tokens attend to each other and to
    nothing else, and the position ids restart at zero inside it.

    Upstream's ``generate_masks_with_special_tokens_and_transfer_map``, with one difference that
    cannot change a number here: the separator scan restarts per batch row, where upstream carries
    the previous row's last separator into the next. mozo runs one caption per image.

    Args:
        input_ids: ``(batch, tokens)``.
        separators: Token ids that end a phrase -- ``[CLS]``, ``[SEP]``, ``.`` and ``?``.

    Returns:
        ``(attention_mask, position_ids, per_row_phrase_masks)``. The attention mask is
        ``(batch, tokens, tokens)`` and True where attention is allowed. The phrase map is one
        ``(phrases, tokens)`` boolean tensor per row, which is what lets a detection be traced
        back to the prompt that produced it.
    """
    batch, tokens = input_ids.shape
    device = input_ids.device

    special = torch.zeros((batch, tokens), device=device, dtype=torch.bool)
    for token in separators.tolist():
        special |= input_ids == token

    attention = torch.eye(tokens, device=device, dtype=torch.bool)[None].repeat(batch, 1, 1)
    position_ids = torch.zeros((batch, tokens), device=device, dtype=torch.long)
    phrases: list[list[Tensor]] = [[] for _ in range(batch)]

    for row in range(batch):
        previous = 0
        for col in torch.nonzero(special[row]).flatten().tolist():
            if col == 0 or col == tokens - 1:
                attention[row, col, col] = True
                position_ids[row, col] = 0
            else:
                attention[row, previous + 1 : col + 1, previous + 1 : col + 1] = True
                position_ids[row, previous + 1 : col + 1] = torch.arange(
                    0, col - previous, device=device)
                belongs = torch.zeros(tokens, device=device, dtype=torch.bool)
                belongs[previous + 1 : col] = True
                phrases[row].append(belongs)
            previous = col

    return attention, position_ids, [torch.stack(row) for row in phrases]


def _valid_ratio(valid: Tensor) -> Tensor:
    """What fraction of each axis is picture rather than padding."""
    _, height, width = valid.shape
    ratio_h = torch.sum(valid[:, :, 0], 1).float() / height
    ratio_w = torch.sum(valid[:, 0, :], 1).float() / width
    return torch.stack([ratio_w, ratio_h], -1)


def _inside(index: Tensor, valid: Tensor) -> Tensor:
    """Whether ``(index + 0.5) / valid`` lies strictly inside (0.01, 0.99), decided in integers.

    Upstream asks this of the float quotient. The answer is the same -- the quotient of two
    integers below a million is never within a rounding of either bound unless it is *on* it, and
    on it both say no -- but a compiled graph need not round a division the way torch does, and the
    question is asked exactly on the bound at every edge whose width is a multiple of 50:
    ``49.5 / 50`` is 0.99. TensorRT 11 answered yes there, which made a column of proposals at each
    such edge selectable, and moved 3-6% of detections on COCO. In integers there is nothing to
    round. ``tests/families/test_mm_grounding_dino.py`` holds the two to agreeing for every width up
    to 2,048.

    Args:
        index: ``(cells,)`` integer positions along one axis.
        valid: ``(batch,)`` how many of them are picture rather than padding.

    Returns:
        ``(batch, cells)`` boolean.
    """
    twice = 100 * (2 * index + 1)
    return (twice[None, :] > 2 * valid[:, None]) & (twice[None, :] < 198 * valid[:, None])


def _proposals(memory: Tensor, padding: Tensor,
               level_shapes: list[tuple[int, int]]) -> tuple[Tensor, Tensor]:
    """Turn every image token into a candidate box centred on itself.

    Each level proposes boxes of a fixed size -- 0.05 of the image, doubling per level -- centred
    on each cell. Tokens that are padding, or whose proposal escapes (0.01, 0.99), are zeroed in
    the memory and pushed to ``inf`` in the proposals so selection cannot pick them.
    """
    batch = memory.shape[0]
    proposals, insides = [], []
    start = 0
    for level, (height, width) in enumerate(level_shapes):
        level_mask = padding[:, start : start + height * width].view(batch, height, width, 1)
        valid_h = torch.sum(~level_mask[:, :, 0, 0], 1)
        valid_w = torch.sum(~level_mask[:, 0, :, 0], 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, height - 1, height, dtype=torch.float32, device=memory.device),
            torch.linspace(0, width - 1, width, dtype=torch.float32, device=memory.device),
            indexing="ij")
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(batch, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(batch, -1, -1, -1) + 0.5) / scale
        size = torch.ones_like(grid) * 0.05 * (2.0**level)
        proposals.append(torch.cat((grid, size), -1).view(batch, -1, 4))
        # The box size is a constant per level, so its half of upstream's bounds test is too.
        sized = 0.01 < 0.05 * 2.0**level < 0.99
        cells = torch.arange(max(height, width), device=memory.device)
        inside = (_inside(cells[:height], valid_h)[:, :, None]
                  & _inside(cells[:width], valid_w)[:, None, :])
        insides.append((inside if sized else torch.zeros_like(inside)).view(batch, -1, 1))
        start += height * width

    output_proposals = torch.cat(proposals, 1)
    inside = torch.cat(insides, 1)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(padding.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~inside, float("inf"))

    queries = memory.masked_fill(padding.unsqueeze(-1), float(0))
    queries = queries.masked_fill(~inside, float(0))
    return queries, output_proposals


class MMGroundingDino(nn.Module):
    """One MM-Grounding-DINO variant, assembled and ready to run.

    Args:
        spec: The variant's geometry.

    Examples:
        >>> model = MMGroundingDino(SPECS["llmdet-tiny"])   # doctest: +SKIP
        >>> logits, boxes = model(image, ids, types, mask, attention, positions)  # doctest: +SKIP
    """

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.spec = spec
        swin = spec.swin
        dim = spec.hidden_dim

        self.backbone = SwinBackbone(swin.embed_dim, swin.depths, swin.num_heads,
                                     swin.window_size, spec.return_stages)
        self.text_backbone = BertEncoder(hidden=spec.text_hidden_dim)
        self.text_projection = nn.Linear(spec.text_hidden_dim, dim)

        # One 1x1 projection per backbone stage, then stride-2 convolutions that invent the
        # coarser levels: the first from the last backbone stage, any further from the level
        # before it.
        channels = spec.backbone_channels
        projections = [nn.Sequential(nn.Conv2d(c, dim, kernel_size=1), nn.GroupNorm(32, dim))
                       for c in channels]
        in_channels = channels[-1]
        for _ in range(spec.num_feature_levels - len(channels)):
            projections.append(nn.Sequential(
                nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, dim)))
            in_channels = dim
        self.input_proj_vision = nn.ModuleList(projections)

        self.query_position_embeddings = nn.Embedding(spec.num_queries, dim)
        self.level_embed = nn.Parameter(torch.zeros(spec.num_feature_levels, dim))
        self.encoder = Encoder(dim, spec.nheads, spec.num_feature_levels, spec.enc_n_points,
                               spec.dim_feedforward, spec.enc_layers)
        self.decoder = Decoder(dim, spec.nheads, spec.num_feature_levels, spec.dec_n_points,
                               spec.dim_feedforward, spec.dec_layers)

        # Query selection's own heads -- separately trained, not the decoder's
        # (``two_stage_bbox_embed_share`` is False).
        self.enc_output = nn.Linear(dim, dim)
        self.enc_output_norm = nn.LayerNorm(dim)
        self.encoder_output_bbox_embed = MLP(dim, dim, 4, 3)
        self.encoder_output_class_embed = ContrastiveEmbedding(spec.max_text_len)

        # One box head per decoder layer, each with its own weights: every one refines a box,
        # so all six run. Only the last layer's score is read, so only its bias is built.
        self.bbox_embed = nn.ModuleList(MLP(dim, dim, 4, 3) for _ in range(spec.dec_layers))
        self.class_embed = ContrastiveEmbedding(spec.max_text_len)

    def forward(
        self,
        image: Tensor,
        input_ids: Tensor,
        token_type_ids: Tensor,
        token_mask: Tensor,
        self_attention_mask: Tensor,
        position_ids: Tensor,
        pixel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run one image against one caption.

        Args:
            image: ``(batch, 3, height, width)``, already normalised.
            input_ids: ``(batch, tokens)``.
            token_type_ids: ``(batch, tokens)``.
            token_mask: ``(batch, tokens)``, True for real tokens.
            self_attention_mask: ``(batch, tokens, tokens)`` from :func:`phrase_masks`.
            position_ids: ``(batch, tokens)`` from :func:`phrase_masks`.
            pixel_mask: ``(batch, height, width)``, True where *image* is padding -- what
                :func:`~.image.fit_square` returns. ``None`` for an image that is all picture,
                which is upstream's single-image inference.

        Returns:
            ``(logits, boxes)`` from the last decoder layer -- ``(batch, queries, max_text_len)``
            of raw scores, and ``(batch, queries, 4)`` of ``cxcywh`` normalised to the picture.
        """
        text = self.text_projection(
            self.text_backbone(input_ids, token_type_ids, position_ids, self_attention_mask))

        batch, _, height, width = image.shape
        # True where the pixel is picture: upstream's convention inside the model, the opposite
        # of this method's argument.
        valid = (torch.ones((batch, height, width), dtype=torch.long, device=image.device)
                 if pixel_mask is None else ~pixel_mask)

        features = self.backbone(image)
        sources = [self.input_proj_vision[level](feature)
                   for level, feature in enumerate(features)]
        for level in range(len(features), self.spec.num_feature_levels):
            previous = features[-1] if level == len(features) else sources[-1]
            sources.append(self.input_proj_vision[level](previous))
        # Each level's mask is the pixel mask resized to it, nearest -- how upstream's backbone
        # masks every level.
        masks = [F.interpolate(valid[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                 for source in sources]
        # Cast to the dtype of the map each was computed for -- the backbone's output for the
        # backbone levels, the projection's for the invented ones -- as upstream casts them.
        dtypes = [feature.dtype for feature in features] + [
            source.dtype for source in sources[len(features):]]
        positions = [image_position(mask, self.spec.hidden_dim // 2, self.spec.pe_temperature)
                     .to(dtype) for mask, dtype in zip(masks, dtypes)]

        flat_sources, flat_masks, flat_positions, level_shapes = [], [], [], []
        for level, (source, mask, position) in enumerate(zip(sources, masks, positions)):
            level_shapes.append((source.shape[2], source.shape[3]))
            flat_sources.append(source.flatten(2).transpose(1, 2))
            flat_masks.append(mask.flatten(1))
            flat_positions.append(position.flatten(2).transpose(1, 2)
                                  + self.level_embed[level].view(1, 1, -1))
        vision = torch.cat(flat_sources, 1)
        vision_valid = torch.cat(flat_masks, 1)
        vision_position = torch.cat(flat_positions, 1)
        # Two forms of the same sizes: a tensor for the arithmetic that divides by them, and the
        # Python integers the loops and splits read.
        spatial_shapes = torch.as_tensor(level_shapes, dtype=torch.long, device=vision.device)
        valid_ratios = torch.stack([_valid_ratio(mask) for mask in masks], 1).float()

        memory, memory_text = self.encoder(
            vision, vision_valid, vision_position, spatial_shapes, level_shapes, valid_ratios,
            text, token_mask, self_attention_mask, position_ids)

        # Language-guided query selection: score every image token against the prompt, keep the
        # best `num_queries`, and start the decoder from their proposals.
        queries, proposals = _proposals(memory, ~vision_valid, level_shapes)
        queries = self.enc_output_norm(self.enc_output(queries))
        scores = self.encoder_output_class_embed(queries, memory_text, token_mask)
        coords = self.encoder_output_bbox_embed(queries) + proposals
        top = torch.topk(scores.max(-1)[0], self.spec.num_queries, dim=1)[1]
        reference = torch.gather(coords, 1, top.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()
        # `embedding_init_target` is True in every published config: the decoder starts from the
        # learned embedding, not from the selected memory.
        target = self.query_position_embeddings.weight.unsqueeze(0).repeat(batch, 1, 1)

        hidden, references = self.decoder(
            target, reference, memory, vision_valid, spatial_shapes, level_shapes, valid_ratios,
            memory_text, token_mask, self.bbox_embed)

        logits = self.class_embed(hidden[-1], memory_text, token_mask)
        boxes = (self.bbox_embed[-1](hidden[-1])
                 + torch.special.logit(references[-2], eps=1e-5)).sigmoid()
        return logits, boxes
