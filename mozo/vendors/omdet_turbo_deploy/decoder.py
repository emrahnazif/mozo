# SPDX-License-Identifier: Apache-2.0
"""The efficient fusion head: where the words meet the image.

Extracted from ``transformers``' ``OmDetTurboDecoder`` (Apache-2.0; the authors' ``ELADecoder``),
inference path only. In order:

1. **Query selection.** Every position of the three encoder levels (8,400 at 640) is projected and
   scored against every class by scaled cosine; the 900 with the best class score become queries,
   their boxes from a regression head plus a fixed anchor grid.
2. **Six decoder layers.** Self-attention runs over the 900 queries *and* the task sentence's
   tokens concatenated -- this is the fusion; the task tokens come out updated and go on to the
   next layer -- then deformable cross-attention into the image, then a feed-forward.
3. **The last layer's boxes and class logits** are the output. Upstream computes intermediate
   class logits only in training; so does this.

Upstream details kept as they are, each one checked bit for bit:

- ``logit_scale`` is ``tensor(1/0.07).log()`` and then ``.exp()``, as upstream writes it (in
  float32 the round trip happens to be exact).
- Norms divide by ``norm.clamp_min(1e-12)``.
- Anchors are ``log(a / (1 - a))`` with ``inf`` where any coordinate is outside (0.01, 0.99), and
  the features at those positions are zeroed with ``torch.where`` before projection.
- ``inverse_sigmoid`` clamps to [0, 1] and then at 1e-5 on both sides.
- The query position embedding is recomputed from the current reference boxes in every layer.
- Deformable attention is the ``grid_sample`` formulation, with ``align_corners=False``.
- At batch one the task sentence has no padding, so the self-attention mask upstream builds is
  ``None``, and nothing is masked here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import Spec
from .encoder import MultiheadAttention

__all__ = ["Decoder", "inverse_sigmoid"]


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def class_similarity(features: Tensor, classes: Tensor) -> Tensor:
    """Scaled cosine between ``(B, N, D)`` features and ``(B, D, C)`` projected classes."""
    features = features / features.norm(dim=2, keepdim=True).clamp_min(1e-12)
    classes = classes / classes.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.tensor(1 / 0.07).log().exp()
    return scale * torch.bmm(features, classes)


class Mlp(nn.Module):
    """``OmDetTurboMLP``: linears with ReLU between, none after the last."""

    def __init__(self, sizes: tuple[int, ...]):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, x: Tensor) -> Tensor:
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < last else layer(x)
        return x


class TaskMlp(nn.Module):
    def __init__(self, spec: Spec):
        super().__init__()
        self.linear1 = nn.Linear(spec.class_embed_dim, spec.task_encoder_hidden)
        self.linear2 = nn.Linear(spec.task_encoder_hidden, spec.class_embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class Residual(nn.Module):
    def __init__(self, spec: Spec):
        super().__init__()
        self.norm1 = nn.LayerNorm(spec.class_embed_dim, eps=spec.layer_norm_eps)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.norm1(x + y)


class TaskEncoder(nn.Module):
    def __init__(self, spec: Spec):
        super().__init__()
        self.mlp = TaskMlp(spec)
        self.res1 = Residual(spec)

    def forward(self, x: Tensor) -> Tensor:
        return self.res1(x, self.mlp(x))


def deformable_sample(value: Tensor, shapes: list[tuple[int, int]], locations: Tensor,
                      weights: Tensor) -> Tensor:
    """Multi-scale deformable attention, ``grid_sample`` form (Deformable DETR's pytorch path).

    Level sizes are Python ints, not read off a tensor, so the graph exporters see constants.
    """
    batch, _, heads, head_dim = value.shape
    _, queries, _, levels, points, _ = locations.shape
    per_level = value.split([height * width for height, width in shapes], dim=1)
    grids = 2 * locations - 1
    sampled = []
    for level, (height, width) in enumerate(shapes):
        level_value = (per_level[level].flatten(2).transpose(1, 2)
                       .reshape(batch * heads, head_dim, height, width))
        grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(level_value, grid, mode="bilinear", padding_mode="zeros",
                                     align_corners=False))
    weights = weights.transpose(1, 2).reshape(batch * heads, 1, queries, levels * points)
    output = ((torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
              .view(batch, heads * head_dim, queries))
    return output.transpose(1, 2).contiguous()


class DeformableAttention(nn.Module):
    def __init__(self, spec: Spec):
        super().__init__()
        self.heads, self.levels, self.points = spec.decoder_heads, spec.levels, spec.decoder_points
        self.d_model = spec.d_model
        self.sampling_offsets = nn.Linear(spec.d_model, self.heads * self.levels * self.points * 2)
        self.attention_weights = nn.Linear(spec.d_model, self.heads * self.levels * self.points)
        self.value_proj = nn.Linear(spec.d_model, spec.d_model)
        self.output_proj = nn.Linear(spec.d_model, spec.d_model)

    def forward(self, hidden: Tensor, memory: Tensor, reference: Tensor,
                shapes: list[tuple[int, int]]) -> Tensor:
        batch, queries, _ = hidden.shape
        _, length, _ = memory.shape
        value = self.value_proj(memory).view(batch, length, self.heads, self.d_model // self.heads)
        offsets = self.sampling_offsets(hidden).view(
            batch, queries, self.heads, self.levels, self.points, 2)
        weights = self.attention_weights(hidden).view(
            batch, queries, self.heads, self.levels * self.points)
        weights = F.softmax(weights, -1).view(batch, queries, self.heads, self.levels, self.points)
        # Reference boxes, (cx, cy, w, h): offsets are scaled by half the box per point.
        locations = (reference[:, :, None, :, None, :2]
                     + offsets / self.points * reference[:, :, None, :, None, 2:] * 0.5)
        return self.output_proj(deformable_sample(value, shapes, locations, weights))


class DecoderLayer(nn.Module):
    def __init__(self, spec: Spec):
        super().__init__()
        width = spec.d_model
        self.self_attn = MultiheadAttention(width, spec.decoder_heads)
        self.norm1 = nn.LayerNorm(width, eps=spec.layer_norm_eps)
        self.cross_attn = DeformableAttention(spec)
        self.norm2 = nn.LayerNorm(width, eps=spec.layer_norm_eps)
        self.linear1 = nn.Linear(width, spec.decoder_feedforward)
        self.linear2 = nn.Linear(spec.decoder_feedforward, width)
        self.norm3 = nn.LayerNorm(width, eps=spec.layer_norm_eps)

    def forward(self, embeddings: Tensor, task: Tensor, reference: Tensor, memory: Tensor,
                shapes: list[tuple[int, int]], position: Tensor) -> tuple[Tensor, Tensor]:
        """One layer. ``task`` is ``(T, B, D)``, as upstream carries it."""
        queries = embeddings.shape[1]
        query = key = embeddings + position
        task = task.transpose(0, 1)
        query = torch.cat((query, task), dim=1)
        key = torch.cat((key, task), dim=1)
        embeddings = torch.cat((embeddings, task), dim=1)
        embeddings = self.norm1(embeddings + self.self_attn(query, key, embeddings))
        task = embeddings[:, queries:, :].transpose(0, 1)
        embeddings = embeddings[:, :queries, :]

        hidden = embeddings + position
        embeddings = self.norm2(
            embeddings + self.cross_attn(hidden, memory, reference.unsqueeze(2), shapes))
        return self.norm3(embeddings + self.linear2(F.relu(self.linear1(embeddings)))), task


class Decoder(nn.Module):
    """Encoder levels + class and task features -> final boxes and class logits.

    Args:
        spec: Geometry.
    """

    def __init__(self, spec: Spec):
        super().__init__()
        self.spec = spec
        width = spec.d_model
        self.channel_projection_layers = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 1, bias=False), nn.BatchNorm2d(width))
            for _ in range(spec.levels))
        self.task_encoder = TaskEncoder(spec)
        self.task_project = nn.Linear(spec.class_embed_dim, width)
        self.layers = nn.ModuleList(DecoderLayer(spec) for _ in range(spec.decoder_layers))
        self.query_position_head = Mlp((4, 2 * width, width))
        self.encoder_vision_features = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width, eps=spec.layer_norm_eps))
        self.encoder_class_head = nn.Linear(spec.class_embed_dim, width)
        self.encoder_bbox_head = Mlp((width, width, width, 4))
        self.decoder_class_head = nn.ModuleList(
            nn.Linear(spec.class_embed_dim, width) for _ in range(spec.decoder_layers))
        self.decoder_bbox_head = nn.ModuleList(
            Mlp((width, width, width, 4)) for _ in range(spec.decoder_layers))
        self._anchors: dict = {}

    def anchors(self, shapes: tuple[tuple[int, int], ...], device: torch.device,
                dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """``generate_anchors``, cached per shape, device and dtype as upstream's ``lru_cache``."""
        key = (shapes, device, dtype)
        if key not in self._anchors:
            spec = self.spec
            anchors = []
            for level, (height, width) in enumerate(shapes):
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(end=height, dtype=dtype, device=device),
                    torch.arange(end=width, dtype=dtype, device=device), indexing="ij")
                grid = torch.stack([grid_x, grid_y], -1)
                size = torch.tensor([width, height], dtype=dtype, device=device)
                grid = (grid.unsqueeze(0) + 0.5) / size
                wh = torch.ones_like(grid, dtype=dtype, device=device) * spec.anchor_size * (2.0**level)
                anchors.append(torch.concat([grid, wh], -1).reshape(-1, height * width, 4))
            anchors = torch.concat(anchors, 1)
            valid = ((anchors > spec.anchor_eps) * (anchors < 1 - spec.anchor_eps)).all(-1, keepdim=True)
            anchors = torch.log(anchors / (1 - anchors))
            self._anchors[key] = (torch.where(valid, anchors, torch.inf), valid)
        return self._anchors[key]

    def forward(self, levels: list[Tensor], classes: Tensor, task: Tensor) -> tuple[Tensor, Tensor]:
        """Decode.

        Args:
            levels: The encoder's three ``(B, 256, h, w)`` maps.
            classes: ``(C, B, 512)`` projected class embeddings.
            task: ``(T, B, 512)`` task-sentence states.

        Returns:
            ``(B, 900, 4)`` boxes as sigmoid (cx, cy, w, h), and ``(B, 900, C)`` class logits.
        """
        spec = self.spec
        flattened, shapes = [], []
        for project, level in zip(self.channel_projection_layers, levels):
            level = project(level)
            height, width = level.shape[2:]
            flattened.append(level.flatten(2).permute(0, 2, 1))
            shapes.append((int(height), int(width)))
        memory = torch.cat(flattened, 1)
        batch = memory.shape[0]

        task = self.task_project(self.task_encoder(task))

        anchors, valid = self.anchors(tuple(shapes), memory.device, memory.dtype)
        features = self.encoder_vision_features(
            torch.where(valid, memory, torch.tensor(0.0, dtype=memory.dtype, device=memory.device)))
        similarity = class_similarity(features, self.encoder_class_head(classes).permute(1, 2, 0))
        boxes = self.encoder_bbox_head(features) + anchors
        top = torch.topk(similarity.max(-1).values, spec.queries, dim=1).indices.view(-1)
        rows = (torch.arange(end=batch, dtype=top.dtype, device=top.device)
                .unsqueeze(-1).repeat(1, spec.queries).view(-1))
        reference = boxes[rows, top].view(batch, spec.queries, -1)
        embeddings = features[rows, top].view(batch, spec.queries, -1)

        reference = reference.sigmoid()
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            embeddings, task = layer(embeddings, task, reference, memory, shapes,
                                     self.query_position_head(reference))
            refined = torch.sigmoid(
                self.decoder_bbox_head[index](embeddings) + inverse_sigmoid(reference))
            if index == last:
                logits = class_similarity(
                    embeddings, self.decoder_class_head[index](classes).permute(1, 2, 0))
                return refined, logits
            reference = refined
        raise AssertionError("unreachable")
