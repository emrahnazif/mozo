# SPDX-License-Identifier: Apache-2.0
"""The numbers that define an EoMT variant, as frozen data.

Upstream spreads them over a YAML config per checkpoint (``configs/dinov2/...``), the defaults of
``models/eomt.py`` (four query blocks), of ``models/vit.py`` (the ViT-L backbone name, patch 16),
of timm's ``vit_*_patch14_reg4_dinov2`` constructors (width, depth, heads, LayerScale, four
registers, no position embedding on the prefix tokens) and of the Lightning modules (the panoptic
thresholds, 0.8 and 0.8). Written out here as one record per variant; the strict load in
:mod:`~mozo.vendors.eomt_deploy.checkpoint` holds the shapes against the checkpoint.

**Patch 16, though timm names the backbones ``patch14``.** ``models/vit.py`` re-creates the
DINOv2 backbone with ``patch_size=16``; the checkpoints carry 16x16 patch embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["COCO_STUFF", "MEAN", "PATCH", "REGISTERS", "SPECS", "STD", "VARIANTS", "Spec"]

#: ``models/vit.py``: ``pixel_mean`` / ``pixel_std``, ImageNet's statistics, applied inside the
#: network to pixels already divided by 255.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: Patch side, as ``models/vit.py`` passes it to timm.
PATCH = 16

#: Register tokens after the class token, as timm's ``*_reg4_dinov2`` constructors build them.
REGISTERS = 4

#: COCO panoptic's stuff classes, in upstream's contiguous ids: every config's ``stuff_classes``.
#: The 80 thing classes come first.
COCO_STUFF = tuple(range(80, 133))


@dataclass(frozen=True)
class Spec:
    """One published variant, in full.

    Attributes:
        variant: mozo's name for it.
        repository: The authors' Hugging Face repository, which upstream's configs name as the
            logger ``name`` and its notebook downloads from.
        config: Upstream's YAML for it, under ``configs/dinov2/``.
        task: ``"panoptic"`` or ``"semantic"``: which post-processing the checkpoint was
            evaluated with.
        width: Transformer width.
        depth: Blocks.
        heads: Attention heads, 64 channels each.
        queries: Learned queries, prepended to the tokens before the last ``query_blocks`` blocks.
        query_blocks: How many of the last blocks see the queries (``num_blocks`` upstream).
        classes: Classes, not counting the no-object slot the class head adds.
        image_size: The square input the network runs at.
        stuff: Classes merged into one segment per class in panoptic post-processing.
    """

    variant: str
    repository: str
    config: str
    task: str
    width: int
    depth: int
    heads: int
    queries: int
    query_blocks: int
    classes: int
    image_size: int
    stuff: tuple[int, ...] = ()

    @property
    def grid(self) -> int:
        """Patches per side."""
        return self.image_size // PATCH

    @property
    def prefix(self) -> int:
        """Tokens between the queries and the patches: the class token and the registers."""
        return 1 + REGISTERS


def _panoptic(variant: str, repository: str, config: str, width: int, depth: int, heads: int,
              query_blocks: int, image_size: int = 640) -> Spec:
    return Spec(variant, repository, f"coco/panoptic/{config}", "panoptic", width, depth, heads,
                queries=200, query_blocks=query_blocks, classes=133, image_size=image_size,
                stuff=COCO_STUFF)


#: What mozo carries: the COCO panoptic S, B and L at 640, L at 1280, and the ADE20K semantic L.
#: Every code path the DINOv2 checkpoints have; see ``PROVENANCE.md`` for what was left out.
SPECS: dict[str, Spec] = {
    spec.variant: spec
    for spec in (
        _panoptic("coco-panoptic-small", "tue-mps/coco_panoptic_eomt_small_640_2x",
                  "eomt_small_640_2x.yaml", 384, 12, 6, query_blocks=3),
        _panoptic("coco-panoptic-base", "tue-mps/coco_panoptic_eomt_base_640_2x",
                  "eomt_base_640_2x.yaml", 768, 12, 12, query_blocks=3),
        _panoptic("coco-panoptic-large", "tue-mps/coco_panoptic_eomt_large_640",
                  "eomt_large_640.yaml", 1024, 24, 16, query_blocks=4),
        _panoptic("coco-panoptic-large-1280", "tue-mps/coco_panoptic_eomt_large_1280",
                  "eomt_large_1280.yaml", 1024, 24, 16, query_blocks=4, image_size=1280),
        Spec("ade20k-semantic-large", "tue-mps/ade20k_semantic_eomt_large_512",
             "ade20k/semantic/eomt_large_512.yaml", "semantic", 1024, 24, 16,
             queries=100, query_blocks=4, classes=150, image_size=512),
    )
}

VARIANTS = list(SPECS)
