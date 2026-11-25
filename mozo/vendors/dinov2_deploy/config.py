# SPDX-License-Identifier: Apache-2.0
"""The numbers that define a DINOv2 variant, as frozen data.

Upstream spreads them across three places: the ``vit_*`` constructors (width, depth, heads), the
hub's ``_make_dinov2_model`` defaults (518 px, patch 14, LayerScale on, no block chunking), and the
``*_reg`` hub entries, which switch the position-embedding interpolation along with the register
tokens. Written out here as one record per variant, because the last of those is exactly the kind
of geometry that looks derivable and is not: the plain models interpolate with a scale factor
nudged by 0.1 and no antialiasing, the register models to an explicit size *with* antialiasing,
and ``transformers`` gets the plain one wrong (``PROVENANCE.md``). The strict load in
:mod:`~mozo.vendors.dinov2_deploy.checkpoint` holds the shapes here against the checkpoint's.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CROP", "MEAN", "RESIZE", "SPECS", "STD", "VARIANTS", "Spec"]

#: Upstream's evaluation transform (``dinov2/data/transforms.py``,
#: ``make_classification_eval_transform``): short side to 256, bicubic, then the central 224.
RESIZE = 256
CROP = 224

#: ImageNet's statistics, as upstream's ``IMAGENET_DEFAULT_MEAN`` / ``_STD``.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Spec:
    """One published variant, in full.

    Attributes:
        variant: mozo's name for it.
        upstream: The torch-hub entry point, which also names the checkpoint file.
        arch: Which of upstream's ``vit_*`` constructors.
        width: Transformer width, which is also the embedding's dimension: the vector is the
            final-norm class token, with no projection after it.
        depth: Blocks.
        heads: Attention heads; 64 channels each in every variant.
        ffn: ``"mlp"`` (GELU) or ``"swiglufused"`` (giant: SwiGLU, hidden width rounded to a
            multiple of 8 from two thirds of 4x).
        registers: Register tokens inserted after the class token, 0 or 4.
        interpolate_antialias: Antialias the bicubic position-embedding resize.
        interpolate_offset: Nudge added to the target grid before dividing by the trained one, to
            form a scale factor. 0 means "resize to an explicit size" instead.
        image_size: The resolution the position embedding was trained at: a 37x37 grid.
        patch: Patch side, in pixels.
    """

    variant: str
    upstream: str
    arch: str
    width: int
    depth: int
    heads: int
    ffn: str = "mlp"
    registers: int = 0
    interpolate_antialias: bool = False
    interpolate_offset: float = 0.1
    image_size: int = 518
    patch: int = 14


def _registers(spec: Spec) -> Spec:
    """The same backbone trained with four registers, as upstream's ``*_reg`` hub entries build it."""
    from dataclasses import replace

    return replace(spec, variant=f"{spec.variant}-reg", upstream=f"{spec.upstream}_reg",
                   registers=4, interpolate_antialias=True, interpolate_offset=0.0)


_SMALL = Spec("small", "dinov2_vits14", "vit_small", width=384, depth=12, heads=6)
_BASE = Spec("base", "dinov2_vitb14", "vit_base", width=768, depth=12, heads=12)

#: What mozo carries: six of upstream's eight. The four plain sizes, and the two most used of the
#: register models. ``large-reg`` and ``giant-reg`` exercise no path these do not. See
#: ``PROVENANCE.md`` for the download counts.
SPECS: dict[str, Spec] = {
    spec.variant: spec
    for spec in (
        _SMALL,
        _BASE,
        Spec("large", "dinov2_vitl14", "vit_large", width=1024, depth=24, heads=16),
        Spec("giant", "dinov2_vitg14", "vit_giant2", width=1536, depth=40, heads=24,
             ffn="swiglufused"),
        _registers(_SMALL),
        _registers(_BASE),
    )
}

VARIANTS = list(SPECS)
