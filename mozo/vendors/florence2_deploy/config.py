# SPDX-License-Identifier: MIT
"""Florence-2's geometry and its fixed decoding settings, written out.

Upstream reads these from ``config.json`` beside the weights (``Florence2Config``, with a
``Florence2VisionConfig`` and a BART-style ``Florence2LanguageConfig`` inside it). mozo publishes
the weights alone, so every number a forward pass depends on is stated here; a strict load holds the
shapes against the checkpoint.

Two geometries, four checkpoints. ``large`` is not ``large-ft`` with other weights: Microsoft
replaced ``Florence-2-large`` in December 2024 with a "continued pretrained ... 4k context length"
checkpoint whose text model has 4,096 positions; ``large-ft`` kept 1,024. See ``PROVENANCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DECODING", "IMAGE_SIZE", "SPECS", "Decoding", "LanguageSpec", "Spec", "VARIANTS",
           "VisionSpec"]

#: Every image is resized to this square, aspect not kept (upstream's ``CLIPImageProcessor``).
IMAGE_SIZE = 768


@dataclass(frozen=True)
class VisionSpec:
    """The DaViT tower: four stages, each a strided convolution then spatial/channel blocks.

    Args:
        dims: Width of each stage.
        heads: Window-attention heads per stage.
        groups: Channel-attention groups per stage.
        depths: Spatial+channel block pairs per stage.
        window: Window side, in tokens, for the spatial blocks.
        patch: Convolution kernel per stage.
        stride: Convolution stride per stage.
        padding: Convolution padding per stage.
        prenorm: Whether a stage's convolution normalises its input (else its output).
        projection: Width of the image features handed to the language model.
    """

    dims: tuple[int, int, int, int]
    heads: tuple[int, int, int, int]
    groups: tuple[int, int, int, int]
    depths: tuple[int, int, int, int] = (1, 1, 9, 1)
    window: int = 12
    patch: tuple[int, int, int, int] = (7, 3, 3, 3)
    stride: tuple[int, int, int, int] = (4, 2, 2, 2)
    padding: tuple[int, int, int, int] = (3, 1, 1, 1)
    prenorm: tuple[bool, bool, bool, bool] = (False, True, True, True)
    projection: int = 768
    #: Rows of the learned 2-D position table (per axis) and of the cosine temporal table.
    pos_rows: int = 50
    temporal_rows: int = 100


@dataclass(frozen=True)
class LanguageSpec:
    """The BART encoder-decoder.

    Args:
        width: ``d_model``.
        layers: Encoder layers, and decoder layers (always equal here).
        heads: Attention heads, encoder and decoder.
        ffn: Feed-forward width.
        positions: Learned positions per side (the table has two more rows: BART's offset).
        vocab: Vocabulary, 50,265 BPE tokens plus the 1,024 Florence-2 adds.
    """

    width: int
    layers: int
    heads: int
    ffn: int
    positions: int
    vocab: int = 51289


@dataclass(frozen=True)
class Spec:
    """One published checkpoint's geometry."""

    vision: VisionSpec
    language: LanguageSpec


_BASE_VISION = VisionSpec(dims=(128, 256, 512, 1024), heads=(4, 8, 16, 32), groups=(4, 8, 16, 32),
                          projection=768)
_LARGE_VISION = VisionSpec(dims=(256, 512, 1024, 2048), heads=(8, 16, 32, 64),
                           groups=(8, 16, 32, 64), projection=1024)

SPECS: dict[str, Spec] = {
    "base": Spec(_BASE_VISION, LanguageSpec(768, 6, 12, 3072, 1024)),
    "large": Spec(_LARGE_VISION, LanguageSpec(1024, 12, 16, 4096, 4096)),
    "base-ft": Spec(_BASE_VISION, LanguageSpec(768, 6, 12, 3072, 1024)),
    "large-ft": Spec(_LARGE_VISION, LanguageSpec(1024, 12, 16, 4096, 1024)),
}

VARIANTS = tuple(SPECS)


@dataclass(frozen=True)
class Decoding:
    """What ``generate()`` does with no arguments, read off every checkpoint's ``text_config``.

    3-beam search, length penalty 1, stopping as soon as three hypotheses are complete, never
    repeating a 3-token sequence, ``<s>`` forced first and ``</s>`` forced at the length limit.
    Microsoft's examples pass ``num_beams=3`` explicitly and get the same thing.
    """

    num_beams: int = 3
    length_penalty: float = 1.0
    early_stopping: bool = True
    no_repeat_ngram: int = 3
    decoder_start: int = 2
    bos: int = 0
    eos: int = 2
    pad: int = 1
    max_new_tokens: int = 1024


DECODING = Decoding()
