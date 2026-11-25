# ------------------------------------------------------------------------
# Depth Anything 3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The interface of a graph exported from :class:`~.net.DepthAnything3Net`.

Written for this package; upstream exports nothing. The graph takes the pixels and the tensors
:meth:`~.net.DepthAnything3Net.shape_inputs` computes from the shape -- float arithmetic on a size
cannot be traced, so they are inputs rather than part of the graph -- and returns the network's
raw outputs. One statement of the names and the shape range, for the exporter and the runner.
"""

from __future__ import annotations

__all__ = ["MAX_PATCHES", "MIN_PATCHES", "input_names", "output_names", "profile"]

from .config import PATCH_SIZE, ModelSpec

#: Patches per side a graph accepts. The long side is always 36 (504 px); the short side may be
#: as small as 3 (42 px), an aspect ratio of 12:1. Below 3 the exporter cannot keep the size
#: symbolic, so a more elongated image is refused by graph runtimes.
MIN_PATCHES = 3
MAX_PATCHES = 36


def input_names(spec: ModelSpec) -> tuple[str, ...]:
    """``pixels``, ``grid``, and for a model with UV embeddings ``uv0``..``uv4``."""
    return ("pixels", "grid") + (() if spec.sky else tuple(f"uv{i}" for i in range(5)))


def output_names(spec: ModelSpec) -> tuple[str, ...]:
    """``depth``, and ``sky`` for a model with a sky head."""
    return ("depth", "sky") if spec.sky else ("depth",)


def _shapes(spec: ModelSpec, h: int, w: int) -> list[tuple[int, ...]]:
    width = spec.head_dim_in // (2 if spec.cat_token else 1)
    shapes = [(1, 3, PATCH_SIZE * h, PATCH_SIZE * w), (1, width, h, w)]
    if not spec.sky:
        shapes += [(1, c, h, w) for c in spec.out_channels]
        shapes.append((1, spec.features // 2, PATCH_SIZE * h, PATCH_SIZE * w))
    return shapes


def profile(spec: ModelSpec, preferred: tuple[int, int] = (27, 36)) -> dict[str, tuple]:
    """``{input: (min, preferred, max)}`` shapes for a TensorRT optimisation profile."""
    low = _shapes(spec, MIN_PATCHES, MIN_PATCHES)
    mid = _shapes(spec, *preferred)
    high = _shapes(spec, MAX_PATCHES, MAX_PATCHES)
    return {name: (a, b, c) for name, a, b, c in zip(input_names(spec), low, mid, high)}
