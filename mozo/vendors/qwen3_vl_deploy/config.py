# SPDX-License-Identifier: Apache-2.0
"""The numbers that define a Qwen3-VL variant, as frozen data.

Upstream reads these from the published ``config.json`` through ``Qwen3VLConfig``, which fills
whatever a file omits from class defaults. Written out here instead, as every family in this tree
writes its geometry down: a spec that is inferred cannot be checked. The strict load in
:mod:`~mozo.vendors.qwen3_vl_deploy.checkpoint` holds the shapes against the checkpoint's.

The two variants share the vision tower exactly and differ in the text decoder: depth, width, and
how many query heads share one key/value head (2 for ``2b``, 4 for ``4b``).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CONTEXT", "SPECS", "VARIANTS", "Spec", "TextSpec", "VisionSpec", "Tokens"]

#: ``max_position_embeddings``: the longest sequence, prompt and answer together, in tokens.
CONTEXT = 262_144


@dataclass(frozen=True)
class VisionSpec:
    """The vision tower. The same for every variant carried.

    Attributes:
        depth: Transformer blocks.
        width: Their width.
        heads: Attention heads; the head dimension is ``width // heads``.
        mlp: Hidden width of each block's MLP (``gelu_pytorch_tanh``).
        out_width: What the mergers project to: the text decoder's width.
        patch: Side of one patch, in pixels.
        temporal_patch: Frames per patch. An image is one frame repeated, so every patch is the same
            16x16 twice.
        merge: Side of the square of patches merged into one text token.
        grid: Side of the learned square position table (``num_position_embeddings ** 0.5``).
        deepstack: Blocks whose output also goes, through its own merger, into the text decoder.
        rope_theta: Base of the vision tower's 2-D rotary embedding.
        norm_eps: LayerNorm epsilon in the blocks and mergers.
    """

    depth: int = 24
    width: int = 1024
    heads: int = 16
    mlp: int = 4096
    out_width: int = 2048
    patch: int = 16
    temporal_patch: int = 2
    merge: int = 2
    grid: int = 48
    deepstack: tuple[int, ...] = (5, 11, 17)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @property
    def factor(self) -> int:
        """Pixels per merged token along a side: images are resized to multiples of this."""
        return self.patch * self.merge


@dataclass(frozen=True)
class TextSpec:
    """The text decoder.

    Attributes:
        layers: Decoder layers.
        width: Hidden width.
        heads: Query heads.
        kv_heads: Key/value heads; ``heads // kv_heads`` queries share each.
        head_dim: Width of one head. Not ``width // heads`` for ``4b`` (2560 / 32 = 80, but 128).
        mlp: Hidden width of the SiLU-gated MLP.
        vocab: Rows of the token embedding, which is also the (tied) output projection.
        rope_theta: Rotary base.
        mrope_section: Frequencies given to time, height and width, interleaved.
        norm_eps: RMSNorm epsilon, everywhere including the per-head q/k norms.
    """

    layers: int
    width: int
    heads: int
    kv_heads: int
    mlp: int
    head_dim: int = 128
    vocab: int = 151_936
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    norm_eps: float = 1e-6


@dataclass(frozen=True)
class Spec:
    """One published variant: its upstream name and both towers."""

    variant: str
    upstream: str
    vision: VisionSpec
    text: TextSpec


@dataclass(frozen=True)
class Tokens:
    """The ids the prompt template and the stopping rule use, from the published tokenizer."""

    endoftext: int = 151_643
    im_end: int = 151_645
    image_pad: int = 151_655

    @property
    def stop(self) -> tuple[int, ...]:
        """``generation_config.json``'s ``eos_token_id``: either ends the answer."""
        return (self.im_end, self.endoftext)


SPECS: dict[str, Spec] = {
    "2b": Spec("2b", "Qwen3-VL-2B-Instruct", VisionSpec(out_width=2048),
               TextSpec(layers=28, width=2048, heads=16, kv_heads=8, mlp=6144)),
    "4b": Spec("4b", "Qwen3-VL-4B-Instruct", VisionSpec(out_width=2560),
               TextSpec(layers=36, width=2560, heads=32, kv_heads=8, mlp=9728)),
}

VARIANTS = tuple(SPECS)
