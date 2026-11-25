# SPDX-License-Identifier: Apache-2.0
"""The numbers that define a Granite Embedding R2 variant, as frozen data.

Upstream reads these from the published ``config.json`` through ``ModernBertConfig``, which fills
whatever a file omits from its own defaults and derives the rest -- the sliding window from
``local_attention``, the layer pattern from ``global_attn_every_n_layers``. Written out here
instead, for the reason every family in this tree writes its geometry down: a spec that is
inferred cannot be checked, and a variant loaded with the wrong geometry inferred is a silent
wrong answer rather than a failed load. The strict load in
:mod:`~mozo.vendors.granite_embedding_r2_deploy.checkpoint` holds the shapes here against the
checkpoint's.

**Two fields of the published config are inert, and are not carried.** ``hidden_act: "gelu"`` and
``position_embedding_type: "relative_key_query"`` are BERT's names. ModernBERT reads
``hidden_activation`` and uses rotary positions, so neither is ever consulted upstream. Carrying
them here would invite someone to implement them.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "VARIANTS", "Spec"]


@dataclass(frozen=True)
class Spec:
    """One published variant, in full.

    Attributes:
        variant: mozo's name for it.
        upstream: IBM's name for it. Where it is *published* is a fact about publishing rather than
            about inference, so it lives in ``tools/`` and not here.
        width: Transformer width, which is also the embedding's dimension: the vector is the
            ``[CLS]`` position's final hidden state, with no projection after it.
        layers: Encoder blocks.
        heads: Attention heads. The head dimension is ``width // heads``.
        mlp: Hidden width of the gated MLP. Its input projection is twice this, because the gate
            and the value come out of one matrix.
        vocab: Rows in the token embedding. English: the 50,280 pieces of the byte-pair
            vocabulary plus the 88 added tokens after them. Multilingual: 179,934 pieces and 66
            added tokens.
        window: How far either side of itself a token sees in a sliding-attention layer:
            ``abs(query - key) <= window``. Half of the published ``local_attention``, which is the
            *whole* window.
        global_every: Every this-many layers, starting with the first, attends to the whole input.
            The rest use the sliding window.
        global_theta: Rotary base for the full-attention layers.
        local_theta: Rotary base for the sliding-attention layers.
        norm_eps: LayerNorm epsilon. torch's default, as it happens, but stated rather than
            inherited, because the next variant need not agree.
        activation: The gated MLP's activation: ``"gelu"`` (exact, erf) for English, ``"silu"``
            for multilingual. The published config's ``hidden_activation``.
        context: The longest input, in tokens, the template's special tokens included.
            ``max_position_embeddings`` in the published config, and the length the model was
            trained to. Upstream does not enforce it -- rotary positions have no table to run off
            the end of, so a longer input runs silently on positions the model never saw. mozo
            refuses instead.
        vocabulary: The tokenizer asset, by name: the two variants share no tokenizer.
    """

    variant: str
    upstream: str
    width: int
    layers: int
    heads: int
    mlp: int
    vocab: int
    window: int
    global_every: int
    global_theta: float
    local_theta: float
    norm_eps: float
    activation: str
    context: int
    vocabulary: str

    @property
    def head_dim(self) -> int:
        """Width of one attention head, which is also the width the rotary embedding turns."""
        return self.width // self.heads

    def is_global(self, layer: int) -> bool:
        """Whether *layer* attends to the whole input. Layer 0 always does."""
        return layer % self.global_every == 0


#: What mozo carries, two of IBM's four R2 embedders:
#:
#: - ``small``, the small English model, which takes 99% of the downloads across IBM's two English
#:   R2 embedders (6.66M a month against 54K, Hugging Face, November 2026). The other,
#:   ``granite-embedding-english-r2``, is the same architecture deeper and wider -- one ``Spec`` and
#:   a checkpoint, no new code -- and exercises no path this one does not.
#: - ``multilingual-small``, the 97M multilingual model: 79% of the multilingual R2 downloads
#:   (235K a month against 61K). The larger ``granite-embedding-311m-multilingual-r2`` is absent on
#:   licence: its tokenizer is subject to the Gemma Terms of Use.
#:
#: See ``PROVENANCE.md``.
SPECS: dict[str, Spec] = {
    spec.variant: spec
    for spec in (
        Spec("small", "granite-embedding-small-english-r2", width=384, layers=12, heads=12,
             mlp=1536, vocab=50368, window=64, global_every=3, global_theta=80000.0,
             local_theta=10000.0, norm_eps=1e-5, activation="gelu", context=8192,
             vocabulary="granite_bpe"),
        Spec("multilingual-small", "granite-embedding-97m-multilingual-r2", width=384, layers=12,
             heads=12, mlp=1536, vocab=180000, window=64, global_every=3, global_theta=150000.0,
             local_theta=160000.0, norm_eps=1e-5, activation="silu", context=32768,
             vocabulary="granite_multilingual_bpe"),
    )
}

VARIANTS = list(SPECS)
