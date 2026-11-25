# Provenance

This package is a deployment-only extraction of the DINOv2 backbones from
[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).

| | |
|---|---|
| Upstream repository | `https://github.com/facebookresearch/dinov2` |
| Upstream commit | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` (2026-06-03, the code `torch.hub.load` serves) |
| Extracted | 2026-11-25 |
| Verified against | `torch` 2.11.0, `torchvision` 0.26.0, Python 3.10, CPU, `MKL_CBWR=AVX2,STRICT` |
| Code licence | Apache-2.0 (`LICENSE`, relicensed from CC-BY-NC on 2023-08-31, commit 81b2b64) |
| Weights licence | Apache-2.0: README ("DINOv2 code and model weights are released under the Apache License 2.0"), `MODEL_CARD.md`, and `license: apache-2.0` on every `facebook/dinov2-*` Hugging Face card, ungated |

## The reference, and why it is this one

The authors publish PyTorch, so the authors' code is the reference (VENDORING §1.1) -- not the
`transformers` port, which was measured to differ from it twice over:

- **Its image processor crops a pixel to the left.** `BitImageProcessor` centres the crop at
  `(w - 224) // 2`; upstream's torchvision transform at `int(round((w - 224) / 2))`. Whenever the
  margin is odd -- a 1920x1281 photograph resizes to 383x256 -- the pixels differ by up to 2.8.
- **Its plain models interpolate the position embedding differently.** `Dinov2Model` resizes to an
  explicit size; upstream (and every plain checkpoint's training) uses a scale factor of
  `(16 + 0.1) / 37`. At 224 x 224, the resolution the evaluation transform produces, the class
  token is 0.13 (giant) to 0.25 (base) away from upstream's. At 518 x 518, where no interpolation
  happens, and for the register models at every size, `transformers`' SDPA path is bit-identical.

End to end, the `transformers` pipeline's vector for `example.jpg` has cosine 0.989 (small), 0.991
(base) and 0.988 (small-reg) with upstream's. An index must not mix the two.

**HEAD, not an older commit.** Upstream replaced its hand-written attention with
`scaled_dot_product_attention` on 2025-06-11 (12592a9, "dino.txt inference code"). The published
k-NN and linear numbers predate that. HEAD and the last commit before it (85a2460) differ by
1.0e-05 to 4.4e-05 on the class token. mozo is identical to HEAD and claims nothing about the
older arithmetic.

## What the checkpoints are

The authors' `dinov2_vit{s,b,l,g}14[_reg4]_pretrain.pth` from `dl.fbaipublicfiles.com/dinov2/`,
backbone only. mozo fetches Meta's Hugging Face mirrors instead (`facebook/dinov2-*`,
`facebook/dinov2-with-registers-*`) because those carry a sha256 and the authors' links carry only
a multipart ETag; renamed back to the authors' key layout, every tensor is `torch.equal` to the
authors' file (175, 176, 343 and 567 tensors for small, small-reg, large and giant;
`tools/verify/dinov2.py` checks it on every run).

`mask_token` is in every checkpoint and used only in training (masked image modelling). The
network declares it so the strict load accounts for it; it never touches a forward pass.

## Variants

Six of upstream's eight. Downloads in the 30 days to 2026-11-25 (Hugging Face API), Meta's repos and
timm's:

| variant | upstream | width | params | `facebook/` | `timm/` |
|---|---|---|---|---|---|
| `small` | `dinov2_vits14` | 384 | 22M | 2.37M | 1.26M |
| `base` | `dinov2_vitb14` | 768 | 87M | 3.97M | 0.80M |
| `large` | `dinov2_vitl14` | 1024 | 304M | 0.65M | 0.18M |
| `giant` | `dinov2_vitg14` | 1536 | 1.14B | 0.26M | 7.5k |
| `small-reg` | `dinov2_vits14_reg` | 384 | 22M | 73k | 435k |
| `base-reg` | `dinov2_vitb14_reg` | 768 | 87M | 291k | 865k |

Left out: `large-reg` (49k + 157k) and `giant-reg` (8k + 9k, the least used of all eight). Between
them the six exercise every distinct path: the plain position-embedding interpolation (scale
factor with a 0.1 nudge, no antialiasing), the register path (four tokens after the class token,
interpolation to an explicit size with antialiasing), and giant's SwiGLU feed-forward.

## What was taken

Verbatim:

```
layers/{layer_scale,mlp,patch_embed}.py
```

Modified, each saying so in its header:

- `layers/attention.py` -- `Attention` only, its forward verbatim. Upstream's backbones use
  `MemEffAttention`, which takes xFormers' `memory_efficient_attention` whenever `xformers` is
  importable (unless `XFORMERS_DISABLED` is set) and falls through to exactly this class
  otherwise. The xFormers kernel is different arithmetic and does not run on the CPU, so
  upstream's vectors depend on what else happens to be installed. Here there is one path.
- `layers/swiglu_ffn.py` -- `SwiGLUFFNFused` always subclasses `SwiGLUFFN`, upstream's choice
  without xFormers; class bodies verbatim. `SwiGLUFFNAligned` (dino.txt) dropped.
- `layers/block.py` -- `Block` only, its inference branch. The hub backbones build with
  `drop_path_rate=0`, so stochastic depth never runs; `NestedTensorBlock`'s list path needs
  xFormers and serves training.
- `vision_transformer.py` -- `interpolate_pos_encoding`, `prepare_tokens_with_masks`,
  `forward_features` and `forward` unchanged; weight initialisation, block chunking (the hub
  passes `block_chunks=0`), the list path, `get_intermediate_layers` and the channel-adaptive
  path (Cell-DINO, non-commercial) dropped. Upstream's own names are kept, including
  `B, nc, w, h = x.shape`, which is really (B, C, H, W).
- the import of `dinov2.layers` made relative.

Written for this package:

- `config.py` -- the six variants as frozen records, gathering what upstream spreads across the
  `vit_*` constructors, the hub's `_make_dinov2_model` defaults and its `*_reg` entries.
- `checkpoint.py` -- builds the network on the `meta` device with the hub's keyword arguments and
  assigns the checkpoint into it, `strict=True`.
- `image.py` -- `make_classification_eval_transform`, through the same torchvision functions.
- `predictor.py` -- one image per forward, then `F.normalize(x, dim=1, p=2)`, which is upstream's
  `ModelWithNormalize` (`dinov2/eval/utils.py`), the wrapper its k-NN evaluation embeds through.

## Where this diverges, and why

- **One image per forward pass.** Upstream leaves batching to the caller. On the CPU, one forward
  over nine images differs from nine single forwards by 1.5e-05 (base) and 1.8e-05 (large) --
  MKL picks kernels by shape; with `MKL_CBWR=AVX2,STRICT` the two agree, but no server runs that
  way -- and on CUDA batched kernels differ too. A stored vector must not depend on what it was
  encoded beside, so every image runs alone. On an RTX 3060 that costs 1.4x (base) to 1.7x
  (small) of the backbone's throughput against a batch of 32.
- **Nothing else.** Pixels, tokens, every block, the norm and the vector are upstream's bit for bit.

## What was deliberately left behind

Training, evaluation scripts, the linear classification heads (`*_lc`), the depth and
segmentation heads, dino.txt, notebooks, Cell-DINO, Channel-Adaptive DINO and XRay-DINO (the last
three under non-commercial terms), `large-reg` and `giant-reg`, and patch features: the network
computes them (`forward_features`), but mozo returns only the class token.

## Verification

`tools/verify/dinov2.py`, against a checkout of the commit above with `XFORMERS_DISABLED=1` set
before import, both sides on the CPU, `MKL_CBWR=AVX2,STRICT`. Per variant: every checkpoint tensor
against the authors' `.pth`; then per image the pixels, the tokens after the position embedding,
every block's output, the final norm, the class token and the normalised vector (read off
`Encoder.encode_image`, all images in one call); then the backbone alone on random input at
518 x 518 (no interpolation) and 224 x 336 (a non-square grid). Images: `example.jpg`, an odd-margin
343x256, 383x257, black, 9x13, a single pixel, 2000x120, 90x1500, greyscale, and 20 random COCO
val2017 photographs.

| variant | comparisons | exact |
|---|---|---|
| small | 670 | 670 |
| base | 670 | 670 |
| large | 1,186 | 1,186 |
| giant | 1,874 | 1,874 |
| small-reg | 671 | 671 |
| base-reg | 671 | 671 |
| **total** | **5,742** | **5,742** |

`--mozo` adds the adapter through the registry and the weights tree, on arrays and on file paths.

The gate was made to fail on purpose, eleven ways, each caught first at the stage it should be
(`--falsify all`, on `base`): the crop at `(w - 224) // 2`, a squashing resize, BGR, the 0.1
offset removed, antialiasing flipped (pixels / tokens); block 0's LayerNorm eps 1e-6 -> 1e-5, a
GELU tanh approximation, the attention scale x (1 + 1e-6) (block 0); block 3's LayerScale skipped
(block 3); the final norm skipped (norm); the vector left unnormalised (vector).
