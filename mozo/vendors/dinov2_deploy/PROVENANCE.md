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

## Runtimes

Measured 2026-11-25. GPU numbers on an RTX 3060 12 GB (vision-003, torch 2.13.0+cu126, TensorRT
11.5, onnxruntime-gpu 1.21.1), CPU numbers on its i5-12400F, against torch fp32 with TF32 off.
"Network" is the backbone alone on one preprocessed image; "call" is `encode_image` on a decoded
1920x1281 photograph, most of which is the PIL resize on the CPU. Parity on 500 random COCO val2017
photographs against torch fp32; k-NN is the authors' protocol (cosine, k = 20, `exp(sim / 0.07)`
votes) on Imagenette 320 px, 100 train images a class as the index, all 3,925 val images as queries.

| variant | runtime | network | call | max abs delta | worst cosine | 10-NN kept | k-NN top-1 |
|---|---|---|---|---|---|---|---|
| small | torch fp32 | 4.21 ms | 17.6 ms | 0 | 1 | 1 | 0.9924 |
| | torch fp16 autocast | 2.43 | 16.1 | 7.7e-04 | 0.999986 | 0.9992 | 0.9924 |
| | tensorrt-fp32 | 2.90 | 16.3 | 1.1e-06 | 1.000000 | 1 | 0.9924 |
| | tensorrt-fp16 | 1.09 | 14.7 | 3.7e-03 | 0.999722 | 0.9958 | 0.9924 |
| small-reg | torch fp32 | 4.33 | 17.7 | 0 | 1 | 1 | 0.9918 |
| | tensorrt-fp32 | 2.94 | 16.4 | 1.0e-06 | 1.000000 | 1 | 0.9918 |
| | tensorrt-fp16 | 1.16 | 14.8 | 5.7e-03 | 0.999537 | 0.9940 | 0.9918 |
| base | torch fp32 | 11.17 | 23.6 | 0 | 1 | 1 | 0.9964 |
| | torch fp16 autocast | 6.30 | 19.7 | 7.1e-04 | 0.999982 | 0.9978 | 0.9964 |
| | tensorrt-fp32 | 9.02 | 21.6 | 6.4e-07 | 1.000000 | 1 | 0.9964 |
| | tensorrt-fp16 | 2.66 | 16.4 | 4.4e-03 | 0.999440 | 0.9954 | 0.9964 |
| base-reg | torch fp32 | 11.33 | 23.5 | 0 | 1 | 1 | 0.9962 |
| | tensorrt-fp32 | 9.03 | 21.6 | 1.1e-06 | 1.000000 | 1 | 0.9962 |
| | tensorrt-fp16 | 2.67 | 16.4 | 4.1e-03 | 0.999556 | 0.9958 | 0.9962 |
| large | torch fp32 | 35.33 | 46.7 | 0 | 1 | 1 | 0.9967 |
| | torch fp16 autocast | 20.44 | 33.2 | 6.5e-03 | 0.998600 | 0.9984 | 0.9967 |
| | tensorrt-fp32 | 30.27 | 41.8 | 1.9e-05 | 1.000000 | 1 | 0.9967 |
| | tensorrt-fp16 | 7.43 | 20.7 | 9.1e-03 | 0.997065 | 0.9954 | 0.9967 |
| giant | torch fp32 | 122.33 | 132.8 | 0 | 1 | 1 | 0.9977 |

ONNX Runtime on CUDA (100 COCO images): small 3.79 / base 9.87 / large 27.22 ms on the network
in fp32, but worst cosine 0.999783 on large (max delta 2.2e-03) -- not torch's arithmetic; fp16
3.09 / 5.65 / 15.42 ms. On the CPU (i5-12400F) ONNX Runtime fp32 is no faster than torch: 23.9
against 24.9 ms (small), 94.3 against 97.3 (base), and slower on large, 477 against 341. Parity
through ONNX Runtime on the CPU: 5.3e-07 to 8.3e-07 on the fixtures.

What mozo offers from this: `onnx-fp32` and `onnx-fp16` graphs for every variant but giant, as the
sources of TensorRT engines. On CUDA `auto` builds `tensorrt-fp32` when TensorRT is installed: it
keeps every nearest neighbour and runs the network 1.17x (large) to 1.45x (small) faster.
`tensorrt-fp16` is offered on request only. torch fp16 is not offered: TensorRT fp16 is faster
at similar drift, and an `auto` that changed vectors by runtime would split an index. The ONNX
graphs are never chosen by `auto` on CPU (torch is preferred and no slower) and are runnable by
name.

Not measured: giant under fp16 (the bench's second copy of the network did not fit beside the
first in 12 GB); giant has no graph at all, being past protobuf's 2 GB limit as one file. Imagenette
k-NN is near its ceiling (99.2-99.8%) and moved on no runtime; the neighbour column is the more
sensitive measure.

The graph is a fixed 1x3x224x224. The register models' position-embedding resize is antialiased
bicubic, which ONNX cannot express, so the export computes the 224x224 embedding once with the
model's own code and the graph adds the constant.

Latency against upstream, same weights, interleaved, synchronised: small 4.19 vs 4.20 ms, base
10.99 vs 10.99, large 34.40 vs 34.45, giant 119.77 vs 119.82 on the RTX 3060; small 26.09 vs
25.97, base 83.79 vs 83.80 on the i5-12400F. Outputs `torch.equal` on CUDA too.

On CUDA, a batch of 9 differs from single forwards by 1.3e-05 to 3.1e-05; one image per forward
costs 1.26x (giant) to 1.71x (small-reg) of a batch of 32's throughput on the network.

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
