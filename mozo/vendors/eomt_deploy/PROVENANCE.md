# Provenance

This package is a deployment-only extraction of [EoMT](https://github.com/tue-mps/eomt)
(Encoder-only Mask Transformer, "Your ViT is Secretly an Image Segmentation Model", CVPR 2025).

| | |
|---|---|
| Upstream repository | `https://github.com/tue-mps/eomt` |
| Upstream commit | `aa017c34afbe023de9462b17ae6067b52fef9867` (2026-10-12) |
| Backbone code | timm 1.0.15 (`huggingface/pytorch-image-models`), which upstream builds its ViT from |
| Extracted | 2026-11-25 |
| Verified against | `torch` 2.7.0, `torchvision` 0.22.0, `timm` 1.0.15, Python 3.12 (upstream's `requirements.txt`) |

## The reference, and why

The authors publish PyTorch, and their checkpoints and PQ/mIoU numbers come from their own
repository, so that repository is the reference. `transformers` has a port
(`EomtForUniversalSegmentation`); it was measured, not trusted:

- its network is `torch.equal` to upstream's on the same pixels (with `attn_implementation="sdpa"`;
  `eager` is up to 2.8e-03 off on the mask logits);
- its **default** image processor, `EomtImageProcessorFast` (what every repository's
  `preprocessor_config.json` names), resizes differently: up to 1.75e-02 on normalised pixels, and
  on the example photograph 2,120 (large) to 6,711 (small) pixels end up in another class, with one
  segment fewer on small;
- with `use_fast=False` its whole pipeline reproduces upstream's partition exactly.

So upstream is the target, and the `transformers` slow pipeline is a second witness in the gate (for
the one number upstream computes but does not return: each panoptic segment's score).

## What was taken

From `tue-mps/eomt` at `aa017c3`:

```
models/eomt.py            -> network.py   EoMT: queries, heads, upscale, forward, _predict
models/vit.py             -> network.py   ViT: the backbone and the pixel statistics
models/scale_block.py     -> network.py   ScaleBlock
training/lightning_module.py -> image.py        scale_img_size_*, resize_and_pad_imgs_instance_panoptic,
                                                window_imgs_semantic
                             -> postprocess.py  revert_resize_and_pad_logits_instance_panoptic,
                                                to_per_pixel_preds_panoptic, to_per_pixel_logits_semantic,
                                                revert_window_logits_semantic
training/mask_classification_panoptic.py         mask_thresh = overlap_thresh = 0.8 (postprocess.py)
configs/dinov2/**/*.yaml                         queries, query blocks, input size, stuff classes (config.py)
```

From timm 1.0.15, the inference path of what upstream's `ViT` instantiates for
`vit_{small,base,large}_patch14_reg4_dinov2` with `patch_size=16`: `VisionTransformer` (patch
embedding, class and register tokens, position embedding with `no_embed_class=True`, blocks, final
norm), `Block`, `Attention`, `LayerScale`, `Mlp`, `PatchEmbed`, `LayerNorm2d`. Module names are
upstream's, so the authors' state dict loads with `strict=True`.

## Modifications

- **Masked attention is not built.** Upstream's forward predicts masks before each of the last
  `num_blocks` blocks and turns them into an attention mask, then `_disable_attn_mask` replaces each
  mask with "attend everywhere" with probability `1 - attn_mask_probs`. Every published checkpoint
  carries `attn_mask_probs = 0`, and the authors' notebook builds the network with
  `masked_attn_enabled=False`. Both paths were measured `torch.equal` on the final logits and maps
  of three checkpoints. The buffer is kept so the strict load accounts for it; the intermediate
  predictions, `_attn_mask` and `_disable_attn_mask` are dropped.
- **Attention is always `scaled_dot_product_attention`.** timm chooses SDPA or a manual softmax
  from `TIMM_FUSED_ATTN` at import, and upstream's `EoMT._attn` follows that flag; SDPA is the
  default and what the gate compares.
- **DINOv3 / RoPE / `transformers`-backbone branches** of `models/eomt.py` and `models/vit.py` are
  dropped: no carried checkpoint uses them (DINOv3 is excluded for its licence).
- **Dropout, drop-path, patch dropout, `pos_drop`, `norm_pre`, the classifier head, the
  `torch.compiler.disable` decorators** are dropped: identities or unused at inference.
- **Post-processing that cannot exhaust memory** (`postprocess.py`). Upstream upsamples all 200
  queries' mask logits to the full picture before keeping the confident ones (about 2 GB of floats
  on a 1920x1281 photograph); here only the kept queries are upsampled, which gives the same numbers
  because bilinear interpolation treats each channel on its own. Upstream upsamples all 150 semantic
  class maps before the argmax (about 1.5 GB); here they are upsampled 16 at a time with a running
  maximum where a later class wins only when strictly greater, `torch.argmax`'s first-index rule.
  Both are in the gate, against upstream's own all-at-once functions.
- **The per-image loop** of upstream's batched functions is unrolled to one image; a semantic
  picture's windows still run as one batch, as upstream runs them.
- **Images come in as RGB `uint8` arrays** from `mozo.image.load_image`; upstream's data modules
  decode their own. The gate hands both sides the same array.

## Weights

The published checkpoint is upstream's network state dict (`encoder.backbone.*`, `encoder.pixel_*`,
`q`, `class_head`, `mask_head`, `upscale`, `attn_mask_probs`; 197 tensors for small and base, 365 for
large). `tools/fetch/eomt.py` builds it from the authors' `model.safetensors` (the `transformers`
layout, sha256 checked against the Hub), concatenating each block's `q/k/v_proj` back into `qkv`,
renaming the rest, adding the two pixel-statistic buffers the Hub's file lacks, and dropping the
loss's `criterion.empty_weight`. The gate holds every tensor of the result `torch.equal` to the
authors' own `pytorch_model.bin`.

| Variant | Repository @ commit | Config |
|---|---|---|
| `coco-panoptic-small` | `tue-mps/coco_panoptic_eomt_small_640_2x` @ `10f5326` | `coco/panoptic/eomt_small_640_2x.yaml` |
| `coco-panoptic-base` | `tue-mps/coco_panoptic_eomt_base_640_2x` @ `1ac4979` | `coco/panoptic/eomt_base_640_2x.yaml` |
| `coco-panoptic-large` | `tue-mps/coco_panoptic_eomt_large_640` @ `dcd130b` | `coco/panoptic/eomt_large_640.yaml` |
| `coco-panoptic-large-1280` | `tue-mps/coco_panoptic_eomt_large_1280` @ `d1bb114` | `coco/panoptic/eomt_large_1280.yaml` |
| `ade20k-semantic-large` | `tue-mps/ade20k_semantic_eomt_large_512` @ `aeb262f` | `ade20k/semantic/eomt_large_512.yaml` |

Class names are not in upstream (its datasets map COCO category ids to contiguous ids without
names); `labels.json` is each repository's own `config.json` `id2label`, at the same commit.

**Licence.** Code MIT (the repository's `LICENSE`). Weights MIT: `license: mit` on every model card,
ungated. The backbones were initialised from DINOv2 with registers (Apache-2.0). Not carried:
the DINOv3 checkpoints (their `pytorch_model.bin` are delta weights over DINOv3, under Meta's own
licence), EoMT-7B (Web-SSL DINO 7B backbone, CC-BY-NC-4.0), Cityscapes (its dataset licence forbids
commercial use of derivative works and names trained models as such), the giants (4.7 GB each, no
new code path but timm's SwiGLU MLP), the COCO instance checkpoints (overlapping masks, a third
post-processing), and ADE20K panoptic (no new code path).

**Training data**, for the user's judgement and not a licence on the weights: COCO 2017 panoptic
(annotations CC-BY-4.0, Flickr images under their owners' licences); ADE20K (annotations
BSD-3-Clause, images for non-commercial research and educational use under MIT CSAIL's terms).

## Verification

**On CUDA (the primary gate).** `tools/verify/eomt.py --device cuda`, upstream and this package in
one process on one RTX 3060 (12 GB, driver 550.144.03), torch 2.7.0+cu126 (upstream's pin), CUDA
12.6, cuDNN 9.5.1, under torch's default global settings (matmul TF32 off, cuDNN TF32 on, cuDNN
benchmark off), which mozo leaves alone: **41,566 of 41,566 comparisons exact** across the five variants
on the same 28 pictures, in fp32 and in fp16 autocast, each side run three times per picture and
byte-identical every time (30,240 of the comparisons are that determinism check). fp16 is gated against upstream's own half path, the notebook's
`autocast(dtype=torch.float16)` (and Lightning's `16-mixed`), which covers the network *and* the
post-processing, the same span `torch-fp16` covers. The CUDA stages add the strides of the
network's input (cuDNN chooses kernels by layout), each parameter's `requires_grad` and
inference-tensor flag (they choose torch's matmul path on CUDA), and the logits and arrays the
adapter's own network produces. Rerun on torch 2.13.0+cu126 (cuDNN 9.10.2), two runs per side: 26,446 of 26,446 exact.
Nineteen falsifiers in fp32 and eighteen in fp16 (37 runs), each caught first at its stage; the GPU-only
ones flip a production setting (channels-last input, SDPA's math backend, matmul TF32 on, cuDNN
TF32 off), build the network inside inference mode, or post-process outside autocast (on the
semantic variant: on the panoptic pictures that change moves nothing, so it cannot be caught there).

The GPU gate found three things the CPU gate cannot see, each fixed here to match upstream:
`torch-fp16` cast the logits to float32 before post-processing, where upstream's autocast keeps
them float16 through the upsample and the merge (it moved semantic pixels; on panoptic pictures it
never changed an answer); a picture needing no padding reached the network with the HWC strides of
the PIL buffer, where upstream's `torch.stack` makes it contiguous; and the network built on the
first `segment` call, inside `torch.inference_mode()`, got inference-tensor parameters that do not
require grad, which on CUDA changes a linear layer's bits on a semantic picture's batched,
non-contiguous queries (class logits 1.8e-05 apart in fp32). The network is now built outside
inference mode.

**On the CPU.** `tools/verify/eomt.py`, CPU, `MKL_CBWR=AVX2,STRICT`, in an environment pinned to upstream's
`requirements.txt`: **6,422 of 6,422 comparisons exact** across the five variants on 28 pictures each
(8 awkward fixtures -- portrait, exactly 640x640, a size where `round` and `floor` differ, 37x50
upsampled, 2000x120, black, grey -- and 20 random COCO val2017 photographs). Stages: every tensor of
the published checkpoint against the authors' `pytorch_model.bin`; the network's `uint8` input;
the patch embedding; the input of every block; the final norm's input; class and mask logits; the
post-processed partition (panoptic segment ids and classes, semantic classes) from this package's
memory-bounded post-processing against upstream's own functions; each panoptic score against the
`transformers` slow pipeline; and mozo's adapter (masks, classes, confidences) through the registry
and the weights tree. Fifteen falsifiers (`--falsify all`) are each caught first at their stage.
(Rerun after the GPU gate's fixes on vision-003's CPU, torch 2.7.0+cu126 with the GPU hidden; the
first run, before the `layout` and `requires_grad` stages existed, counted 5,576 on vision-002.)

## Runtimes

Measured on an RTX 3060 (12 GB) with torch 2.13 and TensorRT 11.5: PQ on 500 COCO val2017 panoptic
images, mIoU on all 2,000 ADE20K validation images. `torch-fp16` (CUDA fp16 autocast, `auto` on
CUDA) moves PQ and mIoU by at most 0.02 at 1.25x (`coco-panoptic-small`) to 4.1x
(`coco-panoptic-large-1280`) a whole call. `tensorrt-fp16`, an engine from the published fp16 ONNX
graph, is faster again (up to 3.1x on the 640/512 large variants) with PQ within 0.06 and mIoU 0.01,
offered on request; not for `coco-panoptic-large-1280`, where the engine (575 ms a call) is slower
than torch-fp16 (500 ms), so no fp16 graph is published for it. A TensorRT fp32 engine matches torch to 3e-07 of pixels but gains at most 1.2x
a call, and is not offered. Full table: `docs/models/eomt.md`.

## Where this diverges, and why

Nothing in the numbers. One thing in what is reported: upstream's panoptic evaluation returns a
`(class, segment)` map and no score; mozo reports each segment's score, the class probability of
the query that founded it -- the value upstream thresholds on (`scores[i][keep]`) and what the
`transformers` port returns as `score`.
