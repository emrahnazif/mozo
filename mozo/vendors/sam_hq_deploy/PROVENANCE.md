# Provenance

This package is a deployment-only extraction of [HQ-SAM](https://github.com/SysCV/sam-hq)
("Segment Anything in High Quality", NeurIPS 2023), reduced to single-image promptable
segmentation.

HQ-SAM is Meta's SAM with its weights frozen and one thing added: a learned *HQ output token* in
the mask decoder, whose mask is predicted from a feature that fuses the decoder's own upsampled
features with an early encoder feature (the first global-attention block of the ViT, or stage 1 of
TinyViT) and the final image embedding. Only the added parameters were trained, on 44K finely
annotated masks.

| | |
|---|---|
| Upstream repository | `https://github.com/SysCV/sam-hq` |
| Upstream commit | `e696978d60352dc9a26b12631cd91781502c6546` (2025-09-11) |
| Checkpoints | `https://huggingface.co/lkeab/hq-sam`, Hub commit `09b02a333b37772133eff3997bdba997867374b7` |
| Extracted | 2026-11-25 |
| Verified against | On CUDA: `torch` 2.13.0+cu126, CUDA 12.6, cuDNN 9.10.2, RTX 3060 (driver 550.144.03), `torchvision` 0.28.0, Pillow 13.0.0, numpy 2.5.4, fp32. On CPU: `torch` 2.13.0, `torchvision` 0.28.0, Pillow 12.3.0, numpy 2.3.5 (also torch 2.6.0 / Pillow 12.1.1 for `tiny`) |

## Why this reference

The authors publish PyTorch and the checkpoints are theirs, so the reference is their repository
(VENDORING §1.1), driven through its own `SamPredictor`. Two other implementations exist and are
not the reference:

- `segment-anything-hq` on PyPI is the same code packaged by the same authors; the repository at a
  pinned commit is the thing that can be checked out and diffed.
- `transformers`' `SamHQModel` (with checkpoints under `syscv-community/`) is an independent port.
  It is not bit-identical to upstream by construction: with `multimask_output=True` it adds the HQ
  mask to all three SAM candidates and returns them sorted, where upstream keeps the best one.
  mozo returns three candidates too (see below), but the gate checks them against upstream's own
  decoder output, not against the port.

## Licences

| What | Licence | Where it is stated |
|---|---|---|
| Code | Apache-2.0 | `LICENSE` at the repository root (byte-identical to the text in this directory) |
| `sam_hq_vit_{b,l,h}.pth` | Apache-2.0 | the `lkeab/hq-sam` model card (`license: apache-2.0`), the repository the README links for every checkpoint |
| `sam_hq_vit_tiny.pth` | Apache-2.0 | same card |
| SAM weights inside the ViT checkpoints | Apache-2.0 | `facebookresearch/segment-anything` README: "The model is licensed under the Apache 2.0 license" |
| TinyViT weights inside `tiny` | Apache-2.0 | `ChaoningZhang/MobileSAM` LICENSE (MobileSAM distilled them from SAM ViT-H) |
| TinyViT code (`tiny_vit.py`) | MIT | `microsoft/Cream` LICENSE; reproduced in `NOTICE` |

Every checkpoint is the authors' file unchanged: the published `torch-fp32.pth` has the sha256 the
Hub records for it (`tools/fetch/sam_hq.py` verifies this at the pinned Hub commit).

## What was taken

Verbatim from `segment_anything/modeling/`, apart from relative imports and an SPDX line:

```
common.py           <- common.py            (LayerNorm2d, MLPBlock)
image_encoder.py    <- image_encoder.py     (the SAM ViT, returning the global blocks' outputs)
prompt_encoder.py   <- prompt_encoder.py
transformer.py      <- transformer.py       (TwoWayTransformer)
```

Changed:

- `mask_decoder.py` <- `mask_decoder_hq.py`. `MaskDecoderHQ.forward` no longer selects: it returns
  the four SAM tokens' logits, the HQ token's logits, and all four IoU predictions, and
  `network.SamHq.decode` does the selection; and the first two lines of upstream's `forward`,
  which fuse the early encoder feature with the image embedding and depend only on the image,
  are `MaskDecoderHQ.hq_features`, run once in `encode` and cached with the embedding rather than
  on every prompt. The arithmetic is upstream's line for line. See "Where this diverges".
- `tiny_vit.py` <- `tiny_vit_sam.py`, reduced: timm (`DropPath`, `to_2tuple`, `trunc_normal_`,
  `register_model`), gradient checkpointing, layer-wise learning-rate decay, weight init, the
  ImageNet classifier head, the model zoo and its download code are gone. `DropPath` was already
  `nn.Identity` at a drop rate of zero. The attention-bias cache is replaced; see the traps.
- `network.py` replaces `Sam`: the same three sub-modules under the same attribute names (so the
  checkpoint loads strictly with no rename table), split into `encode` and `decode`. `Sam.forward`
  (the batched training entry point) is not carried; `Sam.preprocess` is `encode`'s first two lines.
- `image.py` replaces `utils/transforms.py::ResizeLongestSide` and `Sam.postprocess_masks`. The
  resize calls Pillow directly instead of going through torchvision's PIL path -- the same
  `Image.resize(..., BILINEAR)` call -- which is what removes torchvision.
- `predictor.py` replaces `SamPredictor`: a content-keyed encoder cache, as SAM 2's segmenter has,
  and prompts in mozo's shapes. The coordinate scaling is upstream's numpy float64 path.
- `config.py` replaces `build_sam.py`'s inline numbers.

Not carried: `mask_decoder.py` (SAM's baseline decoder, which no HQ checkpoint uses),
`build_sam_baseline.py`, `automatic_mask_generator.py`, `utils/amg.py`, `utils/onnx.py`,
`scripts/export_onnx_model.py`, the `train/` and `seginw/` trees, the demos, and `sam-hq2/`.

## Where this diverges, and why

**Three candidates, not one.** With `multimask_output=True`, upstream's `MaskDecoderHQ.forward`
takes the SAM multimask token with the highest predicted IoU, adds the HQ token's mask to it, and
returns that one mask. mozo's promptable contract is three ranked candidates per click, which every
other family honours and `tests/families/test_promptable.py` holds them to, so this package adds
the HQ mask to each of the three SAM multimask tokens and returns all three with their IoUs. The
top-ranked one is upstream's answer bit for bit; the other two are the same addition applied to the
tokens upstream discards. With `multimask_output=False` nothing differs.

**`hq_token_only` is fixed at upstream's default, `False`.** Upstream's demo suggests `True` for
images of a single object; offering it would be a new argument on a public signature, so it is not
offered. What it would change on HQ-SAM's own validation sets is in the family's docs page.

**The score.** It is SAM's predicted IoU for the SAM token's own mask, before the HQ correction:
HQ-SAM trains no IoU head for the corrected mask. It ranks the three candidates; it does not rate
the final mask.

## Traps, predicted and found

- **The resize is PIL's.** `ResizeLongestSide.apply_image` resizes through
  `torchvision.transforms.functional.resize` *on a PIL image*, which is Pillow's antialiasing
  bilinear on `uint8`, then normalises on the 0..255 scale. Replacing it with `F.interpolate` or
  cv2 changes every pixel (falsifier `resize-filter`: 316 of 395 stages fail).
- **Half-up rounding of the resized size.** `int(x + 0.5)`, not `round`. They disagree only when
  the scaled side lands on exactly .5 -- a 2048x1025 image -- which is why the gate carries one
  (falsifier `resize-rounding`).
- **Letterbox, not squash.** Unlike SAM 2, SAM pads bottom-right to 1024 square *after*
  normalising, and `postprocess_masks` crops the padding before the final resize (falsifier
  `no-crop`).
- **Coordinates in float64.** `apply_coords` scales numpy arrays as float64 from whatever dtype the
  caller passed, then casts to float32. Casting to float32 first moves the prompt embedding
  (falsifier `coords-float32`). The predictor therefore does not cast clicks or boxes on entry.
- **TinyViT's attention biases are cached at `eval()` time, and the builder calls `eval()` before
  loading.** A `vit_tiny` model used straight from `build_sam_vit_t` attends with its zero-initialised
  biases; upstream's light demo calls `eval()` again after the load, which is what makes it right.
  The gate builds the reference the way the demo does, and this package gathers the biases in
  `forward` from the loaded parameter (falsifier `attention-bias`: 313 of 395 fail).
- **LayerNorm epsilon 1e-6** in the ViT, passed explicitly; torch's default is 1e-5 (falsifier
  `layernorm-eps`).
- **The early feature is the *first* global block's output.** The ViT returns one tensor per global
  block and the decoder reads `[0]`; reading the last runs and is wrong (falsifier `interm-index`).
- **Unused weights.** The tiny checkpoint carries TinyViT's ImageNet classifier (`norm_head`,
  `head`); upstream loads with `strict=False`. Here they are dropped by name and the load is
  strict. The ViT checkpoints load strictly with nothing dropped. `num_batches_tracked` counters
  (int64) in the tiny checkpoint load into the BatchNorms and are never read in eval.
- **Which parameters require grad is part of the arithmetic.** Upstream's builders set
  `requires_grad=False` on every SAM weight and leave HQ-SAM's additions (`hf_token`, `hf_mlp`,
  `compress_vit_feat`, `embedding_encoder`, `embedding_maskfeature`) trainable. `torch.matmul`
  decides whether to fold a batched, strided `(B, N, C) @ W.T` into one matrix multiply by whether
  `W` requires grad, under `no_grad` too, and on CUDA the two paths are different sums. With every
  weight trainable the decoder matched upstream for one prompt and drifted 1.8e-02 in the logits
  for a batch of boxes on an RTX 3060 (first in the two-way transformer's `k_proj`); on the CPU the
  two paths agree, so the CPU gate could not see it. `network.TRAINABLE` copies upstream's pattern
  (falsifier `all-trainable`). Found by the GPU gate, 2026-11-26.

## Verification

`tools/verify/sam_hq.py --upstream <checkout>` compares, with `torch.equal`, on CPU, for each
variant: the resized image, the image embedding, the fused HQ features, and for seven prompts (click,
click + negative click, box, box single-mask, box + negative click, lone negative, click
single-mask) the sparse and dense prompt embeddings, the decoder's four raw SAM tokens, HQ token and
IoUs, upstream's final logits / IoU / mask against mozo's best candidate, and all three candidates
against upstream's raw decoder output through upstream's own `postprocess_masks`; then mask-input
refinement, a batch of two boxes through upstream's `predict_torch`, and a repeated prompt through
the cache. Five images: the fixture landscape, its transpose (padding on the right), an odd crop,
a small upscale, and the 2048x1025 half-pixel case. Then mozo end to end through the adapter.
`--falsify` runs the gate under each perturbation and passes only if each is caught.

`--device cuda` is the primary gate: upstream and mozo run in one process on one GPU, in fp32, with
the global switches as mozo leaves them (torch's defaults: TF32 off for matmuls and on for cuDNN,
`cudnn.benchmark` off). Each side first records every stage twice and must repeat itself byte for
byte. It also checks the `sam_hq_boxes` workflow node: each row's mask against upstream's
`predict_torch` on the same boxes, and every other field of the row against what went in. fp32 is
the only torch precision to gate: mozo publishes `torch-fp32` alone and derives no half-precision
torch runtime (`--dtype fp16`/`bf16` refuse, saying so). `--reference-cache` keeps the reference's
digests so the falsifiers reuse one reference run.

## Results (2026-11-25)

**Parity.** 1,596 stages over the four variants (399 each: five images, seven prompts, refinement,
a batch of boxes, the cache, and the end-to-end adapter path), every one `torch.equal`. The upstream
reference itself was measured to move by up to 0.34 on the tiny image embedding when built without
the second `eval()` (trap above).

**Falsification.** All 27 applicable (perturbation, variant) pairs caught, each first at the stage it should be: resize filter and rounding at the resized image; ViT LayerNorm eps and TinyViT's zero attention biases at the image embedding; fusing the last global block at the HQ features; HQ token alone at the final logits; float32 coordinates at the sparse prompt embedding; skipping the crop at the final masks (66-320 of 395 stages fail per run). Run on CPU on the RTX 3060 test server (torch 2.13.0+cu126, Pillow 13.0.0), which also makes it a second machine on which the unperturbed stages agree.

**GPU gate (2026-11-26), the primary one.** On CUDA, RTX 3060 (vision-003, driver 550.144.03),
torch 2.13.0+cu126, CUDA 12.6, cuDNN 9.10.2, fp32, global switches at torch's defaults: 4,988
comparisons over the four variants, every one `torch.equal` -- 1,760 against upstream (440 per
variant: the CPU gate's stages, upstream's requires-grad pattern, and the `sam_hq_boxes` node on
all five images) and 3,228 determinism checks (each side recorded twice: 411 upstream stages and
396 vendor stages per variant, byte-identical). `huge` fits on the 12 GB card beside upstream and
the adapter; no swapping needed. One vendor fix was needed first: every weight was trainable where
upstream freezes SAM's (trap above), which moved batched box prompts on CUDA only.

**GPU falsification.** 47 of 47 (perturbation, variant) pairs caught on CUDA, each first at its
expected stage: the CPU set, plus `all-trainable` (first at the batched boxes' logits; on the CPU
only the requires-grad stage catches it), TF32 matmuls on and TF32 cuDNN off for the vendor alone,
a channels-last encoder input (all three first at the image embedding), and the prompt half run
on the host while the encoder stays on the GPU (first at the final logits). A host-side mask
resize was tried and dropped: CUDA's and the CPU's bilinear `F.interpolate` agreed bit for bit on
every mask (0 of 396 stages moved), so it is not a difference. The reference was recorded once per
variant (`--reference-cache`, digests) and reused by every falsifier. On the CPU the gate still
passes (1,760 comparisons, vision-003's CPU, torch 2.13.0) and the CPU falsifiers are caught.

**Quality on fine structure** (box prompts, single mask, HQSeg-44K's four validation sets, mean of
the four; CUDA fp32, TF32 off): tiny 82.0 mIoU / 73.4 boundary IoU, base 86.5 / 78.7, large
88.0 / 80.9, huge 86.7 / 79.9; SAM 2.1 large through mozo 83.6 / 76.5. Per-set numbers and the
script are in the family docs page and the plan.

**Export.** Each variant splits into an encoder graph (image -> embedding and early feature) and a
decoder graph (embedding, early feature, prompt embeddings -> all SAM tokens, HQ token, IoUs),
both opset 17. Measured, not published: mozo's promptable adapter has no runner that keeps two
graphs apart (SAM 2 and EdgeTAM are torch-only for the same reason).

Encoder / decoder milliseconds per call (median of 10), parity against torch fp32 on CPU as the
final mask's IoU on the fixture with a box prompt, and quality as the four-set mean mIoU / boundary
IoU. GPU rows: RTX 3060 12 GB (vision-002), idle GPU; an earlier run beside another job agreed within 10%.
CPU rows: the 20-core build machine under load from other jobs, indicative.

| runtime | tiny enc / dec ms | tiny mask IoU | base enc / dec ms | base mask IoU | quality tiny; base |
|---|---|---|---|---|---|
| torch fp32, CPU | 548 / 87 | 1.0 (exact) | 5,270 / 97 | 1.0 (exact) | 82.0 / 73.4; 86.5 / 78.7 |
| ONNX Runtime, CPU | 438 / 79 | 1.0 | 2,758 / 104 | 1.0 | not measured |
| torch fp32, CUDA | 35 / 7.1 | 1.0 | 246 / 7.8 | 1.0 | 82.0 / 73.4; 86.5 / 78.7 |
| ONNX Runtime, CUDA | 54 / 12 | 0.9999 | 239 / 14 | 0.9994 | not measured |
| TensorRT 11.5 fp32 | 22 / 4.6 | 1.0 | 159 / 5.4 | 1.0 | 82.0 / 73.4; 86.5 / 78.7 |
| TensorRT 11.5 fp16 | 9.9 / 2.6 | 0.9997 | 64 / 3.1 | 0.9969 | 82.0 / 73.4; 86.4 / 78.7 |

TensorRT fp16 (graph exported under CUDA fp16 autocast, strongly typed) leaves fine-structure
quality where fp32 has it -- 82.00 / 73.44 against 82.01 / 73.43 for tiny, 86.44 / 78.66 against
86.45 / 78.68 for base -- at 3.5x (tiny) and 3.8x (base) torch's encode; TensorRT fp32 is 1.6x and 1.5x, and exact in quality. ONNX Runtime's CUDA
provider drifts more than TensorRT fp32 (it uses TF32 by default). `large` and `huge` were not
exported: they share `base`'s code path, ViT-H's encoder graph is over protobuf's 2 GB limit, and the
12 GB card was shared with another job at the time. None of this is published (see Export above); the measurements are the
case for a two-graph promptable runner, which would serve SAM 2 and EdgeTAM too.

**Size.** 2,049 vendored lines against 2,766 upstream lines in the files taken from (TinyViT 724 ->
390); published bytes per variant 42.5 MB / 379 MB / 1.25 GB / 2.57 GB.

