# Provenance

This package is a deployment-only extraction of **OmDet-Turbo**'s inference path (Om AI Lab,
"Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head",
arXiv:2403.06892).

## The reference, and why

It derives from `transformers/models/omdet_turbo` at **transformers 5.9.0** (Apache-2.0, ©
Om Research Lab and The HuggingFace Inc. team), with its image processor's **PIL backend**, the
timm Swin it builds (**timm 1.0.28**), and torch 2.13. Not from the authors' repository
(`om-ai-lab/OmDet`, Apache-2.0, HEAD `956e2f3`), which would be the default:

1. The checkpoint mozo ships, `omlab/omdet-turbo-swin-tiny-hf`, is the authors' own Hugging Face
   organisation's, in transformers' format, with 1.01M all-time downloads (2026-11-25). The
   original-format repository (`omlab/OmDet-Turbo_tiny_SWIN_T`) reports none.
2. The authors' code needs detectron2 built from source, timm 0.9.16 and transformers 4.21.
3. Its preprocessing is the PIL processor's arithmetic: `PIL.resize((640, 640), BILINEAR)` on the
   uint8 image, then `(x - mean) / std` on the 0-255 scale.

**transformers' default image processor is not the reference.** In transformers 5,
`AutoProcessor` returns the torchvision backend, which resizes differently from PIL (and from the
authors' pipeline): on the gate's ten pictures, seven differ, by at most 0.018 normalised units --
one grey level -- on a few pixels (`tools/verify/omdet_turbo.py --torchvision-distance`). Small,
but not the same input; this package follows the authors.

## What mozo guarantees, and where the chain stops

On the CPU, with the torch runtime, boxes, scores and prompt indices are **bit-identical** to the
reference processor + model + `post_process_grounded_object_detection`, given the same RGB
pixels and an empty language cache. `tools/verify/omdet_turbo.py` checks every stage, exactly.

- **Weights, measured to the authors' release:** every one of the 868 tensors in the published
  safetensors is bit-identical to a tensor in the authors' original `OmDet-Turbo_tiny_SWIN_T.pth`
  (sha256 `f3b6ab59...68ce335`) -- 754 as they are, 114 as one third of a packed attention
  `in_proj` the conversion split into query, key and value. The only original tensors not carried
  are `decoder.denoising_embed_proj`, which is training-only.
- **Implementation, not measured to the authors' code:** that needs detectron2. The link from the
  authors' arithmetic to transformers' is HF's conversion, which this package does not re-check.
- **AP:** the published numbers are the authors'. mozo measures AP only to compare runtimes.

## Provenance of the weights

- `omlab/omdet-turbo-swin-tiny-hf` @ `7fe93cecfb770c4d76cf71163956221249cab566`, model card
  `license: apache-2.0`, ungated. `model.safetensors`, 461,822,656 bytes, sha256
  `439d1575d7e237ad565ed6969ea2a2dfcedf2086155e9ca3ac96cd6180a48cfd` (the Hub's LFS record).
  mozo publishes the same tensors repacked as `torch-fp32.pth` (no runtime `safetensors`).
- Inside it: a Swin-T backbone (Microsoft, MIT; the authors' config starts from
  `swin_tiny_patch4_window7_224`) and CLIP ViT-B/16's text tower (OpenAI, MIT; the original
  repository ships `ViT-B-16.pt` beside the detector), both fine-tuned into this detector and
  released by Om AI Lab under Apache-2.0. 868 tensors, fp32 (42 int64 batch-norm counters),
  115.4M parameters: Swin 27.5M, text 63.4M, encoder and decoder 24.5M.
- The tokenizer vocabulary is CLIP's; the vendored gzip is identical, id for id, to the
  `vocab.json` / `merges.txt` beside the checkpoint (49,408 entries, 48,894 merges).

## What was taken

| from | what |
|---|---|
| `modeling_omdet_turbo.py` | the language backbone's two encodings, the vision backbone's norms, the hybrid encoder (RT-DETR's), the decoder with its query selection, task encoder and deformable attention (Deformable DETR's `grid_sample` form) |
| `processing_omdet_turbo.py` | the default task sentence, tokenisation settings, and the post-processing (`postprocess.py`) |
| `image_processing_pil_detr.py` | the 640x640 PIL bilinear squash and normalisation (`preprocess.py`) |
| `modeling_clip.py` | CLIP's text transformer, SDPA with `is_causal` as transformers 5 runs it |
| `timm/models/swin_transformer.py` | Swin-T as `features_only` builds it at 640 with `always_partition` |
| `openai/CLIP` | the byte-pair tokenizer and `bpe_simple_vocab_16e6.txt.gz` |
| `tokenizers` | the `Lowercase` normaliser's codepoint table (`assets/lowercase.json.gz`) |

## Deliberately left behind

- **Custom task sentences.** Upstream lets a caller write the sentence the decoder reads ("Focus
  on the foreground, detect statue and trees."). Exposing it would add a parameter no other
  open-vocabulary family has; the default sentence is what the authors' demo and transformers use.
- **Batching.** One image per call. Upstream pads task sentences across a batch and masks the
  padding in the decoder's self-attention; at batch one there is none, the mask upstream builds is
  `None`, and nothing here masks.
- **Training:** denoising queries, losses, intermediate class logits, dropout.
- **Upstream's per-class text cache.** See below.

## Where this diverges from the obvious reading, and why

- **The text cache is per vocabulary.** Upstream caches each class prompt across calls and encodes
  only the ones it has not seen, so its bits depend on what was asked before (a batched text
  forward is not bit-identical to the same rows in another batch). This package encodes a whole
  vocabulary as one batch and caches that: always equal to upstream from an empty cache.
- **A class prompt over 75 tokens is refused**; upstream truncates it. The task sentence, which
  mozo builds, is truncated as upstream truncates it: refusing would cap a vocabulary at about
  twenty COCO names.
- **Lowercasing is a table, not `str.lower()`.** The reference lowercases in Rust, codepoint by
  codepoint: no Greek final sigma (`"ΑΣ"` -> `"ασ"`, Python gives `"ας"`), and its own Unicode
  tables (55 codepoints differ from Python 3.12's). Found by fuzzing 20,034 prompts against
  `CLIPTokenizer`: 17 differed with `str.lower()`, none with the table.
- **The checkpoint keys are renamed at load** (`checkpoint.py`, four prefixes), strictly, with a
  dtype check.

## Traps, and where they live

- `TIMM_FUSED_ATTN=2` moves timm's Swin onto `scaled_dot_product_attention`; the default is the
  explicit path, which `vision/swin.py` reproduces. The gate removes the variable.
- timm rolls the map before padding it to a window multiple and computes the shift mask on the
  padded grid (160 -> 161), unlike Microsoft's code.
- transformers 5's `CLIPTextModel` has no `text_model.` level; `from_pretrained` renames the
  published keys, a direct load must too.
- `use_kernel_forward_from_hub` would swap upstream's deformable attention for a downloaded kernel
  if the `kernels` package were installed; the reference environment does not have it.

## Verification

`tools/verify/omdet_turbo.py`, in the `inference` environment (torch 2.13.0+cu130, transformers
5.9.0, timm 1.0.28), CPU, `MKL_CBWR=AVX2,STRICT`. Six photographs and edge cases (the fixture at
three aspect ratios, black, noise, an 20x14 thumbnail) plus COCO images, five vocabularies (one
prompt, three, the 80 COCO names with a truncated task sentence, mixed case / accents / Greek /
CJK / punctuation, and one prompt at the 75-token cap). Per pair: pixels, token ids, task mask,
class embeddings, task states, three Swin levels, three encoder levels, six decoder layers,
boxes, logits, detections at 0.05 and 0.3; then mozo's adapter against the vendor through
PixelFlow. Every comparison `torch.equal`. Falsified with eight deliberate errors, each caught at
its own stage (see the plan's results).

## Export and runtimes

The image path exports through `torch.export` to one ONNX file (opset 18), taking pixels, class
embeddings `(C, 1, 512)` and task states `(T, 1, 512)`, with the class and token axes dynamic.
Measured on an RTX 3090 (torch 2.13, TensorRT 11.5), 500 COCO val2017 images, the 80 class names:

| runtime | image path | COCO AP | detections that change (of 4,843 above 0.3) |
|---|---|---|---|
| torch fp32 (CUDA) | 22.0 ms | 43.75 | |
| fp32, input rounded to fp16 | | 43.74 | 0 |
| torch fp16 autocast | 23.9 ms | 43.74 | 1 |
| ONNX Runtime CUDA | 32.4 ms | 43.74 | 0 |
| TensorRT fp32 | 13.9 ms | 43.74 | 0 |
| TensorRT fp16 (autocast graph) | 7.8 ms | 43.76 | 5 |

Published: `onnx-fp32` (TensorRT fp32's source, and ONNX Runtime's) and `onnx-fp16` (the autocast
graph, TensorRT fp16's source). `tensorrt-fp32` is exact and 1.6x faster, so `auto` takes it on
CUDA; `tensorrt-fp16` is 2.8x and on request. Autocast is slower than fp32 torch, so there is no
`torch-fp16`.
