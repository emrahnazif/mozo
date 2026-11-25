# Provenance

This package is a deployment-only extraction of single-image inference from
[BiRefNet](https://github.com/ZhengPeng7/BiRefNet), as the authors publish it beside each
checkpoint on Hugging Face.

| | |
|---|---|
| Reference | `birefnet.py` and `BiRefNet_config.py`, the remote code in each repository below |
| Project repository | `https://github.com/ZhengPeng7/BiRefNet` at `f7079632d681ecf12161fcea8905682934e59e51` (2026-10-25) |
| Extracted | 2026-11-25 |
| Verified against | `torch` 2.13.0, `torchvision` 0.28.0, `transformers` 5.9.0, `numpy` 2.3.5, Pillow 12, Python 3.12 |

## The reference, and what mozo guarantees

**The reference is the authors' Hugging Face remote code**, at each repository's pinned commit:
what `AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/...", trust_remote_code=True)`
executes, which is the loading path on every model card. It is the authors' own code, flattened
into one file and served beside the weights. The file is byte-identical across the Swin-L
repositories and differs from `BiRefNet_lite`'s in one character: the backbone index, `[6]`
(Swin-L) against `[3]` (Swin-T).

Not the GitHub repository's `main`, because it has moved on from what the checkpoints are served
with: its `config.py` sets `SDPA_enabled = True`, routing Swin attention through
`scaled_dot_product_attention` -- the same mathematics in a different arithmetic order -- where
the remote code pins `SDPA_enabled = False` and runs it eagerly; and it adds DINOv3 backbones.

Pre- and post-processing come from the model card's inference snippet and the repository's
`handler.py`, which agree: torchvision `Resize((S, S))` on a PIL image, `ToTensor`, ImageNet
`Normalize`; `model(x)[-1].sigmoid()`; `ToPILImage`; PIL `resize(image.size)`; `putalpha`.
`handler.py` additionally runs `refine_foreground` on the colours, which is `cutout(refine=True)`.
Its resolution table (2048 for the HR models, 1024 for the others) is what fixes each variant's
size here.

**What this package guarantees:** on the CPU in float32, for the same RGB image, every stage is
`torch.equal` to the reference -- the preprocessed tensor against torchvision's own transforms;
the output of every Swin block, patch merge and stage norm in both backbone passes; the squeeze
block; every patch convolution, deformable convolution, ASPP, decoder, lateral and attention block,
by module name; the logits; the alpha against `ToPILImage` + `resize`; the cut-out against the
card's `putalpha`, and with `refine` against `handler.py`'s own `refine_foreground`. The chain
stops at the published weights: accuracy figures are the authors'; mozo measures accuracy only to
compare its own runtimes.

**The reference has to be pinned to be defined.** `tools/verify/birefnet.py` loads it with
`dtype=torch.float32`, because transformers 5 defaults to `dtype="auto"` and loads the
float16-stored checkpoints (`general`, `hr`) as a float16 model; the authors' own examples then
call `.half()` on CUDA anyway. Neither float16 path is claimed.

## Checkpoints, and why only these four

| mozo | upstream (Hugging Face, commit) | backbone | size | stored | card | downloads/month |
|---|---|---|---|---|---|---|
| `general` | `ZhengPeng7/BiRefNet` `e2bf8e44` | Swin-L | 1024 | float16 | mit | 987,474 |
| `lite` | `ZhengPeng7/BiRefNet_lite` `aa62cd87` | Swin-T | 1024 | float32 | mit | 31,518 |
| `matting` | `ZhengPeng7/BiRefNet-matting` `eccde0a8` | Swin-L | 1024 | float32 | mit | 39,048 |
| `hr` | `ZhengPeng7/BiRefNet_HR` `a7a562f6` | Swin-L | 2048 | float16 | mit | 45,375 |
| | `BiRefNet-portrait` `b6561965` | Swin-L | 1024 | float32 | mit | 88,580 |
| | `BiRefNet_HR-matting` `5d6b6f8a` | Swin-L | 2048 | float16 | mit | 43,202 |
| | `BiRefNet_dynamic` `28030604` | Swin-L | 256-2304 | float16 | mit | 31,475 |
| | `BiRefNet-DIS5K`, `_lite-2K`, `_512x512`, `-HRSOD`, `-COD`, `-legacy`, ... | | | | mit | < 11k each |

Downloads are the Hub API's `downloads` field on 2026-11-25. "Card" is the `license:` field of
the model card; every one is ungated.

The four exercise every distinct path: Swin-L at window 12 (256 tokens a side at 1024: no
padding) and Swin-T at window 7 (256 is not a multiple of 7: every stage pads); a 2048 input;
float16- and float32-stored checkpoints; segmentation-trained and matting-trained weights.
**Left out:** `BiRefNet-portrait`, the largest omission -- the same architecture and size as
`matting`, whose training set (P3M-10k, humans, AM-2k, AIM-500, Human-2k, Distinctions-646,
HIM2K, PPM-100) contains portrait's (P3M-10k, humans). `BiRefNet_HR-matting` would be a fifth
2048 Swin-L. `BiRefNet_dynamic` takes any size, which the other four do not, and needs handling
of sizes that are not multiples of 32. The DIS, HRSOD and COD checkpoints are the paper's
benchmarks. **RMBG-2.0** (briaai) is this architecture under a non-commercial licence and is never
shipped.

Each published `torch-fp32.pth` is the checkpoint's `model.safetensors`, verified against the
sha256 the Hub records, written as an ordinary state dict by `tools/fetch/birefnet.py` **in its
published dtype**, minus twelve tensors only training reads:

- `decoder.conv_ms_spvn_{4,3,2}.*` -- the multi-scale supervision heads.
- `decoder.gdt_convs_pred_{4,3,2}.0.*` -- the gradient-prediction heads.

Upstream builds both and calls them only under `self.training`. The gradient-*attention*
convolutions (`gdt_convs_{4,3,2}`, `gdt_convs_attn_{4,3,2}`) run at inference and are kept. 742
of 754 tensors remain for the Swin-L checkpoints, 574 of 586 for `lite`, and they load strictly.
The float16 tensors of `general` and `hr` are copied into float32 parameters by the load, which
is exact; publishing them as float32 would double the download and change no number.

## Licence

Code: MIT, `LICENSE` in the GitHub repository, Copyright (c) 2024 ZhengPeng; the remote code
carries no header of its own and the model cards link that file. The Swin Transformer originates
with Microsoft (MIT). `refine_foreground` is Photoroom's blur-fusion estimator, credited in
`handler.py`.

Weights: `license: mit` on each shipped model card, stated by the publisher, ungated. The training
sets behind them include P3M-10k, AM-2k and others published under research-only terms of their
own. As for BEN2, mozo takes the publisher's MIT grant on the weights as stated and does not track
the terms of the data behind them -- this is where the chain stops, and the NOTICE beside each
checkpoint says so.

## What was taken

From `birefnet.py`: the Swin v1 backbone (`### models/backbones/swin_v1.py`), the deformable
convolution, ASPP, decoder and lateral blocks, `SimpleConvs`, `image2patches`, `BiRefNet` and
`Decoder`, each reduced to the configuration every published checkpoint uses: `mul_scl_ipt='cat'`,
`cxt_num=3`, `squeeze_block='BasicDecBlk_x1'`, `dec_att='ASPPDeformable'`, `dec_ipt=True`,
`dec_ipt_split=True`, `ms_supervision=True`, `out_ref=True`, `batch_size=4`. From `handler.py`:
`refine_foreground`. Module names are upstream's, because they are the checkpoint's keys.

## What was deliberately left behind

- `transformers` (`PreTrainedModel`, `PretrainedConfig`): loading machinery.
- `timm` (`DropPath`, `trunc_normal_`, `to_2tuple`): identity at eval, initialisation the strict
  load overwrites, one line.
- `einops`: one pattern, `b c (hg h) (wg w) -> b (c hg wg) h w`, written as
  `view`/`permute`/`reshape`; the gate checks it against `einops` at every grid the decoder uses.
- `kornia.filters.laplacian`: computes the gradient label only training reads.
- PVT and VGG/ResNet backbones, the refiners, the plain ASPP, `ResBlk`, the classification head,
  every `nn.Dropout` (all identity at eval), `torch.utils.checkpoint`, backbone freezing.
- The SDPA branch of window attention: disabled in the reference, and it would drop the relative
  position bias and the shift mask, so it is not a faster path to the same numbers.

## Where this diverges from the obvious reading, and why

- **Upstream returns `[logits]`**, a list of one at eval (the last of the multi-scale outputs it
  returns in training). The vendored network returns the tensor.
- **The resize back is named BICUBIC.** Upstream calls `resize(image.size)` and gets Pillow's
  default for a mode-L image, which is BICUBIC since Pillow 7 and was NEAREST before. Naming it
  makes the behaviour independent of the Pillow version and identical to upstream's on any
  current one.

## Upstream behaviour that is reproduced, not fixed

- `ASPPDeformable` has **two** 1x1 deformable branches (`aspp1`, then `[1, 3, 7]`).
- The modulator is `2 * sigmoid(...)`, ranging over `(0, 2)`.
- `SimpleConvs` is two convolutions with nothing between them.
- The last decoder stage resizes the image to its own size (`F.interpolate` with
  `align_corners=True` to the same shape) before the 1x1-grid patch block.
- The window mask is rebuilt on every stage call from `torch.ceil(torch.tensor(H) / window)`.
- `ToPILImage` truncates (`mul(255).byte()`): 0.999 becomes 254.
- The card's snippet does not convert to RGB and fails on an RGBA PNG; `handler.py` converts.
  mozo's image loader always hands over RGB, which is `handler.py`'s behaviour.

## What is not carried, because a library may not do it

- `torch.set_float32_matmul_precision('high')` at import in `handler.py`: a process-wide switch to
  TF32 matrix products on Ampere and later.
- `image.putalpha(mask)`: writes into the caller's own image. mozo returns new arrays.

## Export

### The deformable convolution, and why the graphs decompose it

`torchvision::deform_conv2d` has no ONNX symbolic in torch 2.13 (both exporters refuse it). The
ONNX `DeformConv` operator (opset 19) it would become is not implemented by ONNX Runtime (1.21,
1.23: the model fails to load), and TensorRT 11.5 parses it only through the
`ModulatedDeformConv2d` plugin of `libnvinfer_plugin`, which `mozo`'s TensorRT runner does not
load; with the plugin registry initialised, the parser segfaulted on a one-node graph. So the
exporter swaps in `graph.deform_conv2d`: per kernel tap, `grid_sample` (bilinear, zeros,
`align_corners=True`, torchvision's sampling convention) times the modulator, then a 1x1
convolution, summed. Against torchvision it differs by 5e-6 relative on random tensors and moves
the matte by at most 6e-5 through the whole network (lite, CPU). It is faster than torchvision's
kernel on the CPU (10.2 s against 13.2 s for lite under load) and is never used by the torch
runtime.

### Published graphs

| variant | `onnx-fp32` | `onnx-fp16` |
|---|---|---|
| `general` | 475 MB | 464 MB |
| `lite` | 188 MB | 109 MB |
| `matting` | 893 MB | 464 MB |
| `hr` | not published | 465 MB |

Exported with `torch.onnx.export(dynamo=True)`, opset 18, one fixed shape per variant. The
TorchScript exporter also exports the fp32 graph but cannot export autocast's casts ("tensor does
not have a device"). The fp16 graph is exported under CUDA fp16 autocast, then every weight
autocast only ever casts to float16 is stored in float16. `general`'s fp32 graph stores the 553
weights float16 represents exactly (all of the published ones, since the checkpoint is float16)
in float16 behind a `Cast`: 475 MB instead of 893, the same arithmetic.

### Measured (RTX 3090, TensorRT 11.5, ONNX Runtime 1.23.2, torch 2.13, 2026-11-25)

Network-only median ms on the fixture; accuracy on DIS-VD (470) or P3M-500-NP (500, `matting`),
from the uint8 alpha at the image's size, against the ground truth. "vs fp32" is the alpha against
torch fp32 on CUDA (TF32 off), in grey levels, mean / max over all images.

| variant | runtime | ms | S-measure | MAE | vs fp32 | GPU memory |
|---|---|---|---|---|---|---|
| `lite` | torch fp32 | 83.2 | 0.8793 | 0.0437 | reference | 1.5 GB |
| | torch fp16 | 58.8 | 0.8794 | 0.0437 | 0.021 / 220 | 1.4 GB |
| | ONNX Runtime CUDA | 133.0 | 0.8793 | 0.0437 | 0.008 / 143 | |
| | TensorRT fp32 | 68.6 | 0.8793 | 0.0437 | 0.0001 / 2 | 2.1 GB scratch |
| | TensorRT fp16 | 37.6 | 0.8794 | 0.0437 | 0.128 / 254 | 1.1 GB scratch |
| `general` | torch fp32 | 240.2 | 0.9098 | 0.0330 | reference | 2.3 GB |
| | torch fp16 | 141.6 | 0.9098 | 0.0330 | 0.013 / 249 | 2.2 GB |
| | ONNX Runtime CUDA | 313.4 | 0.9098 | 0.0330 | 0.004 / 138 | |
| | TensorRT fp32 | 183.2 | 0.9098 | 0.0330 | 0.0001 / 2 | 3.1 GB scratch |
| | TensorRT fp16 | 67.0 | 0.9098 | 0.0331 | 0.120 / 254 | 1.5 GB scratch |
| `matting` | torch fp32 | 241.1 | 0.9796 | 0.0050 | reference | |
| | torch fp16 | 141.9 | 0.9796 | 0.0050 | 0.004 / 30 | |
| | ONNX Runtime CUDA | 314.7 | 0.9796 | 0.0050 | 0.002 / 19 | |
| | TensorRT fp32 | 184.2 | 0.9796 | 0.0050 | < 0.0001 / 2 | 2.9 GB scratch |
| | TensorRT fp16 | 67.3 | 0.9796 | 0.0050 | 0.035 / 77 | 1.5 GB scratch |
| `hr` | torch fp32 | 875.0 | 0.9229 | 0.0272 | reference | 9.3 GB |
| | torch fp16 | 497.7 | 0.9229 | 0.0272 | 0.016 / 128 | 8.8 GB |
| | TensorRT fp32 | 658.0 | not measured | | | 12.2 GB scratch |
| | TensorRT fp16 | 224.3 | 0.9227 | 0.0272 | 0.153 / 158 | 6.1 GB scratch |

`general`'s S-measure of 0.910 on DIS-VD agrees with the 0.911 the authors report for it.
TensorRT engines build in 2.5 to 5 minutes. ONNX Runtime's CUDA provider is slower than torch for
every variant and is not what any runtime selects on CUDA. On the CPU it is faster than torch:
`lite` 3.2 s against 9.4 s, `general` 7.0 s against 16.5 s (20 threads, load average 4 to 9 from
other work: indicative). `auto` stays on torch on the CPU, which is mozo's global order and the
exact runtime; `onnx-fp32` is there to be asked for.

**Published:** `onnx-fp32` (`general`, `lite`, `matting`) and `onnx-fp16` (all four). Derived:
`torch-fp16`, `tensorrt-fp32` (from `onnx-fp32`), `tensorrt-fp16` (from `onnx-fp16`). `auto` on
CUDA takes `tensorrt-fp32` where TensorRT 11 is installed and the variant publishes its graph,
else `torch-fp16`. **Not published:** `hr`'s fp32 graph, whose engine needs 12.2 GB of scratch,
more than torch fp32 uses, and runs slower than `torch-fp16`, which `auto` would otherwise lose
to it; `hr`'s TensorRT fp32 accuracy was therefore not measured on the dataset (the export check
on the fixture: 5e-5 on the matte through ONNX Runtime).
