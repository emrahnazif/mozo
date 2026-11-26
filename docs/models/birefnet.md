# BiRefNet: background removal and matting

<!--
summary: BiRefNet in mozo: remove the background of a photograph, with a soft matte for hair, a fast Swin-T variant and a 2048 px one. MIT.
label: BiRefNet
keywords: mozo, birefnet, background removal, alpha matte, image matting, dichotomous segmentation, cutout
-->

BiRefNet removes the background of a photograph. It returns an **alpha matte**, one opacity per
pixel, and `cutout` turns it into an RGBA image with the background transparent. It takes no
prompt and gives one matte for the whole foreground. Four variants trade speed, resolution and how
soft the edge is.

## Quick start

```python
import mozo

model = mozo.get_model("birefnet/general")

alpha = model.predict("example.jpg")               # (1281, 1920) uint8, 255 is foreground
rgba = model.cutout("example.jpg", refine=True)    # (1281, 1920, 4) uint8
```

Over HTTP the answer is the RGBA cut-out as a PNG, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/birefnet/general" -F "file=@example.jpg" --output cutout.png
```

## Variants

| ID | Backbone | Resolution | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|
| `birefnet/general` | Swin-L | 1024 | 445 MB | torch, onnx, tensorrt | MIT |
| `birefnet/lite` | Swin-T | 1024 | 178 MB | torch, onnx, tensorrt | MIT |
| `birefnet/matting` | Swin-L | 1024 | 885 MB | torch, onnx, tensorrt | MIT |
| `birefnet/hr` | Swin-L | 2048 | 445 MB | torch, onnx fp16, tensorrt fp16 | MIT |

- **`general`** is the authors' general-purpose model and the one most people use (nearly a
  million downloads a month). Trained to segment, so its edges are close to binary.
- **`lite`** is the same recipe on a Swin-T backbone instead of Swin-L: about three times faster than `general` on a GPU, for 3 points less S-measure on the authors' validation set.
- **`matting`** is trained on matting data and returns a soft alpha along hair, fur and
  translucent edges. Reach for it when the edge is the point.
- **`hr`** is `general` retrained at 2048×2048, for large photographs where 1024 loses detail.
  About four times the work of `general`.

### What is not offered, and why

- **`BiRefNet-portrait`**, the second most downloaded, is the same architecture and resolution as
  `matting`, and `matting` was trained on its data and more.
- **`BiRefNet_HR-matting`**, **`BiRefNet_dynamic`** (any input size) and the academic DIS, HRSOD
  and COD checkpoints may come later.
- **RMBG-2.0** is BiRefNet's architecture trained by another company under a non-commercial
  licence. Mozo does not ship it.

## Inputs and outputs

```python
model.predict(image)
model.cutout(image, *, refine=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `refine` | `bool` | `False` | `cutout` only: estimate each edge pixel's foreground colour so the old background does not show as a fringe. |

- `predict` returns an `HxW` uint8 array at the input's resolution, 255 for foreground.
- `cutout` returns an `HxWx4` uint8 RGBA array at the input's resolution.

Over HTTP, `/predict` takes only `file` and returns `cutout(image)` as a PNG. For `refine=True`,
use Python or the `birefnet` workflow node.

## How it behaves

### The matte is the model's own probability

The alpha is the network's sigmoid scaled to 0-255, as the authors' code returns it. It is not
stretched per image the way [BEN2](/docs/models/ben2.md)'s default is, so 255 means the model was
certain in every image, and two images' mattes can be compared.

### It runs at a fixed square size

Every image is resized to a square, 1024×1024 (2048×2048 for `hr`), **without** keeping the
aspect ratio, and the matte is resized back to your image. A panorama is squashed, matted and
unsquashed. That is how the weights were trained, and letterboxing instead would move every pixel.

### `refine` changes colour, not opacity

`cutout(refine=True)` runs the authors' foreground refinement: along a soft edge, where every pixel
mixes subject and background, it estimates the subject's own colour so a cut-out placed on
something new carries no fringe of the old background. The alpha is the same either way. It costs
two full-resolution box blurs.

### Runtimes, and which one `auto` picks

Measured on an RTX 3090, network only, with the accuracy of each runtime on the authors'
validation sets: DIS-VD (470 images) for `general`, `lite` and `hr`, P3M-500-NP (500 portraits)
for `matting`. S-measure is higher-better, MAE lower-better.

| Variant | torch fp32 | torch fp16 | TensorRT fp32 | TensorRT fp16 | S-measure / MAE (all four) |
|---|---|---|---|---|---|
| `lite` | 83 ms | 59 ms | 69 ms | 38 ms | 0.879 / 0.044 |
| `general` | 240 ms | 141 ms | 183 ms | 67 ms | 0.910 / 0.033 |
| `matting` | 241 ms | 142 ms | 184 ms | 67 ms | 0.980 / 0.005 |
| `hr` | 875 ms | 498 ms | not offered | 224 ms | 0.923 / 0.027 |

No runtime moved either number by more than 0.0003. The torch fp16 figures were measured before
the 2026-11-26 postprocess fix, which moves its alpha by at most 2 grey levels; re-measuring them
is open. TensorRT fp32 changes the alpha by at most 2
grey levels; the fp16 runtimes move individual edge pixels further, by up to 254 grey levels on a handful of
pixels, without moving the averages.

- **On CUDA with TensorRT 11 installed**, `auto` runs `tensorrt-fp32`, built on first load from the
  published ONNX graph (two to five minutes, then cached). For `hr`, whose fp32 engine needs 12 GB
  of scratch memory and is slower than torch in half precision, it runs `torch-fp16`.
- **On CUDA without TensorRT**, `auto` runs `torch-fp16`: the network under float16 autocast.
  (The authors' own examples call `model.half()` instead, a different computation.)
- **`runtime="tensorrt-fp16"`** is the fastest, 2.2 to 3.9 times torch fp32. `auto` never picks it.
- **On the CPU**, `auto` runs torch, the exact one. `runtime="onnx-fp32"` (with
  `pip install 'mozo[onnx]'`) was 2.4 to 2.9 times faster on a shared 20-thread machine: 3.2 s
  against 9.4 s for `lite`, 7.0 s against 16.5 s for `general`. Measure on yours.

The graphs replace the deformable convolutions, which ONNX Runtime cannot run, with an equivalent
built from bilinear sampling, so they are close to torch rather than identical to it. GPU memory
for `hr`: 9.3 GB on torch fp32, 8.8 GB in fp16, 6.1 GB for the fp16 engine.

### Parity

mozo reproduces the authors' code bit for bit at every stage, for every variant, on CUDA in
float32 and in `torch-fp16` (RTX 3060, torch 2.13, CUDA 12.6), and on the CPU in float32. The fp16
comparison runs the authors' code under the same float16 autocast mozo uses; their own examples
call `model.half()`, which is a different computation. TensorRT and ONNX Runtime use other kernels
and are measured, not claimed exact.

## Weights and licence

Code and weights are MIT, © 2024 ZhengPeng, from `ZhengPeng7/BiRefNet` on GitHub and the four
checkpoints on Hugging Face, whose model cards say `license: mit` and are not gated. Some of the
training data carries research-only terms of its own; mozo takes the authors' MIT grant on the
weights as stated. `general` and `hr` are published in float16, as the authors publish them, and
run in float32. The upstream commits and the measurements are in
`mozo/vendors/birefnet_deploy/PROVENANCE.md`.
