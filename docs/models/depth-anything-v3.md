# Depth Anything 3: relative depth from one photograph

<!--
summary: Depth Anything 3 in mozo: relative depth from a single image, at the image's resolution, on torch, ONNX or TensorRT.
label: Depth Anything 3
keywords: mozo, depth anything 3, da3, depth estimation, monocular depth, relative depth, tensorrt
-->

Depth Anything 3 estimates depth from a single photograph and returns one value per pixel: inverse
depth on a scale of its own, larger meaning nearer. It answers "which pixels are nearer than
which", not "how many metres away". `model.unit` is `None`.

## Quick start

```python
import mozo

model = mozo.get_model("depth_anything_v3/small")
inverse_depth = model.predict("example.jpg")

inverse_depth.shape, inverse_depth.dtype   # ((1281, 1920), dtype('float32')), the image's own size
model.unit                                 # None: relative, larger is nearer
depth = 1 / inverse_depth                  # depth up to an unknown scale, larger is farther
```

Over HTTP the answer is a 16-bit PNG, with how to read it in the headers:

```bash
curl -X POST "http://localhost:8000/predict/depth_anything_v3/small" \
  -F "file=@example.jpg" -D headers.txt --output depth.png
```

The headers carry `x-depth-unit: none`, `x-depth-min` and `x-depth-max`; read a pixel back as
`value = min + png / 65535 * (max - min)`.

## Variants

| ID | Upstream | Parameters | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|
| `depth_anything_v3/small` | DA3-SMALL | 0.08B | 100 MB | torch, onnx, tensorrt | Apache-2.0 |
| `depth_anything_v3/base` | DA3-BASE | 0.12B | 393 MB | torch, onnx, tensorrt | Apache-2.0 |
| `depth_anything_v3/mono-large` | DA3MONO-LARGE | 0.35B | 1,337 MB | torch, onnx, tensorrt fp32 | Apache-2.0 |

`small` and `base` are upstream's "any-view" models, trained on one image or many and run here on
one. `mono-large` is the monocular model; it also finds the sky and sets it to the far end of the
scene, so the sky is never nearer than the horizon.

### What is not offered, and why

- **The large and giant any-view models, and the "nested" metric models**, are CC-BY-NC-4.0:
  non-commercial. Mozo does not ship them.
- **DA3-LARGE-1.1** is labelled Apache-2.0 on its model card but CC BY-NC 4.0 in the authors' own
  model table. Mozo takes the stricter reading and does not ship it.
- **DA3METRIC-LARGE** (Apache-2.0) predicts metres only once multiplied by the camera's focal
  length in pixels, which `predict(image)` has no way to be told. It may come later with a way to
  pass one.
- **Several views at once, camera poses, point clouds and Gaussian splats** are what Depth
  Anything 3 is also for. Mozo's depth task is one image in, one depth map out.

## Inputs and outputs

```python
model.predict(image)
model.unit
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |

`predict` returns an `HxW` float32 array at the input's resolution: inverse depth, larger meaning
nearer, on a scale that changes from image to image. `model.unit` is always `None`.

An image more elongated than about 72:1 is refused with a `ValueError` (its short side would vanish
at the processing size); on the graph runtimes the limit is 12:1.

## How it behaves

### Inverse depth, like the other relative models

Depth Anything 3 itself predicts depth, larger meaning farther. Mozo returns the reciprocal, so that
`unit=None` means one thing across families -- inverse depth, larger is nearer, as Depth Anything
V2's relative variants return -- and a workflow can swap one for the other. `1 / predict(image)`
gives back the model's own depth, still on an unknown scale.

### The image is processed at 504 pixels, then resized back

The longest side is resized to 504 pixels and each side to its nearest multiple of 14, keeping the
aspect ratio, so a 1920x1281 photograph runs at 504x336. Upstream returns the depth at that size;
mozo resizes it back to your image's size (bilinear, as for Depth Anything V2) before inverting it.

### The sky

`mono-large` fills the sky with the 99th percentile of the rest of the scene's depth, as upstream
does. Upstream estimates that percentile from 100,000 pixels drawn from torch's global random
generator; mozo draws the same sample from a generator of its own, seeded the same way every call,
so the same image always gives the same depth and your random state is left alone.

### Runtimes

Measured on an RTX 3090 with torch 2.13 and TensorRT 11.5, over the 654 images of the NYU Depth V2
test split (640x480, so the network runs at 504x378). Times are a whole `predict`, pre-processing
and resize included, median per image. AbsRel and δ1 are against NYU's ground truth after
aligning scale and shift, the usual protocol for relative depth; they are there to compare
runtimes, not to restate the authors' benchmark.

| Variant | Runtime | ms | vs torch | AbsRel | δ1 | GPU memory |
|---|---|---|---|---|---|---|
| `small` | `torch-fp32` | 26.8 | 1.00× | 0.0941 | 0.9042 | 336 MiB |
| | `tensorrt-fp32` (auto) | 18.9 | 1.42× | 0.0941 | 0.9042 | 452 MiB |
| | `tensorrt-fp16` | 8.6 | 3.1× | 0.0941 | 0.9042 | 250 MiB |
| `base` | `torch-fp32` | 29.1 | 1.00× | 0.0732 | 0.9356 | 724 MiB |
| | `tensorrt-fp32` (auto) | 23.3 | 1.25× | 0.0732 | 0.9356 | 907 MiB |
| | `tensorrt-fp16` | 13.4 | 2.2× | 0.0732 | 0.9356 | 640 MiB |
| `mono-large` | `torch-fp32` | 68.4 | 1.00× | 0.0689 | 0.9412 | 1,662 MiB |
| | `tensorrt-fp32` (auto) | 56.3 | 1.21× | 0.0689 | 0.9412 | 1,700 MiB |
| | `tensorrt-fp16` | 22.3 | 3.1× | 0.0690 | 0.9413 | 992 MiB |

- **On CUDA with TensorRT 11 installed, `auto` picks `tensorrt-fp32`**: the inverse depth moves by
  3e-7 to 8e-7 on average and AbsRel not at all. The first load builds the engine from the
  published ONNX graph (21-26 s) and keeps it; every image shape (3 to 36 patches a side) runs
  on that one engine. Without TensorRT, `auto` stays on torch.
- **`tensorrt-fp16` is on request** (`runtime="tensorrt-fp16"`): 2.2 to 3.1 times torch, AbsRel
  within 1e-4. Per pixel it is not fp32: the mean relative difference is 0.08-0.11%, and single
  pixels move by up to a few tens of percent at depth edges. Its engine builds in 37-44 s.
- **`onnx-fp32`** runs on ONNX Runtime anywhere. On the CPU it was about as fast as torch for
  `small` and `base` and half as fast for `mono-large` (a loaded 20-thread machine, so indicative);
  on CUDA it was slower than torch for `small` and `base`. `auto` does not pick it.
- `mono-large`'s sky is chosen by a threshold, so a pixel at the sky's edge can fall on either
  side in different runtimes and come out with a very different value. That is the large
  per-pixel maxima between runtimes; the map as a whole agrees.

The graph runtimes take images up to 12:1 in aspect ratio; `torch-fp32` takes up to about 72:1.

Without the ray branch upstream computes and discards, mozo's network is faster than upstream's
own on the same input: 15.3 against 19.6 ms for `small`, 25.7 against 33.5 for `base`, the same
65.5 ms for `mono-large` (fp32, CUDA, network only).

### Parity

Bit-identical to the authors' repository on all three variants, on CUDA (RTX 3060, torch 2.13,
CUDA 12.6) and on the CPU. With autocast off, mozo's preprocessed tensor, every intermediate of
every module the two networks share (292 per image for `small` and `base`), and the final depth
are equal with no tolerance, on photographs and seven synthetic inputs per variant; with
upstream's own `inference` running under its autocast (bf16 or fp16), mozo under the same
autocast is equal too. `tools/verify/depth_anything_v3.py` re-runs the comparison against a
checkout of the pinned commit, and was made to fail on purpose, at the right stage, by each of
eleven perturbations (nine apply to `small` and `base`, seven to `mono-large`), plus on CUDA five
that change only the GPU kernels (TF32, channels-last, the attention kernel).

On CUDA, `mono-large` is not always bit-for-bit repeatable on very elongated inputs: for a
2000x150 strip, about one gate pass in five gives a result that differs in the last bits of one
attention matrix multiply from the time before (final depth by up to about 1e-4), in the
authors' code exactly as in mozo's. No torch setting has been shown to remove it.

Upstream's API always runs in reduced precision (bf16 on recent GPUs, fp16 otherwise, even on the
CPU); mozo's `torch-fp32` runs in full fp32, the checkpoint's own arithmetic.

## Weights and licence

© 2025 ByteDance Ltd. and/or its affiliates, from `ByteDance-Seed/Depth-Anything-3` and the
checkpoints its authors publish on Hugging Face. Code and the three checkpoints are Apache-2.0, as
stated on each model card and in the authors' model table. The published weights are the
checkpoints' own tensors, minus the camera and ray heads single-image depth never reads; each
variant ships its licence and a NOTICE naming its source file and commit. The upstream commit and
what was changed are in `mozo/vendors/depth_anything_v3_deploy/PROVENANCE.md`.
