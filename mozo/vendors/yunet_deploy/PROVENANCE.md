# Provenance

This package is a deployment-only extraction of **YuNet**'s inference path.

It derives from **libfacedetection.train** (`ShiqiYu/libfacedetection.train`, commit
`dca340aa082c71081a68d17db8e58b33a58a914b`, 2026-06-14), BSD-3-Clause, © 2022-2026 Shiqi Yu -- the
authors' own PyTorch repository, which trained the published checkpoints and states their WIDER
Face AP. Since that commit it is plain PyTorch (the mmdetection fork the checkpoints were trained in
was removed), and it still loads the 2022 checkpoints strictly. Nothing in this package imports it.

**The reference is upstream's own evaluation pipeline**: `build_yunet` with a strict load of the
original checkpoint, `build_eval_transforms(image_size=None, size_divisor=32)` (upstream's "origin"
mode: no resize, zero padding to a multiple of 32, BGR, mean 0 / std 1), `collate_face_samples` and
`YuNetPostprocessor`. Run on CUDA it reproduces the README's AP for `yunet_n` to the last digit
(0.892 / 0.883 / 0.811).

**Why not OpenCV Zoo's ONNX.** OpenCV Zoo's `face_detection_yunet_2023mar.onnx`, the model behind
`cv2.FaceDetectorYN`, is byte-identical (sha256 `8f2383e4…`) to this repository's
`onnx/yunet_s_640_640.onnx`, committed on 2022-12-23 beside `weights/yunet_s.pth`; its unfused
initialisers are bit-equal to the checkpoint's tensors (76 of 181; the rest are batch norm folded
into the convolutions), and its outputs agree with the reference forward to 1.3e-05. So the zoo
model *is* `yunet_s`, exported from the PyTorch this package is checked against (VENDORING §1.1: the
authors publish PyTorch). A `torch.equal` gate against ONNX Runtime's or OpenCV's arithmetic is not
possible; the cross-framework agreement is measured below, which is where the chain stops.

**Weights: BSD-3-Clause.** The checkpoints are files of the repository and carry no terms of their
own, so they are under its LICENSE: BSD-3-Clause at the pinned commit. When they were committed
(2022-12-23 and 2023-01-10) the repository's LICENSE was Apache-2.0, and OpenCV Zoo distributes the
`yunet_s` network under MIT; the NOTICE published beside each checkpoint says all three. They were
trained on WIDER FACE with SCRFD's landmark annotations; WIDER FACE's own terms (research use;
CC BY-NC-ND 4.0 on its Hugging Face card) govern the dataset. `tools/fetch/yunet.py` obtains them,
checking the git blob id GitHub states at the pinned commit and the full sha256.

| variant | upstream | training run | parameters | upstream file sha256 |
|---|---|---|---|---|
| `small` | `weights/yunet_s.pth` | `yunet_yolox_shared_stack_1_step_nophoto_diou_nonorm_layerch32`, epoch 560 | 56,112 floats | `f2688fec…99ffe4d6` |
| `base` | `weights/yunet_n.pth` | `normal_0.5_1.5`, epoch 560 | 77,872 floats | `9c7c9e14…9b93a0d2` |

Upstream's `n` is "normal", not "nano": it is the larger network, hence `base`.

## What was taken

| from (libfacedetection.train `dca340aa`) | what | here |
|---|---|---|
| `models/config.py` | `YUNET_S`, `YUNET_N` | `config.py` |
| `models/layers.py`, `models/backbone.py`, `models/neck.py`, `tasks/face/head.py`, `tasks/face/model.py` | `ConvDPUnit`, `Conv_head`, `Conv4layerBlock`, `YuNetBackbone`, `TFPN`, `YuNetHead`, `YuNet` | `network.py` |
| `tasks/face/transforms.py` (`Pad`, `Normalize`, `ToTensor`), `tasks/face/collate.py` | the origin-mode input | `preprocess.py` |
| `tasks/face/postprocess.py`, `engine/priors.py`, `tasks/face/codec.py`, `engine/nms.py` | `YuNetPostprocessor`, `MlvlPointGenerator`, `bbox_decode`, `kps_decode`, `batched_nms`/`nms` | `postprocess.py` |

Module names are upstream's, so the published state dict loads strictly with no renaming and every
tensor in it is used. mozo republishes `state_dict` alone, without the SGD optimiser state and the
mmdetection training metadata the upstream file carries; no tensor is changed.

## What was deliberately left behind

- **Training**: the dataset readers, augmentations (random square crop, flip), the assigner, the
  EIoU/landmark losses, the scheduler, weight initialisation.
- **The `stacked_convs` branch of the head** and `num_classes > 1`: 0 and 1 in both checkpoints.
- **The ONNX-export branch of the head** (`torch.onnx.is_in_onnx_export()`), which flattens the maps
  and applies the sigmoids inside the graph for OpenCV. mozo's graphs return the raw maps, so one
  decode serves every runtime.
- **The "resize" evaluation mode** (letterbox to 640). The README's AP is measured in origin mode,
  and that is what this package runs.
- **YuNet-pose** (person keypoints): a different task with no released checkpoint.
- **OpenCV Zoo's int8 and block-quantised ONNX files**: quantised exports of `small`, not models of
  their own.

## Where the obvious reading is wrong

Each of these is reproduced, because it is what the checkpoints were evaluated with:

- **BGR, unnormalised.** Upstream reads with `cv2.imread` and never converts; the network takes
  BGR 0-255. mozo's images are RGB, so the channels are reversed (falsifier `rgb`).
- **No resize.** The network sees the image at its own size, padded at the bottom and right.
- **`ConvDPUnit` is pointwise then depthwise**, both biased, the reverse of a MobileNet block, and
  the head's output units have neither batch norm nor ReLU.
- **A prior is its cell's top-left corner**, `(col, row) * stride`, not its centre (falsifier
  `centre`), and a box's size is `exp(offset) * stride` (falsifier `exp`: `exp(offset + ln stride)`
  differs in the last bits).
- **The score is `sigmoid(cls) * sigmoid(obj)`**, and the threshold is `>=`, applied before NMS.
  OpenCV's `FaceDetectorYN` reports the square root (falsifier `sqrt`).
- **Boxes and landmarks are not clipped** to the image.

Not a trap, measured: OpenCV decodes a centre as `(col + offset) * stride` where upstream writes
`offset * stride + col * stride`. They round identically, because every stride is a power of two
and scaling by one is exact; a falsifier built on it changed no comparison.

Two changes of procedure, not of result, in NMS. **Up to 1,024 candidates, every IoU is computed in
one call** and the greedy pass runs over the boolean matrix on the host, where upstream computes one
row per kept box on the device and synchronises each time (1.2 ms of a 3 ms CUDA call). The
elementwise arithmetic is the same on the same pairs, so the kept set is identical; above 1,024
(at thresholds near 0 on large images, up to every cell) upstream's loop runs instead, because the
matrix grows with the square. **The loop keeps plain ints** where upstream keeps `order[0]`, a view
holding all of `order` alive once per kept box: at threshold 0 on an 8K image that exhausted a
24 GB GPU. `tests/families/test_yunet.py` holds the two paths to each other; the gate holds the
result to upstream's.

## Parity

`tools/verify/yunet.py` runs upstream from a checkout at the pinned commit (no install: it is plain
PyTorch) against this package and against mozo's adapter, on CPU, under `MKL_CBWR=AVX2,STRICT`, with
`torch.equal` / `np.array_equal` and no tolerance (torch 2.13.0, OpenCV 4.10.0, numpy 2.3.5). 36
images per variant: the repository's photograph, eleven awkward synthetic cases (blank, noise, 37×23,
one pixel, odd 641×479, 2400×300 panorama, 3000×2000, faces cut by the right and the top edge,
greyscale, 480×320) and 24 random WIDER Face val images (seed 0).

Compared per image: the padded float batch, **every named module's output** (124 module names for
`small`, 143 for `base`, walked side by side with forward hooks), and the final boxes, scores and
landmarks at upstream's evaluation threshold 0.02 and at mozo's default 0.5; then mozo's
`Detections` against upstream's arrays put through the same PixelFlow conversion.

| variant | module outputs compared | result |
|---|---|---|
| small | 4,788 | exact |
| base | 5,436 | exact |

20,160 comparisons in all, every one exact (2026-11-25), on the first build that loaded.

**Falsified** eight ways on `small` (`--falsify all`), each caught at its own stage and nowhere
earlier: channels left RGB (the batch), padding 1 instead of 0 (the batch), stem BN eps 1e-5→1e-3
(`backbone.model0.bn1`), neck upsampling nearest→bilinear (`neck.lateral_convs.1`), box size as
`exp(offset + ln stride)` (the boxes only; network exact), OpenCV's square-root score (the rows),
NMS IoU 0.45→0.4 (the rows; network exact), priors at cell centres (boxes and landmarks).

## Runtimes

The published graph (`onnx-fp32`, `tools/export/yunet.py`) is `YuNet.forward`: the padded BGR batch
in, the twelve raw maps out (`cls_8` .. `kps_32`), height and width dynamic, opset 17. Padding,
decode and NMS stay in this package on every runtime. `tensorrt-fp32` is built from it on the
user's machine, for padded sizes from 32×32 to 4096×4096 (tuned for 720p), and is what `auto` takes
on CUDA with TensorRT 11+; a larger image runs through torch on the same GPU, from the checkpoint a
TensorRT load also fetches.

Measured on an RTX 3090 (GPU 0; TensorRT 11.5.0.63, torch 2.13.0, ONNX Runtime 1.21.1 CPU / 1.23.2
CUDA), WIDER Face val, all 3,226 images, threshold 0.02, AP by upstream's `wider_evaluation`.
"Changed": faces at or above 0.5 with no partner at IoU > 0.9 in torch CPU's answer. Box drift:
worst over matched faces at 0.5.

| variant | runtime | AP easy / medium / hard | changed | boxes max px |
|---|---|---|---|---|
| small | torch CPU (= the reference, gated) | 0.887124 / 0.871049 / 0.768134 | of 21,247 | |
| | TensorRT fp32 | 0.887124 / 0.871049 / 0.768134 | 0 | 0.0002 |
| | torch CUDA (TF32 on, torch's default) | 0.887132 / 0.871045 / 0.768098 | 11 | 3.50 |
| | TensorRT fp16 (autocast graph) | 0.887101 / 0.871040 / 0.768139 | 62 | 9.34 |
| | OpenCV 4.10 `FaceDetectorYN`, zoo ONNX | 0.886864 / 0.870738 / 0.768066 | 37 | |
| base | torch CPU (= the reference, gated) | 0.892320 / 0.883236 / 0.810690 | of 24,793 | |
| | TensorRT fp32 | 0.892320 / 0.883236 / 0.810690 | 0 | 0.0002 |
| | torch CUDA (TF32) | 0.892323 / 0.883244 / 0.810698 | 14 | 4.17 |
| | TensorRT fp16 | 0.892389 / 0.883283 / 0.810723 | 57 | 21.67 |

Upstream's README states 0.892 / 0.883 / 0.811 for `yunet_n`; upstream's own CLI, run here on CUDA,
gives 0.892323 / 0.883244 / 0.810698, the torch CUDA row exactly.

Speed: a whole `predict` (pad, network, decode, NMS) at threshold 0.5 on the fixture photograph
resized, median of 30 after 5 warm-up, GPU otherwise idle:

| variant | image | torch CUDA | TensorRT fp32 | TensorRT fp16 | network alone, torch CUDA → TRT fp32 |
|---|---|---|---|---|---|
| small | 640×480 | 4.49 ms | 2.52 ms | 3.23 ms | 2.26 → 0.52 ms |
| | 1280×720 | 4.73 | 2.79 | 3.51 | 2.31 → 0.80 |
| | 1920×1080 | 5.95 | 3.60 | 4.15 | 3.65 → 1.37 |
| | 3840×2160 | 19.59 | 9.67 | 9.50 | 13.95 → 4.41 |
| base | 640×480 | 4.79 | 2.52 | 3.17 | 2.61 → 0.57 |
| | 1280×720 | 5.06 | 2.95 | 3.41 | 2.67 → 0.92 |
| | 1920×1080 | 6.67 | 3.91 | 4.10 | 4.30 → 1.64 |
| | 3840×2160 | 22.05 | 10.70 | 10.12 | 16.50 → 5.35 |

On the CPU, under load from other jobs (load average 4 to 11 on 20 threads), indicative only:

| variant | image | torch CPU | ONNX Runtime CPU |
|---|---|---|---|
| small | 640×480 | 21.0 ms | 6.8 ms |
| | 1920×1080 | 120.3 | 50.3 |
| | 3840×2160 | 711.3 | 156.3 |
| base | 640×480 | 18.7 | 6.9 |
| | 1920×1080 | 174.9 | 54.4 |
| | 3840×2160 | 884.6 | 181.1 |

ONNX Runtime on CUDA (1.23.2) was slower than torch on CUDA (`small` 1080p 50 ms whole).

What was decided from it:

- **`onnx-fp32` is published and `tensorrt-fp32` derived from it, chosen by `auto` on CUDA.** Not
  one face changed, AP identical to six places, boxes within 0.0002 px of the CPU; a whole call
  1.8× (640×480) to 2.1× (4K) faster than torch on CUDA, which is itself further from the CPU (TF32).
- **The engine's range is 4096×4096**, which reserves 625 MB of device memory (measured for `base`;
  311 MB at 2176×3840, 2.5 GB at 8192×8192), and covers 4K and 12-megapixel photos either way up.
  Beyond it the adapter runs torch: `TensorRTRunner` does not check `set_input_shape`, and an
  engine fed a larger shape silently returns garbage (measured: six WIDER images over 2176 px tall
  lost up to 96% of their detections, AP easy 0.8814 instead of 0.8871).
- **No fp16.** Slower than the fp32 engine for a whole call up to 1080p, 2-6% faster at 4K, and it
  changes ~60 faces at 0.5 with boxes moving up to 22 px.
- **CPU**: ONNX Runtime is 2.5-4.5× faster than torch for a whole call. `auto` stays on torch on the
  CPU, as for every family (no per-family preferences); `runtime="onnx-fp32"` takes it.

Cross-framework, where the chain stops: OpenCV's `FaceDetectorYN` running the zoo ONNX on the same
images scores AP within 0.0003 of the reference, and 37 of 21,247 faces at 0.5 have no IoU > 0.9
partner: the expected disagreement of a different framework (BN folded into the convolutions,
OpenCV's DNN kernels, NMS on integer-truncated boxes, and `sqrt` scores against a threshold).
