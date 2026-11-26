# Provenance

This package is a deployment-only extraction of **RTMO**'s inference path.

It derives from **MMPose v1.3.2** (`open-mmlab/mmpose`, Upstream commit
`5408bc76f5b848cf925a0d1857899011d8c5b497`, 2024-07-12), Apache-2.0, © OpenMMLab — the authors'
own repository, the code that trained the published checkpoints and measured their AP. The mmcv
2.1 and mmengine 0.10 building blocks it is assembled from (`ConvModule`, `Swish`,
`MultiheadAttention`, `FFN`, `Scale`) are Apache-2.0 too, and are written out here in plain torch.
Nothing in this package imports mmcv, mmengine, mmdet or mmpose.

**The reference is MMPose's own inference**, `mmpose.apis.init_model` + `inference_bottomup`, on
the original checkpoint file: the test pipeline (`BottomupResize`, `PackPoseInputs`,
`PoseDataPreprocessor`) and the model in its ordinary evaluation form. Not mmdeploy's
`switch_to_deploy`, which folds batch norm into the DCC's linear layer, rewires the GAU and caches
positional encodings — different arithmetic, and not what the published AP was measured with.

**The weights are taken to be Apache-2.0.** MMPose's model zoo states no separate terms for its
checkpoints, so this is an inference from the repository's licence, and the NOTICE says so in those
words. They were trained on body7 (COCO, AI Challenger, CrowdPose, MPII, sub-JHMDB, Halpe,
PoseTrack18). `tools/fetch/rtmo.py` obtains them.

## What was taken

| from (MMPose v1.3.2 unless noted) | what | here |
|---|---|---|
| `models/backbones/csp_darknet.py`, `models/utils/csp_layer.py` | CSPDarknet P5: Focus, SPP, CSP layers | `network.py` |
| `models/necks/hybrid_encoder.py`, `models/utils/reparam_layers.py`, `models/necks/channel_mapper.py` | the hybrid encoder (one transformer layer on stride 32, FPN, PAN), RepVGG blocks in training form, the projector | `network.py` |
| `models/utils/transformer.py`, mmcv `MultiheadAttention`/`FFN` | the encoder layer, `SinePositionalEncoding`, `ScaleNorm`, `GAUEncoder` | `network.py` |
| `models/heads/hybrid_heads/rtmo_head.py` | `RTMOHeadModule`, `DCC`, `RTMOHead.predict` | `network.py`, `predictor.py` |
| `models/heads/hybrid_heads/yoloxpose_head.py` | `_flatten_predictions`, `decode_bbox` | `postprocess.py` |
| `models/task_modules/prior_generators/mlvl_point_generator.py` | grid cells with `centralize_points` | `postprocess.py` |
| `models/utils/misc.py`, `evaluation/functional/nms.py`, `structures/bbox/` | `filter_scores_and_topk`, `nms_torch`, `bbox_overlaps`, `bbox_xyxy2cs` | `postprocess.py` |
| `datasets/transforms/bottomup_transforms.py`, `structures/bbox/transforms.py` | `BottomupResize`, `get_warp_matrix` | `preprocess.py` |
| `models/pose_estimators/bottomup.py` | `add_pred_to_datasample`, the map back to the frame | `postprocess.py` |
| the four `configs/body_2d_keypoint/rtmo/body7/*.py` | the geometry, as frozen dataclasses | `config.py` |

**There is no checkpoint translation.** Module names are upstream's, so a published checkpoint loads
with `strict=True` after three training-only prefixes are dropped by name (each must be present):
`head.loss_oks.` (the OKS loss's sigmas), `head.dcc.sigma_fc.` (the MLE loss's learned deviation),
`head.head_module.out_kpt_reg.` (proxy keypoints for the assigner, computed only when training).

## What was deliberately left behind

- **Training**: the SimOTA assigner, the five losses, the proxy-keypoint branch, `sigma_fc`,
  batch-sync random resize, the data augmentations.
- **mmdeploy's deploy mode** (`switch_to_deploy` on the head, the DCC, the encoder and the RepVGG
  blocks): see above.
- **Multi-scale and flip test**: off in every RTMO config, and evaluation-time augmentation rather
  than the model.
- **The COCO-only and CrowdPose checkpoints.** Same architecture as body7; the COCO-only ones are
  0.9–1.7 AP worse at every size, the CrowdPose ones predict 14 joints. No consumer of either was
  found (rtmlib ships the body7 l/s/m; Hugging Face's mirrors are body7).
- **RTMW**, MMPose's whole-body model: a different architecture (top-down, CSPNeXt + RTMCC), not a
  variant of this one.

## Where the obvious reading is wrong

Each of these is reproduced, because it is what the checkpoints were evaluated with:

- **Two activations.** mmcv's `Swish` (`x * sigmoid(x)`) in the backbone and head, `nn.SiLU` in the
  neck and GAU. Same function, different rounding; swapping one is caught at the first convolution
  (falsifier `swish`, 3.7e-06).
- **Three batch-norm epsilons**: 1e-3 where the configs say so (backbone, head), torch's 1e-5 where
  they say nothing (neck, projector, DCC).
- **The letterbox is centred**, by an affine warp (`cv2.getAffineTransform` + `cv2.warpAffine`,
  border 114), not resized and padded at the top-left as YOLOX does.
- **BGR, unnormalised.** mean 0, std 1, `bgr_to_rgb` false: the network takes BGR 0–255.
- **The grid cells are pixel centres**: offset 0 plus `(stride - 1) / 2`, 7.5 at stride 16.
- **The DCC encodes bins relative to the grid cell and decodes them relative to the box centre.**
  The same bins, laid over the 1.25× box twice with different origins.
- **The DCC's softmax is hand-written**: clamp to ±5e4, subtract the max, `exp / (sum + 1e-8)`.
- **Boxes are clamped to the input square after the DCC has used them, and joints are never
  clamped.** A joint can land outside the frame.
- **The map back to the frame is numpy, float64.** float32 coordinates divided by a tuple of ints
  promote.
- **`nn.MultiheadAttention` is called sequence-first with `need_weights` at its default**, which
  selects torch's unfused attention path.
- **The joints depend, in the last bits, on how many people survived.** The DCC runs once over all
  survivors; a lower threshold adds rows and the shared matrix products round differently. Boxes,
  scores and visibilities do not move. Measured on the CPU, `small`, the fixture photograph: 0.0017 px
  between thresholds 0.5 and 0.1 under MKL's default kernel dispatch, and zero under
  `MKL_CBWR=AVX2,STRICT` (25 images, every variant), where the kernel no longer depends on the row
  count.

One change of procedure, not of result: **NMS computes every IoU in one call** and runs the greedy
pass over the boolean matrix on the host, where upstream computes one IoU row per kept box on the
device and synchronises each time. Each IoU is the same elementwise arithmetic on the same two boxes,
and the kept set is identical — the gate compares the final rows exactly.

## Parity

`tools/verify/rtmo.py` runs MMPose v1.3.2 in a scratch environment (torch 2.13.0, mmengine 0.10.7,
mmcv-lite 2.1.0, mmdet 3.3.0, numpy 2.3.5, OpenCV 4.10.0) against this package and against mozo's
adapter, in one process on one device, with `torch.equal` / `np.array_equal` and no tolerance.
**Bit-identical on CUDA** (the primary gate, below) **and on CPU** (under `MKL_CBWR=AVX2,STRICT`).
26 frames per variant: the repository's photograph, nine awkward synthetic cases (blank,
noise, 2400×300 panorama, 200×1500 tower, 37×23, odd 641×479, 3000×2000, a person cut by the frame
edge, greyscale) and 16 random COCO val2017 images, each at upstream's `score_thr` 0.1 and at 0.5.

Compared per frame: the letterboxed square, its centre and scale, the float batch, **every named
module's output** (the names are shared, so the two models are walked side by side with forward
hooks: 395 module names for tiny and small, 499 for medium, 591 for large), the DCC's three inputs
and its output, and the final boxes, scores, joints and joint scores; then mozo's `Detections`
against upstream's numbers put through the same PixelFlow call.

| variant | module outputs compared | result |
|---|---|---|
| tiny | 9,955 | exact |
| small | 9,944 | exact |
| medium | 12,593 | exact |
| large | 15,018 | exact |

**Falsified** nine ways on `small`, each caught at its own stage and nowhere earlier: letterbox
border 114→113 (the square), backbone `Swish`→`SiLU` (the stem's activation), neck BN eps
1e-5→1e-3 (the first input projection), position added to the attention's value (the encoder
attention), grid cells moved half a pixel (the DCC's inputs and the boxes; every network module
exact), DCC softmax epsilon 1e-8→1e-7 (the DCC output only, 6.1e-05), DCC window padding
1.25→1.26 (the DCC inputs), NMS threshold 0.65→0.6 (the rows; network exact), boxes unclamped (the
cut-off frame's boxes only).

**mmcv's compiled ops are stubbed in the reference.** `mmpose.models` imports EDPose, which imports
`mmcv.ops.MultiScaleDeformableAttention` at module level, and `mmcv-lite` has none. RTMO calls no
`mmcv.ops`; the module-by-module comparison is what shows the stub reached nothing.

49,719 comparisons in all, every one exact (2026-11-25).

**On CUDA** (`--device cuda`, 2026-11-26): RTX 3060 (vision-002), driver 550.144.03, torch
2.13.0+cu126, CUDA 12.6, cuDNN 9.10.2, mmengine 0.10.7, mmcv-lite 2.1.0, mmdet 3.3.0, OpenCV 5.1.0,
numpy 2.5.4. fp32, the only precision RTMO runs through torch: its `auto` on CUDA is a TensorRT fp32
engine (measured by parity and accuracy below), the DCC that runs after a graph runtime is this
package's torch fp32 code, mozo publishes no torch half runtime, and MMPose has no half inference
path. The settings are torch's defaults, the ones mozo leaves alone: cuDNN may use TF32 for
convolutions, matmuls do not, `cudnn.benchmark` and deterministic algorithms off. Upstream runs as
`init_model(..., device="cuda")` + `inference_bottomup` run it: the letterbox on the host, the
`uint8` batch moved to the GPU and cast there, then network, decode, NMS and DCC on the GPU. The
vendor makes its float batch on the host and moves it (integers 0..255, exact either way). Each side
first ran every frame twice and repeated itself bit for bit (208 checks), then the same stages as on
CPU: 49,927 comparisons, every one exact, with no change to the vendored code (rerun on
vision-002's CPU: 49,719, exact).

GPU falsifiers (`--device cuda --falsify all`, on `small`, against one cached reference run):
all twelve caught. The nine above, each first at the same stage as on CPU (the CPU falsifiers also
rerun on vision-002's CPU: nine of nine), and three only a GPU has: cuDNN TF32 off on the vendor's
network alone (first at `backbone.stage1.1.short_conv.conv`, the first convolution cuDNN runs in
TF32: TF32 is in play), the batch in channels-last memory, values equal (first at
`backbone.stem.conv.conv`: cuDNN picks other kernels), and the DCC run on the host from a CPU copy
of the same module (first at the DCC output, everything before it exact). A falsifier that makes the
vendor's batch on the GPU, as upstream does, is not one: its values are integers, exact on either
side, so it is the same arithmetic. The cache keeps each large intermediate as its shape, dtype and
the sha256 of its bytes (the final rows and the DCC's tensors whole), and an unperturbed run must
pass against it before any falsifier runs. With OpenCV 5.1.0 the unclamped-boxes falsifier is also
caught on the 37×23 frame, whose resized picture now puts a box past the square's edge.

## Runtimes

The published graph (`onnx-fp32`, `tools/export/rtmo.py`) is `RTMO.forward`: the letterboxed BGR
batch in, one row per grid cell out (scores, decoded boxes, visibilities, pose vectors), fixed at
1×3×640×640 (416 for `tiny`), opset 17. The score filter, NMS and the DCC stay in this package on
every runtime, so a graph load also fetches the checkpoint for the DCC's weights. `tensorrt-fp32` is
built from it on the user's machine and is what `auto` takes on CUDA with TensorRT 11+.

Measured on an RTX 3090 (GPU 0; TensorRT 11.5, torch 2.13, ONNX Runtime 1.21 CPU / 1.23.2 CUDA),
500 random COCO val2017 images per variant (seed 0), against torch fp32 on the CPU. AP: COCO
keypoint AP on those images at threshold 0.1. "Changed": people above 0.5 with no IoU>0.5 partner.
Joint drift: worst over matched people, and the 99th percentile over joints with visibility > 0.5.

| variant | runtime | AP | changed | joints max px | visible p99 px |
|---|---|---|---|---|---|
| tiny | torch CPU / ORT CPU / torch CUDA / TRT fp32 | 57.21 (all four) | 0 of 582 | ≤ 0.0026 | 0.0001 |
| | ORT CUDA | 57.22 | 1 | 1.21 | 0.080 |
| | TRT fp16 (grouped convs fp32) | 57.27 | 0 | 23.7 | 0.51 |
| | torch CUDA autocast | 57.13 | 0 | 15.3 | 0.17 |
| small | torch CPU / ORT CPU / torch CUDA / TRT fp32 | 69.56 (all four) | 0 of 665 | ≤ 0.0019 | 0.0002 |
| | ORT CUDA | 69.52 | 0 | 0.88 | 0.068 |
| | TRT fp16 | 69.47 | 0 | 75.4 | 0.91 |
| | TRT fp16 (grouped convs fp32) | 69.66 | 1 | 75.4 | 0.98 |
| | torch CUDA autocast | 69.52 | 0 | 75.3 | 0.63 |
| medium | torch CPU / ORT CPU / torch CUDA / TRT fp32 | 73.82 (all four) | 0 of 663 | ≤ 0.0031 | 0.0002 |
| | ORT CUDA | 73.82 | 2 | 2.92 | 0.042 |
| | TRT fp16 (grouped convs fp32) | 73.86 | 0 | 48.3 | 0.81 |
| | torch CUDA autocast | 73.74 | 2 | 21.5 | 0.55 |
| large | torch CPU / ORT CPU / torch CUDA / TRT fp32 | 74.24 (all four) | 0 of 661 | ≤ 0.0024 | 0.0001 |
| | ORT CUDA | 74.24 | 0 | 4.84 | 0.035 |
| | TRT fp16 | 74.31 | 1 | 109.5 | 0.91 |
| | TRT fp16 (grouped convs fp32) | 74.27 | 1 | 68.1 | 0.72 |
| | torch CUDA autocast | 74.23 | 1 | 109.3 | 0.52 |

Speed, on the fixture photograph (1920×1281), median of 200 graph calls and 50 whole `predict`s,
interleaved, input already on the GPU for the graph number, measured once the machine was quiet:

| variant | torch CUDA graph | TRT fp32 graph | TRT fp16 graph | `predict` torch → TRT fp32 → TRT fp16 |
|---|---|---|---|---|
| tiny | 7.36 ms | 1.81 ms | 2.52 ms (grouped fp32) | 11.73 → 6.10 → 7.47 ms |
| small | 7.52 ms | 2.90 ms | 2.56 ms | 13.79 → 8.68 → 8.97 ms |
| medium | 9.72 ms | 5.19 ms | 3.55 ms (grouped fp32) | 16.80 → 11.24 → 10.67 ms |
| large | 13.01 ms | 8.61 ms | 4.64 ms | 18.68 → 14.33 → 11.25 ms |

What was decided from it:

- **`onnx-fp32` is published and `tensorrt-fp32` derived from it, chosen by `auto` on CUDA.** Not
  one person changed, AP identical, joints within 0.003 px, 1.5–4.1× torch's graph speed.
- **No fp16 graph.** TensorRT fp16 moves visible joints by 0.5–1.0 px at p99 against fp32's 0.0001,
  and shortens a whole call only on `medium` (5%) and `large` (21%). Plain fp16 does not build at
  all for `tiny` and `medium`: TensorRT 11.5 has no fp16 kernel for a grouped convolution with an
  odd number of channels per group (13 and 27); holding those convolutions in fp32 builds it
  (`tools/export/rtmo.py --fp16`). torch autocast is slower than fp32 on three of four variants.
- **ONNX Runtime on CUDA is not exact**: it runs convolutions in TF32 by default (joints up to
  4.8 px). Available on request; `auto` never picks it.
- **CPU**: ONNX Runtime ran the graph faster than torch (e.g. `large` 200 against 288 ms), but those
  numbers were taken while other jobs loaded the machine (load average up to 40 on 20 threads) and
  are indicative only. `auto` stays on torch on the CPU, as for every family.

A whole call costs 6–7 ms beyond the graph on a 1080p-class photo: the warp, the decode and the
DCC. Two changes cut it without moving a bit: the frame is flipped to BGR after the warp, on the
640×640 square instead of the full frame, and NMS computes its IoUs in one call (above).
