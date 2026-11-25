# Provenance

A deployment-only extraction of Mayaku's inference path, for its published checkpoints (so far the
street models), from
[Mayaku](https://github.com/emrahnazif/mayaku).

| | |
|---|---|
| Reference | `mayaku==3.1.1` from PyPI (tag `v3.1.1`, commit `5c9ecf16da85`) |
| Extracted | 2026-11-24 |
| Verified against | `mayaku` 3.1.1, `torch` 2.13.0+cu130, `torchvision` 0.28.0, Python 3.12, CPU |
| Gate | `tools/verify/mayaku.py` |

**Why 3.1.1.** The checkpoints were written by Mayaku 3.0.0 and, by the 3.1.1 changelog, "download
by name in every 3.x version". 3.1.0 is the first release with the prediction decode (a score
threshold, NMS at IoU 0.5, one class per location) that the checkpoints' `predict` blocks
configure, and 3.1.1 is the release that documents the street models. It is also the latest.

## The weights

The publisher's checkpoints, byte for byte, from the Hugging Face repositories the model cards
link to. The same bytes (same sha256) are served by Mayaku's own manifest,
`dtmfiles.com/mayaku/v1/models/manifest.json`, revision `2026-11-23`; checked 2026-11-24.

| mozo variant | upstream name | bytes | sha256 |
|---|---|---|---|
| `street-nano` | `mayaku-n-det-street` | 22,405,191 | `87349e38e049be8117c2e854e3a1352948b1f6a679d8fc88adc37eef16f6f289` |
| `street-medium` | `mayaku-m-det-street` | 137,716,005 | `d8154930fee7dc3fa675a60430d6065f64ea757a6d66e2e8bec8d888dbf14cd6` |
| `street-seg-nano` | `mayaku-n-seg-street` | 22,847,277 | `e79fedd195d1213a7ded66616afd3c370f0f3c3be8445733812e262cf3df447c` |
| `street-seg-medium` | `mayaku-m-seg-street` | 138,813,259 | `700860e6a818a71de1636869f6ad411f3c60b88b4a671df3b0a089bd783ffac3` |

Each holds `{"model": state_dict, "mayaku": sidecar}`; the sidecar is JSON, so the file loads with
`weights_only=True`. A det checkpoint is its seg checkpoint without the 64 `aux.*` keys: every
shared tensor is equal (measured), and both name the same source training checkpoint.

**Present and unused:** every convolution and residual add carries `act_fq.mn` / `act_fq.mx`, the
int8 activation ranges of quantisation-aware training. The fp32 deploy graph never reads them
(upstream's `strip_fake_quant`), so the loader drops exactly those keys by name and loads the rest
strictly.

## Licences

- **Code:** Apache-2.0 (GitHub licence `apache-2.0`; PyPI classifier; `LICENSE` and `NOTICE` in the
  3.1.1 wheel). Mayaku's NOTICE: "Copyright 2026 the mayaku authors ... bundles no third-party
  source code."
- **Weights, per checkpoint:** Apache-2.0, stated by the publisher on each of the four model cards
  (`license: apache-2.0`, ungated, and "The weights are Apache 2.0, like the library") and on the
  docs page `docs/ready-to-use/street.md`. A direct statement on the artifact, not an inference.
- **Training labels were made with SAM 3** (model cards and docs). The SAM License defines the
  "SAM Materials" as what Meta distributes -- models, code, weights, documentation -- and mentions
  outputs only in its warranty disclaimer and its litigation clause; it states no term that
  reaches a model trained on SAM 3's outputs. Recorded as found, not as legal advice. The training
  images are from Objects365, COCO, Wikimedia Commons, Songdo Vision, a roundabout aerial set and
  PDOK aerial photos, under their own terms, as the publisher lists them.

## What was taken

From `mayaku` 3.1.1, reduced to the street models (tiers `n` and `m`, masks at stride 4 from the
box towers):

- `model/blocks.py` -> `blocks.py`: the trained-form blocks and the fusion, verbatim in arithmetic.
- `model/{tiers,backbone,neck,head,aux,detector}.py` -> `network.py`.
- `inference/preprocess.py`, `data/geometry.py` -> `image.py` (letterbox, planar input, inverse).
- `inference/decode.py`, `model/box.py` -> `decode.py` (the prediction decode only).
- `model/mask.py` and `decode._decode_masks` -> `mask.py`, for one image.

Written here: `model.py` (`Spec` per variant, the sidecar check, `detect`, `Detector`).

## What was deliberately left behind

Training, quantisation-aware training and the int8 graph, the evaluation decode (multi-label,
conf 0.001), keypoints, oriented boxes, classification, the other tiers, the stride-8 mask path,
the `tower` aux arm, export, the CLI, the download client, and the dependencies on `pydantic`,
`pyyaml`, `typer`, `onnx`, `pycocotools` and `faster-coco-eval`.

## Where this diverges, and why

- **RGB in.** Upstream takes BGR (OpenCV's order) and reverses the channels while laying out the
  planar input. mozo hands over RGB, so the vendor letterboxes RGB and does not reverse. The
  resize works per channel and the padding is one grey in all three, so the planes are the same;
  the gate compares the canvas and the input tensor directly.
- **One image at a time.** Upstream decodes a batch; the vendor decodes one image, with the same
  operations on the same shapes (the mask chunking is upstream's).
- **The decode runs on the CPU** whatever ran the graph. Upstream decodes on the GPU when its model
  is on CUDA.
- **A threshold outside [0, 1) is refused.** Upstream raises `ZeroDivisionError` at 1 and a math
  domain error at 0; the vendor accepts 0 (keeps everything up to top-k) and refuses the rest with
  a message.
- **The geometry is written down** (`Spec`, `TIERS`) and the checkpoint's sidecar is checked
  against it, rather than the network being built from the sidecar's config as upstream does.
  The seg checkpoints carry no `predict` block; upstream then uses its defaults (0.4 / 0.5 / one
  class per location), the values the det checkpoints record, and the check holds an absent block
  to those defaults.

## Parity

`tools/verify/mayaku.py`, 2026-11-25, `mayaku` 3.1.1 on torch 2.13.0+cu130, CPU: all four
variants, 23 images (the fixture photograph, 16 street-validation images across all six sources,
and six constructed cases: an empty grey frame, a tiny image, extreme wide and tall crops, an
image already 800x800, noise at an odd size), at thresholds 0.4 and 0.05. 184 runs, 8,002
detections, every stage `torch.equal`: the canvas, the input tensor, all 6 or 10 raw maps, the
decoded detections in canvas pixels, boxes, scores, class ids and masks. mozo's own path
(registry, weights, adapter, PixelFlow) matched the vendor exactly on every run. The fused weights
themselves were also compared, all 250-286 tensors equal.

**The gate fails when it should**, at the stage each change reaches and no earlier
(`--falsify all`, street-seg-nano): padding 114 -> 115 fails at the canvas; multiplying by 1/255 instead
of dividing fails at the input tensor; a BatchNorm epsilon x1.001 in the RepVGG fusion fails at
the raw maps; NMS IoU 0.5 -> 0.51 fails at the decoded detections; the mask coordinate scale
8 -> 8.0001 fails at the masks only.

## Upstream behaviour reproduced, not fixed

- **Class-wise NMS can merge two classes at the image edge.** Upstream calls
  `torchvision.ops.batched_nms`, which for at most 1,000 boxes on the CPU (always, after the top
  1,000) separates classes by offsetting them `max coordinate + 1` apart. Canvas boxes can start at negative coordinates (a
  side regressed up to 15 strides out from an anchor near the edge), and then two classes' bands
  can overlap and suppress each other. Measured with torchvision 0.28: two identical boxes at -1000..10 of classes 0 and 1 keep one.
  Rare, and what the published numbers were measured with,
  so kept; it is why this vendor exposes no YOLO-form `suppress` for
  `tests/test_vendor_agreement.py` to check.

## Export

`onnx-fp32`: the fused graph at 1x3x800x800, opset 17, five op types only (Conv, Relu, Add,
nearest Resize, MaxPool: nano 125 / 101 / 40 / 6 / 24). Fixed shape on purpose: the global-context
pooling unrolls `max(h, w) - 1` max-pools at trace time. `onnx-fp16`: exported on CUDA under fp16
autocast, opset 18, fp32 input, for TensorRT. Measured on an RTX 3090 / i9-10900X, 500 street
validation images per variant, against torch fp32 on the CPU (graph only, median):

| variant | torch CPU | ORT CPU | torch CUDA | ORT CUDA | TRT fp32 | TRT fp16 |
|---|---|---|---|---|---|---|
| street-nano | 66 ms | 29 ms | 7.9 ms | 5.3 ms | 4.2 ms | 2.8 ms |
| street-medium | 199 ms | 104 ms | 14.2 ms | 12.9 ms | 10.0 ms | 4.2 ms |
| street-seg-nano | 72 ms | 32 ms | 9.6 ms | 6.8 ms | 5.9 ms | 3.8 ms |
| street-seg-medium | 206 ms | 103 ms | 16.0 ms | 14.7 ms | 11.7 ms | 5.2 ms |

No runtime changed a confident detection (above 0.52 one side, no same-class IoU>0.5 partner above
0.5 the other). Box / mask AP on those images at threshold 0.05: torch, ORT CPU and TRT fp32
identical (nano 38.88 / 31.78, medium 49.87 / 41.62); TRT fp16 within 0.1 (38.84, 38.83 / 31.68,
49.83, 49.88 / 41.67); ORT CUDA within 0.02 but not count-identical (ONNX Runtime's CUDA provider
defaults to TF32 on Ampere). Published: `onnx-fp32` and `onnx-fp16` for all four; `tensorrt-fp32`
is what `auto` picks on CUDA, `tensorrt-fp16` on request.
