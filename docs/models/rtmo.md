# RTMO: everyone's joints in one pass

<!--
summary: RTMO in mozo: give it a frame, get every person's box and COCO's 17 joints, with no detector. Apache-2.0, bit-identical to MMPose.
label: RTMO
keywords: mozo, rtmo, mmpose, pose estimation, keypoints, multi-person, one-stage
-->

RTMO is OpenMMLab's one-stage multi-person pose estimator. You give it a frame and it finds the
people itself, returning each one's box and score with COCO's 17 joints attached. There is no
detector to run first, unlike [ViTPose](/docs/models/vitpose.md), and its cost grows with the
image, not with the number of people. It answers the way
[RF-DETR's keypoint variant](/docs/models/rfdetr.md) does, so it is called like any detector.

## Quick start

```python
import mozo

people = mozo.get_model("rtmo/small").predict("example.jpg")

nose = people[0].keypoints[0]
print(len(people), people[0].class_name, round(people[0].confidence, 2),
      nose.name, nose.x, nose.y, round(nose.confidence, 2))
```

```text
4 person 0.97 nose 1512.53 296.94 0.99
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/rtmo/small?threshold=0.5" -F "file=@example.jpg"
```

## Variants

| ID | Input | COCO val2017 AP (authors) | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|
| `rtmo/tiny` | 416 | 57.4 | 28 MB | torch, onnx; tensorrt on CUDA | Apache-2.0 |
| `rtmo/small` | 640 | 68.6 | 40 MB | torch, onnx; tensorrt on CUDA | Apache-2.0 |
| `rtmo/medium` | 640 | 72.6 | 90 MB | torch, onnx; tensorrt on CUDA | Apache-2.0 |
| `rtmo/large` | 640 | 74.8 | 179 MB | torch, onnx; tensorrt on CUDA | Apache-2.0 |

`runtime="auto"`, the default, picks torch on CPU and on CUDA without TensorRT, and an fp32
TensorRT engine built from the published ONNX graph on CUDA with TensorRT 11 or later installed
(`pip install tensorrt-cu12` or `-cu13`): the same people and joints, faster. See
[Runtimes](#runtimes).

## Inputs and outputs

```python
model.predict(image, threshold=0.5, labels=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `threshold` | `float` | `0.5` | The person score a candidate must exceed. MMPose evaluates at 0.1. |
| `labels` | `list` | `None` | Names for this call: a list of class names, or the published `labels.json` shape to rename joints too. |

Returns a PixelFlow `Detections`, best first: one row per person, `class_id` 0 named `person`, the
box in your frame's pixels, the person score, and 17 joints in COCO's order (nose, eyes, ears,
shoulders, elbows, wrists, hips, knees, ankles). Each joint has `x`, `y`, an `id`, a `name` and a
`confidence`.

A frame with nobody in it is answered with an empty `Detections`, not an error.

## How it behaves

### A joint's confidence is its visibility

Each joint carries the model's estimate that it can be seen, a sigmoid, separate from its position.
RTMO places every joint of every person it finds, including the ones behind a table or outside the
frame, so **filter on the confidence before you draw or measure a joint**. Joints are not clamped to
the image: a foot below the bottom edge is placed below it.

### The threshold is applied before NMS, and that is exact

Each grid cell of two maps (strides 16 and 32) proposes a person; the ones above `threshold` are
sorted and NMS at IoU 0.65 removes duplicates. Because NMS keeps boxes greedily from the best down,
a candidate below the threshold can never suppress one above it, so `threshold=0.5` returns the
same people `threshold=0.1` returns above 0.5, with the same boxes and scores. Their joints can
differ in the last float bits, because the joint decoder runs once over however many people
survived, and MMPose behaves the same way.

### The letterbox is centred

The frame is fitted into a 640×640 square (416 for `tiny`) with grey bars split evenly on both
sides, by the same affine warp MMPose uses. A very wide or tall image therefore gives the model few
pixels per person: a 2400×300 panorama becomes 640×80.

### Every variant is trained on body7

MMPose publishes three sets of RTMO weights. Mozo carries the four trained on body7 (COCO plus six
other pose datasets), which are 0.9–1.7 AP better at every size than the COCO-only ones and are the
ones rtmlib and the Hugging Face mirrors ship. The CrowdPose weights predict 14 joints, not COCO's
17, and are left out.

### Runtimes

The ONNX graph covers the network up to the score filter. The filter, NMS and the joint decoder
(which runs only on the people NMS kept) stay in mozo's code on every runtime, so a graph load also
fetches the checkpoint for the decoder's weights.

Measured on an RTX 3090 with TensorRT 11.5 and 500 random COCO val2017 images per variant, against
torch fp32 on the CPU. AP is COCO keypoint AP on those 500 images at threshold 0.1; graph time is
the network alone, input already on the GPU; a whole `predict` includes the letterbox of a
1920×1281 photo and the decode:

| Variant | torch CUDA graph | TensorRT fp32 graph | whole `predict`, torch → TensorRT | AP | people changed |
|---|---|---|---|---|---|
| `tiny` | 7.4 ms | 1.8 ms | 11.7 → 6.1 ms | 57.21 | 0 of 582 |
| `small` | 7.5 ms | 2.9 ms | 13.8 → 8.7 ms | 69.56 | 0 of 665 |
| `medium` | 9.7 ms | 5.2 ms | 16.8 → 11.2 ms | 73.82 | 0 of 663 |
| `large` | 13.0 ms | 8.6 ms | 18.7 → 14.3 ms | 74.24 | 0 of 661 |

The engine is fp32 throughout (TF32 off): every joint lands within 0.003 px of the CPU's. It is
built on the first load for each GPU and TensorRT version and kept in the cache. The CPU numbers
were taken while other jobs shared the machine and are not reported as speeds.

**No fp16.** An fp16 engine moved visible joints by 0.7 to 1.0 px at the 99th percentile, against
0.0001 for fp32, and made a whole call faster than the fp32 engine only on `medium` and `large`, by
5% and 21%. torch fp16 autocast was slower than fp32 on three of four variants. Neither is
offered.

**ONNX Runtime on CUDA is not exact.** It runs convolutions in TF32 by default: joints moved by up
to 1.1 px and up to 2 of 663 people changed. `auto` never picks it; ask for `onnx-fp32` on CUDA only
if you accept that.

### Parity

Bit-identical to MMPose v1.3.2 on CUDA (RTX 3060, torch 2.13.0, CUDA 12.6, cuDNN 9.10.2) in fp32,
the only precision RTMO runs through torch, with torch's default CUDA settings; and also on CPU.
Compared with no tolerance: the letterboxed input, the output of every one of the network's 395 to
591 modules, the decoder's inputs and outputs, and the final boxes, scores and joints, on 26 images
per variant at thresholds 0.1 and 0.5, and mozo's result through PixelFlow. `tools/verify/rtmo.py`
runs the comparison (`--device cuda` for the GPU); it needs MMPose installed, which mozo does not.

## Weights and licence

MMPose is Apache-2.0. Its model zoo states no separate terms for the checkpoints, so mozo takes them
to be under the repository's licence, and the NOTICE published beside each says that is an
inference. They were trained on COCO, AI Challenger, CrowdPose, MPII, sub-JHMDB, Halpe and
PoseTrack18, whose own terms govern those datasets. Mozo publishes each checkpoint's weights alone,
without the pickled training metadata the original file carries, so it loads without running
arbitrary code; no tensor is changed. The extraction is recorded in
`mozo/vendors/rtmo_deploy/PROVENANCE.md`.
