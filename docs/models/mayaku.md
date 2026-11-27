# Mayaku: real-time detection, and the street models

<!--
summary: Mayaku in mozo, starting with its street models: 22 road-scene classes including faces and licence plates, boxes or masks, Apache-2.0.
label: Mayaku
keywords: mozo, mayaku, street, traffic, face detection, license plate, blur, instance segmentation, drone, tensorrt
-->

Mayaku is a real-time detector built from five op types (Conv, Relu, Add, nearest Resize and
MaxPool), so it exports and compiles cleanly everywhere. In mozo, each published Mayaku model is a
variant of the one `mayaku` family. The first are the **street models** (`street-*`), the rest of
this page; more Mayaku models will join as further variants.

## The street models

The street models detect what moves on or around a road, from every viewpoint a camera there
has: dashcam, handheld, pole and CCTV, rooftop, drone and straight-down aerial. They are finished
models with a class list of their own, 22 classes, among them **face** and **license plate**: find
them, then blur them before footage is stored or shared. The `street-seg-` variants add a mask to
every box.

### Quick start

```python
import mozo

model = mozo.get_model("mayaku/street-nano")
found = model.predict("example.jpg")          # threshold 0.4, the checkpoint's own

for detection in found[:4]:
    print(detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
person 0.96 [1312.45, 19.55, 1917.36, 1272.09]
person 0.95 [1.84, 128.86, 782.61, 1280.44]
face 0.9 [832.28, 255.53, 1021.15, 485.48]
face 0.89 [208.32, 234.23, 443.08, 585.86]
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/mayaku/street-nano" -F "file=@example.jpg"
```

#### Blur every face and plate

```python
import pixelflow as pf
from mozo.image import load_image

image = load_image("street.jpg")
found = mozo.get_model("mayaku/street-nano").predict(image)
private = found.filter_by_class_id([14, 8])        # face, license plate
blurred = pf.annotate.blur(image.copy(), private)
```

In a workflow, the `mayaku` node wired into `blur_regions` blurs everything it found; mozo has no
node yet that keeps only some classes, so for faces and plates alone use Python as above.

### Variants

| ID | Task | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `mayaku/street-nano` | detection | 22 MB | torch, onnx, tensorrt fp32 and fp16 | Apache-2.0 |
| `mayaku/street-medium` | detection | 138 MB | torch, onnx, tensorrt fp32 and fp16 | Apache-2.0 |
| `mayaku/street-seg-nano` | instance segmentation | 23 MB | torch, onnx, tensorrt fp32 and fp16 | Apache-2.0 |
| `mayaku/street-seg-medium` | instance segmentation | 139 MB | torch, onnx, tensorrt fp32 and fp16 | Apache-2.0 |

**medium** when accuracy matters, especially for small and distant objects and aerial views;
**nano** for CPUs and many streams per GPU. A `street-seg-` variant gives exactly the boxes of its
detection variant, plus masks; take one only when you need the masks.

Mayaku reports, on its 17,712-image street validation set (COCO AP@[.50:.95]): nano 43.9 box AP and
35.2 mask AP, medium 56.8 and 45.5. Those are Mayaku's numbers, measured with its evaluation
decode; mozo's own measurements below compare mozo's runtimes with each other.

### The 22 classes

| Group | Classes |
|---|---|
| People | person, face, helmet |
| Micromobility and assistive | bicycle, e-scooter, motorcycle, wheelchair, stroller |
| Vehicles | car, truck, bus, emergency vehicle, train, three-wheeler, construction vehicle, trailer |
| Road objects | license plate, traffic light, safety sign, stop sign, traffic cone |
| Other | animal |

Some are broader or narrower than COCO's: **car** is every light vehicle (vans, SUVs and pickups
included) and **truck** heavy goods vehicles only; **train** includes trams; **animal** is one
class for every animal; **helmet** is any helmet, worn or not. Class ids are 0–21 in the
checkpoint's order (`person` is 0, `face` is 14, `license plate` is 8).

### Runtimes

`runtime="auto"` picks the fastest runtime that measured as accurate as torch: the **fp16
TensorRT engine** on CUDA when TensorRT 11 or later is installed, the **fp16 Core ML package** on
Apple silicon (`pip install 'mozo[coreml]'`), and on the CPU **ONNX Runtime** when it is installed
(`pip install 'mozo[onnx]'`), about twice as fast as torch with the same detections. Without them
it runs torch. Name a runtime to choose another:

```python
from mozo import ModelManager

model = ModelManager().get_model("mayaku", "street-nano", runtime="tensorrt-fp32")
```

Measured on an RTX 3090 and an i9-10900X (10 threads) with torch 2.13, ONNX Runtime 1.21 (CPU)
and TensorRT 11.5, on 500 random images of Mayaku's street validation set. Time is the network's
forward pass on one 800×800 canvas, median; the decode and, for masks, the mask network ran on the
CPU and add 3–11 ms. On CUDA they now run on the GPU, as Mayaku runs them (6.4 against 7.5 ms for
`street-seg-nano`, 7.2 against 7.4 ms for `street-seg-medium`, RTX 3060, vision-002, 16 street
images). AP is COCO box AP (mask AP for `street-seg-`) on those 500 images at
threshold 0.05 with mozo's prediction decode, so it is lower than Mayaku's published numbers and
only for comparing runtimes. A detection "changes" when, against torch fp32 on the CPU, it scores
above 0.52 on one side with no same-class partner (IoU above 0.5) above 0.5 on the other.

| Variant | torch CPU | `onnx-fp32` CPU (`auto`) | torch CUDA | `tensorrt-fp32` | `tensorrt-fp16` (`auto`) |
|---|---|---|---|---|---|
| `street-nano` | 66 ms, AP 38.88 | 29 ms, 2.3×, 0 changes | 7.9 ms | 4.2 ms, 1.9×, AP 38.88, 0 changes | 2.8 ms, 2.8×, AP 38.84, 0 changes |
| `street-medium` | 199 ms, AP 49.87 | 104 ms, 1.9×, 0 | 14.2 ms | 10.0 ms, 1.4×, AP 49.87, 0 | 4.2 ms, 3.4×, AP 49.83, 0 |
| `street-seg-nano` | 72 ms, AP 38.88 / 31.78 | 32 ms, 2.3×, 0 | 9.6 ms | 5.9 ms, 1.6×, AP 38.88 / 31.78, 0 | 3.8 ms, 2.5×, AP 38.83 / 31.68, 0 |
| `street-seg-medium` | 206 ms, AP 49.87 / 41.62 | 103 ms, 2.0×, 0 | 16.0 ms | 11.7 ms, 1.4×, AP 49.87 / 41.62, 0 | 5.2 ms, 3.1×, AP 49.88 / 41.67, 0 |

Speed-ups on CUDA are against torch CUDA. With TensorRT 11 or later installed
(`pip install tensorrt-cu12` or `-cu13`; see [Runtimes](/docs/deployment/runtimes.md#tensorrt)),
`auto` builds the fp16 engine from the published fp16 graph (10–65 MB) on first load and keeps it:
it changed no confident detection on these 500 images and moved AP by at most 0.1, at 1.5–2.4× the
fp32 engine. `runtime="tensorrt-fp32"` builds from the fp32 graph and finds exactly what torch
finds, AP identical. Without TensorRT, `auto` stays on torch.

On Apple silicon (M1 Max, coremltools 9), against torch fp32 on the CPU over 301 photographs (the
fixtures and 300 COCO val2017 images; street-val AP was not measured on the Mac), graph-only median:

| Variant | torch CPU | torch MPS | `coreml-fp32` | `coreml-fp16` (`auto`) |
|---|---|---|---|---|
| `street-nano` | 71 ms | 23.8 ms | 8.8 ms, 0 changes | 5.1 ms, 4.7×, 1 change of 1,455 |
| `street-medium` | 180 ms | 45.0 ms | 26.9 ms, 0 | 22.5 ms, 2.0×, 0 of 1,741 |
| `street-seg-nano` | 77 ms | 26.1 ms | 8.9 ms, 0 | 6.1 ms, 4.3×, 1 of 1,455 |
| `street-seg-medium` | 208 ms | 48.6 ms | 28.3 ms, 0 | 24.3 ms, 2.0×, 0 of 1,741 |

Speed-ups are against torch MPS. The fp16 packages (10–61 MB) run on the Neural Engine; the fp32
ones (19–122 MB) on the GPU, with boxes within 0.002 px of torch and masks overlapping at least
0.998.

The graph is the five op types and nothing else: Conv, Relu, Add, nearest Resize and MaxPool.

### Inputs and outputs

```python
model.predict(image, threshold=0.4, labels=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `threshold` | `float` | `0.4` | Minimum confidence, from 0 up to (not including) 1. 0.4 is the value the checkpoints record, chosen by Mayaku on its validation set as the best F1. |
| `labels` | `list[str]` | `None` | Class names for this call, replacing the published ones. |

Returns a PixelFlow `Detections`, best first. Each detection has:

- `bbox`: `[x1, y1, x2, y2]` in pixels of the image you passed.
- `class_id` and `class_name`: one of the 22 classes above.
- `confidence`: the sigmoid of the winning class's score at that location. One class per location,
  and overlapping boxes of one class are suppressed at IoU 0.5.
- `masks` (`street-seg-` variants only): a list holding one boolean `HxW` array at the image's resolution.

### How it behaves

- **Every image is letterboxed onto an 800×800 canvas**, whatever its size: the scale is uniform
  and the rest is grey. Small objects in very large frames get small.
- **Crowds:** NMS removes some overlapping objects of the same class, such as people side by side.
- Mayaku's own list of known limitations: faces on posters and statues are detected as faces;
  truck and car are the most confused pair; tiny people (under about 8 px) on drone images cause
  most false positives from above.

## Weights and licence

Code and weights are Apache-2.0: the publisher states it on each model card ("The weights are
Apache 2.0, like the library"). `torch-fp32` is the publisher's checkpoint, byte for byte. The
training labels were made with SAM 3; the NOTICE published beside each checkpoint records what
that does and does not mean. Mozo's extraction is checked against `mayaku` 3.1.1 at every stage,
with no tolerance: bit-identical on CUDA (RTX 3060, torch 2.13.0, CUDA 12.6, cuDNN 9.10.2) in
fp32, the only precision Mayaku runs through torch, and also on the CPU; see
`mozo/vendors/mayaku_deploy/PROVENANCE.md`.
