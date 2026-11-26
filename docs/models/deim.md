# DEIM: real-time COCO detection without NMS

<!--
summary: DEIM in mozo: real-time DETR detection of COCO's 80 classes, nano to xlarge plus an Objects365-pretrained xlarge, Apache-2.0, on torch, ONNX and TensorRT.
label: DEIM
keywords: mozo, deim, d-fine, detr, object detection, real-time, coco, tensorrt
-->

DEIM ("DETR with Improved Matching", CVPR 2025) is a real-time transformer detector for COCO's 80
classes. It returns a box, a class and a confidence for each object, and like every DETR it needs
no overlap threshold: each of its 300 queries proposes one box, so there is no non-maximum
suppression to tune. The architecture is D-FINE's (an HGNetv2 backbone, a hybrid encoder, and a
decoder that refines each box edge as a probability distribution); DEIM is the training method
that produced these weights.

## Quick start

```python
import mozo

model = mozo.get_model("deim/medium")
found = model.predict("example.jpg")          # threshold 0.4, upstream's own

for detection in found[:4]:
    print(detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
person 0.96 [0.61, 120.46, 762.99, 1269.21]
person 0.92 [1303.88, 16.84, 1919.15, 1269.05]
cell phone 0.92 [1163.67, 1081.9, 1451.87, 1182.66]
laptop 0.91 [686.89, 777.92, 1357.08, 1136.98]
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/deim/medium" -F "file=@example.jpg"
```

In a workflow, the `deim` node takes an image and returns detections.

## Variants

| ID | Backbone | COCO AP (upstream) | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|
| `deim/nano` | HGNetv2-B0 | 43.0 | 16 MB | torch, onnx, tensorrt | Apache-2.0 |
| `deim/small` | HGNetv2-B0 | 49.0 | 42 MB | torch, onnx, tensorrt | Apache-2.0 |
| `deim/medium` | HGNetv2-B2 | 52.7 | 79 MB | torch, onnx, tensorrt | Apache-2.0 |
| `deim/large` | HGNetv2-B4 | 54.7 | 126 MB | torch, onnx, tensorrt | Apache-2.0 |
| `deim/xlarge` | HGNetv2-B5 | 56.5 | 252 MB | torch, onnx, tensorrt | Apache-2.0 |
| `deim/xlarge-obj365` | HGNetv2-B5 | 59.5 | 252 MB | torch, onnx, tensorrt | Apache-2.0 |

The AP column is the authors' (COCO val2017, their evaluation). `xlarge-obj365` is the same
architecture as `xlarge`, pretrained on Objects365 and then fine-tuned on COCO: the most accurate,
at the same cost. **small** or **medium** for most uses; **nano** for CPUs and many streams per GPU.

## Runtimes

`runtime="auto"` picks a TensorRT fp32 engine on CUDA when TensorRT 11 or later is installed, and
torch otherwise. On the CPU it picks torch, but **`runtime="onnx-fp32"` is faster there**
(`pip install 'mozo[onnx]'`) and gives the same detections:

```python
from mozo import ModelManager

model = ModelManager().get_model("deim", "small", runtime="onnx-fp32")
```

Measured on an RTX 3090 with torch 2.13, ONNX Runtime 1.21 (CPU) and TensorRT 11.5, on 500
random COCO val2017 images. Time is the network's forward pass on one 640×640 image, median; the
resize runs on the CPU and the decode on the device that ran the network. AP is COCO box AP on those 500 images with
all 300 detections, so it only compares runtimes with each other. A detection "changes" when,
against torch fp32 on the CPU, it scores above 0.52 on one side with no same-class partner (IoU
above 0.5) above 0.5 on the other; each variant has 2,200–3,000 detections above 0.5.

| Variant | torch CUDA | `tensorrt-fp32` (`auto`) | `tensorrt-fp16` | AP fp32 / fp16 | fp16 changes |
|---|---|---|---|---|---|
| `nano` | 10.6 ms | 1.9 ms, 5.4× | not published (slower than fp32) | 45.68 / – | – |
| `small` | 11.4 ms | 3.6 ms, 3.2× | 3.1 ms, 3.7× | 51.86 / 51.80 | 3 |
| `medium` | 14.9 ms | 5.7 ms, 2.6× | 4.4 ms, 3.4× | 55.26 / 55.38 | 2 |
| `large` | 25.0 ms | 7.7 ms, 3.2× | 4.5 ms, 5.6× | 56.85 / 56.84 | 0 |
| `xlarge` | 24.8 ms | 13.2 ms, 1.9× | 7.4 ms, 3.4× | 58.54 / 58.49 | 0 |
| `xlarge-obj365` | 31.3 ms | 15.7 ms, 2.0× | 7.3 ms, 4.3× | 61.05 / 60.96 | 4 |

The fp32 engine changed no detection on any variant and its AP is torch's to the second decimal.
With TensorRT 11 or later installed (`pip install tensorrt-cu12` or `-cu13`; see
[Runtimes](/docs/deployment/runtimes.md#tensorrt)), `auto` builds it from the published ONNX graph
on first load (10 s to 2 min) and keeps it. `runtime="tensorrt-fp16"` is faster again, changed at
most 4 confident detections in about 3,000 and moved AP by at most 0.12; it builds from a published
fp16 graph (22–125 MB) and has to be asked for. Without TensorRT, `auto` stays on torch.

On the CPU, torch took 67 ms (`nano`) to 570 ms (`xlarge-obj365`) and `onnx-fp32` 1.2–1.8× less
for nano to large, with no detection changed; those CPU timings were taken while other jobs loaded
the machine, so treat the ratio as indicative.

## Inputs and outputs

```python
model.predict(image, threshold=0.4, labels=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `threshold` | `float` | `0.4` | Minimum confidence. 0.4 is the value upstream's own inference scripts draw at. |
| `labels` | `list[str]` | `None` | Class names for this call, replacing the published ones. |

Returns a PixelFlow `Detections`, best first. Each detection has:

- `bbox`: `[x1, y1, x2, y2]` in pixels of the image you passed.
- `class_id` and `class_name`: one of COCO's 80 classes, ids 0–79 (`person` is 0).
- `confidence`: the sigmoid of one class's score for one query. Classes do not compete for it (it
  is not a softmax), so one box can come back twice with two classes when both clear the threshold.

### Parity

Bit-identical to the authors' code (`ShihuaHuang95/DEIM` at commit `09d35d5`, run as its
`torch_inf.py` runs) on CUDA (RTX 3060, torch 2.13.0, CUDA 12.6, cuDNN 9.10.2) in fp32, the only
precision DEIM runs through torch, with torch's default CUDA settings; and also on the CPU.
Compared with no tolerance on all six variants, 25 images each (a photograph, eight awkward
synthetic cases, 16 COCO val2017 images): the network input, the output of every module (124,225
in all), the encoder outputs, raw logits and boxes, all 300 decoded detections, and what mozo
returns at thresholds 0.4 and 0.05. On the GPU the decode runs on the device, as upstream's does.
`tools/verify/deim.py` runs the comparison against a checkout of that repository (`--device cuda`
for the GPU).

## How it behaves

- **Every image is resized to 640×640 without keeping its aspect ratio**, as upstream does for
  inference and evaluation. Very wide or tall images are squashed, and small objects in large
  frames get small.
- **At most 300 detections per image**: the best 300 (query, class) pairs.
- **No NMS**, so overlapping objects of one class (a crowd, a stack of cups) are not suppressed
  against each other; the model was trained to propose one box per object.

## Weights and licence

The code is Apache-2.0, stated in the authors' repository. The checkpoints are the authors' own
files, byte for byte, published from that repository's README with no terms of their own; that
they are Apache-2.0 too is an inference from the repository's licence, and the NOTICE published
beside each checkpoint says so in those words. `xlarge-obj365` was pretrained on Objects365.
Mozo's extraction is checked against the authors' repository module by module, with no tolerance;
see `mozo/vendors/deim_deploy/PROVENANCE.md`.
