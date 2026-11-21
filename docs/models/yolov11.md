# YOLO11: real-time detection and segmentation

<!--
summary: Ultralytics' YOLO11 in mozo: five detection and five segmentation sizes from 6 MB, on torch and ONNX. AGPL-3.0 weights.
label: YOLO11
keywords: mozo, yolo11, yolo, object detection, instance segmentation, ultralytics
-->

YOLO11 is Ultralytics' real-time detector. It finds objects of COCO's 80 classes and returns a box,
a class and a confidence for each. The `seg-` variants add a mask to every box. Five sizes of each,
from 6 MB to 125 MB. Its weights are AGPL-3.0, which matters if you serve predictions over a
network: read [Weights and licence](#weights-and-licence) before you deploy it.

## Quick start

```python
import mozo

model = mozo.get_model("yolov11/nano")
found = model.predict("example.jpg")

for detection in found[:4]:
    print(detection.class_id, detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
0 person 0.95 [0.99, 121.43, 768.48, 1265.03]
0 person 0.9 [1296.16, 12.43, 1918.35, 1264.79]
67 cell phone 0.82 [1164.86, 1082.15, 1450.9, 1184.56]
0 person 0.78 [668.72, 147.53, 1065.39, 787.12]
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/yolov11/nano?threshold=0.5" -F "file=@example.jpg"
```

The response is the same detections as a JSON list, with every PixelFlow field on every detection.

## Variants

| ID | Task | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `yolov11/nano` | detection | 6 MB | torch, onnx | AGPL-3.0 |
| `yolov11/small` | detection | 19 MB | torch, onnx | AGPL-3.0 |
| `yolov11/medium` | detection | 41 MB | torch, onnx | AGPL-3.0 |
| `yolov11/large` | detection | 51 MB | torch, onnx | AGPL-3.0 |
| `yolov11/xlarge` | detection | 115 MB | torch, onnx | AGPL-3.0 |
| `yolov11/seg-nano` | instance segmentation | 6 MB | torch, onnx | AGPL-3.0 |
| `yolov11/seg-small` | instance segmentation | 21 MB | torch, onnx | AGPL-3.0 |
| `yolov11/seg-medium` | instance segmentation | 45 MB | torch, onnx | AGPL-3.0 |
| `yolov11/seg-large` | instance segmentation | 56 MB | torch, onnx | AGPL-3.0 |
| `yolov11/seg-xlarge` | instance segmentation | 125 MB | torch, onnx | AGPL-3.0 |

`runtime="auto"`, the default, picks torch on every device. To run ONNX instead, load the model
through a `ModelManager`, which takes `runtime`:

```python
from mozo import ModelManager

model = ModelManager().get_model("yolov11", "nano", runtime="onnx-fp32")   # pip install 'mozo[onnx]'
```

There is no CoreML artifact; [No CoreML](#no-coreml) says why.

## Inputs and outputs

```python
model.predict(image, threshold=0.5, labels=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `threshold` | `float` | `0.5` | Minimum confidence to keep a detection. |
| `labels` | `list[str]` | `None` | Class names for this call, replacing the checkpoint's own. |

Returns a PixelFlow `Detections`, best first. Each detection has:

- `bbox`: `[x1, y1, x2, y2]` in pixels of the image you passed.
- `class_id` and `class_name`: COCO's 80 classes numbered 0 to 79, where `person` is 0.
- `confidence`: between 0 and 1.
- `masks` (`seg-` variants only): a list holding one boolean `HxW` array at the image's resolution.

Over HTTP the same parameters are query parameters: `threshold`, and `labels` as a
comma-separated list (`labels=hardhat,vest`).

## How it behaves

### Masks

The `seg-` variants answer the same call with a mask on every detection, aligned with its box:

```python
found = mozo.get_model("yolov11/seg-nano").predict("example.jpg", threshold=0.25)
mask = found[0].masks[0]
mask.shape, mask.dtype      # ((1281, 1920), dtype('bool'))
```

They sit in the same family as the detection variants because what they add is a field, not a
different question. Two behaviours are Ultralytics' own, and mozo reproduces them rather than
tidying them away:

- **A detection whose mask comes out empty is dropped.** Every box you get has a mask, and the
  boxes with nothing in them are not returned at all.
- **The mask crop changes at 50 detections.** Each mask is cut to its box. With fewer than 50
  detections, off CUDA, the box edges are rounded to whole pixels first; with 50 or more, or on
  CUDA, they are not. So the mask you get depends slightly on how many objects are in the picture.

### Overlaps are suppressed

YOLO11 removes duplicate boxes with non-maximum suppression, as Ultralytics does, inside mozo and
identically whichever runtime ran the network. [YOLO26](/docs/models/yolov26.md) is the YOLO that
needs no suppression step.

### Class ids are contiguous

The YOLO families number COCO's classes 0 to 79, so `person` is 0. [RF-DETR](/docs/models/rfdetr.md)
uses COCO's original ids, where `person` is 1. Mind this when you filter by id, for example before
handing people to [ViTPose](/docs/models/vitpose.md). Class names are read from the checkpoint, so a
fine-tuned YOLO11 brings its own vocabulary.

### No CoreML

YOLO11's `C2PSA` block makes Apple's Metal graph compiler abort the process. The configuration that
avoids the crash is slower than torch on `mps`, so publishing it would make `auto` pick the slower
option on a Mac. The `seg-` variants share the backbone and inherit the same limit.
[YOLOv8](/docs/models/yolov8.md) and [YOLO12](/docs/models/yolov12.md) have no such block and do
publish CoreML.

Every variant publishes ONNX. A `seg-` graph has two outputs, the predictions and the prototypes
their mask coefficients refer to; a detection graph has one.

## Weights and licence

The weights are Ultralytics' `ultralytics/assets` release `v8.4.0`, redistributed unmodified. They
are **AGPL-3.0**, or covered by a commercial licence from Ultralytics.

Mozo's own code is Apache-2.0, and so is its YOLO11 inference: it was written independently against
the checkpoint format, imports nothing from `ultralytics`, and was checked stage by stage against
`ultralytics` 8.4.0. The code and the weights are separate works that travel together. But anything
you export from the weights inherits their terms, and **serving predictions from them over a network
places AGPL-3.0 section 13 obligations on you.** Complying is the operator's responsibility.

Every variant ships with the licence and a NOTICE naming the release it came from. To run a server
that never offers AGPL weights, leave the YOLO families out of `MOZO_ENABLE`. The details are in
`mozo/vendors/yolov11_deploy/PROVENANCE.md`.
