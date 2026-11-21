# YOLO12: real-time object detection

<!--
summary: Ultralytics' YOLO12 detector in mozo: five sizes from 6 MB, COCO's 80 classes, on torch, ONNX and CoreML. AGPL-3.0 weights.
label: YOLO12
keywords: mozo, yolo12, yolo, object detection, ultralytics
-->

YOLO12 is Ultralytics' attention-based real-time detector. It finds objects of COCO's 80 classes
and returns a box, a class and a confidence for each. Five sizes, from `nano` at 6 MB to `xlarge` at
119 MB. Its weights are AGPL-3.0, which matters if you serve predictions over a network: read
[Weights and licence](#weights-and-licence) before you deploy it.

## Quick start

```python
import mozo

model = mozo.get_model("yolov12/nano")
found = model.predict("example.jpg")

for detection in found[:4]:
    print(detection.class_id, detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
0 person 0.93 [0.92, 119.15, 777.9, 1262.88]
0 person 0.92 [1304.01, 16.37, 1919.52, 1264.41]
67 cell phone 0.82 [1164.72, 1082.84, 1451.83, 1184.49]
63 laptop 0.81 [730.96, 778.85, 1368.6, 1133.65]
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/yolov12/nano?threshold=0.5" -F "file=@example.jpg"
```

The response is the same detections as a JSON list, with every PixelFlow field on every detection.

## Variants

| ID | Task | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `yolov12/nano` | detection | 6 MB | torch, onnx, coreml | AGPL-3.0 |
| `yolov12/small` | detection | 19 MB | torch, onnx, coreml | AGPL-3.0 |
| `yolov12/medium` | detection | 41 MB | torch, onnx, coreml | AGPL-3.0 |
| `yolov12/large` | detection | 54 MB | torch, onnx, coreml | AGPL-3.0 |
| `yolov12/xlarge` | detection | 119 MB | torch, onnx, coreml | AGPL-3.0 |

`runtime="auto"`, the default, picks CoreML on Apple silicon (`mps`) and torch on CUDA and CPU.
CoreML is by far the fastest way to run YOLO12 on a Mac. To choose a runtime yourself, load
the model through a `ModelManager`, which takes `runtime`:

```python
from mozo import ModelManager

model = ModelManager().get_model("yolov12", "nano", runtime="onnx-fp32")   # pip install 'mozo[onnx]'
```

YOLO12 has no segmentation variants in mozo. For masks from a YOLO, see
[YOLO11](/docs/models/yolov11.md) or [YOLO26](/docs/models/yolov26.md).

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

Over HTTP the same parameters are query parameters: `threshold`, and `labels` as a
comma-separated list (`labels=hardhat,vest`).

## How it behaves

### Overlaps are suppressed

YOLO12 removes duplicate boxes with non-maximum suppression, as Ultralytics does. That step runs
inside mozo, identically whichever runtime ran the network. [YOLO26](/docs/models/yolov26.md) is the
YOLO that needs no suppression step.

### Class ids are contiguous

The YOLO families number COCO's classes 0 to 79, so `person` is 0. [RF-DETR](/docs/models/rfdetr.md)
uses COCO's original ids, where `person` is 1. This matters when you filter by id, for example
before handing people to [ViTPose](/docs/models/vitpose.md):

```python
people = found.filter_by_class_id([0])
```

### Class names come from the checkpoint

The names are read from the weights, so a fine-tuned YOLO12 brings its own vocabulary. Run from an
ONNX or CoreML graph, which carries no names, mozo reads them from a `labels` file published beside
the graph.

## Weights and licence

The weights are Ultralytics' `ultralytics/assets` release `v8.4.0`, redistributed unmodified. They
are **AGPL-3.0**, or covered by a commercial licence from Ultralytics.

Mozo's own code is Apache-2.0, and so is its YOLO12 inference: it was written independently against
the checkpoint format, and it imports, contains and depends on nothing from `ultralytics`. The code
and the weights are separate works that travel together. But anything you export from the weights
inherits their terms, and **serving predictions from them over a network places AGPL-3.0 section 13
obligations on you.** Complying is the operator's responsibility.

Every variant ships with the licence and a NOTICE naming the release it came from. To run a server
that never offers AGPL weights, leave the YOLO families out of `MOZO_ENABLE`. The details are in
`mozo/vendors/yolov12_deploy/PROVENANCE.md`.
