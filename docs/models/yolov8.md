# YOLOv8: real-time object detection

<!--
summary: Ultralytics' YOLOv8 detector in mozo: five sizes from 7 MB, COCO's 80 classes, on torch, ONNX and CoreML. AGPL-3.0 weights.
label: YOLOv8
keywords: mozo, yolov8, yolo, object detection, ultralytics
-->

YOLOv8 is Ultralytics' real-time detector. It finds objects of COCO's 80 classes and returns a box,
a class and a confidence for each. Five sizes, from `nano` at 7 MB to `xlarge` at 137 MB. Its
weights are AGPL-3.0, which matters if you serve predictions over a network: read
[Weights and licence](#weights-and-licence) before you deploy it.

## Quick start

```python
import mozo

model = mozo.get_model("yolov8/nano")
found = model.predict("example.jpg")

for detection in found[:4]:
    print(detection.class_id, detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
0 person 0.92 [0.0, 113.24, 763.19, 1269.33]
0 person 0.92 [1295.3, 17.81, 1920.0, 1269.44]
63 laptop 0.85 [709.09, 773.48, 1366.4, 1143.08]
41 cup 0.64 [632.86, 589.76, 704.29, 724.06]
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/yolov8/nano?threshold=0.5" -F "file=@example.jpg"
```

The response is the same detections as a JSON list, with every PixelFlow field on every detection.

## Variants

| ID | Task | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `yolov8/nano` | detection | 7 MB | torch, onnx, coreml | AGPL-3.0 |
| `yolov8/small` | detection | 23 MB | torch, onnx, coreml | AGPL-3.0 |
| `yolov8/medium` | detection | 52 MB | torch, onnx, coreml | AGPL-3.0 |
| `yolov8/large` | detection | 88 MB | torch, onnx, coreml | AGPL-3.0 |
| `yolov8/xlarge` | detection | 137 MB | torch, onnx, coreml | AGPL-3.0 |

`runtime="auto"`, the default, picks CoreML on Apple silicon (`mps`) and torch on CUDA and CPU.
CoreML is by far the fastest way to run YOLOv8 on a Mac. To choose a runtime yourself, load
the model through a `ModelManager`, which takes `runtime`:

```python
from mozo import ModelManager

model = ModelManager().get_model("yolov8", "nano", runtime="onnx-fp32")   # pip install 'mozo[onnx]'
```

YOLOv8 has no segmentation variants in mozo. For masks from a YOLO, see
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

YOLOv8 predicts many overlapping boxes per object and removes the duplicates with non-maximum
suppression, as Ultralytics does. That step runs inside mozo, identically whichever runtime ran the
network. [YOLO26](/docs/models/yolov26.md) is the YOLO that needs no suppression step.

### Class ids are contiguous

The YOLO families number COCO's classes 0 to 79, so `person` is 0. [RF-DETR](/docs/models/rfdetr.md)
uses COCO's original ids, where `person` is 1. This matters when you filter by id, for example
before handing people to [ViTPose](/docs/models/vitpose.md):

```python
people = found.filter_by_class_id([0])
```

### Class names come from the checkpoint

The names are read from the weights, so a fine-tuned YOLOv8 brings its own vocabulary. Run from an
ONNX or CoreML graph, which carries no names, mozo reads them from a `labels` file published beside
the graph.

## Weights and licence

The weights are Ultralytics' `ultralytics/assets` release `v8.4.0`, redistributed unmodified. They
are **AGPL-3.0**, or covered by a commercial licence from Ultralytics.

Mozo's own code is Apache-2.0, and so is its YOLOv8 inference: it was written independently against
the checkpoint format, and it imports, contains and depends on nothing from `ultralytics`. The code
and the weights are separate works that travel together. But anything you export from the weights
inherits their terms, and **serving predictions from them over a network places AGPL-3.0 section 13
obligations on you.** Complying is the operator's responsibility.

Every variant ships with the licence and a NOTICE naming the release it came from. To run a server
that never offers AGPL weights, leave the YOLO families out of `MOZO_ENABLE`. The details are in
`mozo/vendors/yolov8_deploy/PROVENANCE.md`.
