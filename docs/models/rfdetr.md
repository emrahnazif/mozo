# RF-DETR: detection, segmentation and keypoints

<!--
summary: Roboflow's real-time transformer detector in mozo: boxes, masks or person joints, Apache-2.0, on torch, ONNX and CoreML.
label: RF-DETR
keywords: mozo, rf-detr, object detection, instance segmentation, keypoints, roboflow
-->

RF-DETR is Roboflow's real-time transformer detector. It finds objects of COCO's 80 classes and
returns a box, a class and a confidence for each. The `seg-` variants add a mask to every box, and
`keypoint-preview` returns people with 17 joints each. It needs no overlap threshold: the model
predicts one box per object, so there is no non-maximum suppression to tune.

## Quick start

```python
import mozo

model = mozo.get_model("rfdetr/medium")
found = model.predict("example.jpg", threshold=0.5)

for detection in found[:4]:
    print(detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
person 0.97 [0.61, 122.33, 761.01, 1268.59]
person 0.96 [1303.81, 16.69, 1917.26, 1270.06]
laptop 0.94 [685.36, 783.82, 1354.26, 1135.56]
cell phone 0.93 [1163.78, 1082.97, 1449.97, 1182.17]
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/rfdetr/medium?threshold=0.5" -F "file=@example.jpg"
```

```json
[
  {"bbox": [0.61, 122.33, 761.01, 1268.59], "class_id": 1, "class_name": "person",
   "confidence": 0.97, "masks": null, "segments": null, "keypoints": null, "...": "..."}
]
```

Every detection carries every PixelFlow field, including the ones that are `null`.

## Variants

| ID | Task | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `rfdetr/nano` | detection | 366 MB | torch, onnx, coreml | Apache-2.0 |
| `rfdetr/small` | detection | 386 MB | torch, onnx, coreml | Apache-2.0 |
| `rfdetr/medium` | detection | 405 MB | torch, onnx, coreml | Apache-2.0 |
| `rfdetr/large` | detection | 136 MB | torch, onnx, coreml | Apache-2.0 |
| `rfdetr/seg-nano` | instance segmentation | 135 MB | torch, onnx, coreml | Apache-2.0 |
| `rfdetr/seg-small` | instance segmentation | 135 MB | torch, onnx, coreml | Apache-2.0 |
| `rfdetr/seg-medium` | instance segmentation | 143 MB | torch, onnx, coreml | Apache-2.0 |
| `rfdetr/seg-large` | instance segmentation | 145 MB | torch, onnx, coreml | Apache-2.0 |
| `rfdetr/keypoint-preview` | person keypoints | 164 MB | torch | Apache-2.0 |

`runtime="auto"`, the default, picks CoreML on Apple silicon (`mps`) and torch on CUDA and CPU.
Each graph is checked against the torch model before it is published: the same detections, classes
and count on real photographs, with boxes within 0.2 px and scores within 0.01. To choose one
yourself, ask a `ModelManager` for it by name:

```python
from mozo import ModelManager

model = ModelManager().get_model("rfdetr", "small", runtime="onnx-fp32")   # pip install 'mozo[onnx]'
model.runtime      # 'onnx-fp32'
```

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
- `class_id` and `class_name`: COCO's original ids, where `person` is 1 (they run to 90 with gaps).
- `confidence`: between 0 and 1.
- `masks` (`seg-` variants only): a list holding one boolean `HxW` array at the image's resolution.
- `keypoints` (`keypoint-preview` only): 17 joints, see below.

Over HTTP the same parameters are query parameters: `threshold`, and `labels` as a
comma-separated list (`labels=hardhat,vest`).

## How it behaves

### Class names come from the weights

The names ship with each variant, so they are the vocabulary the checkpoint was trained on, not an
assumption. A checkpoint of your own that carries no names returns `class_id` with `class_name`
unset. Pass `labels=[...]` to name them: mozo never guesses a name.

### Masks

The `seg-` variants answer the same call with a mask on every detection:

```python
found = mozo.get_model("rfdetr/seg-small").predict("example.jpg")
mask = found[0].masks[0]
mask.shape, mask.dtype      # ((1281, 1920), dtype('bool'))
```

### Keypoints

`keypoint-preview` returns people, each with COCO's 17 joints in COCO's order (nose, eyes, ears,
shoulders, elbows, wrists, hips, knees, ankles). Each joint has `x` and `y` in pixels of your image,
a `name`, and a `confidence`:

```python
found = mozo.get_model("rfdetr/keypoint-preview").predict("example.jpg", threshold=0.5)
nose = found[0].keypoints[0]
nose.name, nose.x, nose.y, round(nose.confidence, 3)    # ('nose', 1512.67, 299.17, 0.998)
```

**A joint the model cannot see still has its slot.** It comes back with a confidence near zero and
coordinates that mean nothing, so filter on the confidence before you read a position. Upstream
does the same, and mozo keeps it: dropping the hidden joints would renumber the rest, and a joint's
index is what tells you which joint it is.

Two things differ from the other variants:

- **Its class ids are its own.** The head has two slots, background at 0 and person at 1. The
  detection variants use COCO's ids running to 90.
- **It is torch only.** Roboflow published one operating point, 576×576, and only a checkpoint, so
  `runtime="auto"` gives you torch, and asking for ONNX says it is not published.

The model also predicts an uncertainty ellipse per joint. Mozo does not return it: a PixelFlow
`Detections` has nowhere to carry one, and reducing it to a single number the model never predicted
would be worse than leaving it out.

Roboflow reports 71.8 keypoint AP on COCO at 9.8 ms on a T4 under TensorRT FP16. That is their
measurement, not mozo's. For pose on boxes you already have, from any detector, see
[ViTPose](/docs/models/vitpose.md).

## Weights and licence

Code and weights are Apache-2.0, from Roboflow's `rf-detr` release. Every variant ships with its
licence and a NOTICE naming the exact upstream release, and mozo's extraction records the upstream
commit it was built from in `mozo/vendors/rfdetr_deploy/PROVENANCE.md`.
