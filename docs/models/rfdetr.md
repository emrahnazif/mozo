# RF-DETR: detection, segmentation and keypoints

<!--
summary: Roboflow's real-time transformer detector in mozo: boxes, masks or person joints, Apache-2.0, on torch, ONNX, CoreML and TensorRT.
label: RF-DETR
keywords: mozo, rf-detr, object detection, instance segmentation, keypoints, roboflow, tensorrt
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
| `rfdetr/nano` | detection | 366 MB | torch, onnx, coreml, tensorrt fp32 and fp16 | Apache-2.0 |
| `rfdetr/small` | detection | 386 MB | torch, onnx, coreml, tensorrt fp32 and fp16 | Apache-2.0 |
| `rfdetr/medium` | detection | 405 MB | torch, onnx, coreml, tensorrt fp32 and fp16 | Apache-2.0 |
| `rfdetr/large` | detection | 136 MB | torch, onnx, coreml, tensorrt fp32 and fp16 | Apache-2.0 |
| `rfdetr/seg-nano` | instance segmentation | 135 MB | torch, onnx, coreml, tensorrt fp32 | Apache-2.0 |
| `rfdetr/seg-small` | instance segmentation | 135 MB | torch, onnx, coreml, tensorrt fp32 | Apache-2.0 |
| `rfdetr/seg-medium` | instance segmentation | 143 MB | torch, onnx, coreml, tensorrt fp32 | Apache-2.0 |
| `rfdetr/seg-large` | instance segmentation | 145 MB | torch, onnx, coreml, tensorrt fp32 | Apache-2.0 |
| `rfdetr/keypoint-preview` | person keypoints | 164 MB | torch | Apache-2.0 |

`runtime="auto"`, the default, picks CoreML on Apple silicon (`mps`), a TensorRT engine on CUDA
when TensorRT 11 or later is installed (see [below](#on-cuda-tensorrt)), and torch otherwise.
Each graph is checked against the torch model before it is published: the same detections, classes
and count on real photographs, with boxes within 0.2 px and scores within 0.01. To choose one
yourself, ask a `ModelManager` for it by name:

```python
from mozo import ModelManager

model = ModelManager().get_model("rfdetr", "small", runtime="onnx-fp32")   # pip install 'mozo[onnx]'
model.runtime      # 'onnx-fp32'
```

### On CUDA: TensorRT

With TensorRT 11 or later installed (`pip install tensorrt-cu12` or `-cu13`, matching your torch's
CUDA; see [Runtimes](/docs/deployment/runtimes.md#tensorrt)), `auto` runs RF-DETR on CUDA as
`tensorrt-fp32`: a TensorRT engine built from the published ONNX graph, in full IEEE fp32. It finds
exactly what torch finds, faster. The first load on a machine builds the engine, in about 12 s,
and keeps it in mozo's cache; later loads take under a second. Without TensorRT, `auto` stays on
torch. `runtime="tensorrt-fp16"` is faster again and has to be asked for.

Measured on an RTX 3090 with torch 2.13 and TensorRT 11.5, on 500 random COCO val2017 images.
Time is the model's forward pass. A detection "changes" when, against torch fp32, it scores above
0.52 on one side with no same-class partner (IoU above 0.5) above 0.5 on the other. *Input
rounded* is fp32 with only its input rounded to half precision: how often detections change on
their own.

| Variant | torch fp32 | `tensorrt-fp32` (`auto`) | `tensorrt-fp16` | Input rounded |
|---|---|---|---|---|
| `nano` | 8.6 ms, AP 50.21 | 3.5 ms, 2.4×, AP 50.20, 0 changes | 1.3 ms, 6.4×, AP 50.25, 4 of 2,301 (0.2%) | 0 |
| `small` | 11.0 ms, AP 55.39 | 5.4 ms, 2.0×, AP 55.39, 0 | 1.9 ms, 5.9×, AP 55.39, 8 of 2,531 (0.3%) | 3 |
| `medium` | 12.9 ms, AP 57.22 | 6.9 ms, 1.9×, AP 57.22, 0 | 2.2 ms, 5.9×, AP 57.08, 14 of 2,601 (0.5%) | 3 |
| `large` | 17.6 ms, AP 59.23 | 10.6 ms, 1.7×, AP 59.23, 0 | 2.9 ms, 6.0×, AP 59.26, 6 of 2,648 (0.2%) | 1 |
| `seg-nano` | 12.8 ms, AP 51.40 | 5.0 ms, 2.6×, AP 51.39, 0 | | |
| `seg-small` | 14.7 ms, AP 54.08 | 6.3 ms, 2.3×, AP 54.08, 0 | | |
| `seg-medium` | 18.4 ms, AP 56.65 | 8.5 ms, 2.2×, AP 56.63, 0 | | |
| `seg-large` | 24.2 ms, AP 58.36 | 12.3 ms, 2.0×, AP 58.35, 0 | | |

The segmentation engines' masks differ from torch's on at most 0.05% of pixels. RF-DETR is limited
by launching kernels at one image at a time, which a compiled engine removes, so even the fp32
engine is faster. Torch's own fp16 does not help: under autocast it is slower than fp32 for `nano`
to `medium`.

`tensorrt-fp16` is offered for the four detection variants, the ones it was measured on. It
changes a few borderline detections, about as many as `torch-fp16` does on OWLv2, and AP does not
move. It downloads an fp16 graph of 55–63 MB to build from, and its first build takes about 45 s.

The fp32 engine is closer to exact fp32 than torch's default CUDA path, which runs convolutions in
TF32. Against torch with `torch.backends.cudnn.allow_tf32 = False`, the engine's boxes agree to
0.01 px on the example photograph.

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
