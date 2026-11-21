# ViTPose: person keypoints for boxes you already have

<!--
summary: ViTPose++ in mozo: give it a frame and person boxes from any detector, get COCO's 17 joints on each. Apache-2.0, torch and ONNX.
label: ViTPose
keywords: mozo, vitpose, pose estimation, keypoints, human pose
-->

ViTPose is top-down human pose estimation. You give it a frame and the boxes of the people in it,
and it hands those same detections back with COCO's 17 joints attached to each. It is the only
family in mozo that takes detections as input: pair it with anything that produces boxes, such as
[RF-DETR](/docs/models/rfdetr.md), a YOLO, a tracker, or a rectangle someone drew.

## Quick start

```python
import mozo

found = mozo.get_model("rfdetr/nano").predict("example.jpg")
people = found.filter_by_class_id(1)            # RF-DETR's person id is 1

posed = mozo.get_model("vitpose/small").predict("example.jpg", people)

nose = posed[0].keypoints[0]
print(posed[0].class_name, nose.name, nose.x, nose.y, round(nose.confidence, 2))
```

```text
person nose 424.68 429.45 0.97
```

`class_name` is still the detector's: ViTPose adds joints and changes nothing else.

Over HTTP, with `mozo start` running, pass one `box` per person as `x1,y1,x2,y2`:

```bash
curl -X POST "http://localhost:8000/predict/vitpose/small?box=300,279,615,829&box=843,218,1453,823" \
  -F "file=@example.jpg"
```

A box typed into a URL carries no class, score or tracker id, so those come back empty. Use the
Python API to keep a detector's results intact.

## Variants

| ID | Task | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `vitpose/small` | person keypoints | 133 MB | torch, onnx | Apache-2.0 |
| `vitpose/base` | person keypoints | 502 MB | torch, onnx | Apache-2.0 |
| `vitpose/large` | person keypoints | 1.7 GB | torch, onnx | Apache-2.0 |
| `vitpose/huge` | person keypoints | 3.6 GB | torch | Apache-2.0 |

`runtime="auto"`, the default, picks torch on every device. The ONNX graphs are for serving without
torch, not for speed: see [The ONNX graphs](#the-onnx-graphs-are-for-portability-not-speed).

## Inputs and outputs

```python
model.predict(image, detections, labels=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | The **whole frame**: a file path, encoded image bytes, or an `HxWx3` RGB array. Not a crop. |
| `detections` | PixelFlow `Detections` | | Where the people are. Every row is posed. Required. |
| `labels` | `list[str]` | `None` | Joint names for this call, in the model's joint order, replacing the published ones. |

Returns a copy of `detections` with `keypoints` set on every row: the same boxes, class ids, scores,
names and tracker ids, plus 17 joints each, in COCO's order (nose, eyes, ears, shoulders, elbows,
wrists, hips, knees, ankles). Each joint has `x` and `y` in pixels of your frame, an `id`, a `name`
and a `confidence`.

An **empty** `Detections` is answered with an empty result: a frame with nobody in it is an answer,
not an error.

Over HTTP the only parameter is `box`, repeated for each person.

## How it behaves

### Both arguments are required

There is no shortcut that treats the whole image as one person when you pass no detections. That
would be a guess about what you meant, and on a street scene it returns one confident skeleton
stretched across the frame.

### It does not filter what you give it

Pass the box of a car and you get seventeen confident joints on a car. A heatmap model always puts
a nose somewhere and scores it well, so a low confidence does not save you.

Which boxes are people is a fact about your pipeline, not about this model, so mozo does not decide
it for you. Filter first with `filter_by_class_id`, and mind whose ids you are using: RF-DETR uses
COCO's original ids, where `person` is 1, and the YOLO families use contiguous ids, where it is 0.

### Give it the frame, not a crop

The crop the model wants is **larger than the box it is given**. Each box is first widened or
heightened to the input's 3:4 aspect ratio, then padded by a further 1.25×. For a 50×140 person
that is roughly a 131×175 crop, taken from the surrounding frame.

So a pre-cropped person is not the same input. Those pixels are already gone, the padding falls
back to black, and a wrist just outside the box, which the real crop would have recovered, is lost.
If a crop is all you have, pass it with a box covering its full extent and expect slightly worse
joints near the edges.

### Read the confidence before the position

Each joint carries the peak of its heatmap. It is not a probability and not comparable across
models, but it tracks visibility well. In the example above, the first person sits at a table:
their nose and eyes score 0.85 to 0.97, while their ankles, out of sight, score 0.23 and 0.13.
**Filter on it before you read a coordinate**: a joint the model cannot see still gets a position,
and that position means nothing.

### Every variant is ViTPose++

Seven checkpoints exist upstream. Mozo publishes the four that are ViTPose++, the mixture-of-experts
revision, named by size. The other three are the original ViTPose, which ViTPose++ beats at every
size, and the smallest of them is 344 MB against `small`'s 133 MB.

Every block of ViTPose++ picks one of six dataset experts. Mozo always runs COCO's, because the
published heads match COCO's; the other five give wrong answers with these heads, so there is no
option to pick them.

### Cost

One forward pass handles the whole frame: N boxes are N crops batched together. So the cost of a
frame depends on how many people are in it, and the cost per person is the number that transfers.
Measured on five people, `torch-fp32`, on an Apple M-series laptop:

| Variant | ms per person, MPS | ms per person, CPU |
|---|---|---|
| `small` | 9.9 | 23.1 |
| `base` | 14.8 | 39.3 |
| `large` | 28.5 | 111.6 |
| `huge` | 49.5 | 190.3 |

### The ONNX graphs are for portability, not speed

`small`, `base` and `large` publish an `onnx-fp32` graph, verified to return the same joints as
their checkpoint. On the machine above it is **slower** than torch, 35.5 against 23.1 ms per person
on `small` and 299 against 112 on `large`, because ONNX Runtime has no Metal path and runs on the
CPU. `auto` never picks it on CPU or MPS, so asking for it is a deliberate choice: serving without
torch, or reaching a runtime torch cannot. The graphs are smaller than the checkpoints, 97 MB
against 133 MB for `small`, because fixing the expert to COCO's lets the exporter drop the other
five.

**`huge` has no graph.** Its weights are 2.5 GB even after that, past protobuf's single-file limit,
so ONNX would store them beside the graph in a second file. Mozo publishes every artifact as one
file, so it publishes none.

**There is no CoreML graph either, and that was measured.** It converts and its joints agree to
0.0005 px, but it is no faster than torch on MPS: 22.9 ms against 22.6 for five people on `small`,
44.0 against 44.2 on `base`. Publishing it would make it the `auto` choice on Apple silicon and cost
a `coremltools` install and a second copy of every variant, for nothing.

### Parity

Joints land within **0.005 pixels** of `transformers` on every variant. That is half of PixelFlow's
0.01 rounding step, so it is the floor, and every variant sits on it. Underneath, the heatmaps are
bit-identical.

## Weights and licence

Code and weights are Apache-2.0. Mozo's inference is extracted from `transformers`' ViTPose
implementation (© The University of Sydney and The HuggingFace Team), not from the authors' mmpose
release, because the `transformers` implementation is the one the published PyTorch checkpoints
reproduce. Every variant ships with its licence and a NOTICE naming its source. The extraction,
including the two preprocessing steps rewritten to avoid SciPy, is recorded in
`mozo/vendors/vitpose_deploy/PROVENANCE.md`.
