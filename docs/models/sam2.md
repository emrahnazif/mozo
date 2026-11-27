# SAM 2: segment what you click

<!--
summary: Meta's Segment Anything 2 in mozo: a click or a box in, the mask of what you pointed at out, Apache-2.0 code and weights.
label: SAM 2
keywords: mozo, sam 2, segment anything, promptable segmentation, interactive segmentation, video object tracking
-->

SAM 2 segments the thing you point at. Give it a click, a box, or both, and it returns a mask with
its box and the model's own estimate of how good the mask is. It does not know what it segmented:
the mask is unnamed unless you name it. Mark objects on one frame of a video and it follows them
through the frames after, a mask per object per frame ([tracking](#tracking-objects-through-a-video)). For masks of everything matching a phrase, see
[SAM 3](/docs/models/sam3.md); for a smaller, faster model with the same prompts, see
[EdgeTAM](/docs/models/edgetam.md).

## Quick start

A click on the laptop in the example photograph:

```python
import mozo
from mozo.image import load_image

image = load_image("example.jpg")
model = mozo.get_model("sam2/tiny")

found = model.predict(image, points=[[1020, 960]], labels=[1])

for candidate in found:
    print(round(candidate.confidence, 3), candidate.bbox, candidate.masks[0].shape)
```

```text
0.982 [748.0, 777.0, 1353.0, 1132.0] (1281, 1920)
0.906 [513.0, 376.0, 1529.0, 1259.0] (1281, 1920)
0.886 [749.0, 791.0, 1347.0, 1133.0] (1281, 1920)
```

One click returns three candidate masks, best first: here the laptop, then a larger region
around it, then the laptop again with a slightly different edge. A box is usually unambiguous, so ask for one mask:

```python
found = model.predict(image, boxes=[685, 783, 1355, 1136], multimask_output=False, name="laptop")
found[0].class_name, round(found[0].confidence, 3)     # ('laptop', 0.983)
```

Over HTTP, with `mozo start` running:

```bash
# A click: three candidate masks, best first
curl -X POST "http://localhost:8000/predict/sam2/tiny?point=1020,960&label=1" -F "file=@example.jpg"

# A box, one mask, named by you
curl -X POST "http://localhost:8000/predict/sam2/tiny?box=685,783,1355,1136&multimask=false&name=laptop" \
  -F "file=@example.jpg"
```

## Variants

| ID | Weights (torch) | Runtimes | Licence |
|---|---|---|---|
| `sam2/tiny` | 156 MB | torch | Apache-2.0 |
| `sam2/small` | 184 MB | torch | Apache-2.0 |
| `sam2/base_plus` | 324 MB | torch | Apache-2.0 |
| `sam2/large` | 898 MB | torch | Apache-2.0 |

SAM 2 also publishes ONNX graphs, and CoreML graphs for every size except `base_plus`. Mozo does
not run them yet. A promptable model exports as several graphs and needs a runner that keeps the
image encode and the prompt decode apart, so asking for one is refused rather than quietly
answered with torch:

```text
RuntimeError_: 'onnx-fp32' is published for this model, but this family can only execute torch. Available: torch-fp32
```

## Inputs and outputs

```python
model.predict(image, points=None, labels=None, boxes=None, multimask_output=True, name=None)
model.prepare(image)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `points` | `(N, 2)` or `(B, N, 2)` | `None` | Clicks as `x, y` in pixels of your image. |
| `labels` | `(N,)` or `(B, N)` | `None` | `1` to include a point, `0` to exclude it. Required with `points`. |
| `boxes` | `(4,)` or `(B, 4)` | `None` | Boxes as `x1, y1, x2, y2` in pixels of your image. |
| `multimask_output` | `bool` | `True` | Three candidate masks per prompt instead of one. |
| `name` | `str` or `list[str]` | `None` | What to call what you pointed at: one name, or one per prompt. |

Returns a PixelFlow `Detections` with one row per candidate mask. Each row has:

- `masks`: a list holding one boolean `HxW` array at your image's resolution.
- `bbox`: the box enclosing the mask.
- `confidence`: the model's predicted IoU for the mask, not a class probability.
- `class_id`: the index of the prompt that produced the row, so a batch of prompts stays separable.
- `class_name`: `None`, or the `name` you passed.

Raises a `ValueError` if no prompt is given, if points arrive without labels, or if `name` is a list
whose length is not the number of prompts.

Over HTTP the same prompts are query parameters:

| Parameter | Meaning |
|---|---|
| `point` | A click as `x,y`. Repeat it for several. |
| `label` | `1` to include the matching `point`, `0` to exclude it. One per point, in the same order. |
| `box` | A box as `x1,y1,x2,y2`. Can be combined with points. |
| `multimask` | `true` (default) for three candidate masks, `false` for one. |
| `name` | What to call what you pointed at. |
| `sha256` | The image kept by `/prepare`, sent instead of the file. See below. |

## How it behaves

### Every prompt is a set of points

A click is a point with a label: `1` to include, `0` to exclude. Neither SAM 2 nor EdgeTAM has a
separate box input: a box is written as its two corners carrying reserved labels, and mozo does that
for you. Points and a box can be combined, for example a box plus a click to exclude a part:

```python
found = model.predict(image, boxes=[685, 783, 1355, 1136], points=[[900, 700]], labels=[0])
```

`labels` has no default. Guessing between include and exclude returns a confident mask of the
wrong thing, so points without labels raise an error.

### Three candidates for a click

With `multimask_output=True`, the default, each prompt returns three masks ranked by the model's
predicted IoU. That is the right setting for a single click, which is genuinely ambiguous: did you
mean the handle, the door or the car? Take the first row, or show all three and let the user
choose. A box usually says what you meant, and `multimask_output=False` gives one tighter mask.

### No name unless you give one

A click does not say what it clicked, and mozo will not invent a name: a name comes from the
weights or from you. Pass `name="laptop"` if you know what you pointed at.

### The image encode is cached

The image encoder is the expensive part, and it depends only on the image. Mozo caches its result
by pixel content, so a second prompt on the same image pays only for the small decoder. On the
example photograph on an Apple M1 Max (`mps`), `sam2/tiny` took 0.41 s for the first click and
0.045 s for the box after it. On CPU at 2 megapixels, `sam2/tiny` encodes in about 439 ms.

The cache holds five images per model. Preparing more drops the oldest, which is then encoded again
when it is next used: slower, never wrong.

### Prepare an image before the first prompt

`prepare(image)` runs the encode before the first prompt arrives. Call it when an image opens, on a
background thread if you have one, and the first click costs what every later click does. On Apple
silicon, `sam2/large`'s first click falls from 522 ms to 16 ms, because the encode moved to when the
picture appeared. A click that arrives while `prepare` is still running waits for that encode
instead of starting a second one.

```python
model.prepare(image)                                          # when the image opens
found = model.predict(image, points=[[1020, 960]], labels=[1])
```

Pass the same array to both calls, so neither decodes the file again.

Over HTTP, `POST /prepare/sam2/{variant}` with the file encodes it, keeps the image, and answers with
the sha256 of the file's bytes. Each click then sends `?sha256=` and no file, a few hundred bytes
instead of the picture:

```bash
curl -X POST "http://localhost:8000/prepare/sam2/large" -F "file=@example.jpg"     # {"sha256": "3f9a..."}
curl -X POST "http://localhost:8000/predict/sam2/large?sha256=3f9a...&point=1020,960&label=1"
```

If the server does not hold that image (another worker, a restart, or it was dropped), it answers
404. Send the file again with the same `?sha256=` and it is kept for the clicks after. Only these two
requests keep an image; an ordinary upload is not kept. `MOZO_PREPARED_MB` sets how much memory kept
images may use, 512 MB by default, and the least recently used goes first.

This is for a client on the same machine or network as the server, clicking interactively: an
annotation tool talking to a GPU machine down the hall, for example. Without it, every click uploads
the picture again. Code running mozo in-process needs only `prepare(image)`. A client that never
sends `?sha256=` is answered exactly as before.

[EdgeTAM](/docs/models/edgetam.md), [SAM 3](/docs/models/sam3.md) and
[OWLv2](/docs/models/owlv2.md) can be prepared too; `GET /models` says which families can.

## Tracking objects through a video

`tracker()` starts a session that follows the objects you mark through a video, one frame at a time:

```python
import mozo
import pixelflow as pf

model = mozo.get_model("sam2/tiny")
tracker = model.tracker()

for index, frame in enumerate(pf.VideoReader("street.mp4")):
    if index == 0:
        found = tracker.update(frame, boxes=[[40, 60, 300, 480], [520, 90, 700, 400]],
                               name=["cyclist", "bus"])
    else:
        found = tracker.update(frame)
    for row in found:
        print(index, row.tracker_id, row.class_name, round(row.confidence, 3), row.bbox)
```

- **Prompts start objects.** `update` takes the same `points`, `labels`, `boxes` and `name` as
  `predict`, and each prompt (a box, a set of clicks, or a box with clicks) starts one object on
  the frame it comes with. The first frame needs at least one; any later frame may start more.
- **One row per object in view.** `tracker_id` (and `class_id`) is the object's id, 0, 1, 2… in
  the order objects were started; `class_name` is the name you gave it; `confidence` is SAM 2's
  probability that the object is visible in this frame (PixelFlow keeps three decimals, so a
  confident object reads `1.0`); the mask is at the frame's own size. An object out of view has no
  row, and comes back under the same id when SAM 2 sees it again. (So has one SAM 2 judges present
  but whose mask is empty, which is rare: PixelFlow keeps no detection with an empty box.)
- **Constant memory.** The tracker keeps each object's prompted frame and the last 15 frames, all
  a later frame can read, so a long video or a camera runs in the same memory as a short clip. Any
  number of trackers can share one model, one per stream.
- **Forward only.** SAM 2 reads the frames before a frame, never after. To track backwards from a
  marked frame, give the tracker the frames in reverse order: that is exactly what Meta's
  predictor computes with `reverse=True`.
- **Cost grows with objects.** Every object costs a pass of SAM 2's memory attention per frame, the
  same in every size. On an RTX 3060, in fp32, a frame takes 314 ms for `tiny` with one object
  and 751 ms with three (646 and 1,087 ms for `large`); on a 6-thread CPU, about 2.2 s for `tiny`
  with one object. Tracking is offline work at these speeds, not live video.

In a workflow, the `sam2_track` node takes a detector's boxes on the first frame that has any and
follows those objects through the video, each keeping the class of the detection that started it.
Over HTTP, that workflow runs on an uploaded video through `/workflow/process`.

**What it matches.** The tracker is Meta's `SAM2VideoPredictor` frame by frame, fed the same
frames: bit-identical on CUDA (RTX 3060, torch 2.13, CUDA 12.6) in fp32, every variant, and also on
CPU. The same gate checks `predict` against Meta's `SAM2ImagePredictor` (a click, single-mask
clicks and a box), bit-identical on both too. Two differences are Meta's own: it preprocesses a frame slightly differently
from its image predictor, so the first mask can differ from `predict`'s by a few pixels; and on a
clip shorter than 16 frames its answer depends on the clip's length, which a stream does not have
(the tracker behaves as a video of 16 frames or more). Meta's predictor fills small holes in masks
only when its optional CUDA extension is compiled; the tracker matches it without that extension.

## Weights and licence

Code and weights are Apache-2.0. Mozo's code is an extraction of `facebookresearch/sam2`'s image predictor
and video tracker, and `mozo/vendors/sam2_deploy/PROVENANCE.md` records the upstream commit it was built
from. Every variant ships with its licence and a NOTICE naming the exact upstream release.
