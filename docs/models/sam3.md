# SAM 3: masks for a phrase or a click

<!--
summary: Meta's Segment Anything 3 in mozo: every instance of a phrase with masks, or the mask of one click. SAM License weights.
label: SAM 3
keywords: mozo, sam 3, segment anything, concept segmentation, text prompt, promptable segmentation
-->

SAM 3 segments from words or from a click, off one checkpoint. Name a concept, such as `"a cup"`,
and it returns every instance with a mask, a box and a score. Or click one thing and it returns
candidate masks, as [SAM 2](/docs/models/sam2.md) does. **Its weights carry Meta's SAM License,
not an open-source licence**: read [Weights and licence](/docs/models/sam3.md#weights-and-licence) before you deploy it.

## Quick start

```python
import mozo
from mozo.image import load_image

image = load_image("example.jpg")
model = mozo.get_model("sam3/sam3")

found = model.predict(image, "a laptop")
print(found[0].class_name, round(found[0].confidence, 2), found[0].bbox)
```

```text
a laptop 0.94 [685.97, 776.05, 1353.14, 1132.47]
```

Several concepts in one call return one result carrying several classes:

```python
found = model.predict(image, ["a person", "a cup"])
[detection.class_name for detection in found].count("a cup")     # 6
```

A click returns candidate masks, best first, as SAM 2 does:

```python
found = model.predict(image, points=[[1020, 960]], labels=[1])
round(found[0].confidence, 3), found[0].bbox     # (0.972, [698.0, 626.0, 1351.0, 1132.0])
```

Over HTTP, with `mozo start` running:

```bash
# Every instance of a concept
curl -X POST "http://localhost:8000/predict/sam3/sam3?text=a%20cup" -F "file=@example.jpg"

# One click
curl -X POST "http://localhost:8000/predict/sam3/sam3?point=1020,960&label=1" -F "file=@example.jpg"
```

## Variants

| ID | Weights (torch) | Runtimes | Licence |
|---|---|---|---|
| `sam3/sam3` | 3,450 MB | torch, coreml | SAM License |

Meta ships one model, not a ladder of sizes, so there is one variant. `runtime="auto"` picks the
CoreML image encoder on Apple silicon (`mps`), a further 1,728 MB; the checkpoint is downloaded
whichever runtime runs, because the CoreML graph covers only the image encoder.

## Inputs and outputs

```python
model.predict(image, text=None, threshold=0.5, *, points=None, labels=None, boxes=None,
              multimask_output=True, name=None)
model.prepare(image)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `text` | `str` or `list[str]` | `None` | One concept as a noun phrase, or several. Up to 32 tokens each; longer is truncated. |
| `threshold` | `float` | `0.5` | Minimum score for a concept's instances. Not used for clicks. |
| `points` | `(N, 2)` or `(B, N, 2)` | `None` | Clicks as `x, y` in pixels of your image. |
| `labels` | `(N,)` or `(B, N)` | `None` | `1` to include a point, `0` to exclude it. Required with `points`. |
| `boxes` | `(4,)` or `(B, 4)` | `None` | A box around the one thing to segment. |
| `multimask_output` | `bool` | `True` | Three candidate masks per click instead of one. |
| `name` | `str` or `list[str]` | `None` | What to call what a click pointed at. |

Give `text` or a click (`points`, `boxes`), not both.

For a concept, it returns a PixelFlow `Detections` with a box, a boolean mask and a score per
instance. `class_name` is the phrase that found it and `class_id` its index in `text`. For a click,
it returns what SAM 2 returns: one row per candidate mask, best first, with the model's predicted
IoU as `confidence` and `class_name=None` unless you pass `name`.

Raises a `ValueError` if you give both a phrase and a click, or neither; if a concept is empty; or
if points arrive without labels.

Over HTTP: `text` (repeat it for several concepts) and `threshold`, or the click parameters
`point`, `label`, `box`, `multimask` and `name`, as on
[SAM 2](/docs/models/sam2.md#inputs-and-outputs). `/prepare` and `?sha256=` work as they do there.

## How it behaves

### Name a concept

The phrase you search for is the class name every detection carries, so there is no fixed
vocabulary and nothing for mozo to guess. With several phrases, instances found by different
prompts may overlap: ask for `"car"` and `"vehicle"` and the same car comes back under both names.
That is what you asked for, so mozo does not suppress it.

### Point at one thing

`points` with `labels`, `boxes`, or both, work as on SAM 2: see
[SAM 2's prompts](/docs/models/sam2.md#every-prompt-is-a-set-of-points). Click masks match
`facebookresearch/sam3`'s `predict_inst` bit for bit, its hole filling included.

A phrase and a click in one call are refused:

```text
ValueError: SAM 3 takes a phrase or a click, not both. A box beside a phrase is an example of the concept, which mozo does not take yet; send the phrase alone, or the click alone.
```

In SAM 3 a box beside a phrase means "find more like this example", which mozo does not support
yet, so it refuses the call rather than reading it one way or the other.

Clicks run on every runtime SAM 3 publishes: from revision 2026-11-12 the CoreML graph carries both
halves of the image encoder. A graph pinned to revision 2026-08-20 carries the phrase half only and
refuses a click, saying so.

### One encode serves both

One image encode serves phrases and clicks, as in Meta's own image path. A click on a photograph
you already searched by phrase costs only its decoder, and `prepare(image)` warms both.

Concepts do not batch: the head takes one prompt at a time, so N concepts cost one encode plus N
decodes. Encoded prompts are cached too, so the same words on the next image skip the text tower.

### It wants a GPU

On the example photograph on an Apple M1 Max (`mps`, CoreML encoder), the first phrase took
1.77 s, the next two concepts on the same image 0.70 s together, and a click after that 0.12 s. On
CPU the image encode alone is about 5 s.

### Fixed square input

Inference runs at 1008×1008. SAM 3 squashes the image to that square instead of padding it, so the
aspect ratio is not preserved; masks and boxes come back in pixels of your image.

## Weights and licence

**Read this before deploying.** These weights are **not** open source. They carry Meta's **SAM
License**, which no other family in mozo does.

- It restricts what the weights may be used for: military, nuclear, espionage and weapons uses are
  prohibited.
- Those restrictions flow through to whoever you serve predictions to.
- It binds on use, not on signing.
- It must travel with the weights if you pass them on.

Complying is your responsibility as the operator. To run a server that never offers SAM 3, leave
it out of `MOZO_ENABLE`; a server that never offers a model never downloads or serves its weights.
For words-to-boxes under Apache-2.0, use [OWLv2](/docs/models/owlv2.md) or
[Grounding DINO](/docs/models/grounding-dino.md); for click-to-mask, [SAM 2](/docs/models/sam2.md)
or [EdgeTAM](/docs/models/edgetam.md).

Mozo's SAM 3 code is Apache-2.0. It derives from `transformers/models/sam3` (© The Meta AI Authors
and The HuggingFace Team), not from `facebookresearch/sam3`, whose code ships under the SAM License.
The code and the weights are separate works travelling together. The weights ship with the SAM
License and a NOTICE naming the exact upstream release; `mozo/vendors/sam3_deploy/PROVENANCE.md`
records the details.
