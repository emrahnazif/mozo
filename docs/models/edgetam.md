# EdgeTAM: segment what you click, on less hardware

<!--
summary: Meta's EdgeTAM in mozo: SAM 2's click-and-box segmentation in a 56 MB model, Apache-2.0 code and weights.
label: EdgeTAM
keywords: mozo, edgetam, promptable segmentation, segment anything, on-device
-->

EdgeTAM segments the thing you point at, as [SAM 2](/docs/models/sam2.md) does, from a model built
for phones. It takes the same prompts (clicks, a box, or both) and returns the same result. Its
image path has 9.1M parameters against SAM 2 tiny's 31.4M, and its masks agree with SAM 2 tiny's
at 0.94 IoU on box prompts. Use it when the encode time matters more than the last bit of mask
quality.

## Quick start

A click on the laptop in the example photograph:

```python
import mozo
from mozo.image import load_image

image = load_image("example.jpg")
model = mozo.get_model("edgetam/edgetam")

found = model.predict(image, points=[[1020, 960]], labels=[1])

for candidate in found:
    print(round(candidate.confidence, 3), candidate.bbox)
```

```text
0.977 [746.0, 781.0, 1349.0, 1130.0]
0.437 [512.0, 781.0, 1649.0, 1281.0]
0.113 [756.0, 868.0, 1078.0, 1115.0]
```

```python
found = model.predict(image, boxes=[685, 783, 1355, 1136], multimask_output=False, name="laptop")
found[0].class_name, round(found[0].confidence, 3)     # ('laptop', 0.985)
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/edgetam/edgetam?point=1020,960&label=1" -F "file=@example.jpg"
```

## Variants

| ID | Weights (torch) | Runtimes | Licence |
|---|---|---|---|
| `edgetam/edgetam` | 56 MB | torch | Apache-2.0 |

EdgeTAM is published as one model, so there is one variant.

## Inputs and outputs

```python
model.predict(image, points=None, labels=None, boxes=None, multimask_output=True, name=None)
model.prepare(image)
```

The parameters, the result and the HTTP query parameters (`point`, `label`, `box`, `multimask`,
`name`, `sha256`) are the same as SAM 2's: see
[SAM 2's inputs and outputs](/docs/models/sam2.md#inputs-and-outputs). In short:

- `points` with `labels` (`1` include, `0` exclude), `boxes`, or both.
- One row per candidate mask: a boolean `HxW` mask, its box, and the model's predicted IoU as
  `confidence`.
- `class_name` is `None` unless you pass `name`; `class_id` is the index of the prompt.

## How it behaves

EdgeTAM behaves as SAM 2 does, and SAM 2's page explains each point in full:

- **A click returns three candidates**, best first; a box with `multimask_output=False` returns one.
  See [three candidates for a click](/docs/models/sam2.md#three-candidates-for-a-click).
- **No name unless you give one.** See [no name unless you give one](/docs/models/sam2.md#no-name-unless-you-give-one).
- **The image encode is cached** by pixel content, five images per model, so a second prompt on
  the same image pays only for the decoder.
- **`prepare(image)`, `/prepare` and `?sha256=`** move the encode to when the image opens. See
  [prepare an image before the first prompt](/docs/models/sam2.md#prepare-an-image-before-the-first-prompt).

### Cost

On CPU at 2 megapixels, EdgeTAM encodes an image in 272 ms and decodes a prompt in 33 ms, against
439 ms for SAM 2 tiny's encode. On the example photograph on an Apple M1 Max (`mps`), the first
click took 0.26 s and the box after it 0.05 s.

## Weights and licence

Code and weights are Apache-2.0. Mozo's code is an extraction of `facebookresearch/EdgeTAM`, with its
image trunk from `huggingface/pytorch-image-models`; it is a separate extraction from SAM 2's, and
`mozo/vendors/edgetam_deploy/PROVENANCE.md` records both upstream commits. The variant ships with its
licence and a NOTICE naming the exact upstream release.
