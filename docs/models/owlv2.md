# OWLv2: detect anything you can name

<!--
summary: Google's open-vocabulary detector in mozo: name things in words and get boxes, no class list, Apache-2.0 code and weights.
label: OWLv2
keywords: mozo, owlv2, open-vocabulary detection, zero-shot detection, text prompt
-->

OWLv2 finds objects you name in words. You pass phrases such as `"a laptop"` or `"a coffee cup"`,
and it returns a box for each instance, named by the phrase that found it. There is no class list,
no fine-tuning and no vocabulary agreed in advance. It returns boxes only; for masks, give its
boxes to [SAM 2](/docs/models/sam2.md) or [EdgeTAM](/docs/models/edgetam.md).

## Quick start

```python
import mozo

model = mozo.get_model("owlv2/base-ensemble")
found = model.predict("example.jpg", ["a laptop", "a coffee cup", "a phone"])

for detection in found[:4]:
    print(detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
a laptop 0.72 [761.75, 782.39, 1355.03, 1148.92]
a coffee cup 0.59 [933.21, 481.59, 1044.93, 603.8]
a phone 0.54 [1170.54, 1083.47, 1467.39, 1181.79]
a coffee cup 0.53 [777.02, 642.93, 868.82, 764.45]
```

Over HTTP, with `mozo start` running. Repeat `text` for each phrase:

```bash
curl -X POST "http://localhost:8000/predict/owlv2/base-ensemble?text=a%20laptop&text=a%20coffee%20cup" \
  -F "file=@example.jpg"
```

## Variants

| ID | Weights (torch) | Runtimes | Licence |
|---|---|---|---|
| `owlv2/base-ensemble` | 620 MB | torch | Apache-2.0 |
| `owlv2/base` | 620 MB | torch | Apache-2.0 |
| `owlv2/large-ensemble` | 1,751 MB | torch | Apache-2.0 |
| `owlv2/large` | 1,751 MB | torch | Apache-2.0 |

The `-ensemble` variants average Google's self-trained and fine-tuned checkpoints, and they are
what the paper reports. `base` and `large` are self-training only. Start with `base-ensemble`.

## Inputs and outputs

```python
model.predict(image, text, threshold=0.1)
model.prepare(image)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `text` | `str` or `list[str]` | | One phrase or several, up to 16 tokens each. Longer phrases are truncated. |
| `threshold` | `float` | `0.1` | Minimum score to keep a detection. |

Returns a PixelFlow `Detections`, best first. Each detection has a `bbox` in pixels of your image,
a `confidence`, `class_name` set to the phrase that found it, and `class_id` set to that phrase's
index in `text`. An empty phrase, or no phrase, raises a `ValueError`.

`prepare(image)` encodes the image ahead of the first phrase, see below.

Over HTTP: `text` (repeat it for several phrases) and `threshold` as query parameters, plus
`POST /prepare/owlv2/{variant}` and `?sha256=`, which work as they do for
[SAM 2](/docs/models/sam2.md#prepare-an-image-before-the-first-prompt).

## How it behaves

### Phrases go in as you write them

Mozo passes each phrase to the model verbatim. OWLv2's own examples write `"a photo of a cat"`,
but that wording is yours to choose; mozo applies no template. All phrases share one image pass
and one text pass, so asking for twenty costs barely more than asking for one.

Each candidate box is scored against every phrase and keeps only its best one. So the same box
never comes back twice under two names: ask for `"car"` and `"vehicle"` and each detection picks
one.

### Scores run low

Scores are similarities passed through a sigmoid, not class probabilities. They run lower than a
closed-vocabulary detector's: 0.3 is a confident detection here, which is why the default floor
is 0.1 and not the 0.5 that [RF-DETR](/docs/models/rfdetr.md) uses.

### Nothing is suppressed

The published postprocessing has no non-maximum suppression, so mozo has none either. The model
scores every image patch against every phrase and predicts one box per patch, so a large object
can come back as several overlapping boxes. Removing them is your policy to choose, not
something mozo decides.

### The image encode is cached

The image and text towers are independent, so the image encode depends only on the image. Mozo
caches it by pixel content: asking the same photograph a second question pays only for the text
and the heads. On the example photograph on an Apple M1 Max (`mps`), the first three phrases took
0.71 s and a new phrase on the same image 0.31 s. Call `model.prepare(image)` when the image
opens, and even the first question is fast.

This is the main difference from [Grounding DINO](/docs/models/grounding-dino.md), which mixes
the text into the image features and so has nothing to cache. If you ask one image many questions,
OWLv2 is far cheaper. If your prompt is a description rather than a noun (`"the mug on the left"`),
Grounding DINO reads it better.

### Fixed square input

The image is padded at the bottom and right to a square, then resized to 960 pixels for `base` and
1008 for `large`. The aspect ratio is preserved, but a 4:3 photograph spends a quarter of its
patches on padding.

### Verified on every stage

Mozo's extraction is checked against `transformers` on the tokenizer, preprocessing, both towers,
all three heads and the final boxes, with no tolerance: 226 comparisons on `base-ensemble`, every
one bit-identical. On CPU at 2 megapixels it runs within 1% of `transformers` on the same weights.

## Weights and licence

Code and weights are Apache-2.0, on all four checkpoints. This is mozo's permissively licensed way
to ask a model a question in words: [SAM 3](/docs/models/sam3.md) answers a similar question with
masks, but its weights carry Meta's SAM License.

Mozo's code derives from `transformers/models/owlv2` (© The Google Research Authors and The
HuggingFace Team), not from Google's JAX release in `google-research/scenic`. The tokenizer and its
vocabulary are OpenAI's CLIP (MIT). `mozo/vendors/owlv2_deploy/PROVENANCE.md` records the exact
upstream source.
