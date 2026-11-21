# CLIP: classify from words, embed images and text

<!--
summary: OpenAI's CLIP in mozo: score an image against phrases you write, or get the vectors to search images by text. MIT.
label: CLIP
keywords: mozo, clip, zero-shot classification, image embeddings, text embeddings, image search
-->

CLIP is two networks trained together: an image tower and a text tower. Both end in a vector in
one shared space, so the photo of a forklift and the phrase "a forklift" land near each other.
Mozo gives you two things from it: a score for each phrase you name against an image, and the
vectors themselves, to search a collection of images by words in a database of your own.

## Quick start

```python
import mozo

model = mozo.get_model("clip/base")
scores = model.predict("example.jpg", ["a photo of people at a table",
                                       "a photo of a dog",
                                       "a photo of an empty room"])

for score in scores:
    print(score.class_name, round(score.confidence, 3))
```

```text
a photo of people at a table 0.259
a photo of an empty room 0.181
a photo of a dog 0.165
```

The vectors, one row per image or phrase:

```python
images = model.encode_image("example.jpg")                    # (1, 512)
phrases = model.encode_text(["people working on a laptop",
                             "a dog on a beach"])              # (2, 512)
images @ phrases.T                                            # [[0.244, 0.1008]]
```

Over HTTP, with `mozo start` running. `/predict` scores; `/encode` returns vectors, for images or
for phrases, not both in one call:

```bash
curl -X POST "http://localhost:8000/predict/clip/base?text=a%20photo%20of%20people%20at%20a%20table&text=a%20photo%20of%20a%20dog" \
  -F "file=@example.jpg"
```

```json
[{"class_id": 0, "class_name": "a photo of people at a table", "confidence": 0.259, "metadata": null},
 {"class_id": 1, "class_name": "a photo of a dog", "confidence": 0.165, "metadata": null}]
```

```bash
curl -X POST "http://localhost:8000/encode/clip/base" -F "file=@example.jpg"
curl -X POST "http://localhost:8000/encode/clip/base?text=people%20working%20on%20a%20laptop"
```

```json
{"model": "clip/base", "revision": "2026-08-23", "dim": 512, "embeddings": [[-0.00398, -0.00317, ...]]}
```

## Variants

| ID | Image tower | Vector | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|
| `clip/base` | ViT-B/32 | 512-d | 605 MB | torch | MIT |
| `clip/base-16` | ViT-B/16 | 512-d | 599 MB | torch | MIT |
| `clip/large` | ViT-L/14 | 768-d | 1,711 MB | torch | MIT |
| `clip/large-336` | ViT-L/14 at 336 px | 768-d | 1,712 MB | torch | MIT |

- `base` is the one most people mean by "CLIP".
- `base-16` is the same size with finer patches: slower and stronger.
- `large-336` is the most accurate and by far the slowest.

## Inputs and outputs

```python
model.predict(image, text, threshold=None)
model.encode_image(image)
model.encode_text(text)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. `encode_image` also takes a list of them. |
| `text` | `str` or `list[str]` | | One phrase or several, up to 77 tokens each. |
| `threshold` | `float` | `None` | Minimum score to keep. `None` returns every phrase, scored. |

`predict` returns a PixelFlow `Classifications`, best first. `class_name` is your phrase,
`class_id` its position in the list you passed, `confidence` the score. A blank phrase, no phrase,
or one over 77 tokens raises `ValueError`.

`encode_image` and `encode_text` return a float32 array of shape `(N, 512)` or `(N, 768)`,
L2-normalised, so a dot product is a cosine similarity. Always two-dimensional, even for one
input.

Over HTTP:

- **`/predict`** takes the image as `file`, and `text` repeated once per phrase. It is not
  comma-separated, so a phrase can contain a comma.
- **`/encode`** takes either `file` (repeat it for a batch) or `text` (repeat it), and returns
  `model`, `revision`, `dim` and `embeddings`.

## How it behaves

### The scores are cosine similarities, not probabilities

They are not softmaxed. They do not sum to one, they can be negative, and they are compressed: a
good match sits far below 1.0, so 0.259 does not mean "26% sure". Compare scores against each
other, or against a threshold you calibrate on your own images.

A softmax would be worse here. It is relative to whichever phrases you passed, so one phrase always
scores near 1 and adding a phrase moves every other number. Because mozo does not softmax, **adding
a phrase does not change the other phrases' scores**, and that is what makes a threshold mean
anything. Softmax the scores yourself if you want CLIP's published closed-set behaviour.

If you want a score that means something on its own, for a single yes-or-no question, use
[SigLIP 2](/docs/models/siglip2.md).

### Classify, or hand off the vectors

`predict` is the answer to "is there a person in this frame" when nobody trained a person detector
for your cameras: name the classes in words, no training, no labelled data.

`encode_image` and `encode_text` are a handoff, not an answer. A vector is useful only next to other
vectors, which means a store mozo does not provide. Embed your images once, keep the vectors in a
vector database, then encode a query phrase and let the database find the nearest. Searching a
million frames for "forklift near a person" is then one database query, not a million model runs.
Mozo is the model in that pipeline and nothing else: it does not index, search or persist.

### The towers load independently

Asking for phrases never builds the image tower, and the other way round. An ingest job that only
embeds images holds the image tower alone, and a query service that only embeds phrases holds
63.4M parameters instead of 151.3M (`base`). There is nothing to configure: the first call decides.

### An index is tied to the weights that built it

A vector from `base` means nothing against a vector from `large`, and nothing against a vector from
a different revision of `base`. So a stored index is tied to the variant **and** the revision that
built it, and changing either means embedding the whole collection again. `/encode` returns both
with every response for that reason, and in Python the model has `.revision` (`"2026-08-23"`).

### Parity

Bit-identical to `openai/CLIP` at commit `d05afc43` on all four variants: 104 comparisons across
preprocessing, token ids, image features, text features, similarities and `logits_per_image`, with
no tolerance.

Two traps a code review would miss and only a numeric check catches. The activation is QuickGELU
(`x * sigmoid(1.702x)`), not `nn.GELU`; the standard one moves the features by 0.4 without any
error. And upstream writes `logit_scale * image_features @ text_features.t()`, where `*` and `@`
share precedence, so the scale multiplies the features before the matrix product. Same arithmetic,
different rounding, 1.9e-06 apart.

## Weights and licence

Code and weights are MIT, © 2021 OpenAI, from the authors' own `openai/CLIP` repository. That
repository publishes one `LICENSE` and no separate weights licence, so the weights are taken to be
covered by it; the `NOTICE` beside every checkpoint says so. The five ResNet variants OpenAI also
publishes use a different image tower and are not carried. The extraction and the upstream commit
are recorded in `mozo/vendors/clip_deploy/PROVENANCE.md`.
