# SigLIP 2: a score you can threshold, and embeddings

<!--
summary: Google's SigLIP 2 in mozo: score one phrase against an image as a yes-or-no probability, or get vectors for image search. Apache-2.0.
label: SigLIP 2
keywords: mozo, siglip 2, zero-shot classification, image embeddings, multilingual, image search
-->

SigLIP 2 has the same shape as [CLIP](/docs/models/clip.md): an image tower and a text tower whose
vectors share one space. It was trained with a different loss, and that changes what its score
means. CLIP learned to rank captions against each other, so a CLIP score only means something next
to the other scores. SigLIP learned each image and caption as a yes-or-no pair, so its score means
something on its own: one phrase is a complete question.

## Quick start

```python
import mozo

model = mozo.get_model("siglip2/base-224")
scores = model.predict("example.jpg", ["a photo of people",
                                       "a photo of an elephant in a pool"])

for score in scores:
    print(score.class_name, round(score.confidence, 3))
```

```text
a photo of people 0.035
a photo of an elephant in a pool 0.0
```

The vectors, one row per image or phrase:

```python
images = model.encode_image("example.jpg")                  # (1, 768)
phrases = model.encode_text(["people working on a laptop"])  # (1, 768)
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/siglip2/base-224?text=a%20photo%20of%20people" \
  -F "file=@example.jpg"
```

```json
[{"class_id": 0, "class_name": "a photo of people", "confidence": 0.035, "metadata": null}]
```

```bash
curl -X POST "http://localhost:8000/encode/siglip2/base-224" -F "file=@example.jpg"
```

```json
{"model": "siglip2/base-224", "revision": "2026-08-23", "dim": 768, "embeddings": [[0.00719, 0.06351, ...]]}
```

## Variants

| ID | Towers | Vector | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|
| `siglip2/base-224` | ViT-B/16 at 224 px | 768-d | 1,501 MB | torch | Apache-2.0 |
| `siglip2/base-256` | ViT-B/16 at 256 px | 768-d | 1,501 MB | torch | Apache-2.0 |
| `siglip2/so400m-384` | So400m, patch 14, 384 px | 1152-d | 4,544 MB | torch | Apache-2.0 |
| `siglip2/so400m16-256` | So400m, patch 16, 256 px | 1152-d | 4,543 MB | torch | Apache-2.0 |
| `siglip2/giant-384` | giant image tower, So400m text tower | 1536-d | 7,488 MB | torch | Apache-2.0 |

- `base-224` and `base-256` are the small, fast end.
- `so400m-384` is the quality point; `so400m16-256` is the same tower at roughly a third of the
  cost.
- `giant-384` is the strongest, and the most downloaded of the five.

Google publishes fifteen fixed-resolution variants; these five took 89% of the downloads. The
`-naflex` variable-resolution variants use a different image tower and are not carried.

Every variant is multilingual: the text tower uses Gemma's 256,000-piece vocabulary, so there is no
separate multilingual model to choose.

## Inputs and outputs

```python
model.predict(image, text, threshold=None)
model.encode_image(image)
model.encode_text(text)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. `encode_image` also takes a list of them. |
| `text` | `str` or `list[str]` | | One phrase or several, up to 64 tokens each. |
| `threshold` | `float` | `None` | Minimum score to keep. `None` returns every phrase, scored. |

`predict` returns a PixelFlow `Classifications`, best first: `class_name` is your phrase,
`class_id` its position in the list, `confidence` a probability between 0 and 1 for that one pair.
A blank phrase, no phrase, or one over 64 tokens raises `ValueError`.

`encode_image` and `encode_text` return a float32 array of shape `(N, dim)`, L2-normalised, always
two-dimensional.

Over HTTP, `/predict` takes `file` and `text` repeated once per phrase; `/encode` takes either
`file` or `text`, repeated for a batch, and returns `model`, `revision`, `dim` and `embeddings`.

## How it behaves

### The score is a probability for one pair

The score is `sigmoid(cos × exp(logit_scale) + logit_bias)`, what upstream's own examples print. Two
things follow, and they are the reason to choose this family:

- **A single phrase is a well-posed question.** "Is there a forklift in this frame" with one phrase
  gets an answer that means something. With CLIP you must pass a full set of classes and read the
  ranking.
- **Every phrase can be near zero.** On the example photograph, "a photo of people" scores 0.035
  and "a photo of an elephant in a pool" 0.000. A softmax cannot say "none of these", because
  something always wins.

Adding a phrase does not change the others' scores, as with CLIP.

It is **not** a calibrated class probability. Nothing in training made classes compete, so the
number is how well this phrase matches this image, not P(class | image). The values run low, and a
good match is often a few percent, because the learned bias sits near −17. Read it as a score with
a meaningful zero, and calibrate a threshold on your own images.

### Prompts are lowercased for you

SigLIP 2 was trained on lowercased text, and mozo lowercases every phrase. On the example
photograph, `"A Photo Of People"` scores 0.035, the same as the lowercase phrase. With its case
kept, the same phrase scores 0.0001 on `base-224`, a measurement recorded in the vendor's
`PROVENANCE.md`.

The published tokenizer configuration is misleading here: it names a tokenizer class that ignores
its own `do_lower_case` flag, so the obvious way to load it keeps the case and quietly ruins every
capitalised prompt. Mozo follows what the authors trained on.

### The towers load independently, and that matters more here

Asking for phrases never builds the image tower, and the other way round. The text tower is most of
the checkpoint in this family: Gemma's vocabulary alone is 786 MB of a `base` variant and 1,180 MB
of an `so400m` one. An ingest job that only embeds images avoids most of that memory.

### An index is tied to the weights that built it

A vector from one variant or revision means nothing against a vector from another, and nothing
against a CLIP vector. A stored index is tied to the variant and revision that built it; changing
either means embedding the collection again. `/encode` returns both, and the model has
`.revision`.

### SigLIP 2 does not replace CLIP

They are separate models with separate weights, tokenizers and checks, and mozo carries both. Choose
SigLIP 2 when you want a score you can threshold. Choose CLIP when you need CLIP's own embedding
space, which a lot of existing tooling expects.

### Parity

Bit-identical to the published PyTorch model in `transformers`, with `torch.equal` and no tolerance,
checked by `tools/verify/siglip2.py`. Upstream publishes no measured agreement between these
PyTorch checkpoints and the authors' JAX originals, at any tolerance, so mozo does not claim one.

## Weights and licence

Code and weights are Apache-2.0, stated on every model card and downloadable without a gated
login. The extraction derives from `transformers/models/siglip` (© The Google Research Authors and
The HuggingFace Team), the PyTorch implementation the published checkpoints reproduce. What was
taken and the pinned reference versions are in `mozo/vendors/siglip2_deploy/PROVENANCE.md`.
