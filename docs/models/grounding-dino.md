# Grounding DINO: detect what you describe

<!--
summary: IDEA Research's open-vocabulary detector in mozo: boxes for descriptions as well as nouns, Apache-2.0, torch only.
label: Grounding DINO
keywords: mozo, grounding dino, open-vocabulary detection, zero-shot detection, text prompt
-->

Grounding DINO finds objects you describe in words. It answers the same question as
[OWLv2](/docs/models/owlv2.md), with the same call, the same endpoint and the same result type, so
you can swap one for the other. Reach for it when your prompt is a description rather than a noun:
it mixes the words into the image features six times and lets the decoder look back at them, so
`"the mug on the left"` is read as a phrase, not as a bag of words.

## Quick start

```python
import mozo

model = mozo.get_model("grounding_dino/tiny")
found = model.predict("example.jpg", ["a laptop", "a coffee cup", "a phone"])

for detection in found:
    print(detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
a phone 0.89 [1158.58, 1068.93, 1455.86, 1190.51]
a laptop 0.81 [680.65, 768.13, 1360.23, 1138.77]
a coffee cup 0.54 [908.0, 471.76, 1055.76, 608.64]
```

Over HTTP, with `mozo start` running. Repeat `text` for each phrase:

```bash
curl -X POST "http://localhost:8000/predict/grounding_dino/tiny?text=a%20laptop&text=a%20phone" \
  -F "file=@example.jpg"
```

## Variants

| ID | Weights (torch) | Runtimes | Licence |
|---|---|---|---|
| `grounding_dino/tiny` | 694 MB | torch | Apache-2.0 |
| `grounding_dino/base` | 938 MB | torch | Apache-2.0 |

Upstream publishes no others. `tiny` (Swin-T) scores 48.4 box AP on COCO zero-shot and `base`
(Swin-B) 56.7, for 35% more weights, by the authors' numbers. `tiny` is 82% of upstream's own
downloads, but 8.3 AP is a wide enough gap that `base` is worth trying.

## Inputs and outputs

```python
model.predict(image, text, threshold=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `text` | `str` or `list[str]` | | One phrase or several. No `.` or `?` inside a phrase; about 60 phrases fit. |
| `threshold` | `float` | `None` | Minimum score to keep. `None` applies upstream's default of 0.35. |

Returns a PixelFlow `Detections`, best first. Each detection has a `bbox` in pixels of your image,
a `confidence`, `class_name` set to the phrase you passed, and `class_id` set to its index in
`text`.

Raises a `ValueError` if no phrase is given, a phrase is empty, a phrase contains `.` or `?`, or
the phrases together pass the model's 256-token budget.

Over HTTP: `text` (repeat it for several phrases) and `threshold` as query parameters.

## How it behaves

### Three rules for prompts

- **No `.` or `?`.** The model separates concepts with them, so a phrase carrying one would be
  split in two and its detections reported against the wrong phrase. Mozo refuses it:

  ```text
  ValueError: 1 prompt(s) became 2 phrase(s); a prompt may not contain '.' or '?', which separate concepts.
  ```

- **Case is not kept.** The caption is lowercased before tokenizing, as upstream does.
- **About 60 phrases fit** in the model's 256-token budget. Upstream silently cuts the rest; mozo
  raises.

All your phrases are joined into one caption, so several phrases cost one forward pass.

### The name you get back is the phrase you asked for

Upstream decodes the matching tokens back into a string, which can return `"yellow school"` for
`"a yellow school bus"`. Mozo reports which prompt matched instead, exactly, from the phrase map the
tokenizer already builds. So `class_name` is your string and `class_id` its index. This is the one
place mozo deliberately differs from upstream, and it changes a string, not a number.
`PROVENANCE.md` has the reasoning.

### Nothing is suppressed

As with OWLv2, two prompts describing the same thing can both find it, and overlapping boxes for
one prompt are the model's answer, not a failure to deduplicate.

### It costs the same every time you ask

Measured on the example photograph, `mps`, three prompts:

| | first look at an image | asking the same image again |
|---|---|---|
| `owlv2/base-ensemble` | 458 ms | 13 ms |
| `grounding_dino/tiny` | 472 ms | 472 ms |
| `grounding_dino/base` | 656 ms | 656 ms |

On a first look the two are comparable. Asking again, OWLv2 is about 40 times faster, and that gap
comes from the architecture. OWLv2's image encode depends only on the image, so mozo caches it.
Grounding DINO mixes the text into the image features, so its image encode depends on the prompt
and there is nothing to keep. If you ask one image many questions, that is the whole comparison.

### Torch only

The image is resized to a short side of 800, with the long side capped near 1333: the aspect ratio
is preserved and nothing is padded. So unlike most families there is no fixed input size, and a
graph exported at one shape cannot take another. No ONNX or CoreML artifact is published.

### No `transformers` at run time

The text encoder is BERT, but the published checkpoint carries its own fine-tuned BERT weights.
Mozo rebuilds that shape and ships the WordPiece vocabulary in the wheel, so tokenizing a prompt
needs no network and no `transformers`.

### Verified on every stage

Mozo's extraction is checked against the authors' implementation on the tokenizer,
preprocessing, the BERT tower, all three Swin levels, the encoder, all six decoder layers and the
final logits and boxes, with no tolerance: 138 comparisons across both variants, every one
bit-identical.

## Weights and licence

Code and weights are Apache-2.0. Mozo's code derives from `IDEA-Research/GroundingDINO`, the
authors' own PyTorch release (© 2023 – present IDEA Research). The weights' licence is stated by
the authors on the Hugging Face repositories that serve the files; the NOTICE published with each
variant records exactly where. The tokenizer vocabulary is Google's `bert-base-uncased`
(Apache-2.0). `mozo/vendors/grounding_dino_deploy/PROVENANCE.md` records the upstream commit.
