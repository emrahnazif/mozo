# Grounding DINO: detect what you describe

<!--
summary: IDEA Research's open-vocabulary detector in mozo: boxes for descriptions as well as nouns, Apache-2.0, torch, fp16 and TensorRT on CUDA.
label: Grounding DINO
keywords: mozo, grounding dino, open-vocabulary detection, zero-shot detection, text prompt, tensorrt, fp16
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
| `grounding_dino/tiny` | 694 MB | torch; fp16 and TensorRT on CUDA | Apache-2.0 |
| `grounding_dino/base` | 938 MB | torch; fp16 and TensorRT on CUDA | Apache-2.0 |

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

### On CUDA: half precision and TensorRT

On CUDA, `auto` runs Grounding DINO as `torch-fp16`: the same checkpoint, run under fp16
autocast. There is nothing extra to download. `tensorrt-fp16` is faster still and has to be asked
for, because it runs every image at one fixed square size (see below).

Measured on an RTX 3090 with torch 2.13 and TensorRT 11.5, on 500 random COCO val2017 images.
*Forward* is the model alone, with the 80 class names as one caption. *predict* is the whole call
from a decoded image to detections, with 80 phrases. A detection "changes" when, compared with
fp32 at the same image size, it crosses the 0.35 threshold one way or the other and is more than
0.02 away from it. *Input rounded* is fp32 itself with only its input rounded to half precision:
how often detections change on their own. *Resize* is upstream's preprocessing (short side 800, long
side up to 1333); *800 square* fits the image into an 800×800 square and pads the rest.

| Variant | Runtime | Images | Forward | predict | Speed | COCO AP | Detections that change |
|---|---|---|---|---|---|---|---|
| `tiny` | fp32 | resize | 121.9 ms | 136.5 ms | 1.0× | 50.77 | |
| | input rounded | resize | | | | 50.74 | 0 of 4,118 |
| | `torch-fp16` (`auto`) | resize | 79.8 ms | 94.5 ms | 1.45× | 50.76 | 4 (0.1%) |
| | `tensorrt-fp16` | 800 square | 43.2 ms | 53.1 ms | 2.57× | 49.24 | 5 of 3,882 (0.1%) |
| `base` | fp32 | resize | 167.0 ms | 182.4 ms | 1.0× | 60.16 | |
| | input rounded | resize | | | | 60.06 | 3 of 3,616 |
| | `torch-fp16` (`auto`) | resize | 103.2 ms | 118.0 ms | 1.55× | 60.08 | 3 (0.1%) |
| | `tensorrt-fp16` | 800 square | 49.5 ms | 59.9 ms | 3.05× | 58.25 | 2 of 3,593 (0.1%) |

Half precision barely moves anything: detections change about as rarely as they do on their own,
on torch and on TensorRT alike. What `tensorrt-fp16` costs in AP comes from the square, not from
TensorRT: fp32 torch at 800 square scores the same 49.19 and 58.37. Ask for
`runtime="torch-fp32"` to get exactly the fp32 results. On torch 2.6, `torch-fp16` gained less:
1.28× on `tiny` and 1.42× on `base`.

### Choosing the square size

A TensorRT engine takes one image shape, so every image is fitted into a square, aspect ratio
kept, and the rest padded. The model is told which pixels are padding, so it does not read the
border as part of the scene. Without that mask AP fell by 6 to 12 points. Pick the square with
`input_size`; it defaults to 800 on `tensorrt-fp16`:

```python
from mozo import ModelManager

model = ModelManager().get_model("grounding_dino", "base", runtime="tensorrt-fp16", input_size=1024)
```

A bigger square finds more small objects and costs time. Same machine and images as above,
`predict` with 80 phrases, AP against upstream's resize on fp32 (50.77 `tiny`, 60.16 `base`):

| Square | `tiny` predict | `tiny` AP | `base` predict | `base` AP |
|---|---|---|---|---|
| 800 (default) | 53.1 ms | 49.24 (−1.5) | 59.9 ms | 58.25 (−1.9) |
| 1024 | 76.4 ms | 50.34 (−0.4) | 86.1 ms | 59.47 (−0.7) |
| 1333 | 139.0 ms | 51.10 (+0.3) | 150.1 ms | 59.61 (−0.6) |

1024 gives back most of the AP and is still about twice as fast as fp32. At 1333 the square holds
more pixels than upstream's resize does for most photographs, so it is barely faster than fp32.
`input_size` takes 64 to 2048 and also works on torch, which is how the table compares
preprocessings on one runtime.

### What TensorRT needs

- **TensorRT 11 or later.** TensorRT 10 miscompiles the model's position encoding, so boxes come
  out wrong, and mozo refuses it with that reason. TensorRT 11 needs a torch built for CUDA 12 or
  13: `pip install tensorrt-cu12` or `tensorrt-cu13`, matching your torch.
- **`onnx` and `onnxscript`**, to export the graph the engine is built from:
  `pip install onnx onnxscript`. A machine that already has its engine needs neither.
- **About 3.5 minutes the first time** for each GPU and square size: mozo exports the graph from
  the checkpoint and builds the engine (193–223 s on the RTX 3090). The engine is kept beside the
  checkpoint in mozo's cache and the graph is deleted. Every later load takes about half a second.
  `weights.remove` deletes engines with the model. To pay that cost ahead of time, load the model
  once when you build the machine or the image.

No ONNX graph is published for Grounding DINO, because the square size is yours to choose. The
graph is written on your machine for the size you ask for. Any caption up to the model's 256
tokens runs on one engine.

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
