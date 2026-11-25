# OmDet-Turbo: real-time detection from words

<!--
summary: Om AI Lab's real-time open-vocabulary detector in mozo: name things in words, get boxes, Apache-2.0, on torch, ONNX and TensorRT.
label: OmDet-Turbo
keywords: mozo, omdet-turbo, open-vocabulary detection, zero-shot detection, text prompt, tensorrt
-->

OmDet-Turbo finds objects you name in words, like [OWLv2](/docs/models/owlv2.md) and
[Grounding DINO](/docs/models/grounding-dino.md), and is built to be the fast one. You pass
phrases such as `"a laptop"` or `"a coffee cup"`, and it returns a box for each instance, named by
the phrase that found it. The text is encoded once per list of phrases, so a fixed vocabulary run
over many images pays only for the image. It returns boxes only; for masks, give its boxes to
[SAM 2](/docs/models/sam2.md) or [EdgeTAM](/docs/models/edgetam.md).

## Quick start

```python
import mozo

model = mozo.get_model("omdet_turbo/tiny")
found = model.predict("example.jpg", ["a laptop", "a coffee cup", "a phone"])

for detection in found[:4]:
    print(detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
a laptop 0.8 [678.35, 769.81, 1364.17, 1137.75]
a phone 0.56 [1163.01, 1080.7, 1453.52, 1182.59]
a coffee cup 0.53 [932.39, 479.65, 1052.36, 604.28]
a coffee cup 0.31 [598.1, 724.19, 721.9, 837.02]
```

Over HTTP, with `mozo start` running. Repeat `text` for each phrase:

```bash
curl -X POST "http://localhost:8000/predict/omdet_turbo/tiny?text=a%20laptop&text=a%20coffee%20cup" \
  -F "file=@example.jpg"
```

## Variants

| ID | Weights (torch) | Runtimes | Licence |
|---|---|---|---|
| `omdet_turbo/tiny` | 462 MB | torch, onnx; TensorRT fp32 and fp16 on CUDA | Apache-2.0 |

Om AI Lab publishes one OmDet-Turbo model, a Swin-T. The ONNX and TensorRT runtimes also download a
218 MB graph of the image path; the text half always runs on torch, from the checkpoint.

## Inputs and outputs

```python
model.predict(image, text, threshold=0.3)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `text` | `str` or `list[str]` | | One phrase or several, up to 75 tokens each. |
| `threshold` | `float` | `0.3` | Minimum score to keep a detection. |

Returns a PixelFlow `Detections`, best first. Each detection has a `bbox` in pixels of your image,
a `confidence`, `class_name` set to the phrase that found it, and `class_id` set to that phrase's
index in `text`. No phrase, an empty one, or one longer than 75 tokens raises a `ValueError`.

Over HTTP: `text` (repeat it for several phrases) and `threshold` as query parameters.

## How it behaves

### The phrases are read twice

Each phrase becomes an embedding that every candidate box is scored against. All the phrases
together also become one sentence, `"Detect a laptop, a coffee cup, a phone."`, which the decoder
reads while it refines the boxes: that is the model's "efficient fusion head". The sentence is the
one Om AI Lab's demo and `transformers` use; mozo does not take a custom one.

The sentence is cut at 77 tokens, as upstream cuts it, which is about twenty COCO-length names.
Every phrase is still scored, however long the list; only the sentence the decoder reads is cut.

### Phrases do not compete

Every candidate is scored against every phrase separately, so the same box can come back under
`"car"` and `"vehicle"`. Overlapping boxes under one phrase are suppressed at IoU 0.5, as upstream
does. A score is the sigmoid of a scaled cosine similarity, not a probability over your phrases.

### The text is encoded once

The prompt embeddings depend only on the words. Mozo keeps them per list of phrases, so the second
image with the same list costs only the image path. The list is encoded as one batch and cached
whole: asking for `["cat"]` after `["cat", "dog"]` gives exactly what a fresh model gives, which
is not true of upstream's per-phrase cache.

### Fixed square input

Every image is squashed to 640×640 (the aspect ratio is not kept), as upstream does.

### Compared with OWLv2 and Grounding DINO

On an RTX 3090, OmDet-Turbo's image path takes 22 ms in torch fp32 and 14 ms as `tensorrt-fp32`,
against 23 ms for OWLv2 `base-ensemble`'s image tower in `torch-fp16` and 80 ms for Grounding DINO
`tiny` in `torch-fp16`. On 500 COCO val2017 images with the 80 class names as the phrases it
scores 43.8 AP, where OWLv2 `base-ensemble` scores 50.0 and Grounding DINO `tiny` 50.8 on the same
images. Pick it when speed over many images matters more than the last points of recall.

### On CUDA: TensorRT

With TensorRT 11 or later installed (`pip install tensorrt-cu12` or `-cu13`, matching your torch's
CUDA; see [Runtimes](/docs/deployment/runtimes.md#tensorrt)), `auto` runs OmDet-Turbo on CUDA as
`tensorrt-fp32`: an engine built from the published graph, in full IEEE fp32, which finds what
torch finds. Without TensorRT, `auto` stays on torch. `runtime="tensorrt-fp16"` is faster again
and has to be asked for. The first load builds the engine (about 40 s for fp32 and 90 s for fp16
on the RTX 3090) and keeps it in mozo's cache; later loads reuse it.

Measured on an RTX 3090 with torch 2.13 and TensorRT 11.5, on 500 random COCO val2017 images with
the 80 class names. Time is the image path, the text already encoded. A detection "changes" when,
against torch fp32, it scores above 0.32 on one side with no same-class partner (IoU above 0.5)
above 0.3 on the other; the count is out of the 4,843 detections torch fp32 makes above 0.3.

| Runtime | Image path | Speed | COCO AP | Detections that change |
|---|---|---|---|---|
| torch fp32 | 22.0 ms | 1.0× | 43.75 | |
| input rounded to fp16 | | | 43.74 | 0 |
| `tensorrt-fp32` (`auto`) | 13.9 ms | 1.6× | 43.74 | 0 |
| `tensorrt-fp16` | 7.8 ms | 2.8× | 43.76 | 5 (0.1%) |

Torch's own fp16 (autocast) is slower than fp32 here, so mozo does not offer it. ONNX Runtime on
CUDA (32 ms) is slower than torch.

### On the CPU

`auto` runs torch. `runtime="onnx-fp32"` runs the same graph through ONNX Runtime: on 100 COCO
images it found exactly what torch found, and its image path was 5–18% faster than torch's (eight
threads each, on a shared and heavily loaded 20-thread CPU, so indicative: 671–775 ms against
814–834 ms). The text tower stays on torch either way.

### No `transformers` at run time

The text tower is CLIP's, fine-tuned into the checkpoint, and the backbone is timm's Swin-T. Mozo
rebuilds both in plain torch and ships CLIP's vocabulary in the wheel, so tokenizing a prompt
needs no network, no `transformers` and no `timm`.

### Verified on every stage

Mozo's extraction is checked against `transformers`' implementation on the tokenizer, the
preprocessing, both text encodings, all three Swin levels, the encoder, all six decoder layers,
the boxes, the logits and the detections, with no tolerance, on the CPU: every comparison
bit-identical (`tools/verify/omdet_turbo.py`).

## Weights and licence

Code and weights are Apache-2.0: Om AI Lab states it on the model card of the checkpoint mozo
publishes (`omlab/omdet-turbo-swin-tiny-hf`), and its GitHub repository is Apache-2.0. The
checkpoint holds a Swin-T backbone and CLIP's text tower, both MIT, fine-tuned into this detector.
Every tensor is bit-identical to the authors' original-format release. Mozo publishes the same
tensors as a `.pth`. `mozo/vendors/omdet_turbo_deploy/PROVENANCE.md` records the sources.
