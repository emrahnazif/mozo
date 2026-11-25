# MM-Grounding-DINO: LLMDet, detect what you describe

<!--
summary: LLMDet on OpenMMLab's MM-Grounding-DINO in mozo: open-vocabulary boxes, strongest on rare objects, Apache-2.0, torch and fp16 on CUDA.
label: MM-Grounding-DINO
keywords: mozo, llmdet, mm-grounding-dino, mm grounding dino, open-vocabulary detection, zero-shot detection, text prompt, lvis, fp16
-->

MM-Grounding-DINO finds objects you describe in words. It answers the same question as
[Grounding DINO](/docs/models/grounding-dino.md) and [OWLv2](/docs/models/owlv2.md), with the same
call, the same endpoint and the same result type, so you can swap one for another. The weights are
LLMDet's (iSEE Laboratory, CVPR 2025): Grounding DINO's design, re-trained in the open by OpenMMLab,
then fine-tuned with a large language model writing a detailed caption for every training image.
Reach for it over Grounding DINO when the things you look for are uncommon: on the authors'
zero-shot LVIS benchmark, which counts 1,203 kinds of object, its smallest size scores 44.7 AP.

## Quick start

```python
import mozo

model = mozo.get_model("mm_grounding_dino/llmdet-tiny")
found = model.predict("example.jpg", ["a laptop", "a coffee cup", "a phone"])

for detection in found:
    print(detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
a laptop 0.6 [680.07, 764.62, 1365.29, 1139.25]
a phone 0.54 [1160.47, 1072.67, 1455.71, 1188.2]
a coffee cup 0.42 [907.83, 468.74, 1057.26, 611.46]
```

Over HTTP, with `mozo start` running. Repeat `text` for each phrase:

```bash
curl -X POST "http://localhost:8000/predict/mm_grounding_dino/llmdet-tiny?text=a%20laptop&text=a%20phone" \
  -F "file=@example.jpg"
```

## Variants

| ID | Weights (torch) | Runtimes | Licence |
|---|---|---|---|
| `mm_grounding_dino/llmdet-tiny` | 692 MB | torch; fp16 on CUDA | Apache-2.0 |
| `mm_grounding_dino/llmdet-base` | 936 MB | torch; fp16 on CUDA | Apache-2.0 |
| `mm_grounding_dino/llmdet-large` | 1.4 GB | torch; fp16 on CUDA | Apache-2.0 |

`llmdet-tiny` is Swin-T, `llmdet-base` Swin-B and `llmdet-large` Swin-L. By the authors' numbers
they score 44.7, 48.3 and 51.1 AP on LVIS minival without having been trained on it, where
Grounding DINO's `tiny` scores 28.8 by MM-Grounding-DINO's model card. MM-Grounding-DINO's own checkpoints, which LLMDet was
fine-tuned from, run on the same code and are not published: LLMDet beats each at its size.

## Inputs and outputs

```python
model.predict(image, text, threshold=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `text` | `str` or `list[str]` | | One phrase or several. No `.` or `?` inside a phrase; about 60 phrases fit. |
| `threshold` | `float` | `None` | Minimum score to keep. `None` applies 0.4, the authors' own default. |

Returns a PixelFlow `Detections`, best first. Each detection has a `bbox` in pixels of your image,
a `confidence`, `class_name` set to the phrase you passed, and `class_id` set to its index in
`text`.

Raises a `ValueError` if no phrase is given, a phrase is empty, a phrase contains `.` or `?`, or
the phrases together pass the model's 256-token budget.

Over HTTP: `text` (repeat it for several phrases) and `threshold` as query parameters.

## How it behaves

### The same prompt rules as Grounding DINO

No `.` or `?` inside a phrase (they separate concepts, so mozo refuses a phrase carrying one), case
is not kept, and about 60 phrases fit in 256 tokens, past which mozo raises where upstream cuts the
rest silently. All your phrases are joined into one caption, so several cost one forward pass.
`class_name` is the phrase you asked for, not a string decoded from the model's tokens. Nothing is
suppressed: overlapping boxes for one phrase are the model's answer.

### Scores are not Grounding DINO's

A score is the best per-token probability for a box, as in Grounding DINO, but from a differently
scaled head, so the two families' numbers do not transfer. The default threshold is 0.4 here,
0.35 there.

### On CUDA: half precision

On CUDA, `auto` runs MM-Grounding-DINO as `torch-fp16`: the same checkpoint under fp16 autocast,
nothing extra to download. Ask for `runtime="torch-fp32"` to get exactly the fp32 results.

Measured on an RTX 3090 with torch 2.13, on 500 random COCO val2017 images, with the machine's
CPU shared with other jobs (so treat the milliseconds as indicative). *Forward* is the model alone
with the 80 class names as one caption (195 tokens); *predict* is the whole call from a decoded
image to detections with those 80 phrases. A detection "changes" when, compared with fp32, it
crosses the 0.4 threshold one way or the other and is more than 0.02 from it. *Input rounded* is
fp32 with only its input rounded to half precision: how often detections change on their own.
COCO train2017 is in LLMDet's training data, so this AP is a yardstick, not a zero-shot score.

| Variant | Runtime | Forward | predict | Speed | COCO AP | Detections that change |
|---|---|---|---|---|---|---|
| `llmdet-tiny` | fp32 | 128.1 ms | 220.5 ms | 1.0× | 58.54 | |
| | input rounded | | | | 58.64 | 0 of 2,514 |
| | `torch-fp16` (`auto`) | 86.4 ms | 169.1 ms | 1.48× | 58.79 | 1 |
| `llmdet-base` | fp32 | 177.8 ms | 258.1 ms | 1.0× | 61.63 | |
| | input rounded | | | | 61.53 | 1 of 2,589 |
| | `torch-fp16` (`auto`) | 109.0 ms | 197.5 ms | 1.63× | 61.55 | 0 |
| `llmdet-large` | fp32 | 554.6 ms | 631.1 ms | 1.0× | 62.52 | |
| | input rounded | | | | 62.57 | 5 of 2,731 |
| | `torch-fp16` (`auto`) | 350.3 ms | 414.4 ms | 1.58× | 62.57 | 4 |

Half precision moves detections no more often than fp32 moves on its own, and AP does not move.

### No TensorRT, and why

Unlike Grounding DINO, this family offers no TensorRT runtime. Measured the same way, on the same
800×800 square Grounding DINO's engine uses, TensorRT 11.5 engines were 2.9 to 3.5 times faster
than fp32 torch, and changed 3 to 6% of detections: 150, 104 and 77 of about 2,500 for the three
sizes, against 0 to 4 for `torch-fp16` on the same square. AP fell by up to 2.5 points
(`llmdet-tiny`, 57.09 on torch to 54.54). An fp32 engine drifted as much as an fp16 one, so it is
not precision: ONNX Runtime running the very graph TensorRT was built from changed none of 112
detections. One cause was found and removed (a bounds test asked exactly on its bound, which mozo
now decides in integers); the rest is inside TensorRT's compilation of the encoder and is not yet
explained. Grounding DINO's engine moves 0.1%.

`input_size` still fits every image into a square on torch, which is faster and finds less: on
`llmdet-tiny`, an 800 square ran the 80-phrase `predict` in 182.4 ms against 220.5 ms and scored
57.09 against 58.54 AP.

### No `transformers` at run time

The checkpoints are published for `transformers`, and mozo runs them without it. That matters:
`transformers` 5 loads these files with all six decoder layers using the first layer's box head,
which runs and gives different boxes (fixed upstream after 5.12). Mozo's extraction matches
`transformers` 4.57 and 4.55, the version the LLMDet authors name.

### Verified on every stage

Mozo's extraction is checked against `transformers` 4.57.3 on preprocessing, the tokenizer, the BERT
tower, every Swin level, the encoder, each decoder layer, the final logits and boxes and the
detections after post-processing, with no tolerance: 1,520 comparisons across the three
variants, every one bit-identical. The tokenizer is also run on 20,020 prompts in every script
Unicode has, and agrees on every codepoint.

## Weights and licence

Code and weights are Apache-2.0. The weights are LLMDet's, published by the authors at
`huggingface.co/iSEE-Laboratory/llmdet_{tiny,base,large}`, each model card stating
`license: apache-2.0`; the LLMDet code repository is Apache-2.0 too. Mozo republishes the same
tensors as a PyTorch checkpoint, with the NOTICE naming the file and commit they came from. The
training data includes COCO train2017, so COCO scores are not zero-shot for these weights.
`mozo/vendors/mm_grounding_dino_deploy/PROVENANCE.md` records the reference and every divergence.
