# Florence-2: detection in the model's own words

<!--
summary: Microsoft's Florence-2 object detection in mozo: boxes named in the model's own words, no class list, MIT code and weights.
label: Florence-2
keywords: mozo, florence-2, florence2, object detection, open vocabulary, microsoft
-->

Florence-2 is one Microsoft model that does a dozen vision tasks, picked by a task prompt. Mozo
serves its object detection task (`<OD>`): give it a photograph and it writes out every object it
can name, each with a box. There is no class list. The model writes the label, then the box, then
the next label, so `class_name` is its own word for the thing ("human face", "glasses", "laptop")
and `class_id` is empty.

## Quick start

```python
import mozo

model = mozo.get_model("florence2_detect/base")
found = model.predict("example.jpg")

for detection in found[:5]:
    print(detection.class_name, round(detection.confidence, 2), detection.bbox)
```

```text
glasses 0.27 [177.6, 309.36, 469.44, 420.81]
glasses 0.37 [991.68, 358.04, 1220.16, 425.93]
human face 0.39 [212.16, 237.63, 452.16, 566.84]
human face 0.37 [1456.32, 135.15, 1690.56, 414.4]
human face 0.49 [830.4, 258.12, 1001.28, 475.89]
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/florence2_detect/base" -F "file=@example.jpg"
```

## Variants

| ID | Weights | Runtimes | Licence |
|---|---|---|---|
| `florence2_detect/base` | 463 MB | torch; fp16 on CUDA | MIT |
| `florence2_detect/large` | 1,554 MB | torch; fp16 on CUDA | MIT |
| `florence2_detect/base-ft` | 463 MB | torch; fp16 on CUDA | MIT |
| `florence2_detect/large-ft` | 1,541 MB | torch; fp16 on CUDA | MIT |

`base` (0.23B parameters) and `large` (0.77B) are Microsoft's pretrained models. They name things
freely: "man", "woman", "human face", "taillight". The `-ft` variants were fine-tuned by Microsoft
on a set of downstream tasks that includes COCO detection, so they mostly use COCO's 80 names and
find COCO's objects better. Start with `base-ft` if your objects are everyday COCO things, and with
`base` if you want the model's wider vocabulary.

Microsoft replaced `Florence-2-large` in December 2024 with a version trained further with a
4,096-token context, and says on its model card that it "might not be trained well". Mozo serves
that current version: on 500 COCO images it scored 36.98 AP, against 36.77 for the original release
(both in fp32, scored the same way, see below).

## Inputs and outputs

```python
model.predict(image, threshold=None, labels=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `threshold` | `float` or `None` | `None` | Keep only boxes scoring at least this, between 0 and 1. `None` keeps every box. |
| `labels` | | `None` | Not supported. Passing anything raises a `ValueError`: there is no class list to rename. |

Returns a PixelFlow `Detections`, in the order the model wrote them: grouped by label, not sorted by
score. Each detection has a `bbox` in pixels of your image, `class_name` set to the model's label,
`class_id` set to `None`, and a `confidence`. Non-ASCII characters are dropped from labels, as
Microsoft's own post-processing does.

Over HTTP: `threshold` as a query parameter.

## How it behaves

### The score is how sure the decoder was of the box

The model writes its answer token by token: a label, then four location tokens for the box's
corners. The confidence is Microsoft's own score for `<OD>`: the average probability the decoder
gave the box's four location tokens, once it had written the label. It is not the probability that
the object is there, and it runs low: 0.10 to 0.60 on the example photograph. Do not compare it with
another detector's score. Mozo keeps every box by default, as Microsoft's code does; pass a
`threshold` to cut.

### Decoding is fixed, and deterministic

Each call runs Microsoft's default decoding: beam search over three candidate answers, never
repeating the same three tokens, up to 1,024 tokens. There is nothing to configure, and the same
image always gives the same boxes. A crowded image takes longer, because every box is about five
more tokens to write.

### The image is squashed to a square

Every image is resized to 768 × 768 pixels, without keeping its aspect ratio, and boxes are mapped
back to your image. A very wide or tall image is squeezed, so thin objects along its long side get
fewer pixels.

### On CUDA: half precision

The weights are published in half precision (fp16), as Microsoft releases them. On the CPU mozo
converts them to fp32 when loading, which changes no value. On CUDA, `auto` runs them in fp16, about
twice as fast. Measured on an RTX 3060 (12 GB) against fp32, on 500 COCO val2017 images:

| Variant | fp32 AP | fp16 AP | fp32 boxes fp16 keeps | Time per image, fp32 → fp16 |
|---|---|---|---|---|
| `base` | 34.19 | 34.41 | 97.5% | 0.34 → 0.18 s |
| `large` | 36.98 | 36.89 | 98.1% | 0.82 → 0.40 s |
| `base-ft` | 44.40 | 44.47 | 98.6% | 0.25 → 0.14 s |
| `large-ft` | 46.09 | 45.81 | 99.0% | 0.63 → 0.29 s |

A box is "kept" when fp16 returns one with the same label and an overlap (IoU) of at least 0.9.
The model names things freely, so a box only counts for COCO when its label is exactly a COCO
category name: "person" scores, "man" does not. That is harsh on `base` and `large`, so their AP is
a lower bound; it compares runtimes and variants fairly, not this model with others. Ask for
`runtime="torch-fp32"` to get exactly the fp32 results.

### Verified on every stage

Mozo's extraction is checked against Microsoft's own code (the remote code in its Hugging Face
repositories, under transformers 4.53.3) with no tolerance: the pixels, each stage of the vision
tower, the encoder, every decoding step's scores for all three beams, the chosen sequence, and the
final boxes, labels and scores. On CUDA (an RTX 3060, torch 2.13, CUDA 12.6) all four variants are
bit-identical to Microsoft's code in fp32 and in fp16, the precision `auto` uses there: 4,305
comparisons, every one identical, with both sides run twice to show they repeat themselves. On the
CPU, in fp32, another 2,096. On a CPU it runs 3–5% faster than Microsoft's code on the same weights.

## Weights and licence

Code and weights are MIT, on all four checkpoints. Microsoft states `license: mit` on each model
card and ships the MIT licence file in each repository. Mozo's extraction derives from the code in
those repositories, plus a transcription of transformers' beam search (Apache-2.0).
`mozo/vendors/florence2_deploy/PROVENANCE.md` records the exact sources.
