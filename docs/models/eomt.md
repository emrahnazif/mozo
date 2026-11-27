# EoMT: a class for every pixel

<!--
summary: EoMT in mozo: semantic and panoptic segmentation from a plain DINOv2 ViT, one mask per segment, never overlapping. COCO panoptic and ADE20K, MIT.
label: EoMT
keywords: mozo, eomt, semantic segmentation, panoptic segmentation, scene parsing, coco panoptic, ade20k, dinov2
-->

EoMT gives every pixel of a picture a class from a fixed vocabulary: the people, the laptop and
the cups, and also the wall, the floor and the sky. The answer is one mask per segment, and no two
masks overlap. The COCO panoptic variants also give each countable thing its own segment, so five
people are five detections, while all the wall is one. The ADE20K variant labels 150 kinds of scene
content without splitting instances. EoMT ("Your ViT is Secretly an Image Segmentation Model",
CVPR 2025) is a plain DINOv2 ViT with learned queries in its last blocks, and no decoder.

## Quick start

```python
import mozo

model = mozo.get_model("eomt/coco-panoptic-large")
found = model.predict("example.jpg")

for detection in found[:6]:
    print(detection.class_name, detection.confidence, detection.bbox)
```

```text
floor-other-merged 0.967 [1837.0, 1057.0, 1920.0, 1281.0]
person 1.0 [1318.0, 21.0, 1920.0, 1270.0]
laptop 0.999 [697.0, 779.0, 1359.0, 1137.0]
cup 0.999 [937.0, 485.0, 1046.0, 606.0]
wall-other-merged 0.99 [0.0, 0.0, 1920.0, 528.0]
person 1.0 [0.0, 122.0, 765.0, 1268.0]
```

Each detection's `masks[0]` is an `HxW` boolean array at the picture's size. Over HTTP, with
`mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/eomt/coco-panoptic-large" -F "file=@example.jpg"
```

The response is a list of detections with every PixelFlow field; each mask travels as a base64
PNG (`{"format": "png", "data": "...", "shape": [1281, 1920], "dtype": "bool"}`).

## Variants

| ID | Task | Classes | Input | Weights (torch) | PQ / mIoU (authors) | Runtimes | Licence |
|---|---|---|---|---|---|---|---|
| `eomt/coco-panoptic-small` | panoptic | COCO, 133 | 640 | 96 MB | PQ 46.7 | torch; fp16 and tensorrt-fp16 on CUDA | MIT |
| `eomt/coco-panoptic-base` | panoptic | COCO, 133 | 640 | 375 MB | PQ 51.6 | torch; fp16 and tensorrt-fp16 on CUDA | MIT |
| `eomt/coco-panoptic-large` | panoptic | COCO, 133 | 640 | 1.3 GB | PQ 56.0 | torch; fp16 and tensorrt-fp16 on CUDA | MIT |
| `eomt/coco-panoptic-large-1280` | panoptic | COCO, 133 | 1280 | 1.3 GB | PQ 58.3 | torch; fp16 on CUDA | MIT |
| `eomt/ade20k-semantic-large` | semantic | ADE20K, 150 | 512 windows | 1.3 GB | mIoU 58.4 | torch; fp16 and tensorrt-fp16 on CUDA | MIT |

`coco-panoptic-small` is the default and the fastest; `coco-panoptic-large` is the one most people
use. COCO's 133 classes are its 80 things (person, car, cup, ...) and 53 kinds of stuff (sky, wall,
road, ...; ids 80 to 132, many of them merged COCO-stuff classes such as `wall-other-merged`).
ADE20K's 150 are scene classes: wall, floor, ceiling, tree, building, road, sky, person, and so on.

### On CUDA: half precision, and TensorRT

On CUDA, `auto` runs EoMT as `torch-fp16`: the same checkpoint with the network and its
post-processing under fp16 autocast, the way the authors' own half-precision evaluation runs, and
bit-identical to it. Nothing extra is downloaded, and PQ and mIoU do not move. `runtime="tensorrt-fp16"` is
faster again for every variant but `coco-panoptic-large-1280`: a TensorRT 11 engine built from a
published fp16 ONNX graph (50–640 MB), which needs
TensorRT installed (`pip install tensorrt-cu12` or `-cu13`; see
[Runtimes](/docs/deployment/runtimes.md#tensorrt)) and is only used when asked for.
`runtime="torch-fp32"` is bit-identical to the authors' code in full precision.

Measured on an RTX 3060 (12 GB) with torch 2.13 and TensorRT 11.5. PQ on the first 500 COCO
val2017 panoptic images, mIoU on all 2,000 ADE20K validation images, computed the way the COCO
panoptic API and upstream do. Time is a whole `predict` on the 1920x1281 example photograph, which
includes resizing it and bringing every mask back to full size (about 60 ms whatever the runtime).
*Moved* is the mean share of pixels whose class differs from torch fp32's. The `torch-fp16` accuracy
was measured again once its post-processing moved under autocast (RTX 3060, vision-003): the same
PQ, mIoU and moved pixels, except ADE20K's moved share (0.02% to 0.05%).

| Variant | torch fp32 | `torch-fp16` (`auto`) | `tensorrt-fp16` |
|---|---|---|---|
| `coco-panoptic-small` | 91 ms, PQ 47.24 | 73 ms, 1.25×, PQ 47.24, moved 0.05% | 67 ms, 1.35×, PQ 47.19, moved 0.25% |
| `coco-panoptic-base` | 158 ms, PQ 52.12 | 97 ms, 1.6×, PQ 52.14, moved 0.03% | 84 ms, 1.9×, PQ 52.15, moved 0.55% |
| `coco-panoptic-large` | 341 ms, PQ 56.61 | 160 ms, 2.1×, PQ 56.61, moved 0.04% | 114 ms, 3.0×, PQ 56.67, moved 0.15% |
| `coco-panoptic-large-1280` | 2,028 ms, PQ 58.78 | 500 ms, 4.1×, PQ 58.78, moved 0.02% | not offered: 575 ms, slower than `torch-fp16` (PQ 58.79) |
| `ade20k-semantic-large` | 346 ms, mIoU 58.40 | 171 ms, 2.0×, mIoU 58.39, moved 0.05% | 111 ms, 3.1×, mIoU 58.39, moved 0.11% |

`tensorrt-fp16` changes more pixels than `torch-fp16` and is not the default for that reason: on an
occasional picture a large segment whose confidence sits right at the 0.8 threshold drops to void
in one runtime and not the other (the worst of 500 COCO images changed 92% of its pixels on
`coco-panoptic-base`), although PQ does not move. An fp32 TensorRT engine was measured too and is
not offered: it matches torch to within one pixel in three million, but a whole call is at most
1.2 times faster than torch fp32, slower than `torch-fp16`.

## Inputs and outputs

```python
model.predict(image, threshold=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `threshold` | `float` | `0.8` | Panoptic variants only: the class probability a segment needs before segments are merged. A semantic variant raises `ValueError` (`400` over HTTP). |

Returns a PixelFlow `Detections`, one detection per segment, in the model's own order:

| Field | Panoptic | Semantic |
|---|---|---|
| `masks[0]` | the segment's pixels, `HxW` bool | the class's pixels, `HxW` bool |
| `class_id`, `class_name` | the class; names from the published labels | the same |
| `confidence` | the class probability of the query that found the segment | `None` |
| `bbox` | the tight box around the mask | the same |

The model's `.labels` lists every class name in id order.

## How it behaves

### Segments partition the picture

No pixel is in two masks. For a semantic variant every pixel is in exactly one; for a panoptic
variant a pixel may be in none ("void"), where no segment survived. A dense class map is one line
away:

```python
import numpy as np

canvas = np.full(image.shape[:2], -1)          # -1 = void
for d in found:
    canvas[d.masks[0]] = d.class_id
```

The box of a stuff segment can span the whole picture (the wall behind everyone); it is still the
true box of its mask. A semantic class present in two places is one detection whose box covers both.

### Things and stuff

In a panoptic answer every thing class is split into instances and every stuff class is merged into
one segment, so the class says which it is: ids below 80 are things. Counting people is
`sum(d.class_name == "person" for d in found)`.

### What `threshold` does, and what it does not

The model proposes up to 200 segments, each with a class probability. Those above `threshold` are
merged into a partition: where two overlap, the more confident one takes the pixel. Raising the
threshold drops segments *before* that merge, and their pixels become void. That is not the same as
filtering the result afterwards with `filter_by_confidence`, which removes a segment but leaves its
pixels to nobody, rather than to the neighbour that would have won them.

### Semantic scores are not probabilities

A semantic variant ranks classes per pixel by a sum over its 100 queries of mask probability times
class probability. That number is not a probability, can exceed 1, and is not reported: a semantic
detection's `confidence` is `None`, and mozo does not invent a region score.

### How the picture is fed to the network

Panoptic variants scale the picture so its long side is 640 (or 1280) and pad the rest with black;
the masks are scaled back to the picture. The semantic variant scales the short side to 512 and
slides square 512 windows along the long side, averaging where they overlap. Both are the authors'
evaluation pre-processing, bit for bit.

### transformers gives slightly different masks

The model cards' `transformers` snippet uses `EomtImageProcessorFast`, whose resize differs from the
authors' by about one grey level. On the example photograph that moves 2,120 pixels to another
class on `coco-panoptic-large`, and drops a segment on `coco-panoptic-small`. With
`use_fast=False` it matches the authors, and mozo, exactly.

### Parity

The torch path is bit-identical to `tue-mps/eomt` at commit `aa017c3`, the authors' own code, on
CUDA (RTX 3060, torch 2.7.0 and 2.13.0, CUDA 12.6) in fp32 and in fp16 autocast, and on the CPU:
the checkpoint against the authors' file, the pixels and their memory layout, every block, the
logits, the final partition, and what `predict` returns, on 28 pictures per variant. The
memory-saving post-processing (only the confident segments are scaled to full size; the semantic
classes are scaled a few at a time) is in that comparison. See
`mozo/vendors/eomt_deploy/PROVENANCE.md`.

## Weights and licence

Code and weights are MIT, stated by the authors on the repository and on every model card. The
backbones were initialised from Meta's DINOv2 (Apache-2.0). EoMT's DINOv3-based checkpoints (Meta's
own licence), its 7B checkpoint (CC-BY-NC-4.0 backbone) and its Cityscapes checkpoint (the dataset's
licence forbids commercial use of derivative works, trained models included) are not offered.

**About the training data.** The COCO variants were trained on COCO 2017 panoptic, whose
annotations are CC-BY-4.0 and whose images are Flickr photographs under their owners' licences.
`ade20k-semantic-large` was trained on ADE20K, whose annotations are BSD-3-Clause but whose images
MIT CSAIL makes available for non-commercial research and educational use only. Neither is a
licence on the weights; judge your own use with them in mind. The NOTICE beside each checkpoint
says the same.
