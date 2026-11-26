# DINOv2: find images that look alike

<!--
summary: Meta's DINOv2 in mozo: an image in, one vector out, for near-duplicates, visual search, clustering and few-shot classification. Six variants, Apache-2.0.
label: DINOv2
keywords: mozo, dinov2, image embeddings, visual search, near-duplicate detection, clustering, retrieval, k-nn
-->

DINOv2 turns an image into one vector, and images that look alike land close together. Use it to
find near-duplicates, search a catalogue by example ("more like this one"), cluster a collection,
or classify from a handful of labelled examples by nearest neighbour. Meta trained it on 142M images
with no labels and no captions, so it captures what the picture shows, not what a caption would
say. It is mozo's only image-only embedder: it takes no text.

## Quick start

```python
import mozo

model = mozo.get_model("dinov2/base")
vectors = model.encode_image(["shelf_monday.jpg", "shelf_tuesday.jpg", "parking_lot.jpg"])

vectors.shape            # (3, 768), each of unit length
vectors[0] @ vectors[1]  # high: the same shelf
vectors[0] @ vectors[2]  # lower: a different scene
```

Over HTTP it is `/encode` and nothing else, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/encode/dinov2/base" -F "file=@shelf_monday.jpg" -F "file=@shelf_tuesday.jpg"
```

```json
{"model": "dinov2/base", "revision": "2026-11-25", "dim": 768, "embeddings": [[0.01873, -0.04211, ...], [...]]}
```

`/predict` answers `501` and says where to go instead, because the model returns vectors, not an
answer:

```json
{"detail": "dinov2 returns vectors, not an answer about an image, so it has no /predict. Use /encode/dinov2/base with an uploaded image."}
```

## Variants

| ID | Vector | Parameters | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|
| `dinov2/small` | 384-d | 22M | 88 MB | torch, onnx; tensorrt on CUDA | Apache-2.0 |
| `dinov2/base` | 768-d | 87M | 346 MB | torch, onnx; tensorrt on CUDA | Apache-2.0 |
| `dinov2/large` | 1024-d | 304M | 1.2 GB | torch, onnx; tensorrt on CUDA | Apache-2.0 |
| `dinov2/giant` | 1536-d | 1.1B | 4.5 GB | torch | Apache-2.0 |
| `dinov2/small-reg` | 384-d | 22M | 88 MB | torch, onnx; tensorrt on CUDA | Apache-2.0 |
| `dinov2/base-reg` | 768-d | 87M | 346 MB | torch, onnx; tensorrt on CUDA | Apache-2.0 |

`base` is the most used by far, and the default. The `-reg` variants are the same backbones trained
with four extra "register" tokens. Registers mainly clean up the model's per-patch features, which
mozo does not return. On the whole-image vector they change ImageNet k-NN accuracy by about a tenth
of a point either way, by the authors' numbers. Pick them if your existing index was built with
them.

## Inputs and outputs

```python
model.encode_image(image)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray`, or a list of them | | A file path, encoded image bytes, or an `HxWx3` RGB array. |

Returns a float32 array of shape `(N, D)`, in the order given, always two-dimensional. `D` is 384,
768, 1024 or 1536 depending on the variant. The vectors are L2-normalised, so a dot product is a
cosine similarity. The model's `.revision` names the weights that produced them.

`ValueError` if no image is given or one cannot be decoded.

Over HTTP, `/encode` takes `file`, repeated once per image, and returns `model`, `revision`, `dim`
and `embeddings`. Sending `?text=` answers `400`: DINOv2 does not embed text.

## How it behaves

### Images only, and not CLIP's space

DINOv2 learned from images alone, so its vectors cannot be searched by words, and they mean nothing
against [CLIP](/docs/models/clip.md)'s or [SigLIP 2](/docs/models/siglip2.md)'s. Use those two to
find images from a phrase. DINOv2 is for image against image, and there it keeps more of what the
picture itself shows (the object, its pose, texture and layout) than a model trained to agree with
captions.

### It sees the middle of the image

Each image is scaled so its short side is 256 pixels, and the central 224 x 224 is kept. This is
the authors' evaluation transform, the one their published accuracy was measured with. On a 16:9
frame, only the middle 49% of the width reaches the model. A difference near the edges, such as a
watermark in a corner, does not move the vector. Crop first if what matters is off-centre.

### A vector does not depend on what was encoded beside it

Each image runs through the network on its own. Running several in one batch is faster, but
changes each vector slightly (by about 1e-05) depending on its batch-mates. So the same image
always gets the same vector, whatever arrived with it. On an RTX 3060 this costs 1.4 times (`base`)
to 1.7 times (`small`) the backbone time a batch of 32 would take.

### The scores are cosine similarities

A dot product between two vectors is their cosine. It is not a probability and has no calibrated
zero: compare scores against each other, or against a threshold you calibrate on your own data.
For near-duplicates, look at the scores of a few pairs you know are duplicates and a few you know
are not, and set the line between them.

### An index is tied to the weights that built it

A vector from one variant or revision means nothing against a vector from another. A stored index
is tied to both, and changing either means embedding the collection again. `/encode` returns
`model` and `revision` with every response for that reason, and the model has `.revision`.

It is also tied to the preprocessing. The `transformers` port of DINOv2 crops a pixel to the left
of the authors' crop and, for the four plain variants, resizes the position embedding differently.
For the same photograph and weights its vector has a cosine of about 0.99 with mozo's, enough to
move a near-duplicate threshold. Do not mix vectors from the two in one index.

### On CUDA it runs through TensorRT

With TensorRT 11 or later installed (`pip install tensorrt-cu12` or `-cu13`), `auto` builds a
TensorRT engine in full fp32 from the published ONNX graph, for every variant but `giant`. The
first load on a machine builds it in a few seconds and keeps it in the cache. Its vectors are the
torch vectors to within 2e-05, and on 500 COCO photographs every image kept all ten of its nearest
neighbours. Without TensorRT, `auto` stays on torch.

`runtime="tensorrt-fp16"` runs an fp16 engine, built from a published fp16 graph, 3.8 to 4.8
times faster than torch for the network itself. Its vectors move a little (the worst cosine to
fp32 was 0.9971 on `large`), and about 0.5% of nearest neighbours change. `auto` never picks it,
and an index built in fp32 should be queried in fp32.

Measured on an RTX 3060 (vision-003), one 1920x1281 photograph, against torch fp32 on the same GPU.
The network alone, then the whole call, which includes resizing the photograph on the CPU:

| Variant | torch fp32 | tensorrt-fp32 | tensorrt-fp16 | whole call, torch / trt-fp32 / trt-fp16 |
|---|---|---|---|---|
| `small` | 4.2 ms | 2.9 ms | 1.1 ms | 17.6 / 16.3 / 14.7 ms |
| `base` | 11.2 ms | 9.0 ms | 2.7 ms | 23.6 / 21.6 / 16.4 ms |
| `large` | 35.3 ms | 30.3 ms | 7.4 ms | 46.7 / 41.8 / 20.7 ms |

On a large photograph most of a call is the resize, on the CPU, whichever runtime runs the
network. k-NN accuracy on Imagenette (10 classes, 1,000 index images, 3,925 queries, the authors'
k-NN protocol) did not move on any variant or runtime: 99.2% (`small`), 99.6% (`base`), 99.7%
(`large`).

`giant` runs on torch only. At 4.5 GB its graph is past ONNX's 2 GB limit for a single file.

### Parity

Bit-identical to the authors' torch-hub code (`facebookresearch/dinov2`, commit `7764ea0`) on CUDA
(RTX 3060, torch 2.13, CUDA 12.6) in fp32, the only precision mozo runs through torch, and also on
the CPU, at every stage: the checkpoint against the authors' file, the pixels, the tokens, every
block, the final norm, the class token and the unit vector, compared with `torch.equal` and no
tolerance. On CUDA that is 13,750 comparisons across the six variants on 29 images, giant included,
every one identical, with torch's default CUDA settings and both sides running the same attention
kernel. The `tensorrt` runtimes are a different engine and are measured by closeness instead
(above).

Upstream switches to xFormers' attention kernel whenever the `xformers` package happens to be
installed, which is different arithmetic. Mozo's copy has no such path, so its vectors do not
depend on what else is installed.

## Weights and licence

Code and weights are Apache-2.0: the project's README states it for both, and so do its model card
and every `facebook/dinov2-*` Hugging Face card, which download without a gated login. The same
repository also publishes Cell-DINO and XRay-DINO under non-commercial terms; mozo takes neither.
What was taken and how it was checked is in `mozo/vendors/dinov2_deploy/PROVENANCE.md`.
