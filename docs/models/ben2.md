# BEN2: background removal with a soft matte

<!--
summary: BEN2 in mozo: remove the background of a photograph with a per-pixel opacity matte that keeps hair, or a ready RGBA cut-out. MIT.
label: BEN2
keywords: mozo, ben2, background removal, alpha matte, image matting, cutout
-->

BEN2 removes the background of a photograph. It returns an **alpha matte**: an opacity for every
pixel, not a yes-or-no mask. That is the difference between keeping a head of hair and cutting it
off. It takes no prompt and gives one matte for the whole foreground; `cutout` turns it into an
RGBA image with the background transparent.

## Quick start

```python
import mozo

model = mozo.get_model("ben2/base")

alpha = model.predict("example.jpg")               # (1281, 1920) uint8, 255 is foreground
rgba = model.cutout("example.jpg", refine=True)    # (1281, 1920, 4) uint8
```

Over HTTP the answer is the RGBA cut-out as a PNG, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/ben2/base" -F "file=@example.jpg" --output cutout.png
```

## Variants

| ID | Output | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `ben2/base` | `HxW` uint8 alpha matte | 381 MB | torch | MIT |

Upstream publishes one checkpoint.

## Inputs and outputs

```python
model.predict(image, *, stretch=True)
model.cutout(image, *, stretch=True, refine=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `stretch` | `bool` | `True` | Stretch the matte per image, as upstream does. `False` returns the model's own sigmoid, where 0.5 means something. No effect when `refine=True`. |
| `refine` | `bool` | `False` | `cutout` only: estimate each edge pixel's foreground colour so the old background does not show as a fringe. |

- `predict` returns an `HxW` uint8 array at the input's resolution, 255 for foreground.
- `cutout` returns an `HxWx4` uint8 RGBA array at the input's resolution.

Over HTTP, `/predict` takes only `file` and returns `cutout(image)` with the defaults as a PNG. For
`stretch=False` or `refine=True`, use Python.

## How it behaves

### A matte, not a mask

Every other model in mozo that outlines something decides per pixel: in or out. Threshold that
around hair and the hair goes with the background, because a strand thinner than a pixel was never
fully either. BEN2 gives the mixing fraction instead, which is what you need to put the subject on a
new background.

That is also why it sits beside [SAM 3](/docs/models/sam3.md) rather than replacing it. SAM 3
answers "where is every cow", from a prompt, one instance at a time. BEN2 answers "how much of this
pixel is foreground", with no prompt, for the whole image.

### The default matte is not a probability

The network outputs a sigmoid, and upstream then **stretches it per image**: the most-foreground
pixel in this picture becomes 255 and the least becomes 0. Mozo does the same by default, because it
is what upstream's users get and what its examples show.

```python
alpha = model.predict("example.jpg")                  # stretched, upstream's default
alpha = model.predict("example.jpg", stretch=False)   # the model's own sigmoid
```

Compare a stretched matte only within one image, never across two. An image whose most confident
pixel scored 0.6 still comes back with pixels at 255. Use `stretch=False` when you threshold,
compare frames, or feed the matte to something else.

Upstream divides by `max - min` with no guard, so an image it reads as uniform gives `nan` cast to
uint8: silent garbage. Mozo skips the stretch when the matte is flat and returns the constant.

### `refine` changes colour, not opacity

Along a soft edge every pixel mixes foreground and background, so a cut-out placed on something new
shows a fringe of the old background. `cutout(refine=True)` estimates the foreground colour on its
own and rewrites RGB; the alpha is left alone. It costs two full-resolution box blurs, which is why
it is off by default.

The refined path also uses a slightly different alpha, and that is upstream's doing: the plain path
resizes then stretches, the refined path converts the raw 1024×1024 sigmoid to uint8 and resizes
that. Mozo reproduces both exactly, which is why `stretch` has no effect with `refine=True`.

### It runs at 1024×1024, squashed

Each image is resized to a 1024×1024 square, **without** keeping the aspect ratio, then split into
four 512×512 quadrants that are matted alongside a 512×512 view of the whole frame. The matte is
resized back to your image afterwards. A 4000×500 panorama is squashed to a square, matted and
unsquashed. That looks wrong and is correct: the weights were trained that way, and letterboxing
instead moves every pixel.

### Torch only, and the device matters most

Both an ONNX and a CoreML graph were built and measured, and neither is published. ONNX was slower
than torch (0.88×). CoreML was 1.56× faster on Apple silicon, but 7.8% of its alpha pixels differed
by more than one grey level, all of them on the edges, which are the reason to run a matte at all.
`tools/export/ben2.py` re-runs that measurement.

On Apple silicon, torch takes about 600 ms for a 1920×1281 photograph, against about 5.5 s on the
CPU.

### Parity

On the CPU in float32, mozo reproduces upstream bit for bit at every stage. Upstream's CUDA path
runs in float16, which is a different computation and is not claimed.

## Weights and licence

Code and weights are MIT, © 2025 Prama LLC, from `PramaLLC/BEN2` on GitHub and its checkpoint on
Hugging Face. The code's licence is in the repository's `LICENSE`; the weights' is on the model
card, which is not gated. Mozo publishes the 380 MB of weights from upstream's safetensors file, not
the 1.13 GB training checkpoint that also carries optimiser state. The upstream commit and the
measurements are in `mozo/vendors/ben2_deploy/PROVENANCE.md`.
