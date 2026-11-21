# Moebius: remove an object from a photo

<!--
summary: Moebius inpainting in mozo: an image and a mask in, the masked object painted out, everything else byte-identical. Apache-2.0.
label: Moebius
keywords: mozo, moebius, object removal, inpainting, image editing
-->

Moebius removes things from photographs. Give it an image and a mask, and it repaints the masked
region so the thing was never there. It answers with a picture: there is no `Detections`, no class
and no confidence number. Pair it with a model that finds what to remove, such as
[SAM 3](/docs/models/sam3.md) for a phrase or [EdgeTAM](/docs/models/edgetam.md) for a click.

## Quick start

Remove every cup from the example photograph:

```python
import mozo
from mozo.image import load_image

image = load_image("example.jpg")
cups = mozo.get_model("sam3/sam3").predict(image, "a cup")

clean = mozo.get_model("moebius/general").predict(image, cups, seed=0, dilate=8)
clean.shape == image.shape      # True: same size, and outside the mask, the same bytes
```

Over HTTP the selection is a rectangle, and the answer is a PNG:

```bash
curl -X POST "http://localhost:8000/predict/moebius/general?box=632,589,706,694&seed=0" \
  -F "file=@example.jpg" -o clean.png
```

Repeat `box` to remove several things. For a mask that follows an object's outline rather than a
rectangle, run a segmenter and use the Python API.

## Variants

| ID | Weights (torch) | Runtimes | Licence |
|---|---|---|---|
| `moebius/general` | 1,073 MB | torch | Apache-2.0 |
| `moebius/places2` | 1,073 MB | torch | Apache-2.0 |

`general` is for arbitrary photographs, `places2` for scenes and backgrounds. Upstream also
publishes two face-specific checkpoints; mozo does not carry them. The weights are a UNet and an
autoencoder, downloaded together. `general` also publishes ONNX graphs, which mozo does not run.

## Inputs and outputs

```python
model.predict(image, mask, *, seed=0, steps=20, guidance=2.0, dilate=0, feather=3)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str` or `numpy.ndarray` | | A file path or an `HxWx3` RGB array. |
| `mask` | `numpy.ndarray` or `Detections` | | A boolean `HxW` array, or detections: each one's mask, else its polygon, else its box. |
| `seed` | `int` | `0` | Which sample to draw. The same seed gives the same picture. |
| `steps` | `int` | `20` | Denoising steps requested. Nineteen of twenty run, as upstream does. |
| `guidance` | `float` | `2.0` | Classifier-free guidance scale, upstream's own value. |
| `dilate` | `int` | `0` | Grow the mask by this many pixels first. |
| `feather` | `int` | `3` | Blur radius on the mask edge before compositing. `0` for a hard edge. |

All the regions are removed in one pass over their union. A detection with only a box removes the
whole rectangle, so prefer a segmenter's masks.

Returns an `HxWx3` uint8 RGB array, the same size as `image`. An empty mask returns the image
unchanged without running the model.

Over HTTP: `box` (repeatable, `x1,y1,x2,y2`) is required, and `seed` is optional. The response is an
8-bit PNG, lossless so two seeds can be compared honestly.

## How it behaves

### It draws a sample, not an estimate

Every other model in mozo describes an image. This one rewrites one, and the model does not
estimate anything: it draws a sample. Run it again with `seed=1` and you get a different, equally
valid removal; neither is more correct. That is why `seed` is an argument and not a hidden
constant.

### Everything the seam does not reach comes back byte-identical

Upstream returns the decoder's reconstruction of the whole frame, which changes every pixel,
including the ones nobody selected. Mozo composites instead: the hole is the model's, and
everything else is your own array, unchanged. That is what makes it safe to point at a photograph
you care about.

At the border, the seam is feathered with a 3-pixel blur so it reads as a gradient rather than a
cut-out, and the blend spreads a few pixels past your selection. On the example photograph, with
a cup's box as the mask, no pixel more than 7 px outside the box changed. Pass `feather=0` and
nothing outside the mask changes at all, at the cost of a visible cut-out.

### If the thing is still faintly there, raise `dilate`

An object's shadow and its antialiased edge sit outside the mask that found it, and removing the
object without them leaves an outline. This is the most common surprise, and 8 to 16 pixels
usually settles it.

The same setting fixes the other cause. On a mask only a few pixels across, the feather pulls the
blend below full strength even at the centre, so a little of the original survives everywhere.
Growing the mask fixes that; shrinking the feather only makes the seam obvious.

### It runs at 512×512 and nothing else

This is a property of the weights, not a setting: the cross-attention's position table has a row
per latent cell, so there is nowhere to put the positions of a larger image. `predict` resizes for
you and composites back at your resolution, which means detail inside the hole is capped at 512.
On a large photograph that is visible if you look for it.

### Cost

On the example photograph (1920×1281) on an Apple M1 Max (`mps`), removing six cups took 18.3 s
with the default 20 steps.

## Weights and licence

Code and weights are Apache-2.0, from `hustvl/Moebius`. The authors state the weights' licence as
Apache-2.0 on GitHub and MIT on the Hugging Face model card; the NOTICE published with each variant
records both. The autoencoder is `hustvl/PixelHacker`'s, MIT, a separate work published beside it
with its own NOTICE. `mozo/vendors/moebius_deploy/PROVENANCE.md` records the upstream source and
why the face checkpoints are left out.
