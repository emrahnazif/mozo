# Depth Anything V2: depth from one photograph

<!--
summary: Depth Anything V2 in mozo: a depth map from a single image, in metres or relative, at the image's resolution.
label: Depth Anything V2
keywords: mozo, depth anything v2, depth estimation, monocular depth, metric depth
-->

Depth Anything V2 estimates depth from a single photograph and returns one value per pixel. It
comes in two groups that answer different questions: **metric** variants give distances in metres,
**relative** variants only say which pixels are nearer than others. `model.unit` tells you which
one you have.

## Quick start

```python
import mozo

model = mozo.get_model("depth_anything_v2/indoor-small")
depth = model.predict("example.jpg")

depth.shape, depth.dtype     # ((1281, 1920), dtype('float32')), the image's own size
model.unit                   # 'metres'
depth.min(), depth.max()     # about 0.67 and 6.85 metres
```

Over HTTP the answer is a 16-bit PNG, with what it means in the headers:

```bash
curl -X POST "http://localhost:8000/predict/depth_anything_v2/indoor-small" \
  -F "file=@example.jpg" -D headers.txt --output depth.png
```

```text
content-type: image/png
x-depth-unit: metres
x-depth-min: 0.6743873953819275
x-depth-max: 6.845202922821045
```

Read a pixel back as `depth = min + png / 65535 * (max - min)`. `x-depth-unit` is `metres`, or
`none` for a relative variant.

## Variants

| ID | Output | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `depth_anything_v2/small` | relative | 99 MB | torch | Apache-2.0 |
| `depth_anything_v2/base` | relative | 390 MB | torch | **CC-BY-NC-4.0** |
| `depth_anything_v2/large` | relative | 1,341 MB | torch | **CC-BY-NC-4.0** |
| `depth_anything_v2/indoor-small` | metres, 0–20 m | 99 MB | torch | Apache-2.0 |
| `depth_anything_v2/indoor-base` | metres, 0–20 m | 390 MB | torch | Apache-2.0 |
| `depth_anything_v2/indoor-large` | metres, 0–20 m | 1,341 MB | torch | Apache-2.0 |
| `depth_anything_v2/outdoor-small` | metres, 0–80 m | 99 MB | torch | Apache-2.0 |
| `depth_anything_v2/outdoor-base` | metres, 0–80 m | 390 MB | torch | Apache-2.0 |
| `depth_anything_v2/outdoor-large` | metres, 0–80 m | 1,341 MB | torch | Apache-2.0 |

`base` and `large` (relative) are **non-commercial**. Every other variant is Apache-2.0. The indoor
variants are fine-tuned on Hypersim, the outdoor ones on Virtual KITTI 2.

Measured on Apple silicon (MPS) at the time of extraction, the whole pipeline takes about 70 ms for
`small`, 150 ms for `base` and 400 ms for `large`, on photographs about 518×790 after resizing.

## Inputs and outputs

```python
model.predict(image)
model.unit
model.max_depth
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |

`predict` returns an `HxW` float32 array at the input's resolution.

- `model.unit` is `"metres"` for the metric variants and `None` for the relative ones. Mozo never
  guesses a unit.
- `model.max_depth` is the metric ceiling in metres (`20.0` indoor, `80.0` outdoor), or `None`.

Over HTTP, `/predict` takes only `file`, and answers with the PNG and headers shown above.

## How it behaves

### Relative depth is not a distance

A relative variant returns inverse depth on a scale that changes from image to image. Larger means
nearer, and that is all it means: no value is a distance, and two images cannot be compared. On the
example photograph `small` returns values from 0.0 to about 7.2, which are not metres. If you need
distances, use an `indoor-` or `outdoor-` variant, whichever matches the scene.

### Over HTTP, 16 bits rather than 8

The PNG has 16 bits per pixel because six of the nine variants predict metres, and squeezing those
into 256 levels would throw the measurement away. The range travels in `x-depth-min` and
`x-depth-max`, so you get the original values back to within 1/65535 of the range.

### The aspect ratio is kept

The shorter side is resized to 518 pixels and the longer side follows, both rounded to a multiple of
14, so a 1920×1281 photograph runs at 770×518, not squashed to a square. The map is resized back to
your image's size afterwards. Upstream does the same, and its published numbers were measured that
way.

That is also why there is no ONNX or CoreML graph: the input shape follows the photograph, and a
graph exported at one shape fails at every other.

### Parity

Bit-identical to the authors' repository, on all nine variants: same state dict, forward pass
`max|delta| = 0.0`, and end to end on real photographs `max|delta| = 0.0`. `tools/verify/depth_anything_v2.py`
re-runs the comparison against a checkout of the pinned upstream commit.

## Weights and licence

© 2024 TikTok and The University of Hong Kong, from `DepthAnything/Depth-Anything-V2` and the
checkpoints its authors publish on Hugging Face. The code and seven of the nine checkpoints are
Apache-2.0; the relative `base` and `large` checkpoints are CC-BY-NC-4.0, for non-commercial use
only. Each variant ships its own licence and a NOTICE naming its source file. The upstream commit
and what was changed are in `mozo/vendors/depth_anything_v2_deploy/PROVENANCE.md`.
