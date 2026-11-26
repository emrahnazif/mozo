# HQ-SAM: segment what you click, thin parts included

<!--
summary: HQ-SAM in mozo: SAM with an extra output token for fine structure, click-and-box segmentation that keeps spokes, legs and wires, Apache-2.0 code and weights.
label: HQ-SAM
keywords: mozo, hq-sam, sam-hq, segment anything in high quality, promptable segmentation, fine structure
-->

HQ-SAM segments the thing you point at, as [SAM 2](/docs/models/sam2.md) does, and keeps its thin
parts: the spokes of a bicycle, the legs of an insect, the rails of a chair, which SAM and SAM 2
tend to round off into a blob. It is Meta's SAM with its weights frozen and one learned output token
added, whose mask also reads an early, high-resolution encoder feature. It takes the same prompts
as SAM 2 (clicks, a box, or both) and returns the same result.

Use it when the outline matters more than the encode time: annotation, cut-outs, anything with
fine structure. For speed, see [EdgeTAM](/docs/models/edgetam.md); for masks of everything matching
a phrase, see [SAM 3](/docs/models/sam3.md).

## Quick start

A click on the laptop in the example photograph:

```python
import mozo
from mozo.image import load_image

image = load_image("example.jpg")
model = mozo.get_model("sam_hq/tiny")

found = model.predict(image, points=[[1020, 960]], labels=[1])

for candidate in found:
    print(round(candidate.confidence, 3), candidate.bbox)
```

```text
0.987 [746.0, 776.0, 1352.0, 1133.0]
0.932 [519.0, 776.0, 1438.0, 1212.0]
0.912 [744.0, 777.0, 1440.0, 1154.0]
```

```python
found = model.predict(image, boxes=[685, 783, 1355, 1136], multimask_output=False, name="laptop")
found[0].class_name, round(found[0].confidence, 3)     # ('laptop', 0.98)
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/sam_hq/base?box=685,783,1355,1136&multimask=false" -F "file=@example.jpg"
```

In a workflow, `sam_hq_boxes` adds a mask to every box a detector found.

## Variants

| ID | Trunk | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `sam_hq/tiny` | TinyViT (MobileSAM's), "Light HQ-SAM" | 43 MB | torch | Apache-2.0 |
| `sam_hq/base` | SAM ViT-B | 379 MB | torch | Apache-2.0 |
| `sam_hq/large` | SAM ViT-L | 1.3 GB | torch | Apache-2.0 |
| `sam_hq/huge` | SAM ViT-H | 2.6 GB | torch | Apache-2.0 |

`tiny` is the one that runs on a CPU; from `base` up it is a GPU model.

### How they compare

On HQ-SAM's four fine-structure validation sets (HQSeg-44K: DIS-VD, COIFT, HRSOD and
ThinObject5k-TE, 1,537 images), each object prompted with its box, one mask out, the mean over
the four sets of mask IoU and boundary IoU:

| Model | Mean IoU | Boundary IoU | CPU, one image + box |
|---|---|---|---|
| [EdgeTAM](/docs/models/edgetam.md) | 74.2 | 64.6 | 0.34 s |
| [SAM 2](/docs/models/sam2.md) `tiny` | 82.0 | 73.3 | 0.72 s |
| SAM 2 `large` | 83.6 | 76.5 | |
| HQ-SAM `tiny` | 82.0 | 73.4 | 0.55 s |
| HQ-SAM `base` | 86.5 | 78.7 | 3.5 s |
| HQ-SAM `large` | **88.0** | **80.9** | |
| HQ-SAM `huge` | 86.7 | 79.9 | |

`base` outlines thin structure better than SAM 2 `large`; `large` is the best HQ-SAM on these
sets. `huge` is the strongest of the four on everyday objects in the authors' COCO evaluation
(about 49.9 mask AP against `large`'s 49.5 and `base`'s 46.7, read off the authors' chart), not on fine structure. `tiny` is as good as
SAM 2 `tiny` here, in less CPU time and a quarter of the download. CPU times are a 1920x1281
photograph on a shared 20-core machine, so read them as a ranking.

## Inputs and outputs

```python
model.predict(image, points=None, labels=None, boxes=None, multimask_output=True, name=None)
model.prepare(image)
```

The parameters, the result and the HTTP query parameters (`point`, `label`, `box`, `multimask`,
`name`, `sha256`) are the same as SAM 2's: see
[SAM 2's inputs and outputs](/docs/models/sam2.md#inputs-and-outputs). In short:

- `points` with `labels` (`1` include, `0` exclude), `boxes`, or both.
- One row per candidate mask: a boolean `HxW` mask, its box, and the model's predicted IoU as
  `confidence`.
- `class_name` is `None` unless you pass `name`; `class_id` is the index of the prompt.

## How it behaves

- **A click returns three candidates**, best first; a box with `multimask_output=False` returns one.
  Each candidate is one of SAM's three masks with HQ-SAM's correction added. HQ-SAM's own code keeps
  only the best of the three; mozo returns all three, as it does for every promptable model, and
  the first is exactly what HQ-SAM's code returns.
- **The score is SAM's predicted IoU for its own mask**, before the correction. HQ-SAM trains no
  score for the corrected mask, so the score ranks the candidates rather than rating the outline.
- **The picture is letterboxed, not squashed**: the long side goes to 1024 and the rest is padded,
  where SAM 2 stretches to a square. A very wide or tall image therefore gets fewer pixels on its
  short side than it would in SAM 2.
- **The image encode is cached** by pixel content, five images per model, and `prepare(image)`,
  `/prepare` and `?sha256=` move it to when the image opens. See
  [prepare an image before the first prompt](/docs/models/sam2.md#prepare-an-image-before-the-first-prompt).
- **HQ-SAM's `hq_token_only` option is not offered.** mozo runs the setting the authors evaluate
  with (SAM's mask plus the correction).

### Cost

On a GPU the encode is the whole cost: on an RTX 3060, `tiny` encodes an image in 35 ms and `base`
in 244 ms in torch fp32, and a prompt then costs 7-8 ms. On a CPU only `tiny` is practical.

## Weights and licence

Code and weights are Apache-2.0. The checkpoints are the authors' files from
`huggingface.co/lkeab/hq-sam`, unmodified; they hold Meta's SAM weights (Apache-2.0) or, for `tiny`,
MobileSAM's TinyViT (Apache-2.0), with HQ-SAM's trained parameters on top. Mozo's code is an
extraction of `SysCV/sam-hq`, bit-identical to it on CUDA (RTX 3060, torch 2.13.0, CUDA 12.6) in
fp32, mozo's only torch precision for HQ-SAM, and also on CPU; the TinyViT code in it is MIT
(Microsoft), reproduced in the package NOTICE. `mozo/vendors/sam_hq_deploy/PROVENANCE.md` records
the upstream commit, what was left out (including HQ-SAM 2), and the measurements.
