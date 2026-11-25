# All 85 models, by task

<!--
summary: Every model mozo serves, grouped by task, with its variants, weights licence, runtimes and download size.
label: All models
keywords: mozo, model catalogue, computer vision models, licences, runtimes
-->

Mozo serves 85 variants across 19 families. A model is named `family/variant`, the same in Python
(`mozo.get_model("rfdetr/medium")`) and over HTTP (`/predict/rfdetr/medium`). Each family's page
has examples, every parameter, and how the model behaves.

Sizes are the torch weights a first load downloads. The runtimes are the ones mozo can execute for
that family; `runtime="auto"` picks among them for your device.

## Object detection

A box, a class and a confidence for each object.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [RF-DETR](/docs/models/rfdetr.md) | `nano` `small` `medium` `large` | Apache-2.0 | torch, onnx, coreml | 136–405 MB |
| [YOLOv8](/docs/models/yolov8.md) | `nano` `small` `medium` `large` `xlarge` | AGPL-3.0 | torch, onnx, coreml | 7–137 MB |
| [YOLO11](/docs/models/yolov11.md) | `nano` `small` `medium` `large` `xlarge` | AGPL-3.0 | torch, onnx | 6–115 MB |
| [YOLO12](/docs/models/yolov12.md) | `nano` `small` `medium` `large` `xlarge` | AGPL-3.0 | torch, onnx, coreml | 6–119 MB |
| [YOLO26](/docs/models/yolov26.md) | `nano` `small` `medium` `large` `xlarge` | AGPL-3.0 | torch, onnx | 6–119 MB |

## Instance segmentation

A box and a mask for each object, in one pass. Same call as detection: the variant decides whether
masks come back.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [RF-DETR](/docs/models/rfdetr.md) | `seg-nano` `seg-small` `seg-medium` `seg-large` | Apache-2.0 | torch, onnx, coreml | 135–145 MB |
| [YOLO11](/docs/models/yolov11.md) | `seg-nano` `seg-small` `seg-medium` `seg-large` `seg-xlarge` | AGPL-3.0 | torch, onnx | 6–125 MB |
| [YOLO26](/docs/models/yolov26.md) | `seg-nano` `seg-small` `seg-medium` `seg-large` `seg-xlarge` | AGPL-3.0 | torch, onnx | 7–142 MB |

## Keypoints

COCO's 17 joints for each person, in pixels of your image. RF-DETR finds people and their joints in
one pass; ViTPose adds joints to boxes you already have.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [RF-DETR](/docs/models/rfdetr.md) | `keypoint-preview` | Apache-2.0 | torch | 164 MB |
| [ViTPose](/docs/models/vitpose.md) | `small` `base` `large` `huge` | Apache-2.0 | torch, onnx (not `huge`) | 133 MB–3.6 GB |

## Detection from words

Boxes for anything you can name, from a phrase instead of a fixed class list.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [OWLv2](/docs/models/owlv2.md) | `base` `base-ensemble` `large` `large-ensemble` | Apache-2.0 | torch | 620 MB–1.8 GB |
| [Grounding DINO](/docs/models/grounding-dino.md) | `tiny` `base` | Apache-2.0 | torch | 694–938 MB |
| [SAM 3](/docs/models/sam3.md) | `sam3` (masks as well as boxes) | SAM License | torch, coreml | 3.5 GB |

## Segmentation from a click or a box

Point at something and get a mask of the thing you pointed at.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [SAM 2](/docs/models/sam2.md) | `tiny` `small` `base_plus` `large` | Apache-2.0 | torch | 156–898 MB |
| [EdgeTAM](/docs/models/edgetam.md) | `edgetam` | Apache-2.0 | torch | 56 MB |
| [SAM 3](/docs/models/sam3.md) | `sam3` | SAM License | torch, coreml | 3.5 GB |

## Classification from words, and image embeddings

Name your classes in words and each one is scored against the image, or take the vectors to search
images by text in a database of your own.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [CLIP](/docs/models/clip.md) | `base` `base-16` `large` `large-336` | MIT | torch | 599 MB–1.7 GB |
| [SigLIP 2](/docs/models/siglip2.md) | `base-224` `base-256` `so400m-384` `so400m16-256` `giant-384` | Apache-2.0 | torch | 1.5–7.5 GB |

## Text embeddings

English text in, one vector out, for searching documents by meaning. Text only: these vectors do
not share a space with CLIP's or SigLIP 2's.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [Granite Embedding R2](/docs/models/granite-embedding-r2.md) | `small` | Apache-2.0 | torch, coreml | 95 MB |

## Text recognition

Every line of text in the image, read. A variant is a script, not a language: `latin` covers 41
languages.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [EasyOCR](/docs/models/easyocr.md) | `english` `latin` `chinese-simplified` `japanese` `korean` | Apache-2.0 | torch | 98–105 MB |

## Depth

A depth map at the image's resolution, in metres or relative.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [Depth Anything V2](/docs/models/depth-anything-v2.md) | `small` (relative) | Apache-2.0 | torch | 99 MB |
| | `base` `large` (relative) | CC-BY-NC-4.0 | torch | 390 MB–1.3 GB |
| | `indoor-` and `outdoor-`, three sizes each (metres) | Apache-2.0 | torch | 99 MB–1.3 GB |
| [Depth Anything 3](/docs/models/depth-anything-v3.md) | `small` `base` `mono-large` (relative) | Apache-2.0 | torch, onnx, tensorrt | 100 MB–1.3 GB |

## Background removal

An opacity matte for the foreground, soft enough to keep hair.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [BEN2](/docs/models/ben2.md) | `base` | MIT | torch | 381 MB |

## Object removal

An image and a mask in, the masked thing painted out.

| Family | Variants | Licence | Runtimes | Size |
|---|---|---|---|---|
| [Moebius](/docs/models/moebius.md) | `general` `places2` | Apache-2.0 | torch | 1.1 GB |

## Ask mozo instead of this page

The same facts are available in code, without loading a model or touching the network:

```python
import mozo
from mozo import weights

mozo.MODEL_REGISTRY["rfdetr"]["variants"]     # ['nano', 'small', ..., 'keypoint-preview']
weights.license_of("yolov8", "nano")          # 'AGPL-3.0'
```

Over HTTP, `GET /models` lists every model the server offers, with its task and licence.
