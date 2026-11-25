# Mozo

<!--
summary: Run 85 computer vision models, and workflows built from them, from one pip install, with the original's exact numbers.
label: Mozo
keywords: mozo, computer vision, model serving, object detection, segmentation, embeddings
template: docs
-->

Mozo runs computer vision models from one `pip install`: detection, segmentation, keypoints,
open-vocabulary detection, OCR, depth, embeddings, background removal and object removal. That is
85 models across 19 families, served over HTTP or called from Python, plus workflows that chain
them.

Each of these models normally comes with its own package, and each package pins its own torch,
numpy and OpenCV. Put a few in one environment and something breaks. Mozo ships none of those
packages. Every model's inference code is copied into mozo and checked to return **exactly** the
original's numbers, compared with no tolerance. So one environment runs all 85.

## Install

```bash
pip install mozo
```

Mozo needs Python 3.10 or newer. No model needs an extra package: weights are downloaded the first
time you use a model, checked against their sha256, and cached in `~/.cache/mozo`.

## Quick start

The examples in these docs use one photograph, five people around a table, and show what mozo
actually returned for it. Download it to follow along:

```bash
curl -O https://raw.githubusercontent.com/emrahnazif/mozo/main/mozo/static/example.jpg
```

From Python:

```python
import mozo

model = mozo.get_model("rfdetr/medium")

for found in model.predict("example.jpg"):
    print(found.class_name, round(found.confidence, 2), found.bbox)
```

```text
person 0.97 [0.61, 122.33, 761.01, 1268.59]
person 0.96 [1303.81, 16.69, 1917.26, 1270.06]
laptop 0.94 [685.36, 783.82, 1354.26, 1135.56]
cell phone 0.93 [1163.78, 1082.97, 1449.97, 1182.17]
...
```

`predict` takes a file path, encoded bytes or an RGB array, and returns a PixelFlow `Detections`:
the same result type from every detection family, so filtering and drawing are written once with
[PixelFlow](https://emrah.fr/pixelflow/).

Or as a server, for other machines and other languages:

```bash
mozo start
```

```bash
curl -X POST "http://localhost:8000/predict/rfdetr/medium" -F "file=@example.jpg"
```

The response is the same detections as JSON. The server documents every route at
`http://localhost:8000/docs`.

## Next steps

- [Installation](/docs/installation.md): GPUs, Apple silicon, the ONNX and CoreML runtimes, and
  where weights are cached.
- [Python](/docs/python.md): loading models, reading and drawing results, memory, and your own
  checkpoints.
- [HTTP server](/docs/server.md): serving models to other machines and other languages.

## Choose a model

[All models](/docs/models/all-models.md) lists every family and variant by task, with its licence,
size and runtimes. Each family has its own page with examples, parameters and how it behaves:

| Task | Families |
|---|---|
| Object detection | [RF-DETR](/docs/models/rfdetr.md), [YOLOv8](/docs/models/yolov8.md), [YOLO11](/docs/models/yolov11.md), [YOLO12](/docs/models/yolov12.md), [YOLO26](/docs/models/yolov26.md) |
| Instance segmentation | [RF-DETR](/docs/models/rfdetr.md), [YOLO11](/docs/models/yolov11.md), [YOLO26](/docs/models/yolov26.md) |
| Keypoints | [RF-DETR](/docs/models/rfdetr.md), [ViTPose](/docs/models/vitpose.md) |
| Detection from words | [OWLv2](/docs/models/owlv2.md), [Grounding DINO](/docs/models/grounding-dino.md), [MM-Grounding-DINO](/docs/models/mm-grounding-dino.md), [SAM 3](/docs/models/sam3.md) |
| Segmentation from a click or box | [SAM 2](/docs/models/sam2.md), [EdgeTAM](/docs/models/edgetam.md), [SAM 3](/docs/models/sam3.md) |
| Classification from words, image embeddings | [CLIP](/docs/models/clip.md), [SigLIP 2](/docs/models/siglip2.md) |
| Text embeddings | [Granite Embedding R2](/docs/models/granite-embedding-r2.md) |
| Text recognition | [EasyOCR](/docs/models/easyocr.md) |
| Depth | [Depth Anything V2](/docs/models/depth-anything-v2.md) |
| Background removal | [BEN2](/docs/models/ben2.md) |
| Object removal | [Moebius](/docs/models/moebius.md) |

## Licences

Mozo's code is Apache-2.0. The weights are separate works with their own licences: 47 variants
are Apache-2.0, 5 are MIT, 30 are AGPL-3.0 (every YOLO), 2 are CC-BY-NC-4.0 and 1 carries Meta's
SAM License. Each family page says which, and `mozo.weights.license_of("yolov8", "nano")` answers
before anything is downloaded.
