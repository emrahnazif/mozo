# Model weights and their licences

<!--
summary: Which licence each model's weights carry, what AGPL-3.0, CC-BY-NC-4.0 and the SAM License ask of you, and how to serve only permissive weights.
label: Licences
keywords: mozo, licences, agpl, model weights, commercial use, compliance
-->

Mozo's own code is Apache-2.0, and so is every model's vendored inference code. The weights are
separate works with their own licences, published unmodified beside their licence text and a NOTICE
naming the exact upstream release. Which licence applies to you depends on the weights you use, not
on mozo.

## The licences

| Licence | Variants | Families |
|---|---|---|
| Apache-2.0 | 47 | RF-DETR, ViTPose, OWLv2, Grounding DINO, MM-Grounding-DINO, SAM 2, EdgeTAM, SigLIP 2, EasyOCR, Granite Embedding R2, Moebius, Depth Anything V2 (7 of its 9) |
| MIT | 5 | CLIP, BEN2 |
| AGPL-3.0 | 30 | YOLOv8, YOLO11, YOLO12, YOLO26 |
| CC-BY-NC-4.0 | 2 | Depth Anything V2 `base` and `large` |
| SAM License | 1 | SAM 3 |

Ask before downloading anything:

```python
from mozo import weights

weights.license_of("yolov8", "nano")      # 'AGPL-3.0'
```

`GET /models` carries the same identifier for every variant a server offers, and
`workflow.check()` lists the licence of every model a workflow loads before it runs.

## What the restrictive ones ask

Nothing here is legal advice, and complying with a model's licence is the responsibility of whoever
runs it.

- **AGPL-3.0, every YOLO.** The weights are AGPL-3.0, or covered by a commercial licence from
  Ultralytics. Anything you export from them inherits those terms, and **serving predictions from
  them over a network places AGPL-3.0 section 13 obligations on you**. Mozo's YOLO inference code is
  Apache-2.0: it was written independently and imports nothing from Ultralytics.
- **CC-BY-NC-4.0, Depth Anything V2 `base` and `large`.** Non-commercial use only. The other seven
  Depth Anything V2 variants, including the metric ones of the same sizes, are Apache-2.0.
- **SAM License, SAM 3.** Meta's own licence, not an open-source one. It prohibits military,
  nuclear, espionage and weapons uses, those restrictions pass to whoever you serve predictions to,
  it binds on use rather than on signing, and it must travel with the weights if you pass them on.
  [SAM 3](/docs/models/sam3.md#weights-and-licence) has the details.

## Serve only permissive weights

`MOZO_ENABLE` decides what a server offers. Set it to this list in the server's environment:

```bash
MOZO_ENABLE=ben2,clip,easyocr,edgetam,granite_embedding_r2,grounding_dino,mm_grounding_dino,\
moebius,owlv2,rfdetr,sam2,siglip2,vitpose,depth_anything_v2/small,depth_anything_v2/indoor-small,\
depth_anything_v2/indoor-base,depth_anything_v2/indoor-large,depth_anything_v2/outdoor-small,\
depth_anything_v2/outdoor-base,depth_anything_v2/outdoor-large
```

That is the 52 Apache-2.0 and MIT variants. Depth Anything V2 is named variant by variant because
it is the one family whose licence is not the same for every variant. Because the list names what
to offer rather than what to leave out, it still offers 49 after an upgrade that adds a family.

A server set this way never downloads or serves the weights it leaves out, through any route:
a workflow that names one fails at that node instead of loading it.

## Your own checkpoints

A checkpoint you supply is under whatever terms you got it under. Mozo cannot know them, so
`license_of` answers only for weights mozo publishes.
