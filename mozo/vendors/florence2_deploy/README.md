# florence2_deploy

Deployment-only Florence-2: Microsoft's DaViT vision tower and BART encoder-decoder, its default
3-beam decoding, and its post-processor, extracted from the remote code in Microsoft's Hugging Face
repositories. MIT code and weights. No `transformers`, `tokenizers`, `timm` or `einops` dependency.

One checkpoint, several tasks, picked by a task prompt. This package carries two:

| method | task | returns | served by mozo as |
|---|---|---|---|
| `detect` | `<OD>` | boxes, the model's own label, upstream's score | `florence2_detect` |
| `read` | `<OCR_WITH_REGION>` | each line's four corners and text | not yet (no confidence; see below) |

```python
from mozo.vendors.florence2_deploy import Florence2Predictor
from mozo.image import load_image

model = Florence2Predictor("torch-fp16.pth", "base")    # device="cpu", dtype=torch.float32
for box in model.detect(load_image("kitchen.jpg")):
    box.box, box.label, box.score     # xyxy in pixels, "human face", 0.49
```

`torch-fp16.pth` is Microsoft's `model.safetensors`, re-saved with `torch.save`: the same 666 (base)
or 918 (large) float16 tensors under the same names. Loading it in float32 is an exact upcast.

## What the score is

`<OD>`'s score is upstream's own (`SCORE_MODE: avg_loc_scores`): `exp` of the mean log-probability
of the box's four location tokens on the beam that wrote them. It says how sure the decoder was of
the coordinates once it had written the label. It is not a probability that the object is there,
and it runs low (0.1-0.6 on the example photograph).

## Why `read` is not served yet

`<OCR_WITH_REGION>` has no score at all upstream. PixelFlow's array entry point
(`pf.from_arrays`) requires one per detection, and mozo does not invent one; the family lands when
PixelFlow accepts `scores=None`. The decode is gated exactly like `detect`'s.

See `PROVENANCE.md` for what was taken, what was left behind, and where the chain stops.
