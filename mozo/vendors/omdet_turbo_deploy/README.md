# omdet_turbo_deploy

Deployment-only OmDet-Turbo: real-time open-vocabulary detection. Apache-2.0 code and weights,
extracted from `transformers/models/omdet_turbo` (and the timm Swin it runs) and reduced to the
inference path, with no `transformers`, `timm` or `tokenizers` dependency.

You name things in words and it returns boxes for them.

```python
from mozo.vendors.omdet_turbo_deploy import SPECS, Detector
from mozo.image import load_image

detector = Detector("torch-fp32.pth", SPECS["tiny"])     # device="cpu"
image = load_image("kitchen.jpg")

boxes, scores, labels = detector(image, ["kettle", "a mug", "window"])
# boxes  (n, 4) xyxy in the source image's pixels, best first
# scores (n,)   sigmoid of a scaled cosine between a query and one prompt
# labels (n,)   which prompt, indexing the list you passed
```

## The two halves

The prompts are read twice: each one becomes a class embedding that queries are scored against,
and all of them together become a task sentence, `"Detect kettle, a mug, window."`, which the
decoder's queries attend to. Both depend on the words alone, so `encode_text` runs once per
vocabulary and is cached; every later image pays only for the image path (Swin-T, the hybrid
encoder, the decoder). The image path takes the text features as inputs, which is why a graph
runtime can replace all of it (`forward=`, `export.py`).

| | |
|---|---|
| Prompt length | up to 75 tokens each; longer is refused. |
| Task sentence | truncated at 77 tokens, as upstream does: about twenty COCO-length names. Every prompt is still scored. |
| Case | ignored, as the reference's tokenizer lowercases. |
| Image | squashed to 640x640 with PIL's bilinear filter; nothing else runs. |

**Prompts do not compete.** Each (query, prompt) pair is a candidate, so one box can come back
under two prompts. Overlapping boxes under one prompt are suppressed at IoU 0.5, as upstream does.

## What this is not

No training path, no batching, no custom task sentence. `PROVENANCE.md` says what was left
behind, how the extraction was checked, and where the chain to the authors' code stops.
