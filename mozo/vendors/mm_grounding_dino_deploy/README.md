# mm_grounding_dino_deploy

Deployment-only MM-Grounding-DINO open-vocabulary detection, running LLMDet's checkpoints.
Apache-2.0 code *and* weights, extracted from `transformers`' port and reduced to the detection
path — with no `transformers`, `tokenizers` or `safetensors` dependency, and no network round trip
to tokenize a prompt.

You describe a thing in words and it returns boxes for it. There is no class list and no
fine-tuning.

```python
from mozo.vendors.mm_grounding_dino_deploy import Predictor, SPECS
from mozo.image import load_image

model = Predictor("torch-fp32.pth", SPECS["llmdet-tiny"])   # device="cpu"
image = load_image("kitchen.jpg")

found = model(image, ["a kettle", "the mug on the left", "a window"])

found[0].box            # (x1, y1, x2, y2) in the source image's pixels
found[0].score          # the query's best per-token probability
found[0].prompt_index   # which prompt that was, indexing the list you passed
```

## What it is

MM-Grounding-DINO is OpenMMLab's open re-training of IDEA Research's Grounding DINO, with a
better-scaled contrastive head and a separate box head per decoder layer. LLMDet (iSEE Laboratory,
CVPR 2025) fine-tunes it with a large language model writing captions for every training image,
and is the stronger detector at every size on the authors' zero-shot LVIS benchmark.

| variant | backbone | LVIS minival AP (authors, zero-shot) | weights |
|---|---|---|---|
| `llmdet-tiny` | Swin-T | 44.7 | 692 MB |
| `llmdet-base` | Swin-B | 48.3 | 936 MB |
| `llmdet-large` | Swin-L | 51.1 | 1376 MB |

## Prompting

The same contract as `grounding_dino_deploy`, because the two answer one task: a prompt is a
description, read with the image rather than embedded alone; prompts may not contain `.` or `?`
(they separate concepts, so a prompt carrying one is refused); case is not preserved; about 60
prompts fit the 256-token budget, and this raises past it where upstream truncates silently.

## Shapes

By default nothing is letterboxed: the short side goes to 800 and the long side is capped at 1333,
aspect kept, nothing padded — `transformers`' `GroundingDinoImageProcessor`, whose rounding differs
from Grounding DINO's own on panoramas (167×1333, not 167×1336). For TensorRT the image is fitted
into a square instead (`fit_square`) and the padding is handed to the model as a mask.

## Fidelity

Bit-identical to `transformers` 4.57.3 (and 4.55.0) on CPU, every stage from pixels and token ids
to the final logits, boxes and post-processed detections, for all three checkpoints. `transformers`
5.x is not a usable reference: it loads these files with every decoder layer running the first
layer's box head. Run the gate with

```bash
MKL_CBWR=AVX2,STRICT python tools/verify/mm_grounding_dino.py llmdet-tiny llmdet-base llmdet-large --reference
```

See `PROVENANCE.md` for what was taken, what was left, where this diverges, and the traps.
