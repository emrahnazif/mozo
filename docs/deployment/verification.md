# How mozo proves it matches the original models

<!--
summary: What bit-identical means in mozo, how each model is checked against its original implementation, the few stated exceptions, and how to re-run a check.
label: Verification
keywords: mozo, verification, parity, bit-identical, reproducibility, provenance
-->

Mozo does not wrap the original packages; it carries a copy of each model's inference code. A copy
can quietly drift from the original, so every family is checked against the implementation it came
from, and the check uses **no tolerance**: two outputs pass only if they are exactly equal. A
tolerance would hide exactly the small drift the check exists to catch.

## What is compared

Not only the final answer. A gate compares every intermediate stage against the original, on real
inputs: the preprocessed tensor, each block of the network, the decoded output. A wrong stage is
then found where it goes wrong, not inferred from a final number that happens to look close.

| Family | Compared against | Comparisons |
|---|---|---|
| EasyOCR | the `easyocr` package | 1,275 across five variants, from the preprocessed tensor to each decoded string |
| OWLv2 | `transformers` | 226 on `base-ensemble`: tokenizer, preprocessing, both towers, three heads, boxes |
| Grounding DINO | the authors' implementation | 138 across both variants, from the tokenizer to the final logits and boxes |
| OmDet-Turbo | `transformers` | 877 on six images and five vocabularies: tokenizer, preprocessing, both text encodings, Swin, encoder, six decoder layers, detections |
| CLIP | `openai/CLIP` | 104 across four variants, from preprocessing to `logits_per_image` |
| Mayaku | `mayaku` 3.1.1 | every stage on the four street variants, 22 images at two thresholds: canvas, input tensor, each raw map, the decoded detections, boxes, scores, classes and masks |
| Granite Embedding R2 | `transformers` | 1,153,207: every stage of the model on 20 texts, and the tokenizer on 1.15 million texts, on Python 3.10 to 3.14 |

Sixteen of the nineteen families ship a gate in `tools/verify/` in mozo's repository. RF-DETR,
SAM 2 and ViTPose have their parity measured and recorded the same way, in the files below.

## The stated exceptions

Where a family is not bit-identical everywhere, its page says so and why:

- **ViTPose** joints land within 0.005 pixels of `transformers`. The heatmaps are bit-identical; the
  difference is PixelFlow rounding coordinates to 0.01.
- **EasyOCR** on a GPU: strings and boxes are exact, confidences move by up to 2.2e-05. The gate is
  a CPU claim; pass `device="cpu"` for exactly its numbers.
- **Grounding DINO** reports the phrase you asked for as `class_name`, where upstream decodes the
  matching tokens back into a string that can be cut short. That is the one deliberate difference
  in any family, and it is in a string, not in a number.
- **ONNX and CoreML graphs** are separate artifacts, checked against the torch model before they are
  published, within a stated bound: for RF-DETR, the same detections with boxes within 0.2 px. Each
  family's page states its bound. The torch path is the one compared bit for bit.

Being a copy rather than a wrapper costs nothing at run time: EasyOCR runs within 1% of the
published package on the same weights, and Granite Embedding R2 at the reference's speed.

## Where it is recorded

Every vendored model in `mozo/vendors/<family>_deploy/` has a `PROVENANCE.md`, which records:

- the exact upstream commit the code was extracted from, and what was changed or left out;
- the parity measured, and how;
- for the traps a code review would miss, what they were. CLIP's activation is QuickGELU, not the
  standard GELU, and substituting it moves the features by 0.4 without an error. Granite's
  checkpoint is stored in bfloat16 and loads that way by default, 3e-03 away from float32.

The weights are published with their licence and a NOTICE naming the upstream release they came
from, and every file is checked against a sha256 recorded in the package before mozo uses it.

## Run a check yourself

The gates need mozo's repository and the original package, not a network beyond the downloads:

```bash
git clone https://github.com/emrahnazif/mozo
cd mozo
pip install -e . transformers
python tools/verify/owlv2.py
```

Each gate prints its comparisons and ends with PASS, or fails if any is not identical. Some need a
checkout of the upstream repository instead; the top of each gate says what it needs.
