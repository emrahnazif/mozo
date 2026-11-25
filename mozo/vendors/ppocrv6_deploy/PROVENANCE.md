# Provenance

## Where this came from

| | |
|---|---|
| Upstream pipeline | PaddleOCR 3.7.0 (`https://github.com/PaddlePaddle/PaddleOCR`, tag `v3.7.0` = `b03f4642`), on PaddleX 3.7.2 |
| Upstream networks | transformers 5.20.2: `pp_lcnet_v4`, `pp_ocrv6_medium_det`, `pp_ocrv6_small_det`, `pp_ocrv6_small_rec`, `pp_ocrv6_tiny_rec` -- PaddlePaddle's own PyTorch implementation |
| Code licence | Apache-2.0, all three |
| Weights | `PaddlePaddle/PP-OCRv6_<size>_<det\|rec>_safetensors` on the Hugging Face Hub, at the commits in `tools/fetch/ppocrv6.py`; each card `license: apache-2.0`, ungated |
| Paper | Zhang et al., *PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks*, arXiv:2606.13108 |
| Extracted | 2026-11-25 |

## The reference, and why it is two things

PaddlePaddle publishes every PP-OCRv6 network three ways: a Paddle inference graph, an ONNX export
of it, and a `model.safetensors` for its PyTorch implementation, which PaddleOCR runs with
`engine="transformers"`. PaddleOCR's default engine is Paddle, and its published numbers come from
its Paddle pipeline.

Neither PaddleOCR engine can be the reference as it ships:

- `engine="paddle"` computes the networks with Paddle's arithmetic. At the same weights and the
  same input its probability maps differ from torch's by ~1e-4 (table below). No torch code can be
  bit-identical to it.
- `engine="transformers"` is torch end to end, but its pre- and post-processing are transformers'
  image processors, *not* the Paddle pipeline's code: torchvision resizing, an unclip re-derived
  without pyclipper (different integer boxes), and every line in a batch stretched to the widest
  one's width, where the Paddle pipeline pads each line at its own aspect. On the fixtures that
  changes strings (`'E'` against `'_'`) and confidences (0.96 against 0.99995). On ICDAR 2015
  (medium, 500 images, word recall as defined in `tools/bench/ppocrv6.py`): 0.650 for
  `engine="transformers"`, 0.667 for this package.

So the reference is **PaddleOCR 3.7.0's own pipeline, with each network forward handed to
PaddlePaddle's PyTorch implementation on PaddlePaddle's safetensors.** Every stage is upstream
code: the networks are transformers' modules, everything around them is PaddleX's. The gate drives
`PaddleOCR.predict` itself, with the Paddle runner replaced.

**The guarantee:** on the CPU, this package's quadrilaterals, strings and confidences are identical
to that reference. **Where the chain stops:** Paddle's own arithmetic (measured below), and the tiny
recogniser, whose safetensors checkpoint is not the snapshot in PaddlePaddle's Paddle and ONNX
files (below).

## The pipeline configuration

PaddleOCR's `OCR.yaml` with the three extra networks off, which is what every PP-OCRv6 model card's
Python example does (`use_doc_orientation_classify=False, use_doc_unwarping=False,
use_textline_orientation=False`):

| | |
|---|---|
| page resize | short side up to at least 64, long side down to at most 4000, both rounded to 32 |
| DB | binarise at 0.3, keep boxes scoring 0.6, grow by 1.5, up to 3000 contours (inference.yml) |
| recogniser | height 48, each line at its own aspect, width at least 320 and at most 3200; six lines a batch, shortest first, padded to the widest |
| kept | every line, including empty reads (score threshold 0) |

PaddleOCR's pipeline defaults turn the three extra networks *on* (document orientation, UVDoc
unwarping, text-line orientation). They are separate model families with their own weights, and
unwarping returns boxes in the unwarped image's frame. Not included.

## Verified

`tools/verify/ppocrv6.py`, exact, no tolerance. Three paths: this package, the reference above, and
mozo's adapter through PixelFlow. Stages: the page tensor and the resize record, the probability
map, every DB candidate's unclip through both pyclipper and this package, the quads and their
scores, the sorted order, every crop, every recogniser batch's probabilities, and each line's quad,
string and confidence from `PaddleOCR.predict`. GATE_SUMMARY

The networks were also gated alone against transformers at two shapes each, random inputs:
identical, all six, on the first attempt.

## What changed, and why

Each change was verified at zero delta before it was kept.

1. **One checkpoint per variant.** `tools/fetch/ppocrv6.py` puts the detector's and recogniser's
   state dicts in one file; the tensors are PaddlePaddle's, byte for byte, and load strictly with
   no renaming -- the module names are transformers'.
2. **No transformers machinery.** Configs are frozen dataclasses (`config.py`) copied from the
   published `config.json`; the attention calls `scaled_dot_product_attention` directly, which is
   what transformers' interface dispatches to with PaddleOCR's settings (sdpa, verified).
3. **No pyclipper.** DB grows each box with Clipper's round-join offset. `boxes._offset` and
   `boxes._union` reproduce Clipper 6.4.2's integer output for the convex quadrilaterals DB hands
   it, including the vertex order of the union's output, which `cv2.minAreaRect` is sensitive to
   (see Measured traps). Checked against pyclipper on every candidate of every gated page and on
   FUZZ_COUNT random boxes. Clipper is Boost-1.0; its terms are in NOTICE.
4. **No shapely.** Upstream's unclip measures area and perimeter with OpenCV, not shapely, so
   shapely never arises.
5. **BGR once.** mozo hands RGB; the reader converts once and works in BGR thereafter, as the
   pipeline does from its decoder on.
6. **The pipeline is a function, not PaddleX's predictor framework**: no batch sampler, no result
   classes, no benchmark decorators, no model downloads. One page at a time, as PaddleOCR's own
   pipeline runs (`batch_size: 1`).

## The Paddle files, measured

Same preprocessed input, CPU, network output against transformers on the safetensors:

| network | Paddle (static) | Paddle (dygraph, safetensors) | PaddlePaddle's ONNX |
|---|---|---|---|
| medium det | 6.0e-04 | 7.0e-04 | 1.5e-04 |
| small det | 9.6e-04 | | |
| tiny det | 8.7e-03 | 9.3e-03 | 9.7e-04 |
| medium rec | 1.4e-04 | | |
| small rec | 3.6e-05 | 1.7e-05 | 3.9e-05 |
| tiny rec | **0.15** | 1.9e-05 | **0.15** |

For five networks the Paddle and ONNX files hold the same tensors as the safetensors -- every
unfused, untransposed initialiser in the ONNX is bit-equal to a safetensors tensor -- and the
differences are arithmetic. **The tiny recogniser is a different training snapshot**: none of its
89 ONNX initialisers equals any safetensors tensor (its first convolution's first weight is
-0.0542 in one and -0.0599 in the other), and the Paddle graph agrees with the ONNX. This package
serves the safetensors snapshot, the one PaddleOCR serves with `engine="transformers"`.

PADDLE_E2E

## Runtimes

RUNTIMES

## Measured traps

1. **`cv2.minAreaRect` depends on where the polygon starts.** The same points rotated by one
   vertex can come back as the same rectangle described from another corner, 90 degrees apart, and
   DB's box then differs. Clipper's union drops repeated and collinear vertices, keeps the
   orientation, and ends at the lowest-y vertex with the largest x. Of 20,000 random boxes every
   hull agreed and 308 rectangles still differed until that order was reproduced.
2. **pyclipper truncates float coordinates toward zero** before offsetting.
3. **`max_candidates` is 3000**, from each detector's inference.yml; DBPostProcess's default of
   1000 is not what the pipeline uses. The thresholds, on the other hand, are `OCR.yaml`'s
   (0.3 / 0.6 / 1.5), not inference.yml's (0.2 / 0.45 / 1.4).
4. **The normalisation applies ImageNet's RGB-ordered mean to a BGR image**: blue gets 0.485. It is
   a multiply-add per plane (`x * (1/255/std) + (-mean/std)`), which rounds differently from
   `(x/255 - mean)/std`.
5. **Batching is part of the arithmetic.** Lines are sorted by aspect and read six at a time, each
   batch zero-padded to its widest line.
6. **CTC confidence is a float32 numpy mean** of the kept steps' maxima; an empty read is 0.0 and
   is kept.
7. **A crop 1.5 times taller than wide is turned a quarter** before reading.
8. **torch on CUDA uses TF32 for convolutions by default** (`torch.backends.cudnn.allow_tf32`),
   which mozo does not change; see Runtimes.

## Not included

Training. The document orientation classifier, UVDoc and the text-line orientation classifier.
Seal text and polygon boxes (`text_type="seal"`). Word boxes (`return_word_box`). PP-OCRv5 and
older. Arabic display reordering (`python-bidi`), which applies to v5's Arabic recogniser only.
