# EasyOCR: read the text in an image

<!--
summary: EasyOCR in mozo: find every line of text in an image and read it, five scripts covering dozens of languages. Apache-2.0, torch.
label: EasyOCR
keywords: mozo, easyocr, ocr, text recognition, text detection
-->

EasyOCR finds every line of text in an image and reads it. Each line comes back as a detection
carrying the string it says, its confidence, and the four corners of the text as found, so rotated
text keeps its orientation. A variant is a script, not a language: `latin` alone covers 41
languages.

## Quick start

```python
import mozo

model = mozo.get_model("easyocr/english")
found = model.predict("example.jpg")

for line in found:
    print(repr(line.text), round(line.confidence, 2), line.bbox)
```

```text
'8' 0.82 [879.0, 25.0, 955.0, 139.0]
```

The example photograph has almost no text in it: EasyOCR finds one line, a digit at the top of the
frame. On a page, a sign or a screenshot you get one detection per line.

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/easyocr/english" -F "file=@example.jpg"
```

```json
[
  {"bbox": [879.0, 25.0, 955.0, 139.0],
   "segments": [[879.0, 25.0], [955.0, 25.0], [955.0, 139.0], [879.0, 139.0]],
   "text": "8", "confidence": 0.816, "class_id": null, "class_name": null, "...": "..."}
]
```

## Variants

| ID | Script | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `easyocr/english` | English | 98 MB | torch | Apache-2.0 |
| `easyocr/latin` | Latin, 41 languages | 99 MB | torch | Apache-2.0 |
| `easyocr/chinese-simplified` | Simplified Chinese | 105 MB | torch | Apache-2.0 |
| `easyocr/japanese` | Japanese | 100 MB | torch | Apache-2.0 |
| `easyocr/korean` | Korean | 99 MB | torch | Apache-2.0 |

These five are 88% of upstream's own downloads, out of the seventeen recognisers it publishes.
Torch is the only runtime: [Why no ONNX or CoreML](#why-no-onnx-or-coreml) says why.

## Inputs and outputs

```python
model.predict(image)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB `uint8` array. |

Returns a PixelFlow `Detections`, one per line of text. Each detection has:

- `text`: the string read. A line that was found but read as empty is kept, with `text` set to `""`.
- `confidence`: the recogniser's confidence, between 0 and 1.
- `segments`: the four corners as found, for rotated text.
- `bbox`: the axis-aligned box around those corners.
- `class_id` and `class_name`: always `None`.

Over HTTP there are no parameters beyond the file.

## How it behaves

### A variant is a script, not a language

`latin` reads every character its character set holds, in all 41 of its languages: ask it for
`café` and you get `café`. Upstream instead picks a checkpoint from a list of languages, then
suppresses characters outside those languages while decoding. Its output therefore depends on
something that is not a property of the weights. Mozo's variants are the weights, and nothing is
suppressed.

### Text, not a class name

Every other detection family names a class from a fixed vocabulary. OCR produces content that
belongs to no vocabulary, and PixelFlow keeps the two apart: the string is in `text`, and
`class_id` and `class_name` are `None`.

### Lines are not in reading order

Level lines come back top to bottom, followed by tilted ones. That is upstream's ordering, and mozo
keeps it, but it is not a reading order: a two-column page comes back interleaved. If you need
reading order, sort the detections yourself.

### Two models, no suppression

CRAFT finds the text and a CRNN reads it. There is no non-maximum suppression: a detection is a
connected region of two heatmaps, one scoring "inside a character" and one "between two characters
of the same word". A page takes about 200 ms on CPU and 31 ms on Apple silicon, within 1% and 22%
of the published `easyocr` package running the same weights on the same device.

### The GPU path is not bit-identical to the CPU one

Strings and corners are exact on every device, but confidences on a GPU move by up to 2.2e-05.
Mozo's parity check is a CPU claim: pass `device="cpu"` if you need exactly those numbers.

```python
model = mozo.get_model("easyocr/english", device="cpu")
```

### Why no ONNX or CoreML

The input shape of both networks depends on the image. The detector sees the page scaled to fit
2,560 pixels and padded to a multiple of 32, so its shape follows the photograph. The recogniser's
width is `ceil(aspect) × 64` per line, and padding a line to a fixed width adds decode steps, which
can change what it reads.

With flexible shapes neither network converts. At fixed shapes both do, and CoreML was the fastest
runtime measured, 13.8 ms against torch on MPS's 18.8 for the detector. But the recogniser then
differs by 1.6e-05, above the 1.4e-05 already known to flip a character, and fixing shapes for the
detector alone would mean 6,241 combinations.

## Weights and licence

Code and weights are Apache-2.0, from JaidedAI's [EasyOCR](https://github.com/JaidedAI/EasyOCR)
repository at commit `363afb18` (version 1.7.2, the last release to PyPI). Mozo's extraction was
checked against the published `easyocr` package at every stage, from the preprocessed image to the
decoded string, with no tolerance: 1,275 comparisons across the five variants, every one identical.
Every variant ships with its licence and a NOTICE naming its source; the extraction is recorded in
`mozo/vendors/easyocr_deploy/PROVENANCE.md`.
