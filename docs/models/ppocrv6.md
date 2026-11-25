# PP-OCRv6: read text in about 50 languages

<!--
summary: PP-OCRv6 in mozo: find and read every line of text, about 50 languages including Chinese, Japanese and Korean, in three sizes. Apache-2.0.
label: PP-OCRv6
keywords: mozo, pp-ocrv6, paddleocr, ocr, text recognition, text detection, chinese, japanese, korean, multilingual
-->

PP-OCRv6 is PaddlePaddle's OCR system, released with PaddleOCR 3.7.0 in June 2026. It finds every
line of text in an image and reads it. Each line comes back as a detection carrying the string,
a confidence and the four corners of the line, so rotated text keeps its orientation. One
recogniser reads about 50 languages at once — Chinese, Japanese, Korean, English and other Latin
scripts, Cyrillic, Greek, Arabic and more — so a variant is a size, not a language.

## Quick start

```python
import mozo

model = mozo.get_model("ppocrv6/medium")
found = model.predict("sign.png")

for line in found:
    print(repr(line.text), round(line.confidence, 2), line.bbox)
```

```text
'EXIT 42' 1.0 [38.0, 54.0, 245.0, 106.0]
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/ppocrv6/medium" -F "file=@sign.png"
```

```json
[
  {"bbox": [38.0, 54.0, 245.0, 106.0],
   "segments": [[38.0, 54.0], [245.0, 54.0], [245.0, 106.0], [38.0, 106.0]],
   "text": "EXIT 42", "confidence": 0.999, "class_id": null, "class_name": null, "...": "..."}
]
```

## Variants

| ID | Detector + recogniser | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|
| `ppocrv6/medium` | 22.0M + 19.1M parameters | 165 MB | torch, onnx; tensorrt-fp16 on CUDA | Apache-2.0 |
| `ppocrv6/small` | 2.5M + 5.3M | 31 MB | torch, onnx | Apache-2.0 |
| `ppocrv6/tiny` | 0.4M + 1.1M | 6 MB | torch, onnx | Apache-2.0 |

`medium` is PaddleOCR's own default. `tiny` reads 6,904 characters and no Japanese kana; the
other two read 18,708.

## Inputs and outputs

```python
model.predict(image)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB `uint8` array. |

Returns a PixelFlow `Detections`, one per line of text. Each detection has:

- `text`: the string read. A line that was found but read as empty is kept, with `text` set to `""`
  and confidence 0.
- `confidence`: the recogniser's confidence, the mean probability of the characters it kept,
  between 0 and 1.
- `segments`: the four corners, clockwise from the top-left.
- `bbox`: the axis-aligned box around those corners.
- `class_id` and `class_name`: always `None`.

Over HTTP there are no parameters beyond the file.

## How it behaves

### PaddleOCR's pipeline, with its extra models off

PaddleOCR can run three more networks around these two: a document orientation classifier, a
document unwarper and a text-line orientation classifier. Mozo runs none of them, which is the
configuration every PP-OCRv6 model card's Python example uses. Upside-down text is therefore read
upside down, and a photographed page is read as photographed, not flattened first.

### Lines in reading order, roughly

Lines come back top to bottom, and left to right among lines whose top-left corners are within 10
pixels of each other in height. That is PaddleOCR's ordering. A two-column page comes back
interleaved.

### A small page is scaled up, a large one is not scaled down

The detector sees the page at its own size, rounded to a multiple of 32, scaled up only when the
short side is under 64 pixels and down only when the long side is over 4,000. Small text on a
large page is read at full resolution; a 4K photograph costs what 4K costs.

## Speed and accuracy

Measured on the 500 test images of ICDAR 2015 (street scenes), one RTX 3090, median per page.
"Graphs" is the two networks; "page" includes the OpenCV work around them. Word recall is the share
of the ground-truth words found inside a line and read exactly; PP-OCRv6 reads lines and the ground
truth is words, so compare these numbers with each other, not with leaderboards.

| | tiny | small | medium |
|---|---|---|---|
| torch, CUDA: graphs / page | 15.4 / 31.9 ms | 18.6 / 34.6 ms | 37.7 / 54.6 ms |
| tensorrt-fp16, CUDA: graphs / page | | | 18.7 / 33.6 ms |
| word recall, torch | 0.449 | 0.591 | 0.667 |
| word recall, tensorrt-fp16 | | | 0.666 |

`medium` is PaddleOCR's default and the most accurate; `small` reads 89% as many words with a fifth
of its parameters.

### On the CPU, use ONNX for speed

```python
model = mozo.get_model("ppocrv6/small", device="cpu", runtime="onnx-fp32")
```

ONNX Runtime runs the detector 1.4 to 2 times faster than torch on the CPU (the recogniser about
the same), and reads the same text: on 100 images, 99.5% of lines identical and the rest moved by a
pixel or one character. `auto` stays on torch, the path the parity check covers.

### On CUDA, torch is not bit-identical to the CPU

torch on CUDA computes convolutions in TF32 by default. It changes 1 to 2.5% of the lines read
compared to the CPU (accuracy is the same within noise). Pass `device="cpu"` for exactly the
verified numbers.

### TensorRT fp16, `medium` only, pages up to 2048 pixels

```python
model = mozo.get_model("ppocrv6/medium", runtime="tensorrt-fp16")
```

Needs TensorRT 11 or later; the first load builds an engine and keeps it. Accuracy does not move,
but about 10% of lines come out slightly differently from fp32. A page that reaches the detector
larger than 2048 pixels a side is refused with an error naming the runtimes that take it. `small`
in fp16 was slower than in fp32, and TensorRT builds a `tiny` fp16 detector that finds nothing, so
neither is offered. An fp32 TensorRT engine reads exactly what the CPU reads but is only 8% faster
for `medium`, so it is not offered either.

## Weights and licence

Code and weights are Apache-2.0. The weights are PaddlePaddle's own PyTorch checkpoints
(`PaddlePaddle/PP-OCRv6_<size>_<det|rec>_safetensors` on the Hugging Face Hub), the tensors
unmodified. Mozo's extraction was checked against PaddleOCR 3.7.0's own pipeline, with the networks
run by PaddlePaddle's PyTorch implementation, at every stage from the page tensor through each box,
crop and recogniser batch to the decoded string, with no tolerance: 1,976 comparisons across the
three sizes, every one identical. PaddleOCR's own Paddle engine computes the same networks with Paddle's
arithmetic: on ICDAR 2015 it reads 99.95% of lines exactly as mozo does, with the same accuracy; `tiny`'s recogniser is a different
training snapshot from the one in PaddlePaddle's Paddle and ONNX files. Both are measured in
`mozo/vendors/ppocrv6_deploy/PROVENANCE.md`.
