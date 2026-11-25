# ppocrv6_deploy

PaddlePaddle's PP-OCRv6 inference path, extracted for deployment. Two graphs, no PaddleOCR, no
Paddle, no pyclipper.

```python
from mozo.vendors.ppocrv6_deploy import Reader, SPECS

reader = Reader("torch-fp32.pth", SPECS["medium"])
for region in reader(rgb_array):
    print(region.text, region.confidence, region.quad)
```

Most callers want `mozo.adapters.ppocrv6.PPOCRv6Predictor` instead, which resolves the weights
and returns PixelFlow detections.

## How it works

| | |
|---|---|
| `lcnetv4.py` | PP-LCNetV4, the backbone under all six networks. |
| `det.py` | The detectors: backbone, a feature pyramid (two kinds), DB's probability head. |
| `rec.py` | The recognisers: backbone pooled to one row, then an SVTR encoder or 1-D convolutions, then CTC. |
| `image.py` | The page into the detector; each quad cropped upright; each line into the recogniser. |
| `boxes.py` | Probability map to quads (DB), the Clipper offset that grows them, the reading order. |
| `text.py` | The alphabet and the CTC collapse. Alphabets in `assets/`. |
| `predictor.py` | The seam: `find` the quads, `read` them six lines at a time, shortest first. |
| `config.py` | Each variant's geometry, copied from PaddlePaddle's `config.json`. |
| `checkpoint.py` | The fused checkpoint into both networks, strictly. |

Everything between the two networks is OpenCV and numpy, in BGR, as upstream works.

## Variants

| Variant | Detector | Recogniser | Alphabet |
|---|---|---|---|
| `medium` | 22.0M parameters, large-kernel PAN | 19.1M, SVTR | 18,708 (about 50 languages) |
| `small` | 2.5M, residual-SE neck | 5.3M, SVTR | 18,708 |
| `tiny` | 0.4M, residual-SE neck | 1.1M, conv CTC head | 6,904 (no Japanese kana) |

See `PROVENANCE.md` for the reference, what is verified and how, and what each runtime measured.
