# Install mozo on CPU, CUDA or Apple silicon

<!--
summary: Install mozo with pip, add the optional ONNX or CoreML runtimes, and run it on a CUDA GPU or Apple silicon.
label: Installation
keywords: mozo, install, pip, cuda, apple silicon, onnx, coreml
-->

Mozo is one package on PyPI. It needs Python 3.10 or newer, and installs no model packages: every
model's inference code is inside mozo.

```bash
pip install mozo
```

That installs twelve dependencies: torch, torchvision, numpy, OpenCV (headless), Pillow,
[PixelFlow](https://emrah.fr/pixelflow/), FastAPI, uvicorn, python-multipart, click, regex and
ftfy. Check it worked:

```bash
mozo version
```

```text
Mozo version 1.10.0
```

## Weights arrive on first use

Installing downloads no weights. The first time you load a model, mozo downloads its weights,
checks them against the sha256 recorded in the package, and caches them in `~/.cache/mozo`. The
next load reads the cache. A first load can take minutes for a multi-gigabyte model; every load
after that takes seconds.

To download ahead of time, for example while building a container image:

```python
from mozo import weights

weights.fetch("rfdetr", "medium", device="cuda")
```

Pass the device the model will run on: it decides which runtime a load picks, and so which files it
needs.

To keep the cache somewhere else, set `MOZO_CACHE`. To refuse downloads entirely, set
`MOZO_OFFLINE=1`. [Configuration](/docs/deployment/configuration.md) has both, and how to feed an
offline machine from a mirror.

## Optional runtimes

Every model runs on torch. Some families also publish ONNX or CoreML graphs, which run without
torch's overhead on some devices. The extras install what those need:

```bash
pip install 'mozo[onnx]'     # onnxruntime
pip install 'mozo[coreml]'   # coremltools, macOS only
```

No model needs an extra. Without one, mozo simply never picks those graphs. With one installed,
`runtime="auto"` may pick them where they are the faster choice: CoreML for RF-DETR, YOLOv8, YOLO12
and SAM 3 on Apple silicon. [Runtimes](/docs/deployment/runtimes.md) lists which family publishes
what.

## NVIDIA GPUs

Mozo uses whichever torch is installed. On Linux, the default `pip install torch` already includes
CUDA. Where it does not, install the CUDA build of torch first, from
[pytorch.org](https://pytorch.org/get-started/locally/), then mozo:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install mozo
```

Mozo picks the best device it finds: `cuda`, then `mps`, then `cpu`. Ask for one yourself with
`device=`:

```python
import mozo

model = mozo.get_model("rfdetr/medium", device="cpu")
```

For ONNX on a GPU, install `onnxruntime-gpu` instead of the `onnx` extra.

## Apple silicon

Torch's `mps` device runs on Apple silicon with no extra install. A few operations torch has not
implemented for `mps` yet raise an error unless they are allowed to fall back to the CPU.
`mozo start` allows it by itself. From Python, set it before torch is imported:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Install the `coreml` extra for the families that publish CoreML graphs. On Apple silicon those are
the fastest way to run them, and `auto` picks them by itself.

## From source

```bash
git clone https://github.com/emrahnazif/mozo
cd mozo
pip install -e .
```

The workflow editor's built files are committed, so you need npm only to change the editor, never to
install mozo.
