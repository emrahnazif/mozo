# Torch, ONNX, CoreML and TensorRT runtimes

<!--
summary: Which runtimes each mozo model can run on, how runtime="auto" chooses for your device, half precision and TensorRT on CUDA, and how to pick one yourself.
label: Runtimes
keywords: mozo, onnx, coreml, tensorrt, fp16, pytorch, runtime, apple silicon, cuda
-->

Every model runs on torch. Some families also publish ONNX or CoreML graphs of the same weights,
and some run in half precision or through TensorRT on CUDA. Mozo picks one for your device by
itself; this page says what exists, how the choice is made, and when to override it.

## What each family runs on

| Family | torch | ONNX | CoreML |
|---|---|---|---|
| RF-DETR | every variant | every variant except `keypoint-preview` | every variant except `keypoint-preview` |
| YOLOv8, YOLO12 | every variant | every variant | every variant |
| YOLO11, YOLO26 | every variant | every variant | none |
| ViTPose | every variant | every variant except `huge` | none |
| SAM 3 | yes | none | yes (the image encoder) |
| Granite Embedding R2 | yes | none | yes |
| OWLv2 | yes, and fp16 on CUDA | every variant (the image tower in fp16, to build TensorRT engines from) | none |
| Grounding DINO | yes, and fp16 on CUDA | none (exported on your machine, to build TensorRT engines from) | none |
| BEN2, CLIP, Depth Anything V2, EasyOCR, EdgeTAM, Moebius, SAM 2, SigLIP 2 | yes | none | none |

Some of those families publish graphs mozo does not run yet, such as SAM 2's ONNX and CoreML and
Moebius's ONNX, or runs only through TensorRT, such as OWLv2's; asking for one says so instead of
quietly running torch. Where a family has no graph,
it is usually a measured decision, and its page says why: a graph that was slower than torch, or
that could not take the input sizes the model needs.

## How `auto` chooses

`runtime="auto"`, the default, takes the first of these the variant publishes and the installed
packages can run:

| Device | Preference |
|---|---|
| Apple silicon (`mps`) | CoreML, then torch, then ONNX |
| CUDA | TensorRT fp32 where the family offers it and TensorRT is installed, then torch fp16 where offered, then torch, then ONNX |
| CPU | torch, then ONNX |

CoreML needs `pip install 'mozo[coreml]'` and ONNX needs `pip install 'mozo[onnx]'`. Without them,
`auto` simply never picks those graphs. In practice: on Apple silicon with the CoreML extra, RF-DETR,
YOLOv8, YOLO12, SAM 3 and Granite Embedding R2 run as CoreML graphs, and everything else runs on
torch. On CUDA, RF-DETR runs as `tensorrt-fp32` when TensorRT 11 or later is installed, OWLv2 and
Grounding DINO run as `torch-fp16`, and everything else runs on torch in fp32.

## Choose one yourself

Ask a `ModelManager` for a runtime by name:

```python
from mozo import ModelManager

models = ModelManager()
model = models.get_model("rfdetr", "small", runtime="onnx-fp32")
model.runtime        # 'onnx-fp32'
```

A runtime the variant does not publish, or one its family cannot run, raises an error listing what
is available. `mozo.get_model` has no `runtime` argument; it always uses `auto`.

Reasons to override `auto`:

- **Serve without torch's overhead on a CPU machine.** On some models ONNX Runtime is faster on a
  CPU; on others it is slower. [ViTPose](/docs/models/vitpose.md)'s graphs, for example, are slower
  than torch on CPU. Measure on your machine.
- **Run on CUDA with ONNX Runtime.** Install `onnxruntime-gpu` instead of the `onnx` extra; mozo
  asks ONNX Runtime for its CUDA provider on a `cuda` device.
- **Compare runtimes.** Every graph is checked against the torch model before it is published, and
  each family's page states how closely they agree.

ONNX Runtime's CoreML provider is never used: on RF-DETR it placed fewer than half the graph's nodes
and failed at run time. On Apple silicon, mozo reaches the accelerators through real CoreML graphs
instead.

## Half precision and TensorRT on CUDA

Three runtimes exist only on CUDA, and only for families where they were measured. None has a file
of its own to download: each is built on your machine from the weights a variant already publishes.

| Runtime | What it is | Chosen by `auto` |
|---|---|---|
| `tensorrt-fp32` | a TensorRT engine built on your machine, in full fp32 | yes, where offered and TensorRT is installed |
| `torch-fp16` | the torch checkpoint, with the heavy layers run in half precision | yes, where offered |
| `tensorrt-fp16` | a TensorRT engine built on your machine, in half precision | no, ask for it |

[RF-DETR](/docs/models/rfdetr.md#on-cuda-tensorrt) offers `tensorrt-fp32` on every variant that
publishes a graph and `tensorrt-fp16` on its four detection variants. Its fp32 engine changed no
detection on 500 COCO images and runs 1.7 to 2.6 times faster than torch, because RF-DETR is
limited by kernel launches at one image at a time; its fp16 engine is about 6 times faster and
changes 0.2–0.5% of detections.

[OWLv2](/docs/models/owlv2.md#on-cuda-half-precision-and-tensorrt) and
[Grounding DINO](/docs/models/grounding-dino.md#on-cuda-half-precision-and-tensorrt) offer both;
their pages have the measurements. In short, on an RTX 3090, against fp32 on 500 COCO images:
`torch-fp16` is 4.3 to 4.8 times faster on OWLv2 and 1.45 to 1.55 times on Grounding DINO (torch
2.13), with the same AP, and changes 0.1–1.1% of detections across a threshold, a little more than
fp32 itself does when only its input is rounded.
`tensorrt-fp16` is faster again, but `auto` never picks it, for a reason that differs by family.
On OWLv2 it is 7.2 to 8.2 times faster with the same AP, but its fp16 arithmetic is less exact
than torch's, so 2.5–4.8% of confident detections come out differently than in fp32. On Grounding
DINO it is 2.6 to 3.0 times faster and changes as few detections as `torch-fp16`, but it runs every
image at one square size, 800 unless you pick another with `input_size`, which costs 1.6 to 1.8 AP
at 800.
Half precision changes numbers a little on every model, and on detection models such as RF-DETR
and YOLO it was measured to lose detections, so it is offered model by model, not everywhere.

To go back to exactly the fp32 results, ask for `runtime="torch-fp32"`.

### TensorRT

TensorRT is not installed with mozo, because NVIDIA ships one wheel per CUDA version and it has to
match your torch's. Install the one for your torch's CUDA major (`python -c "import torch;
print(torch.version.cuda)"`):

```bash
pip install tensorrt-cu12     # or tensorrt-cu13
```

Mozo needs **TensorRT 11 or later** and refuses older ones. TensorRT 10 miscompiles Grounding
DINO's position encoding, and its fp16 was the builder's choice rather than the graph's. TensorRT
11 has wheels for CUDA 12 and 13 only, so a torch built for CUDA 11 cannot use it. Grounding DINO
also needs `pip install onnx onnxscript`, to export its graph on your machine.

Engines are built strongly typed: an fp16 engine runs exactly the types its graph carries, which
are `torch-fp16`'s. fp32 stays IEEE fp32, without TensorRT's default TF32 shortcut.

Then ask for it by name:

```python
model = ModelManager().get_model("owlv2", "base-ensemble", runtime="tensorrt-fp16")
```

The first load on a machine builds the engine and keeps it in mozo's cache. That takes 12 s for an
RF-DETR fp32 engine (about 45 s in fp16), 17 s for OWLv2 `base` and 26 s for `large` on an RTX
3090, and about 3.5 minutes for Grounding DINO, which exports its graph first. Every later load reuses it and takes about a second. An engine only fits the GPU model and the TensorRT version it
was built with, so a different GPU or a TensorRT upgrade builds a new one. To build it ahead of
time, load the model once when you build the machine or the image, as
[Production](/docs/deployment/production.md#load-models-before-the-first-request) suggests for
downloads. `weights.remove` deletes engines with the model, and `weights.cached()` lists them as
`"built"`.

## A checkpoint of your own

A checkpoint you trained always runs on torch in fp32. Graphs are built from mozo's published
weights, so a checkpoint of your own has none.
