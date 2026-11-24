# Torch, ONNX and CoreML runtimes

<!--
summary: Which runtimes each mozo model can run on, how runtime="auto" chooses for your device, and how to pick one yourself.
label: Runtimes
keywords: mozo, onnx, coreml, pytorch, runtime, apple silicon
-->

Every model runs on torch. Some families also publish ONNX or CoreML graphs of the same weights.
Mozo picks one for your device by itself; this page says what exists, how the choice is made, and
when to override it.

## What each family runs on

| Family | torch | ONNX | CoreML |
|---|---|---|---|
| RF-DETR | every variant | every variant except `keypoint-preview` | every variant except `keypoint-preview` |
| YOLOv8, YOLO12 | every variant | every variant | every variant |
| YOLO11, YOLO26 | every variant | every variant | none |
| ViTPose | every variant | every variant except `huge` | none |
| SAM 3 | yes | none | yes (the image encoder) |
| Granite Embedding R2 | yes | none | yes |
| BEN2, CLIP, Depth Anything V2, EasyOCR, EdgeTAM, Grounding DINO, Moebius, OWLv2, SAM 2, SigLIP 2 | yes | none | none |

Some of those families publish graphs mozo does not run yet, such as SAM 2's ONNX and CoreML and
Moebius's ONNX; asking for one says so instead of quietly running torch. Where a family has no graph,
it is usually a measured decision, and its page says why: a graph that was slower than torch, or
that could not take the input sizes the model needs.

## How `auto` chooses

`runtime="auto"`, the default, takes the first of these the variant publishes and the installed
packages can run:

| Device | Preference |
|---|---|
| Apple silicon (`mps`) | CoreML, then torch, then ONNX |
| CUDA | torch, then ONNX |
| CPU | torch, then ONNX |

CoreML needs `pip install 'mozo[coreml]'` and ONNX needs `pip install 'mozo[onnx]'`. Without them,
`auto` simply never picks those graphs. In practice: on Apple silicon with the CoreML extra, RF-DETR,
YOLOv8, YOLO12, SAM 3 and Granite Embedding R2 run as CoreML graphs, and everything else runs on
torch.

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

## A checkpoint of your own

A checkpoint you trained always runs on torch. Graphs are built from mozo's published weights, so a
checkpoint of your own has none.
