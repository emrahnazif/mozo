# mayaku_street_deploy

Deployment-only Mayaku street detection and instance segmentation, extracted from
[`mayaku`](https://github.com/emrahnazif/mayaku) 3.1.1. Apache-2.0. Depends on torch, torchvision
(for NMS), numpy and OpenCV; it does not import `mayaku`.

```python
from mozo.vendors.mayaku_street_deploy import Detector
from mozo.image import load_image

detector = Detector("mayaku-n-seg-street.pth", "seg-nano")      # device="cpu"
found = detector.predict(load_image("crossing.jpg"))             # conf defaults to 0.4

found.boxes        # (n, 4) float32 x1, y1, x2, y2 in the source image's pixels
found.scores       # (n,) float32
found.class_ids    # (n,) int64, into detector.names
found.names        # (n,) the class name of each detection
found.masks        # (n, h, w) bool for a seg- variant, else None
```

Most users reach this through `mozo.adapters.mayaku_street`, which resolves the weights, picks the
runtime and returns a PixelFlow result.

## How it works

The checkpoint is the publisher's own `.pth`: the trained network (QARepVGG blocks, a BatchNorm
after every convolution) and a JSON sidecar describing it. `Detector` builds the trained form,
loads it strictly (dropping only the int8 range buffers of quantisation-aware training, which the
fp32 graph never reads), checks the sidecar against the variant's written-down `Spec`, and fuses
it with upstream's own arithmetic into the deploy graph: five op types, Conv, Relu, Add, nearest
Resize and MaxPool.

`detect(image, forward, spec, conf)` does everything except the forward pass: the letterbox onto
the 800x800 canvas, the prediction decode (best class per location, a score threshold on the
logit, top 1000, class-wise NMS at IoU 0.5, at most 300 kept) and, for a seg variant, the
per-instance mask network (169 dynamic weights per detection, run over a stride-4 mask feature
inside the detection's box). The torch module, an ONNX session and a TensorRT engine all plug into
`forward`, so they share one pre- and post-processing path.

Variants: `nano` / `medium` (boxes), `seg-nano` / `seg-medium` (boxes and masks; the same boxes).

See `PROVENANCE.md` for what was taken from upstream, what was left behind and how the two are
held equal.
