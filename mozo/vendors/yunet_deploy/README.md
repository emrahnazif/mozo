# yunet_deploy

Deployment-only YuNet face detection, extracted from the authors' PyTorch training repository
(ShiqiYu/libfacedetection.train, commit `dca340aa`) and reduced to the inference path in plain
torch. BSD-3-Clause code and weights (see `NOTICE`).

You give it an image and it finds the faces, each with five landmarks.

```python
from mozo.vendors.yunet_deploy import Predictor
from mozo.image import load_image

model = Predictor("torch-fp32.pth", "small")   # device="cpu"
faces = model.predict(load_image("team.jpg"), threshold=0.5)

faces.boxes.shape       # (N, 4) xyxy, image pixels, not clipped
faces.scores            # (N,) sigmoid(cls) * sigmoid(obj), best first
faces.keypoints.shape   # (N, 5, 2): right eye, left eye, nose tip, right and left mouth corner
```

## At the image's own size

Nothing is resized. The image is padded with zeros at the bottom and right to a multiple of 32,
flipped to BGR (what the network was trained on) and fed as 0-255 floats -- upstream's evaluation
in "origin" mode. The network is fully convolutional, so any size works, and the cost grows with
the pixel count. Faces from about 10 to 300 px are what it was trained to find; shrink an image
whose faces are larger.

## The network

Six depthwise-separable stages (2x2 max pools between them), a tiny top-down FPN over the stride
8/16/32 maps, and a decoupled head that predicts, per cell of each map, a class logit, an
objectness logit, a box (centre offset and log size, in strides) and five landmark offsets.
`small` (upstream's `yunet_s`) and `base` (`yunet_n`; the `n` is "normal") differ in the widths of
the first stages and in one shared convolution per head level.

## Decode

A cell's prior is its top-left corner, `(col, row) * stride`; a box's centre is
`offset * stride + corner`, its size `exp(log size) * stride`, each landmark
`offset * stride + corner`. The score is `sigmoid(cls) * sigmoid(obj)`; cells with a score of at
least `threshold` go through greedy NMS at IoU 0.45, and the survivors come back best first.
OpenCV's `FaceDetectorYN` reports the square root of the same score.

## Swapping the forward pass

`Predictor(None, variant, forward=fn)` replaces the network with `fn(batch)`, returning the twelve
raw maps flat (`cls`, `bbox`, `obj`, `kps`, each at strides 8, 16, 32), which is how an exported
graph runs. Padding and decode stay this package's either way.

See `PROVENANCE.md` for what was taken, what was left behind, and how it is checked.
