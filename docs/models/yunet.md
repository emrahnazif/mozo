# YuNet: fast faces, with five landmarks

<!--
summary: YuNet in mozo: every face with its eyes, nose tip and mouth corners, from a network under 80k parameters. BSD-3-Clause, exact to the authors' code.
label: YuNet
keywords: mozo, yunet, face detection, facial landmarks, opencv, FaceDetectorYN, face blur
-->

YuNet is a tiny face detector by Wei Wu, Hanyang Peng and Shiqi Yu, the model behind OpenCV's
`FaceDetectorYN`. Give it an image and it returns every face it finds, with a box, a score and five
landmarks: both eyes, the nose tip and both corners of the mouth. The network has under 80,000
parameters and runs at the image's own size, so it is fast enough to run on every frame of a video
call and finds faces a dozen pixels wide in a crowd. Use it for selfies and video, for blurring the
faces in any picture, or for finding faces to align before recognition.
Detectors trained on road scenes find faces too, among everything else in the street; reach for
YuNet when the faces are the point.

## Quick start

```python
import mozo

faces = mozo.get_model("yunet/small").predict("example.jpg")

eye = faces[0].keypoints[0]
print(len(faces), faces[0].class_name, round(faces[0].confidence, 2), eye.name, eye.x, eye.y)
```

```text
4 face 0.88 right_eye 855.19 364.69
```

Over HTTP, with `mozo start` running:

```bash
curl -X POST "http://localhost:8000/predict/yunet/small?threshold=0.5" -F "file=@example.jpg"
```

Blurring every face is a two-node workflow: the `yunet` node into `blur_regions`. The same
detections work with any PixelFlow annotator.

## Variants

| ID | Upstream name | Parameters | WIDER Face val AP (easy / medium / hard) | Weights (torch) | Runtimes | Licence |
|---|---|---|---|---|---|---|
| `yunet/small` | `yunet_s`, OpenCV Zoo's `face_detection_yunet_2023mar` | 56k | 0.887 / 0.871 / 0.768 | 0.3 MB | torch, onnx; tensorrt on CUDA | BSD-3-Clause |
| `yunet/base` | `yunet_n` | 78k | 0.892 / 0.883 / 0.811 | 0.4 MB | torch, onnx; tensorrt on CUDA | BSD-3-Clause |

AP measured by mozo at the image's own size, threshold 0.02, with the authors' evaluation code.
`small` is the network OpenCV ships; `base` is a little larger and better on small faces. Upstream
calls it `yunet_n`, for "normal": it is not a nano model.

`runtime="auto"`, the default, picks torch on CPU and on CUDA without TensorRT, and an fp32
TensorRT engine built from the published ONNX graph on CUDA with TensorRT 11 or later installed
(`pip install tensorrt-cu12` or `-cu13`). On the CPU, `runtime="onnx-fp32"` is the fastest; see
[Runtimes](#runtimes).

## Inputs and outputs

```python
model.predict(image, threshold=0.5, labels=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str`, `bytes` or `numpy.ndarray` | | A file path, encoded image bytes, or an `HxWx3` RGB array. |
| `threshold` | `float` | `0.5` | The score a face must reach. The authors evaluate at 0.02. |
| `labels` | `list` | `None` | Names for this call: a list of class names, or the published `labels.json` shape to rename the landmarks too. |

Returns a PixelFlow `Detections`, best first: one row per face, `class_id` 0 named `face`, the box
in your image's pixels, the score, and five landmarks in this order: `right_eye`, `left_eye`,
`nose_tip`, `right_mouth_corner`, `left_mouth_corner`. "Right" is the subject's, so on a face
looking at the camera the right eye is the one on the left of the picture. Each landmark has `x`,
`y`, an `id` and a `name`, and its `confidence` is `None`: the model places landmarks, it does not
score them. PixelFlow's annotators draw a landmark with no confidence.

An image with no face in it is answered with an empty `Detections`, not an error.

## How it behaves

### It runs at your image's size

Nothing is resized. The image is padded to a multiple of 32 and the whole of it goes through the
network, so the time grows with the pixel count: on an RTX 3090, a 640×480 frame costs about the
same as 720p, and 4K a few times more. The model was trained on faces from about 10 to 300 pixels
across. **A face much larger than that is missed**: a close-up selfie from a phone camera, where
the face fills the frame, needs shrinking first, for example with PixelFlow's resize, and the boxes
scaling back.

### The score, and OpenCV's

The score is the class probability times the objectness, as the authors compute it. OpenCV's
`FaceDetectorYN` reports the square root of the same number, so OpenCV's default threshold of 0.9 is
0.81 here, and mozo's 0.5 is OpenCV's 0.71.

### The threshold is applied before NMS, and that is exact

Faces scoring at least `threshold` go through NMS at IoU 0.45. NMS keeps boxes greedily from the
best down, so a face below the threshold can never remove one above it: `threshold=0.5` returns
exactly the faces `threshold=0.02` returns above 0.5, with the same boxes and landmarks.

### Boxes are not clipped

A face cut by the edge of the picture gets a box that runs past it, as the authors' code returns it.
Clip it yourself if a consumer needs boxes inside the image.

### Runtimes

The ONNX graph covers the whole network; padding, decode and NMS stay in mozo's code on every
runtime. Measured on an RTX 3090 with TensorRT 11.5 against torch fp32 on the CPU, on WIDER Face's
3,226 validation images at threshold 0.02:

| Runtime | `small` AP (easy / medium / hard) | `base` AP | boxes, worst |
|---|---|---|---|
| torch, CPU | 0.8871 / 0.8710 / 0.7681 | 0.8923 / 0.8832 / 0.8107 | |
| TensorRT fp32 | identical | identical | 0.0005 px |
| torch, CUDA | 0.8871 / 0.8710 / 0.7681 | 0.8923 / 0.8832 / 0.8107 | 4.2 px (TF32) |
| TensorRT fp16 | 0.8871 / 0.8710 / 0.7681 | 0.8924 / 0.8833 / 0.8107 | 22 px |
| OpenCV `FaceDetectorYN`, zoo ONNX | 0.8869 / 0.8707 / 0.7681 | | |

A whole `predict` on the fixture photograph resized to each size, median, faces at 0.5:

| Variant | Image | torch CUDA | TensorRT fp32 | TensorRT fp16 |
|---|---|---|---|---|
| `small` | 640×480 | 4.5 ms | 2.5 ms | 3.2 ms |
| | 1280×720 | 4.7 ms | 2.8 ms | 3.5 ms |
| | 1920×1080 | 6.0 ms | 3.6 ms | 4.2 ms |
| | 3840×2160 | 19.6 ms | 9.7 ms | 9.5 ms |
| `base` | 640×480 | 4.8 ms | 2.5 ms | 3.2 ms |
| | 1280×720 | 5.1 ms | 3.0 ms | 3.4 ms |
| | 1920×1080 | 6.7 ms | 3.9 ms | 4.1 ms |
| | 3840×2160 | 22.1 ms | 10.7 ms | 10.1 ms |

The TensorRT engine is fp32 throughout, with TF32 off, and lands within 0.0005 px of the CPU. torch
on CUDA runs its convolutions in TF32 by default, which moves boxes by up to 4 px; the AP does not
notice. The engine is built on the first load for each GPU and TensorRT version, takes about 20
seconds, and reserves about 625 MB of GPU memory, for images up to 4096 pixels on a side; a larger
image runs through torch on the same GPU instead, from the same 0.3 MB checkpoint.

**No fp16.** YuNet is so small that its time goes to launching kernels, not to arithmetic: an fp16
engine is slower than fp32 for a whole call up to 1080p and 2 to 6% faster at 4K, and it changes
about 60 of 21,000 faces at threshold 0.5 and moves boxes by up to 22 px. It is not offered.

**On the CPU, ONNX Runtime is 2.5 to 4.5 times faster than torch** for a whole call (measured while
other jobs loaded the machine, so indicative). `auto` stays on torch on the CPU, as it does for every
family; pass `runtime="onnx-fp32"` with `pip install 'mozo[onnx]'` to take it. ONNX Runtime on CUDA
was slower than torch on CUDA and is not worth asking for.

### Parity

Bit-identical to the authors' PyTorch (`ShiqiYu/libfacedetection.train` at commit `dca340aa`) on
CPU: the padded input, the output of every one of the network's 124 (`small`) and 143 (`base`)
modules, and the final boxes, scores and landmarks, compared with no tolerance on 36 images per
variant at thresholds 0.02 and 0.5, and mozo's result through PixelFlow: 20,160 comparisons.
`tools/verify/yunet.py` runs the comparison against a checkout of that repository.

## Weights and licence

The code and the checkpoints come from the authors' training repository, which is BSD-3-Clause and
states no separate terms for its checkpoints; the NOTICE published beside each says so, and that
the repository was Apache-2.0 when they were committed and that OpenCV Zoo distributes `small` under
MIT. They were trained on WIDER FACE, whose own terms (research use) govern the dataset. Mozo
publishes each checkpoint's weights alone, without the optimiser state and training metadata the
original file carries; no tensor is changed. The extraction is recorded in
`mozo/vendors/yunet_deploy/PROVENANCE.md`.
