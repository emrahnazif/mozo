# rtmo_deploy

Deployment-only RTMO multi-person pose estimation, extracted from MMPose v1.3.2 and reduced to the
inference path in plain torch: no `mmcv`, `mmengine`, `mmdet` or `mmpose`. Apache-2.0 code; the
weights are taken to be Apache-2.0 too (see `NOTICE`).

You give it a frame and it finds the people and their joints, in one pass.

```python
from mozo.vendors.rtmo_deploy import Predictor
from mozo.image import load_image

model = Predictor("torch-fp32.pth", "small")   # device="cpu"
people = model.predict(load_image("street.jpg"), threshold=0.5)

people.boxes.shape       # (N, 4) xyxy, frame pixels
people.scores            # (N,) person scores, best first
people.keypoints.shape   # (N, 17, 2) x, y in frame pixels
people.visibility.shape  # (N, 17) per-joint visibility
```

## One stage, no detector

Unlike a top-down model such as ViTPose, RTMO is not told where people are. Every cell of two grids
(strides 16 and 32) proposes a person: a score, a box, seventeen visibilities and a pose vector. The
cells above `threshold` are sorted, NMS at IoU 0.65 removes duplicates, and only the survivors' pose
vectors go through the DCC, which turns each into joint positions inside its box. So the cost of the
joints grows with the number of people found, not with the image.

## Input

RGB `uint8`, any size. It is letterboxed — centred, grey 114 bars — into a 640×640 square (416 for
`tiny`) by an affine warp, exactly as MMPose's `BottomupResize`, and flipped to BGR, which is what
the network was trained on. Nothing is normalised: the network takes 0–255.

## Threshold

`threshold` is the person score a cell must exceed before NMS. MMPose evaluates at 0.1. Filtering
before NMS is not an approximation: NMS keeps boxes greedily by descending score, so a box below the
threshold can never suppress one above it, and raising the threshold only removes rows from the
bottom of the answer. The rows that stay keep their boxes and scores exactly; their joints can move
in the last float bits, because the DCC runs as one batch over the survivors and a matrix product
over twelve rows does not round like one over four. MMPose does the same.

## Visibility

Each joint's third number is the sigmoid of the visibility branch: the model's estimate that the
joint can be seen. Its position is predicted regardless, so a hidden joint still has coordinates.
Filter on visibility before drawing.

## Swapping the forward pass

`Predictor(..., forward=fn)` replaces the dense network (everything before the score filter) with
`fn(batch) -> (scores, boxes, visibility, pose)`, which is how an exported graph runs. The
letterbox, filter, NMS, DCC and the map back to the frame stay this package's either way; the DCC
still loads from the checkpoint.

See `PROVENANCE.md` for what was taken, what was left behind, and how it is checked.
