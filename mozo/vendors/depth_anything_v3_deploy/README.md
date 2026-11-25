# depth_anything_v3_deploy

Depth Anything 3, reduced to single-image depth.

```python
from mozo.vendors.depth_anything_v3_deploy import Predictor

predictor = Predictor.from_pretrained("small", weights="torch-fp32.pth", device="cuda")
depth = predictor.predict(rgb)       # H'xW' torch tensor: depth, larger is farther
```

Three variants, the Apache-2.0 checkpoints that predict depth from one image without a camera:

```
small        DA3-SMALL       any-view model, run on one view
base         DA3-BASE        any-view model, run on one view
mono-large   DA3MONO-LARGE   monocular model, with a sky head
```

The depth comes back at the processed resolution (504 px on the long side, each side a multiple
of 14), as upstream returns it, on a scale of the model's choosing. `resize` takes it to any size.

A graph exported from `DepthAnything3Net` takes the pixels and `shape_inputs(...)` -- the tensors
the network derives from the shape alone, which cannot be traced -- see `graph.py`.

Depends on `torch`, `torchvision`, `numpy` and `cv2`. Every import inside the package is
relative. See `PROVENANCE.md` for what was extracted, what changed, and how it is verified.
