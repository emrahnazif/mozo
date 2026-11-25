# deim_deploy

Inference-only DEIM-D-FINE (DEIM, CVPR 2025): real-time DETR detection of COCO's 80 classes, in
one package that depends on **`torch`, `torchvision`, `numpy` and `pillow`** and nothing else, and
imports nothing from mozo.

```python
from deim_deploy import load, postprocess, preprocess

model = load("small", "deim_dfine_hgnetv2_s_coco_120e.pth")   # strict load, then deploy()
batch, size = preprocess(rgb)                                  # (1,3,640,640), (1,2) as (w, h)
labels, boxes, scores = postprocess(model(batch), size)        # top 300, xyxy pixels, by score
keep = scores[0] > 0.4
```

## Models

| Name | Backbone | Upstream checkpoint | Notes |
|---|---|---|---|
| `nano` | HGNetv2-B0 | `deim_dfine_hgnetv2_n_coco_160e.pth` | two feature levels, 128 wide |
| `small` | HGNetv2-B0 | `deim_dfine_hgnetv2_s_coco_120e.pth` | |
| `medium` | HGNetv2-B2 | `deim_dfine_hgnetv2_m_coco_90e.pth` | |
| `large` | HGNetv2-B4 | `deim_dfine_hgnetv2_l_coco_50e.pth` | |
| `xlarge` | HGNetv2-B5 | `deim_dfine_hgnetv2_x_coco_50e.pth` | |
| `xlarge-obj365` | HGNetv2-B5 | `deim_dfine_hgnetv2_x_obj2coco_24e.pth` | Objects365 pretraining; frozen BN, ReLU decoder |

All run at 640x640 (a squash, not a letterbox). Geometry is in `config.py`.

## Files

| File | What |
|---|---|
| `hgnetv2.py` | backbone, and `FrozenBatchNorm2d` |
| `hybrid_encoder.py` | hybrid encoder: one transformer layer on the coarsest level, then FPN/PAN |
| `decoder.py` | D-FINE decoder: deformable attention, fine-grained distribution refinement |
| `utils.py` | deformable attention core, weighting function, box conversion |
| `model.py` | assembly, strict loader, `deploy()` |
| `preprocess.py`, `postprocess.py` | upstream's inference transform and deploy-mode decode |

See `PROVENANCE.md` for what was taken, what changed, and how it is checked.
