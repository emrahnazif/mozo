# eomt_deploy

Inference-only EoMT: semantic and panoptic segmentation from a plain DINOv2 ViT with learned
queries in its last blocks. Depends on `torch`, `numpy` and `Pillow` and nothing else.

```python
from mozo.vendors.eomt_deploy import SPECS, Segmenter

segmenter = Segmenter("torch-fp32.pth", SPECS["coco-panoptic-small"])
found = segmenter.segment(rgb)        # an HxWx3 uint8 RGB array
found.masks      # (N, H, W) bool, never overlapping
found.class_ids  # (N,) int64, COCO panoptic's 133 classes in upstream's contiguous order
found.scores     # (N,) float32 class probabilities; None for a semantic variant
```

| Variant | Task | Input | Queries | Backbone |
|---|---|---|---|---|
| `coco-panoptic-small` | panoptic, 133 classes | 640 | 200 | ViT-S/16 + 4 registers |
| `coco-panoptic-base` | panoptic, 133 classes | 640 | 200 | ViT-B/16 + 4 registers |
| `coco-panoptic-large` | panoptic, 133 classes | 640 | 200 | ViT-L/16 + 4 registers |
| `coco-panoptic-large-1280` | panoptic, 133 classes | 1280 | 200 | ViT-L/16 + 4 registers |
| `ade20k-semantic-large` | semantic, 150 classes | 512 windows | 100 | ViT-L/16 + 4 registers |

Bit-identical to `tue-mps/eomt` at `aa017c3` on CUDA, in fp32 and fp16 autocast, and on the CPU: `tools/verify/eomt.py`. What was taken, what
was left and where it diverges: `PROVENANCE.md`.
