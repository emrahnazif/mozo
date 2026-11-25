# sam_hq_deploy

Deployment-only HQ-SAM promptable segmentation for single images. Apache-2.0 code, extracted from
`SysCV/sam-hq` and reduced to the image path, with no `torchvision` or `timm` dependency.

```python
from mozo.vendors.sam_hq_deploy import Segmenter
from mozo.image import load_image

segmenter = Segmenter("sam_hq_vit_b.pth")     # device="cpu"; the variant is read off the file
image = load_image("photo.jpg")

found = segmenter.predict(image, points=[[820, 640]], labels=[1])
found = segmenter.predict(image, boxes=[40, 60, 300, 480], multimask_output=False)

found.masks     # (b, c, h, w) bool, in the source image's pixels
found.scores    # (b, c) the SAM token's predicted IoU for each mask
found.logits    # (b, c, 256, 256) low-res logits, to refine with as mask_input
```

## Prompting

A click is a point with a label, `1` to include and `0` to exclude; `labels` is required with
`points` and has no default. A box is `x1, y1, x2, y2` and needs no labels: SAM has a box input of
its own, so unlike SAM 2 a box is not folded into the click list. Clicks and a box may be given
together, one point set per box.

## What comes back

Every mask is a SAM output token's mask plus the HQ output token's -- upstream's default
(`hq_token_only=False`), and what its evaluations use.

- `multimask_output=False`: one mask per prompt, from SAM's single-mask token. Use it for a box,
  which is not ambiguous.
- `multimask_output=True`: three masks per prompt, one per SAM multimask token, each with the HQ
  correction. **Upstream returns only the best of these three**; this package returns all three
  so that it answers in the shape every promptable family in mozo does. The one with the highest
  score is exactly upstream's answer.

The score is SAM's predicted IoU for the token's own mask. HQ-SAM trains no score for the
corrected mask, so it ranks candidates rather than rating the final mask.

## Variants

| Variant | Trunk | Upstream file |
|---|---|---|
| `tiny` | MobileSAM's TinyViT ("Light HQ-SAM") | `sam_hq_vit_tiny.pth` |
| `base` | SAM ViT-B | `sam_hq_vit_b.pth` |
| `large` | SAM ViT-L | `sam_hq_vit_l.pth` |
| `huge` | SAM ViT-H | `sam_hq_vit_h.pth` |

See `PROVENANCE.md` for what was extracted, what was left behind, and how it is verified.
