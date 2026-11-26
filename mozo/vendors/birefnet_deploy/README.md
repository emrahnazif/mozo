# birefnet_deploy

Deployment-only BiRefNet for single images. MIT code, MIT weights, extracted from the authors'
Hugging Face remote code (`birefnet.py`, 2,250 lines, one file per repository), with no
`transformers`, `timm`, `einops` or `kornia` dependency at runtime.

> **Verified bit-identical to the published model** at every stage -- the preprocessed tensor,
> every backbone block of both passes, every decoder block and deformable convolution, the logits,
> the alpha and both cut-outs -- on CUDA in float32 and under float16 autocast, and on CPU in
> float32. `PROVENANCE.md` says where the guarantee stops.

```python
from birefnet_deploy import Predictor

predictor = Predictor.from_pretrained("general", "torch-fp32.pth")
alpha  = predictor.matte(rgb)                  # (H, W)    uint8
cutout = predictor.cutout(rgb, refine=True)    # (H, W, 4) uint8 RGBA
```

## The model in one paragraph

The image is squashed to a square (1024, or 2048 for `hr`) and run through a Swin v1 backbone
**twice**, at full and at half resolution; each level's two feature maps are concatenated, and the
three shallower levels are folded into the deepest. A squeeze block and four decoder stages follow.
Every decoder stage sees the image again -- cut into a grid of patches stacked as channels, 32x32 at
stride 32 down to the image itself at full resolution -- and every one runs a deformable ASPP
(modulated deformable convolution, `torchvision.ops.deform_conv2d`) and a gradient-attention gate.
One 1x1 convolution makes the logits; the sigmoid is the matte.

## Variants

| variant | backbone | resolution | trained for |
|---|---|---|---|
| `general` | Swin-L | 1024 | general background removal (segmentation) |
| `lite` | Swin-T | 1024 | the same, about three times faster |
| `matting` | Swin-L | 1024 | soft alpha, trimap-free matting |
| `hr` | Swin-L | 2048 | general use at high resolution |

## Exporting

`torchvision::deform_conv2d` has no ONNX symbolic, ONNX Runtime has no `DeformConv` kernel, and
TensorRT runs one only through a plugin. `graph.decomposed()` swaps in a decomposition built from
`grid_sample` and 1x1 convolutions for the duration of an export. It is not bit-identical to
torchvision (6e-5 on the matte), so the torch runtime never uses it.

## Dependencies

`torch`, `torchvision` (`ops.deform_conv2d` only), `numpy`, `cv2`, `Pillow`. Pillow is
load-bearing: the resize in is PIL's antialiasing BILINEAR and the resize out PIL's BICUBIC on the
8-bit matte, as upstream does both. `cv2` is the box blur of the foreground refinement.
