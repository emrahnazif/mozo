# dinov2_deploy

Deployment-only DINOv2. Apache-2.0 code *and* weights, extracted from
`facebookresearch/dinov2` and reduced to the backbone's inference path, with no `xformers`,
`omegaconf`, `fvcore` or `dinov2` package at inference.

An image in, one vector out, and images that look alike land close together. Image only: there is
no text tower, so these vectors do not share a space with CLIP's or SigLIP 2's.

```python
from mozo.vendors.dinov2_deploy import Encoder, SPECS

encoder = Encoder("torch-fp32.pth", SPECS["base"])           # device="cpu"
encoder.encode_image([rgb_a, rgb_b])
# (2, 768) float32, L2-normalised: a dot product is the cosine
```

Six variants: `small`, `base`, `large`, `giant` (384, 768, 1024 and 1536-d) and `small-reg`,
`base-reg`, the same backbones trained with four register tokens. See `PROVENANCE.md` for why
`large-reg` and `giant-reg` are absent.

## What the vector is

The final-norm class token, L2-normalised: what upstream's own k-NN evaluation embeds. The image is
preprocessed with upstream's evaluation transform -- short side to 256 (bicubic), the central
224x224, ImageNet statistics -- so only the middle of a long image is seen.

## What it guarantees

- **Bit-identical to the authors' torch-hub code** at commit `7764ea0`, on the CPU, at every stage,
  with `torch.equal` and no tolerance. `tools/verify/dinov2.py` is the gate.
- **The same image gets the same vector, whatever it was encoded beside.** One image per forward.
- **The same vector whether or not `xformers` is installed.** Upstream switches kernels on it;
  this package has no such path.
