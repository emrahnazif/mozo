# Provenance

This package is a deployment-only extraction of [DEIM](https://github.com/ShihuaHuang95/DEIM)
(DEIM-D-FINE: D-FINE's architecture trained with DEIM's matching).

| | |
|---|---|
| Upstream repository | `https://github.com/ShihuaHuang95/DEIM` |
| Upstream commit | `09d35d53d39ee3145a1e61e3a989b28b9468d1dd` (2026-03-24) |
| Model code last changed | `c7ed52d` (2025-03-12); later commits touch README and LICENSE only |
| Extracted | 2026-11-25 |
| Verified against | CUDA: `torch` 2.13.0+cu126, CUDA 12.6, cuDNN 9.10.2, `torchvision` 0.28.0, Pillow 13.0.0, numpy 2.5.4, RTX 3060 (2026-11-26); CPU: `torch` 2.13.0, `torchvision` 0.28.0, Pillow 12.3.0, numpy 2.3.5, Python 3.12 |

## The reference

Upstream is not on PyPI; the reference is the repository at the commit above, run as
`tools/inference/torch_inf.py` runs it: `YAMLConfig(config)` with `HGNetv2.pretrained` off,
`load_state_dict(checkpoint["model"])`, `model.deploy()`, `postprocessor.deploy()`, and the
transform `T.Compose([T.Resize((640, 640)), T.ToTensor()])` on a PIL image. The COCO evaluator's
transform is the same arithmetic. `tools/inference/onnx_inf.py` letterboxes instead, and is not
the reference.

**Guarantee:** on CUDA in fp32 (DEIM's only torch precision) and on the CPU, for the same RGB pixels, the vendor's network input, every module's
output, the raw logits and boxes and all 300 decoded detections are bit-identical to upstream's,
for all six checkpoints. The chain stops there: the AP numbers in upstream's README are the
authors'; mozo has not reproduced their full COCO evaluation, only measured its runtimes against
each other on a 500-image subset.

## What was taken

| Upstream | Here | Changes |
|---|---|---|
| `engine/backbone/hgnetv2.py`, `FrozenBatchNorm2d` from `engine/backbone/common.py` | `hgnetv2.py` | pretrained download, freezing and initialisers removed; arch configs B1/B3/B6 dropped; `HG_Block` reduced to the `agg='se'` branch every config takes |
| `engine/deim/hybrid_encoder.py` | `hybrid_encoder.py` | the `version='dfine'` branch only; post-norm only; position embedding always the precomputed eval one |
| `engine/deim/dfine_decoder.py` | `decoder.py` | denoising, aux outputs, wide layers (`layer_scale>1`), `discrete` attention, `one2many`/`agnostic` selection, initialisers removed; the decoder returns the last layer's boxes and logits unstacked |
| `engine/deim/utils.py`, `dfine_utils.py`, `box_ops.py` | `utils.py` | the default deformable-attention path and the deploy-form weighting function only |
| `engine/deim/postprocessor.py` | `postprocess.py` | the deploy-mode focal-loss branch, as a function |
| `engine/deim/deim.py` | `model.py` | plus a strict, dtype-checked loader |
| `tools/inference/torch_inf.py` (transform) | `preprocess.py` | on an RGB array, through PIL as upstream |
| the configs' resolved values | `config.py` | written down as frozen `Spec`s |

Module and attribute names are upstream's, so the checkpoints load strictly and the gate compares
module outputs by name.

## The checkpoints

The authors' Google Drive files, linked from the README at the commit above, published by mozo
byte for byte as `torch-fp32.pth`. `{"model": state_dict}`, loadable with `weights_only=True`.

- `decoder.denoising_class_embed.weight` (contrastive denoising's class embedding, training only)
  is the one key with no module here. Dropped by name; any other unexpected key fails the load.
- `decoder.up` / `decoder.reg_scale` are registered twice upstream (also as
  `decoder.decoder.up` / `decoder.decoder.reg_scale`), and are here too, so both names load.
- `reg_scale` is **int64** (the YAML says `4` or `8`). `load_state_dict` casts dtypes silently, so
  the loader checks them: a spec written `reg_scale=4.0` is refused rather than computed in float.
- `decoder.anchors` / `decoder.valid_mask` are buffers in the checkpoint and are loaded from it.
  The encoder's position embedding is not in the checkpoint: computed at build, as upstream.

## Licence

Code: Apache-2.0, upstream's `LICENSE` (copied here), with a preamble naming Intellindust. DEIM's
ancestors D-FINE, RT-DETR and PaddleDetection (HGNetv2) are Apache-2.0; their notices are kept in
the file headers and in `NOTICE`.

Weights: the repository states no separate terms, and there are no model cards. That the weights
are Apache-2.0 is an inference from the repository's licence (as for CLIP), and the NOTICE published
beside each checkpoint says it in those words. OpenVINO's Hugging Face conversions of L/M/X are
tagged `apache-2.0`, a redistributor's claim. `xlarge-obj365` was pretrained on Objects365.

## Where this diverges from the obvious reading, and why

Reproduced from upstream, recorded here so nobody "fixes" them:

- **The real aggregation is `se`, not `ese`.** `HG_Block` defaults to `agg='ese'`, but `HG_Stage`
  passes its own default `'se'` and `HGNetv2` never overrides it.
- **`deim.yml` overrides the L and X D-FINE configs' `freeze_norm: True` back to `False`**, and the
  Objects365 config sets it `True` again. `FrozenBatchNorm2d` (`x * scale + shift` with the
  statistics folded) is different arithmetic from eval-mode `BatchNorm2d`.
- **The deploy-form weighting function is built in doubles**, then cast once; the training form
  stays in float32. They differ in the last bits; upstream's inference runs the deploy form.
- **`output + output_detach` starts from the integer 0**, and `norm3` sees
  `target.clamp(-65504, 65504)`: both kept.
- **Upstream calls `dec_score_head[0]` (an `nn.Identity` after deploy) for a pre-score nothing
  reads.** Not called here; no arithmetic.
- **Class ids are contiguous 0-79.** Upstream remaps to COCO's sparse category ids only inside its
  evaluator (`remap_mscoco_category`); the deploy post-processor returns 0-79, and so does this.
- **One query may yield several detections** of different classes: top-k runs over every
  (query, class) pair. Upstream's decode, kept.

## Verification

`tools/verify/deim.py --upstream <clone>` runs each variant in a process of its own (upstream's
`YAMLConfig` leaks one config's keys into the next built in the same process), with upstream, this
package and mozo side by side in it, on one device (`--device cuda` or the CPU), and compares, on
the fixture photograph, eight synthetic cases (blank, black, tiny, both extreme aspect ratios,
exactly 640, an upscale, noise) and 16 random COCO val2017 images:

- the network input, the encoder outputs, raw logits and boxes, and the 300 decoded detections,
  with `torch.equal`;
- every module's output, in call order, by the sha256 of its bytes;
- mozo's served result against the vendor's at thresholds 0.4 and 0.05: exactly, on the arrays
  mozo's adapter hands to PixelFlow, and on the rows PixelFlow returns after its rounding.

On the CPU, every process runs with `MKL_CBWR=AVX2,STRICT`. Without it, CPU results are not reproducible from
one process to the next: MKL chooses its sgemm kernel by buffer alignment, and the same model on
the same pixels gave a different encoder attention output in about one process in ten (measured
on medium on an idle machine; logits up to about 5e-4 apart). That is a property of torch on this
CPU, not of either implementation, and it affects mozo's CPU results the same way.

**Result (2026-11-25):** all six variants, 25 images each, every comparison equal: input, encoder
outputs, logits, boxes, 300 detections, and 120,625 module outputs (456 to 1,145 module calls per
image per variant), plus mozo at both thresholds. The gate is falsifiable, and was made to fail
13 ways on `nano` and `xlarge-obj365`, each at its own stage: PIL bicubic instead of bilinear and
`x * (1/255)` instead of `x / 255` (input), the encoder's GELU as tanh (first module: the encoder
activation), decoder LayerNorm eps 1e-6 (first module: `decoder.enc_output.norm`; the encoder
still equal), the training-form weighting function (first module: the second layer's query
position head; boxes and logits fail, the encoder does not), box scale with x and y swapped
(detections only), and, on `xlarge-obj365` only, `FrozenBatchNorm2d` computed as eval
`F.batch_norm` (first module: `backbone.stem.stem1.bn`).

**On CUDA (2026-11-26), the primary gate:** RTX 3060 (vision-003), driver 550.144.03, torch
2.13.0+cu126, CUDA 12.6, cuDNN 9.10.2, fp32. That is DEIM's only torch precision: its `auto` on
CUDA is a TensorRT engine (measured by parity and accuracy below), mozo publishes no torch half
runtime, and upstream's inference script has no half path. The settings are torch's defaults, which
mozo leaves alone: cuDNN may use TF32 for convolutions; matmul TF32, `cudnn.benchmark` and
deterministic algorithms off. Upstream runs as `torch_inf.py` does with `-d cuda`: the batch made on
the host and moved to the device, the network and the post-processor on the device, with autograd
on (the script never turns it off). The attention modules take torch's unfused `need_weights=True`
path on both sides; no `scaled_dot_product_attention` call is made (counted), so there is no SDPA
backend to pin. Each side ran every image twice and repeated itself bit for bit (150 images, each
side). Then the same 25 images and stages as on the CPU, every one exact on all six variants:
124,225 module outputs, plus per image the input, encoder outputs, logits, boxes, 300 detections
and mozo at both thresholds.

One fix to mozo's adapter came out of it, none to this package: the adapter copied the raw logits
and boxes to the host and decoded there, while upstream decodes on the device. CUDA's `sigmoid`
differs from the CPU's in the last bit on about a quarter of the logits (up to 6e-08), so the 300
scores differed on every image, and on one the top-k order changed. The adapter now decodes where
the network ran. The check of mozo against PixelFlow's rounded rows passed before the fix; the
exact check on the arrays handed to PixelFlow was added because of it.

**GPU falsifiers** (`--device cuda --falsify all`, against one cached reference run per variant,
after an unperturbed run passed against the same cache): 67 of 67 caught on all six variants. The
seven above, each first at the same stage as on the CPU, and five that perturb only what a GPU
does, on the vendor's side alone: cuDNN TF32 off (first at the first convolution cuDNN runs in
TF32, e.g. `backbone.stem.stem2a.conv` on `nano`), matmul TF32 on (first at
`encoder.encoder.0.layers.0.self_attn`: the backbone is all convolutions), the batch in
channels-last memory with equal values (first at `backbone.stem.stem1.conv`: cuDNN picks other
kernels), attention through `scaled_dot_product_attention` (`need_weights=False`; first at the
encoder's self-attention), and the decode on the host (detections only). The cache keeps the input
and encoder outputs as their dtype, shape and sha256, module outputs as digests, and the raw
outputs and 300 detections whole.

The CPU gate (all six variants, 25 images) and its 37 falsifiers (all six variants, 11 images)
were rerun after the change, on vision-003's CPU, 2026-11-26: all exact, all caught.

## Export

The graph is the deploy-form network alone (`tools/export/deim.py`): `1x3x640x640`, opset 17,
outputs `pred_logits` and `pred_boxes`; preprocessing and decode stay in this package on every
runtime. Measured on an RTX 3090, 500 COCO val2017 images per variant, against torch fp32 on the
CPU: ONNX Runtime CPU and CUDA and the TensorRT fp32 engine changed no detection above 0.5 and left
AP identical on all six. The TensorRT fp32 engine runs 1.9x (`xlarge`) to 5.4x (`nano`) faster than
torch CUDA, and `auto` takes it. The fp16 graph (exported under torch autocast, opset 18) changes at
most 4 detections in about 3,000 and AP by at most 0.12, and is 1.2-2.1x faster than the fp32
engine for `small` and up; for `nano` it is slower than fp32 (2.4 against 1.9 ms), so `nano`
publishes none. A whole-graph `convert_float_to_float16` cast does not parse in TensorRT. Torch
autocast on CUDA was slower than fp32 for every variant, so there is no `torch-fp16`.
