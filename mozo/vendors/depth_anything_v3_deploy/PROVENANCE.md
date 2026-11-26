# Provenance

This package is a deployment-only extraction of single-image depth from
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).

| | |
|---|---|
| Upstream repository | `https://github.com/ByteDance-Seed/Depth-Anything-3` |
| Upstream commit | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| Extracted | 2026-11-25 |
| Verified against | `torch` 2.13.0, `torchvision` 0.28.0, `numpy` 2.3.5, `opencv` 4.10.0, Python 3.12 |

The commit is HEAD of 2026-07-27. The model and pre-processing code last changed in `a71fc79`
(2025-12-11); every commit since touches the Gradio app, the backend service or the CLI.

## The reference

The authors' repository, because it is what trained and published the checkpoints; there is no
other implementation (`transformers` 5.20 has no Depth Anything 3). Single-image inference there
is `DepthAnything3.inference([image])`: `InputProcessor`, then `DepthAnything3Net.forward`, then
`OutputProcessor`, whose `depth[0]` is the answer.

**What this package guarantees:** with autocast off, for the same RGB image, the depth this
package returns at the processed resolution is `torch.equal` to upstream's
`DepthAnything3Net` through upstream's `InputProcessor` and `OutputProcessor` -- and so is every
intermediate the two networks share by name. With autocast on, as upstream's `inference` always
runs, this package under the same autocast equals upstream's unmodified `inference`. Both hold
on CUDA (RTX 3060, torch 2.13.0, CUDA 12.6, cuDNN 9.10.2), in fp32 and under upstream's bf16 and
fp16 autocast, and on the CPU. The chain stops there: accuracy numbers are the authors'.

## Checkpoints, and why only these three

| mozo | upstream (Hugging Face, commit) | params | card | authors' table | taken |
|---|---|---|---|---|---|
| `small` | `depth-anything/DA3-SMALL` `e08cab65` | 0.08B | apache-2.0 | Apache 2.0 | yes |
| `base` | `depth-anything/DA3-BASE` `f4a6c9b3` | 0.12B | apache-2.0 | Apache 2.0 | yes |
| `mono-large` | `depth-anything/DA3MONO-LARGE` `f465978e` | 0.35B | apache-2.0 | Apache 2.0 | yes |
| | `DA3METRIC-LARGE` `4010e39f` | 0.35B | apache-2.0 | Apache 2.0 | no: metres need the focal length |
| | `DA3-LARGE-1.1` `0e109ae3` | 0.35B | **apache-2.0** | **CC BY-NC 4.0** | no: the claims conflict |
| | `DA3-LARGE`, `DA3-GIANT(-1.1)`, `DA3NESTED-GIANT-LARGE(-1.1)` | | cc-by-nc-4.0 | CC BY-NC 4.0 | no: non-commercial |

"Card" is the `license:` field of the model card; "authors' table" is the model table in the
README at the pinned commit. DA3-LARGE-1.1's card says apache-2.0 while the same authors' table
says CC BY-NC 4.0, as for every other LARGE and GIANT; the stricter reading wins.
DA3METRIC-LARGE is architecturally `mono-large` (identical config) and its output is metres only
after `focal_px * output / 300` (upstream README, FAQ), which needs the camera's focal length.

Each published `torch-fp32.pth` is the checkpoint's `model.safetensors`, verified against the
sha256 the Hub records, converted to an ordinary state dict by `tools/fetch/depth_anything_v3.py`
with upstream's `model.` wrapper prefix removed and these tensors left out, because no
single-image depth computation reads them:

- `cam_enc.*` (64 tensors) -- the camera encoder, for pose-conditioned input.
- `cam_dec.*` (10 tensors) -- the camera decoder, which predicts the pose.
- `head.scratch.{refinenet1..4,output_conv1,output_conv2}_aux.*` -- the any-view
  head's ray branch, a second fusion chain that shares nothing with depth but its inputs
  (94 tensors).

`small` and `base` keep 269 of 437 tensors; `mono-large` keeps all 406. Every tensor kept is
checked `torch.equal` to the source and loads strictly. `pos_embed.pth` is the one tensor
`backbone.pretrained.pos_embed` again, alone, for the graph runtimes (below).

Upstream loads its checkpoints through `PyTorchModelHubMixin`, which is **not strict**: for
`small` and `base` it leaves `head.scratch.output_conv2_aux.{1,2,3}.2` (LayerNorm weight and
bias) at their initialisation. They belong to the ray branch, which returns only its last level,
so no output reads them.

## What was taken

Verbatim from `src/depth_anything_3/model/`:

```
dinov2/layers/{attention,block,drop_path,layer_scale,mlp,patch_embed}.py
```

Modified (each file says how at its top):

- `dinov2/layers/rope.py` -- the position and frequency caches removed (recomputed per call,
  same values), and the frequency table sized by the token count instead of by reading the largest
  position back from the tensor. A dict keyed by shape and an `int(tensor)` cannot be traced, and
  the caches grew by one entry per image size in a long-running process.
- `dinov2/layers/__init__.py` -- does not import SwiGLU (xformers; giant only).
- `dinov2/dinov2.py` -- `vitg` removed, import made relative.
- `dinov2/vision_transformer.py` -- one view's forward. Removed: reference-view selection and
  reordering (fires only for three or more views), exported auxiliary features, the mask token,
  SwiGLU and identity feed-forwards, `vit_giant2`, the logger. `einops.rearrange` is spelled as
  the reshapes it performs. The resampling of the position grid moved into a function of
  `pos_embed` and the shape (`interpolate_patch_pos_embed`) that a caller may run beforehand; the
  final `.view` is a `.reshape` (same values whatever the strides).
- `dpt.py` -- one view's forward (no chunk loop, no Gaussian-splat image plumbing), plain dicts
  instead of `addict`, `nn.quantized.FloatFunctional().add` spelled as the `torch.add` it is, and
  the head's output size as integer division (`int(ph * 14 / 1)` is `ph * 14 // 1`; the float
  form freezes the size into a traced graph).
- `dualdpt.py` -- the depth branch only, the same integer division, and the UV embeddings as
  functions of the shape (`uv_embedding`, `uv_embeddings`) that a caller may run beforehand.
- `head_utils.py` (from `model/utils/`) -- `activate_head_gs` removed; `custom_interpolate`
  keeps an integer size integer (the fusion blocks pass `scale_factor=2`).

Written for this package:

- `config.py` -- the three checkpoints' YAML as frozen dataclasses.
- `net.py` -- `DepthAnything3Net` reduced to backbone and head; the sky fill
  (`_process_mono_sky_estimation`, `compute_sky_mask`, `set_sky_regions_to_max_depth`); and
  `shape_inputs`, the tensors a graph runtime is handed.
- `predictor.py` -- upstream's `InputProcessor` for one image at its defaults
  (`process_res=504`, `upper_bound_resize`) on the array rather than a PIL image, and the
  post-process.
- `graph.py` -- the exported graph's inputs, outputs and shape range.

## Where this diverges from the obvious reading, and why

- **Autocast.** Upstream's `DepthAnything3.forward` always runs under `torch.autocast` -- bf16 if
  `torch.cuda.is_bf16_supported()`, else fp16 -- on whatever device, including the CPU. This
  package runs in fp32 unless the caller enters autocast. Both are gated.
- **The sky fill's random sample.** Upstream takes the 0.99 quantile of the non-sky depth over
  100,000 pixels drawn with `torch.randint` from torch's *global* generator, so its depth depends
  on the caller's RNG state and moves it. Here the sample comes from a generator seeded 0 on every
  call, on the depth's device: the same image gives the same depth, and the caller's RNG is not
  touched. With the global generator seeded 0, upstream draws the same sample.
- **The resolution of the answer.** Upstream returns depth at the processed resolution. That is
  this package's answer too; mozo's adapter resizes it to the input's size and inverts it.
- **Extreme aspect ratios.** A side that rounds to under 14 px (beyond 72:1) fails an assertion
  upstream; here it raises `ValueError` before the network.
- `torch.backends.cudnn.benchmark = False`, set by upstream's `api` at import, is not carried: a
  library does not set process-wide flags.

The DINOv2 attention here calls `F.scaled_dot_product_attention` (upstream's `fused_attn=True`);
there is no xformers or flash-attention path in the model code upstream ships beyond SwiGLU,
which only the giant model uses.

## Verification

`tools/verify/depth_anything_v3.py` reproduces all of the following against a checkout of the
commit above and the Hugging Face snapshots, in one process with upstream: on CUDA
(`--device cuda`, the primary gate) and on the CPU with `MKL_CBWR=AVX2,STRICT`:

- **Standalone** -- nothing imported outside stdlib, torch, torchvision, numpy and cv2.
- **Every stage** -- the preprocessed tensor, every call of every module the two networks share
  (292 calls over 249 modules for `small`/`base`, per image), and the final depth: `torch.equal`.
- **Upstream's own API under autocast** -- equal.
- **mozo** -- the adapter through the registry and the weights resolver: equal to the vendor's
  depth, resized and inverted.
- **Deterministic**, and the precomputed shape inputs give exactly the network's own result.

Measured 2026-11-25 on all three variants, over four photographs and seven synthetic inputs each
(noise, an enlarged 90x60, a 504-wide image that skips the first resize, portrait, a 2000x150 strip,
a tie in the multiple-of-14 rounding, black): every image passes every stage. The gate is falsified
by eleven perturbations, each caught at the stage it reaches and not before: LayerNorm eps of a
block and of the final norm, the position-grid offset, `INTER_AREA` to `INTER_LINEAR` and the
multiple-of-14 tie rule (at the input), and for `small` and `base` the RoPE frequency, the QK-norm
eps, swapped camera tokens and the head's UV-embedding ratio x1.0000001; for `mono-large` the
sky seed and the sky threshold (at the final depth).

**On CUDA**, measured 2026-11-26 on an RTX 3060 (vision-002; torch 2.13.0+cu126, CUDA 12.6,
cuDNN 9.10.2, driver 550.144.03), with mozo's production settings (torch's defaults: cuDNN TF32
on, matmul TF32 off, every SDPA backend enabled, `cudnn.benchmark` off): six photographs and the
seven synthetic inputs per variant, in fp32 (mozo's `torch-fp32`) and under upstream's own bf16
and fp16 autocast -- every stage of every image of every variant byte-identical, 42,120
comparisons. Upstream's RoPE caches its
cos/sin table under a key that ignores autocast, so whichever precision the first call in a
process had sticks; the gate empties that cache before each upstream run, which is the answer a
fresh process gives. On CUDA the same eleven perturbations are caught, and five that change only
the hardware arithmetic on the vendor's side: TF32 matmuls on, cuDNN TF32 off, channels-last
convolution inputs, SDPA forced to its math kernel (the default is the memory-efficient fp32
kernel), and the attention's unfused path.

**Not always deterministic on CUDA.** With mozo's production settings, `mono-large` on a 42x504
input (a 2000x150 strip) now and then computes one attention `qkv` Linear differently from the
time before: same input, same weights, same cuBLAS kernel (`ampere_sgemm_128x32_tn`), a
different last bit, after which every later stage differs (final depth up to ~1e-4). It happens
on upstream and on this package alike, so it is not a difference between them. Measured
2026-11-26/27 on the RTX 3060 above: 4 occurrences in 22 gate passes over the 13 images; 1 in 24
with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; 0 in 24 with `torch.use_deterministic_algorithms(True)`
as well, although that setting launches the same `qkv` kernel and only changes the head's
bilinear upsampling (other bits, 14% slower), so it is not shown to be the cure. In isolation
the same GEMM on the same input never differed in 800,000 calls, nor 9,000 whole forwards, nor
2,160 forwards of that input alone in the gate's loop. No other variant, image or module has
shown it. The gate reports it as a failure, naming the side and the module.

## Export

`tools/export/depth_anything_v3.py` publishes `onnx-fp32` and `onnx-fp16` (fp16 autocast, the
head kept in fp32 as upstream keeps it) with dynamic height and width, 3 to 36 patches a side. Float
arithmetic on the input's size cannot be traced, so the position grid and the UV embeddings are
graph inputs (`graph.py`), computed by `shape_inputs` exactly as the network computes them. Four
places in upstream's code froze the size into a traced graph and were rewritten with the same
integers (listed above). Measured against torch fp32 on an RTX 3090 over the 654 NYUv2 test images:
TensorRT fp32 3e-7 to 8e-7 mean relative difference, AbsRel identical; TensorRT fp16 0.08-0.11%,
AbsRel within 1e-4.
