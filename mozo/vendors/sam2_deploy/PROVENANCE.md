# Provenance

This package is a deployment-only extraction of [SAM 2](https://github.com/facebookresearch/sam2),
reduced to promptable segmentation of single images and to tracking objects through a video,
frame by frame.

| | |
|---|---|
| Upstream repository | `https://github.com/facebookresearch/sam2` |
| Upstream commit | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Extracted | 2026-08-20 |
| Verified against | `torch` 2.11.0, `torchvision` 0.26.0, Python 3.10, on CPU |

Record the commit before re-syncing: without it, diffing a later upstream against this copy
means guessing which revision it started from.

## What was taken

Verbatim, apart from rewriting imports to relative form:

```
backbones/{hieradet,image_encoder,utils}.py
sam/{mask_decoder,prompt_encoder,transformer}.py
position_encoding.py
memory_attention.py      (added 2026-11-25, for video)
memory_encoder.py        (added 2026-11-25, for video)
```

Modified:

- `sam2_utils.py` — truncated after `LayerNorm2d`. Everything below it (`sample_box_points`,
  `sample_random_points_from_errors`, `sample_one_point_from_error_center`, `get_next_point`)
  samples training clicks, and it was the only thing importing `sam2.utils.misc.mask_to_box`.
  The image path uses three names from this file: `DropPath`, `MLP`, `LayerNorm2d`.
- `backbones/hieradet.py` — `iopath`'s `g_pathmgr.open` replaced with the builtin `open`. It
  appeared once, in a `weights_path` branch that loads a pretrained trunk on its own; mozo loads
  the whole checkpoint instead, so the branch never runs. Dropping it removes `iopath` from the
  install.

Written for this package:

- `config.py` — the four variants' geometry as frozen dataclasses, replacing the Hydra YAML.
  Values were read out of `sam2/configs/sam2.1/sam2.1_hiera_*.yaml`. This is what removes
  `hydra-core` and `omegaconf`.
- `network.py` — the image-mode model, replacing `SAM2Base`, split into `encode` and `decode`.
- `image.py` — preprocessing, prompt scaling, and mask resizing, replacing
  `sam2/utils/transforms.py`. This is what removes `torchvision`.
- `predictor.py` — checkpoint loading, prompt assembly, and the encoder-output cache.
- `tracking.py` (2026-11-25) — `SAM2Base`'s per-frame methods (`_forward_sam_heads`,
  `forward_image`, `_prepare_backbone_features`, `_prepare_memory_conditioned_features`,
  `_encode_new_memory`, `track_step` with `_track_step`, `_use_multimask`), copied in upstream's
  order of operations. Settings fixed at their SAM 2.1 values are folded, and the training,
  reverse-tracking and mask-prompt branches are gone; nothing else changed.
- `video.py` (2026-11-25) — `SAM2VideoPredictor` rewritten as a session that takes one frame at a
  time and keeps only what a later frame can read, plus the video loader's frame preprocessing.

## What was deliberately left behind

From the video predictor: `sam2_video_predictor_legacy.py`; whole-video loading
(`init_state` and the mp4 path through `decord`); prompts added to a frame already tracked
(corrections), `add_new_mask`, clearing and removing objects, `reset_state`; reverse
propagation (feeding frames backwards to a forward session is the same arithmetic, measured);
`SAM2VideoPredictorVOS` and `torch.compile`; the non-overlap constraint (off in every published
configuration). Also the training code, the evaluation harness, `automatic_mask_generator.py`, the
CUDA `connected_components` extension used for hole filling, and the Hydra build path.

Dropping these removes `hydra-core`, `omegaconf`, `iopath` and `torchvision` from the install.
Since 2026-11-25 every tensor in the checkpoint has a module and the load is strict with nothing
filtered (it used to drop the 7.5 M tracker parameters by prefix). One module is built and never
read: `mask_downsample`, used upstream only for mask prompts.

One tensor that *looks* like tracker state is used on the image path too. `no_mem_embed` is trained to mean "there is
no memory to attend to", which is exactly the situation a single image is in, and upstream's
image predictor adds it to the lowest-resolution feature map. Leaving it out costs 4.5e-02 on the
embedding and a few hundred pixels per mask — close enough to look correct, which is why the
parity suite pins it.

## Measured parity

Against upstream `SAM2ImagePredictor` built through Hydra from the same checkpoint, same process,
same torch, on CPU. Eight prompt configurations per variant: a single positive point, a box, a
box combined with a point, positive-and-negative pairs, and five random multi-point sets.

| Variant | Encoder features | IoU predictions | Low-res logits | Mask pixels differing |
|---|---|---|---|---|
| tiny | 0.0 | 0.0 | 0.0 | 0 / 2,459,520 |
| small | 0.0 | 0.0 | 0.0 | 0 / 2,459,520 |
| base_plus | 0.0 | 0.0 | 0.0 | 0 / 2,459,520 |
| large | 0.0 | 0.0 | 0.0 | 0 / 2,459,520 |

Bit-identical, not merely within tolerance. Two things had to be right for that, and neither was
obvious:

- **The resize antialiases.** Upstream resizes with `torchvision.transforms.Resize`, which on a
  tensor defaults to `antialias=True`. `cv2.INTER_LINEAR` does not, and mozo decodes with cv2
  everywhere else. `image.py` calls the same `F.interpolate` torchvision delegates to.
- **Prompt coordinates are normalised before they are scaled.** Upstream divides by the image
  width and then multiplies by 1024. Multiplying by the combined ratio instead differs in the
  last bits of a float, which was enough to move one box corner across a pixel boundary and put
  one mask pixel on the wrong side of the threshold.

## Video tracking (added 2026-11-25)

**Reference:** `SAM2VideoPredictor` at the commit above, built by `build_sam2_video_predictor`
with its defaults (dynamic multimask fallback, `binarize_mask_from_pts_for_mem_enc=True`,
`fill_hole_area=8`), fed a folder of frames.

**Guarantee:** for the same RGB frames and prompts, `Tracker.update` returns, on CPU, logits at
the video's size equal (`torch.equal`) to what upstream yields from `add_new_points_or_box` and
`propagate_in_video`. An object prompted on frame *t* equals upstream's predictor given that
prompt at frame *t* part way through tracking. Measured by `tools/verify/sam2_video.py`:
GATE_RESULTS

Why a session can be exact: tracking frame *t* reads, per object, its prompted frame, the outputs
of frames *t-1*…*t-6* (memories) and *t-1*…*t-15* (object pointers), and nothing later. Driving
upstream's own predictor one frame at a time, with later frames unreadable, gives the same logits
as its offline run (measured before extraction). Objects are independent in this predictor (batch
size one each), which is what makes adding an object on a later frame exact.

**Where the chain stops:**

- **Holes.** Upstream fills mask holes of up to 8 low-res pixels only if its optional CUDA
  extension `sam2._C` was built; otherwise it catches the ImportError and skips. This package
  reproduces the skip. Upstream installs that built the extension fill holes and differ from it.
- **Clips shorter than 16 frames.** Upstream normalises the object-pointer temporal encoding by
  `min(num_frames, 16) - 1`, so its offline answer on a short clip depends on the clip's length.
  A stream has none: the session behaves as a video of 16 frames or more. `Segmenter.tracker`
  takes `num_frames` so the gate can check short clips (equal when given).
- **The mp4 path.** Upstream's `.mp4` loader resizes inside `decord`/ffmpeg. The guarantee is
  against the folder-of-frames path, upstream's documented route; mozo decodes video with
  PixelFlow and applies that path's resize to the decoded frame.
- **The video loader is not the image predictor's preprocessing.** PIL's default (bicubic) resize
  of the 8-bit frame, `/ 255.0` in float64, stored as float32, normalised in float32 -- against
  `SAM2ImagePredictor`'s antialiased bilinear resize. Upstream's two predictors therefore give
  slightly different first masks for the same prompt on the same frame (95 pixels on a box,
  measured). Both are kept as upstream has them.
- **Mask memories are stored in bfloat16** on every device, as upstream does, and read back into
  float32 attention.

**The decoder's stability fallback.** Upstream's `build_sam2` and `build_sam2_video_predictor`
both switch on `dynamic_multimask_via_stability` (with delta 0.05, threshold 0.98) by Hydra
override: on a single-mask call, when the single mask is unstable, take the best of the three
candidates instead. It carries no weights and is not in the YAML, so the strict load could not
notice it missing, and the image path built before 2026-11-25 lacked it. `config.STABILITY` now
sets it for both paths; the gate checks single-mask image calls against `SAM2ImagePredictor`.

**Runtimes for tracking:** torch only. Memory attention -- the cost per object per frame, the
same in every size -- does not export to ONNX: its rotary encoding multiplies complex tensors
(`torch.view_as_complex`), which neither ONNX exporter accepts ("ScalarType ComplexFloat is an
unexpected tensor scalar type"; dynamo: conversion error), and a real-arithmetic rewrite would
be a different float function. So no TensorRT for it either. TensorRT fp32 on the image encoder
alone (from the published ONNX graph) saves 24 ms of a 314 ms `tiny` frame on an RTX 3060 (8%),
not worth a split-graph runner. Reduced precision is measured but not published; see the plan.

## Runtimes

`torch-fp32` and `onnx-fp32` are exported here. **CoreML is not: Apple publishes it.**

`apple/coreml-sam2.1-{tiny,small,large}` are Apache-2.0 converted packages, and mozo
redistributes them rather than producing a second set of the same thing. They differ from the
other artifacts in three ways worth knowing before using them: they are **fp16**, they split
**three** ways rather than two (image encoder, prompt encoder, mask decoder), and their encoder
takes a **CoreML image input** with normalisation baked into the graph rather than a normalised
tensor. Hence the `coreml-fp16-*` names. `base_plus` has no CoreML at all, because Apple did not
publish one.

The fp16 costs accuracy against the fp32 reference, measured on the fixture with a box prompt:

| Variant | Mask pixels differing from torch-fp32 |
|---|---|
| tiny | 49,064 / 2,459,520 (1.99%) |
| small | 36,047 / 2,459,520 (1.47%) |
| large | 19,955 / 2,459,520 (0.81%) |

Some of that is not fp16 rounding. Apple's encoder scales the image by the single scalar
`0.01735207`, which is `1 / (255 * 0.226)` -- the *mean* of the three ImageNet standard
deviations -- where the correct scaling is per channel, `[0.017125, 0.017507, 0.017429]`. CoreML's
image input accepts a scalar scale with a per-channel bias and nothing richer, so the conversion
could not express it. The bias is right; only the scale is averaged. Pre-compensating the input
recovers little of the gap, so this is a contributing error rather than the whole of it.

Speed, tiny variant at 1024x1024 on this machine:

| Runtime | Encode | Decode | Agreement with torch-fp32 CPU |
|---|---|---|---|
| torch fp32, CPU | 463 ms | 30 ms | — |
| torch fp32, MPS | 109 ms | — | **wrong: max abs 9.7e-01** |
| onnx fp32, CPU | 940 ms | 20 ms | 1.2e-05 |
| CoreML fp16 (Apple's), all units | 109 ms | 46 ms | see the table above |

The two halves have different best runtimes, and because they are separate artifacts they can be
mixed: CoreML encodes fastest and ONNX decodes fastest. ONNX is slower than torch on the encoder
and faster on the decoder, so neither runtime wins outright.

**MPS is refused for this family.** Not because it is slow or unsupported — it runs, and it is
the fastest torch path — but because on `torch` 2.11.0 it returns wrong numbers, and only under
`torch.no_grad()` or `torch.inference_mode()`, which is exactly how inference runs. Under
`enable_grad` the same code on the same device agrees to 2.6e-05. The divergence is inside the
Hiera trunk; the neck and the decoder are clean. A silent 0.37 percent of mask pixels move, so
nothing raises and nothing looks broken.

Both halves export separately, which is the point of the `encode`/`decode` split: the encoder is
a fixed-shape graph and the decoder takes a variable number of prompt points, as a dynamic axis in
ONNX. The decoder graph returns all four mask tokens and the slice that picks one or three is done
in Python, so a single graph serves both `multimask_output` settings: **token 0 is the
single-mask output and tokens 1 to 3 are the multimask candidates**, so a caller driving
`onnx-fp32-decoder.onnx` wants `low_res_masks[:, 1:]` to match `multimask_output=True` and
`[:, :1]` to match `False`. That convention is recorded here because it is not in the graph.
Apple's decoder does not offer the choice -- it emits three masks only.

Converting SAM 2 to CoreML ourselves does work, if a reason to ever appears: Hiera interpolates
its position embedding with `bicubic`, which coremltools cannot convert, but at a fixed 1024 input
that interpolation is a constant and folds away exactly. That is the only obstacle.
