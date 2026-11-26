# Provenance

This package is a deployment-only extraction of SAM 3's **image** path and its **video tracker**
(see "Video tracking" below).

It derives from `transformers/models/sam3` (Apache-2.0, © The Meta AI Authors and The
HuggingFace Team) — **not** from `facebookresearch/sam3`, whose code ships under Meta's SAM
License. Choosing the Apache-2.0 implementation is what keeps `pip install mozo` free of
SAM-Licensed source. The published model is used as a black box for verification: it is run, and
its tensors are compared against, but its source is not the source of this code.

The tokenizer and its vocabulary are OpenAI's CLIP (MIT). See `NOTICE`.

**The weights are a different matter.** SAM 3's checkpoints are covered by the SAM License,
which is not open source and carries field-of-use restrictions that flow to downstream users.
This package does not redistribute them.

## What was taken

| from | what |
|---|---|
| `modeling_sam3.py` | the RoPE ViT trunk, the FPN neck, the geometry encoder, the DETR encoder and decoder, the dot-product scoring and the mask head |
| `configuration_sam3.py` | the numeric geometry, rewritten as frozen dataclasses in `config.py` |
| `convert_sam3_to_hf.py` | the checkpoint key-rename table, adapted in `checkpoint.py` |
| `openai/CLIP` | the byte-pair tokenizer and `bpe_simple_vocab_16e6.txt.gz` |

The checkpoint is consumed **exactly as Meta publishes it** — no repacking, no pruning, no
mozo-format artifact. `checkpoint.py` translates key names at load time instead.

**The reference is `facebookresearch/sam3` at commit `5eb25fb`** (2025-12-21). It is the commit
the image gate was recorded against: all 140 `.py` files under `sam3/` of the checkout used for
`tools/verify/sam3_reference.py` are byte-identical to it. Later commits are not this model's
arithmetic any more: the SAM 3.1 release (`9f22cb9`, 2026-03-27) rewrote the shared ViT's MLP as a
fused bfloat16 `addmm`+GELU whenever gradients are off (`sam3/perflib/fused.py`), so SAM 3's own
image path became a bfloat16 function there, and refuses float32.

## What was deliberately left behind

Multi-GPU inference, interactive refinement of a tracked object (Meta's `add_prompt` with points
on a concept session), SAM 3.1, the training code and the evaluation harness.

Weights present in the checkpoint that this package does not build:

- `detector.backbone.language_backbone.encoder.text_projection` — upstream builds it, applies it,
  and the caller discards the result. Dropping it saves a matmul per prompt and 2 MB. The parity
  gate covers the claim: if it were live, the outputs would not match.
- `detector.geometry_encoder.points_{direct,pool,pos_enc}_project` — the geometry encoder supports
  **point** exemplars as well as boxes, but neither `Sam3Processor` nor `transformers` exposes
  them, and there is no Apache-licensed implementation to derive the behaviour from.

Weights that are built, because a strict load needs somewhere to put them, but never run:

- `segmentation_head.pixel_decoder.{conv_layers,norms}.2` — the pixel decoder allocates three
  upsampling stages and runs one per *gap* between pyramid levels. Three levels means two gaps.
- The lowest-resolution FPN level in both neck stacks, which upstream builds and then scalps.

## Where this diverges from `transformers`, and why

Every item below was found by comparing numbers against the published model, not by reading.
Each one is a place where following the Apache-2.0 source faithfully would have produced
confident, plausible, wrong output.

| divergence | cost of following `transformers` |
|---|---|
| The geometry encoder runs on **every** prompt, emitting a CLS token even with no boxes. `transformers` skips it when `input_boxes` is empty. | The prompt becomes 32 tokens where the weights expect 33 — on every text prompt. |
| The decoder's normalisations are numbered in a different order than they run: `norm2` follows self-attention, `norm1` follows cross-attention. | Two normalisations on the wrong residuals. Measured **6.2e-01** on boxes. |
| `freqs_cis` is loaded from the checkpoint and applied in complex arithmetic. `transformers` rebuilds real `cos`/`sin` tables. | Rebuilding is 5.96e-08 off; the real-valued rotation compounds to **1.35e-02** by block 31. |
| `qkv` stays fused, as the checkpoint stores it. | Splitting is algebraically identical, numerically not. |
| Layout is chosen per module — sequence-first in the geometry encoder and decoder, batch-first in fusion. | `nn.MultiheadAttention` transposes internally for `batch_first=True`, selecting a different kernel. One ulp at layer 0, growing through the stack. |
| `need_weights` differs per attention call: default for the two unbiased attentions, `False` for the box-biased one. | Each selects a different path inside PyTorch. Getting it uniform costs 5e-06 either way. |
| Padding masks are `True` for padding. `transformers` uses `True` for valid. | Attends to exactly the tokens meant to be ignored. |
| The trunk hands the neck a permuted view, not a contiguous tensor. | `conv2d` picks a different kernel for channels-last. **2.15e-05** on the neck output while the trunk stays bit-identical. |
| `layer_norm_eps` is 1e-5. `transformers`' `Sam3VitConfig` defaults to 1e-6. | The weights were trained at 1e-5. |
| Preprocessing multiplies by `1/255` rather than dividing by 255, and resizes in uint8 with antialiasing. | Dividing moves 1.2M of 3M pixels by one ulp; skipping antialias reaches 72 grey levels. |
| The presence logit is returned **unclamped**. `transformers` clamps it to ±10 in the decoder loop. | Upstream carries a clamp setting and leaves the returned logit untouched by it — measured −10.719295 for an absent concept, where clamping gives −10. Only visible on prompts that find nothing. |
| The pixel decoder is restored to channels-last before each convolution. | `F.interpolate` drops the format; `conv2d` and `GroupNorm` are layout-sensitive while interpolation and addition are not. 9.57e-06 on pixel embeddings, 2.7e-04 on masks. |
| `need_weights` in the mask head's prompt cross-attention is `False`, where the decoder's prompt cross-attention takes the default — despite both carrying a key-padding mask. | 2.5e-06 at that call, compounding through the pixel decoder. Which path reproduces upstream is not derivable from the call signature; it was measured at each of the four sites. |

## Measured parity

Against the published model on `tests/fixtures/images/example.jpg`, CPU, float32.

| stage | result |
|---|---|
| preprocessing | max abs diff **0** |
| vision trunk + dual neck (all 6 pyramid tensors + positions) | max abs diff **0** |
| tokenizer, 9 prompts incl. casing, punctuation, non-ASCII, empty, truncation | identical ids |
| text tower (mask, features, embeddings) | max abs diff **0** |
| geometry encoder | max abs diff **0** |
| fusion encoder | max abs diff **0** |
| DETR decoder (`pred_boxes`, `presence`) | max abs diff **0** |
| dot-product scoring (`pred_logits`) | max abs diff **0** |
| mask head (`pred_masks`) | max abs diff **0** |
| **whole concept path, end to end** — 5 prompts incl. a multi-word phrase and an absent concept | masks, boxes, logits and presence all max abs diff **0** |
| exemplar boxes — positive, negative, and two with mixed labels | max abs diff **0** |
| `Segmenter`, through both caches — masks, boxes and scores after thresholding | max abs diff **0** |

Every other gate prompts with no boxes, so the exemplar row is what exercises `roi_align` and the
three box projections at all. The two-box case is what makes it meaningful: with a single box the
reference's sequence-first prompt is indistinguishable from a batch-first one, so a transpose
would pass unnoticed.

The click path is gated too, on seven prompt shapes: one positive point, positive with negative,
two positives, a box, a box with a negative point, a lone negative point, and
`multimask_output=False`. The last two exist because a simpler set misses them -- a lone negative
is the only prompt with nothing to include, and the single-mask call is the only one that reaches
the stability fallback. Masks, scores and low-resolution logits are bit-identical to the published
model on all seven, run the way Meta's own example runs it: `Sam3Processor.set_image`, then
`predict_inst`.

## The click path

Derived from `transformers/models/sam3_tracker` (Apache-2.0), which is SAM 3's own tracker --
the same provenance as the concept path, and the reason none of it comes from `sam2_deploy`. No
vendor imports another vendor: each has to be reproducible against its own upstream, and a shared
substrate would let one family's re-sync move another's masks.

An earlier version of this package did import SAM 2's `PromptEncoder`, `MaskDecoder` and
`TwoWayTransformer`, on the argument that the architectures are the same and a second copy would
have to stay bit-identical forever. The architectures *are* the same; the argument was still
wrong. It left SAM 3's numbers descending from SAM 2's extraction of a different upstream,
inheriting whatever divergences that extraction carries, with no gate on the SAM 2 side to catch
a re-sync. `sam3_tracker` is the right source and was available the whole time.

Deriving from it also turned two findings from guesses into facts: `dynamic_multimask_via_stability`
is a declared config field rather than something bisected out of 24 parity prompts, and the
geometry comes from `Sam3TrackerPromptEncoderConfig` rather than from reading the reference's
runtime attributes.

Configured for 1008 pixels over a 72x72 grid instead of 1024 over 64x64. Every value was read
back from the published model's own attributes and then confirmed by a strict load:
`image_size 1008`, `backbone_stride 14`, `sam_image_embedding_size 72`, `low_res_mask_size 288`,
`sam_prompt_embed_dim 256`, four mask tokens, object scores on, high-resolution features on.

One prompt structure, not several. Every click prompt is a set of labelled points: `1` include,
`0` exclude, `2`/`3` reserved for a box's two corners, since the network has no box input.
Points, a box, and a box with points are the same array filled differently.

Three divergences from `transformers` were needed, each found by measuring rather than reading,
and each in the same category as the concept path's thirteen:

- **`LayerNorm2d` is written out** rather than permuting into `nn.LayerNorm`. Same normalisation,
  different last bits, and the difference survives the upscaling path into the mask -- 7.6e-06 on
  a logit, two ulp, enough to move pixels across the threshold.
- **The prompt encoder always pads a not-a-point token.** Upstream pads whenever no separate box
  input is given, and this package never gives one: a box arrives folded into the points as its
  two corners carrying labels 2 and 3.
- **`repeat_interleave` runs only when there is something to repeat.** `transformers` calls it
  unconditionally; repeating once still materialises a contiguous copy, which changes the
  attention kernel torch selects and moves the result by 8e-06.

Three further things could not be read off the weights at all, and each was found by measuring:

- **`dynamic_multimask_via_stability` is on.** It carries no parameters, so the strict load could
  not catch it, and it only changes an answer when the single-mask token is unstable. It survived
  23 of 24 parity prompts before one image caught it.
- **The two heads share one preprocessing and one encode.** Meta's image path,
  `Sam3Processor.set_image` then `predict_inst`, runs the trunk once on the concept transform and
  feeds both halves of the dual neck from it. `transformers`' `sam3_tracker`, and Meta's own
  standalone `inst_interactive_predictor.set_image`, prepare the image the way SAM 2 does instead
  -- resize in float, divide by 255 -- and so need a trunk pass of their own. An earlier version
  of this package followed that, with a second encode and a second cache. The two preparations
  differ by half a grey level, which moves a click mask by about 900 pixels on average (30 COCO
  images, 720 masks) against 0.1 pixels between CPU and MPS. On 15 COCO images and 225 masks,
  with hole filling off on both sides, the shared encode is bit-identical to `predict_inst` and
  the separate one is not: mean mask IoU 0.970, worst 0.598, a different best candidate on 1.3
  percent of prompts.
- **`no_mem_embed` applies here too**, as it does on SAM 2's image path -- trained as "there is no
  memory to attend to", which is a single image's situation exactly.

`tracker.*` is 309 tensors; 149 of them (4.2 M parameters) are the click path. The other 160
(7.5 M) are the video tracker's memory attention, memory encoder and pointer projections; they load
with the rest (`checkpoint.tracker_state_dict`), so the checkpoint loads strictly with nothing
left over.

**Holes are filled as upstream fills them.** Meta's interactive predictor sets every background
region of at most 256 pixels in the 288x288 logits just above the threshold before resizing
(`max_hole_area=256`), labelling regions through all eight neighbours with a CUDA kernel or, on
CPU, scikit-image. `image.fill_holes` does the same with OpenCV, already a mozo dependency; it is
this package's own code, written to the rule rather than copied. Measured on 720 masks from 30 COCO
images, it matches upstream's fill bit for bit on all 720, and the fill changes 593 of them. The
returned logits stay unfilled, as upstream's do. `sam2_deploy` still omits it: SAM 2's own image
predictor does not fill by default.

## Video tracking

`tracker/` follows concepts or clicked objects through a video, one frame at a time. It derives
from `transformers/models/sam3_tracker_video` (memory attention, memory encoder, the tracking step;
© The HuggingFace Team, Apache-2.0) and `transformers/models/sam3_video` (the concept tracker:
detection, matching, probation, re-seeding, occlusion; © The Meta AI Authors and The HuggingFace
Team, Apache-2.0), the decoder and prompt encoder being the click head's own. The reference is
Meta's video model at `5eb25fb`, `build_sam3_video_model()` with its defaults (temporal
disambiguation on), run as a black box by `tools/verify/sam3_video.py`.

**Licence approach (approved 2026-11-26).** In several places Meta's model behaves otherwise than
its Apache-2.0 port, and mozo follows Meta: per-arrival batches, working NMS and hole filling, one
session per phrase. That behaviour was established by running Meta's model and comparing numbers,
and the code is written from the Apache source or from the rule, as the image path's thirteen
divergences were. No code from `facebookresearch/sam3` is included.

### Where this diverges from `transformers`, and why

| divergence | why |
|---|---|
| NMS and hole filling are this package's own code (`tracker/ops.py`). | `transformers` takes both from a compiled kernel downloaded from the Hub at run time (`kernels-community/cv-utils`) and silently skips them without it: no NMS (duplicates become objects; it ran out of memory on "hand" on a 12 GB GPU where Meta needs 8.1 GB) and no hole filling. Meta's CPU path is the rule: greedy NMS in NumPy's `argsort` order, 8-connected components (scikit-image there, OpenCV here). |
| Objects are tracked in batches, one per arrival frame (`tracker/group.py`). | `transformers` keeps every object in one batch. A batched forward is not bit-identical to another batching, and Meta's re-seed and memory selection act on a whole batch. |
| Several phrases are independent sessions sharing the image encode. | `transformers` merges several phrases' detections into one matching and suppression; Meta takes one phrase per session, so only the independent form has a reference. |
| The phrase is encoded batched with `"visual"`. | Meta's video detector always encodes its text batch, the phrase and the exemplar placeholder, together. A batch of two is not the same float computation as one: measured 8.5e-06 on the text features, 6e-04 on the masks. |
| The tracker's image features are rounded through bfloat16. | Meta casts them "for all-gather" on every run, one GPU included (accommodation 2 below). |
| Frames are stored in float16 (`ops.load_frame`). | Meta's video loader resizes with PIL (bilinear, antialiased), stores the frame in float16 and normalises it in float16. Not the image path's transform. |
| The memory attention's rotary encoding is complex, the excluded keys rotated in place. | As the trunk's (`vision/vit.py`). `transformers` rotates with real `cos`/`sin` and notes the 1e-07 drift itself. |
| The first frame of a phrase session is visited twice. | Meta's API runs `add_prompt` on the frame, then `propagate_in_video` starts from that same frame; the second visit re-seeds the objects the first found and re-encodes their memory. |
| The detection score folds in presence as Meta's video detector does: `sigmoid(clamp(logit(sigmoid(q) · sigmoid(presence)), ±10))`. | The image path keeps them apart (`predictor.instances`); Meta's video model builds its detector with joint scores. |
| A frame's output is computed from that frame and the ones before only. | Meta holds output back 15 frames ("hotstart"), so that an object it removes as spurious within its first 15 frames is never shown. Its decisions are reproduced; the delay is not: such an object is shown until removed. On eight runs over Meta's two example clips (2,000+ frames, up to 8 objects), with Meta's thresholds, it never happened; with the thresholds forced from 8 to 2, 10 object-frames of 90 differed. |
| Memory is bounded (`group.prune`). | Meta keeps every past frame. A frame reads at most the four closest conditioning frames and, through memory selection, the fifteen most recent selectable frames; five and sixteen are kept, plus the last 20 frames (removal re-scores a batch, and only happens within probation). |

### Accommodations of the reference

Meta's video model is a bfloat16 (AMP) system: building it enters a process-wide CUDA autocast it
never leaves, and it casts the tracker's features to bfloat16 explicitly. Without autocast it
refuses float32. The gate runs it on the CPU in float32 with four accommodations:

1. the autocast context is exited (inert on CPU anyway);
2. the explicit bfloat16 cast of the tracker features is **kept**, and handed on as float32 where
   AMP would have kept bfloat16 -- the only one that touches a value;
3. hardcoded `"cuda"` allocations, `pin_memory`, `.cuda()` calls and the tracker's state storage
   stay on the CPU (storage only);
4. Meta's CPU connected components (`perflib/connected_components.py`) fails on an empty batch, a
   frame with no objects; it returns the empty result the CUDA path returns.

### Measured

**On CUDA** (the primary gate; `--device cuda`): Meta and mozo in one process on an RTX 3060
(vision-003), torch 2.13.0+cu126, CUDA 12.6, cuDNN 9.10.2, driver 550.144.03, fp32, PyTorch's
default TF32 settings (matmul off, cuDNN on: what mozo runs with), Meta's triton connected
components and NMS unpatched. Each side run twice first: byte-identical (147 comparisons each).
Then **852 of 852 comparisons bit-identical**, the same stages and cases as on CPU below.
Found on the way: importing Meta's `sam3.model_builder` turns TF32 on for the whole process on an
Ampere GPU (`_setup_tf32`); the gate restores the flags, and before it did, the trunk's features
differed by 0.06. All 10 falsifiers caught on CUDA too (`--device cuda --falsify`, 17 frames), at the same stages as on CPU.

**On CPU** (`--device cpu`): float32 (vision-003, torch 2.13.0, 6
threads, `MKL_CBWR=AVX2,STRICT`), 20 frames per clip: **852 of 852 comparisons bit-identical**,
every stage of every frame (frame tensor, tracker features, detections after NMS, propagated tracks
after hole filling, the update plan, the masks built, the frame's output after the overlap rule) on:
"person" on `0001` (four objects, the frame-16 re-seed); "hand" on `0001` (objects arriving in
several batches); "toy" on `bedroom` (never found); "person" and "hand" in one session, each
against Meta with that phrase alone; a click, a box and a click pair on `bedroom` through Meta's
tracker without the detector; "person" on 10 frames with the length given; and end to end through
`mozo.get_model` -> `tracker()` -> PixelFlow, length not given.

**Falsified** (`--falsify`, 17 frames), each perturbation caught at its own stage, 10 of 10:
float16 frames stored as float32 (frame 0 pixels, 4.9e-04); tracker features not rounded through
bfloat16 (frame 0 features, 8.7e-04); the phrase encoded without `"visual"` (frame 0 detections,
9.5e-05); no NMS (frame 8, five detections instead of four); no hole filling (frame 0 masks, 48
pixels); float32 memories (frame 1 propagation, 0.05); the first frame visited once (one
propagation fewer); re-seeding every 15 frames (frame 16, 8.9); memory selection at 0.9 (frame 3 of
"hand", 14.6); one batch per object (frame 0 propagation, 193). **A predicted trap that was not
one:** rotating in real arithmetic (`x·cos + rot(x)·sin`, from the same table) is bit-identical to
the complex multiply in memory attention, on CPU and here; the trunk's divergence came from
rebuilding the tables, not from the arithmetic. Its falsifier was never caught and was dropped;
"memory selection off" (every past frame a candidate) was equivalent on these clips, where every
frame scores above 0.01, and was replaced by the threshold one.

Against Meta's own bfloat16 demo default on the RTX 3060, mozo's float32 masks agree to a mean
IoU of 0.991 ("person") and 0.977 ("hand") (Phase 0).

Speed, RTX 3060, float32, 120 frames of `0001`, measured while a CPU job ran on the same machine:
2.95 s a frame with four people, 4.57 s with eight to nine hands; Meta in float32 2.64 and 4.20 s
under the same conditions. Peak CUDA memory 4.7 GB (people) and 5.3 GB (hands), flat once the
objects are there (Meta, which keeps every frame: 6.7 and 8.1 GB at 120 frames). Two phrases over
270 frames: 6.9 s a frame, 22 objects by the end, 5.6 GB.

## Dependencies

`regex` and `ftfy` are core dependencies, needed by the tokenizer: CLIP's split pattern uses
Unicode categories (`\p{L}`, `\p{N}`) that the standard library's `re` cannot express, and `ftfy`
repairs mis-decoded input the way upstream does.

`torchvision` is a declared core dependency, and `grounding/geometry.py` imports it at module
scope rather than defending against its absence. It was already a de-facto dependency before SAM 3
— `rfdetr_deploy` and `depth_anything_v2_deploy` both import it at module scope — and SAM 3 needs
`roi_align` for exemplar-box pooling. Declaring it is what makes a missing install fail at install
time instead of inside a forward pass.

Writing `roi_align` against `grid_sample` to drop the dependency was considered and rejected: it
would remove nothing from the install, since two other families already require torchvision.
