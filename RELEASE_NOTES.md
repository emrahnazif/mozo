## Unreleased

- **RF-DETR runs 1.7 to 2.6 times faster on CUDA by default, with the same detections**, when
  TensorRT 11 or later is installed (`pip install tensorrt-cu12` or `-cu13`). `auto` then builds a
  TensorRT engine in full fp32 from the published ONNX graph: on 500 COCO images no detection
  changed and AP was identical, on every detection and segmentation variant. RF-DETR spends most of
  a call launching kernels, which a compiled engine removes. The first load on a machine builds
  the engine in about 12 s and keeps it in the cache. Without TensorRT, nothing changes: `auto`
  stays on torch. `keypoint-preview` publishes no graph and stays on torch.
- **RF-DETR's detection variants can run through TensorRT in fp16, about 6 times faster than
  fp32.** Ask for `runtime="tensorrt-fp16"`. AP does not move and 0.2–0.5% of detections above
  0.5 come out differently than in fp32. `auto` never picks it. Each of `nano`, `small`, `medium`
  and `large` now publishes an fp16 ONNX graph, 55–63 MB, to build it from; only those loads
  download it.

- **OWLv2 runs about 4.3 times faster on CUDA by default (4.8 times for `large`)**, measured with
  torch 2.13; 4.4 and 5.5 times with torch 2.6. `auto` now
  picks `torch-fp16` on CUDA: the same checkpoint, with the image tower's blocks in half precision
  and its norms, heads and text tower in fp32. COCO AP is unchanged on every variant. A few
  borderline detections come out differently than in fp32: 0.5–1.1% of those above 0.3. `runtime="torch-fp32"` gives exactly the old results. Nothing new to download.
- **Grounding DINO runs 1.45 times faster on CUDA by default (1.55 times for `base`)**, measured
  with torch 2.13; 1.3 and 1.4 times with torch 2.6. `auto` now
  picks `torch-fp16` on CUDA: the same checkpoint under fp16 autocast. COCO AP is unchanged, and
  about 0.1% of detections above 0.35 come out differently than in fp32, about as often as fp32
  itself changes when only its input is rounded. `runtime="torch-fp32"` gives exactly the old
  results.
- **Grounding DINO can run through TensorRT on CUDA, 2.6 times faster than fp32 (3.0 times for
  `base`)** for a whole `predict` with 80 phrases. Ask for `runtime="tensorrt-fp16"`. It needs
  TensorRT 11 or later (TensorRT 10 miscompiles the model's position encoding, and mozo refuses it)
  and `pip install onnx onnxscript`. Every image is fitted into one square, 800 pixels by default,
  and padded, with the padding masked so the model ignores it. At 800 that costs 1.6 AP on `tiny`
  and 1.8 on `base` against upstream's resize. Detections change no more than on `torch-fp16`.
  The first load for each GPU and size exports the graph and builds the engine, in about
  3.5 minutes; later loads take half a second. `auto` never picks it.
- **Grounding DINO takes `input_size`**, a square to fit every image into instead of upstream's
  resize: 1024 gives back most of the AP (0.4 below upstream on `tiny`, 0.7 on `base`) at about
  twice fp32's speed on TensorRT. It works on torch too.
- **Grounding DINO exports to ONNX correctly, at any caption length.** The Swin backbone's ceiling
  division traced to a truncating division, so an exported graph computed one row of windows too
  few. Deformable attention now carries the feature-map sizes as plain integers instead of reading
  them back from the GPU, which `torch.export` needs; the results are bit-identical. The torch path
  is otherwise unchanged.
- **OWLv2 can run through TensorRT on CUDA, 7.2 to 8.2 times faster than fp32.** Ask for
  `runtime="tensorrt-fp16"` after installing TensorRT 11 or later for your torch's CUDA
  (`pip install tensorrt-cu12` or `-cu13`). The first load on a machine builds an engine for that
  GPU, in 17 s for `base` and 26 s for `large`, and keeps it in the cache; later loads take about a
  second. COCO AP is unchanged, but TensorRT's fp16 arithmetic is less exact than torch's, so
  2.5–4.8% of confident detections differ from fp32. `auto` never picks it. Each OWLv2 variant now
  publishes its image tower as an fp16 ONNX graph (178 MB for `base`, 618 MB for `large`) to build
  engines from; only TensorRT loads download it.
- **`mozo.runtimes.TensorRTRunner`** builds an engine from an ONNX graph, caches it beside the
  graph for the GPU and TensorRT version, and runs it on torch CUDA tensors, safely from several
  threads. It needs TensorRT 11 or later and refuses older versions: engines are strongly typed,
  so precision comes from the graph, and fp32 stays IEEE fp32 rather than TensorRT's default
  TF32.
  It can export the graph itself the first time and delete it after the build, take dynamic input
  shapes, and check a fresh engine on inputs you supply. `weights.cached()` lists built engines with the state `"built"`, and `weights.remove`
  deletes them with the model.
- **Runtimes can be built at load time** from another published artifact, rather than downloaded,
  and can be limited to one kind of device: `torch-fp16` and `tensorrt-fp16` exist only on CUDA,
  and asking for them elsewhere says so.

# Mozo v1.14.0: Granite Embedding R2 on CoreML, at half the download

- **Granite Embedding R2 downloads half as much: 95 MB instead of 191 MB.** The new revision,
  `2026-11-24`, stores the weights in bfloat16 as IBM publishes them, and mozo still runs them in
  float32. The vectors are bit-identical to `2026-11-20`'s, so an index you built keeps working.
- **Granite Embedding R2 runs as a CoreML graph on Apple silicon.** With the `coreml` extra on
  macOS 15 or later, `auto` picks it: 0.27–0.68× torch's time on the GPU, e.g. 394 ms against
  1,471 ms for 8,192 tokens, and within 1.8e-07 of torch's vectors, so an index built on either
  runtime serves queries from the other. Each text is padded to one of eight fixed lengths and the
  padding is masked out; a text just past a length runs at up to 1.3× torch's time.
- **`CoreMLRunner(path, function=)`** runs one function of a multifunction CoreML package, and
  `mozo.runtimes.coreml_functions(path)` lists them with their input shapes. A package is compiled
  once per process, however many of its functions are loaded.
- **`Workflow.save` never leaves a half-written file.** It writes the whole file beside the old one
  and then moves it into place, so a save cut short by a killed process or a full disk leaves the
  previous workflow as it was. The file keeps its permissions.

# Mozo v1.13.0: Short workflow files and an editor that lays itself out

- **A shorter workflow file.** A workflow can be written as its nodes alone, one per line with its
  parameters beside its id and type, and no edges: an input that is not written takes the latest
  earlier node with an output of its type, and a node writes `"inputs": {"image": "read"}` only
  where it wants something else. The video example in the docs goes from seven edges to one
  written input. Unknown parameters and inputs are refused, naming the node, where they used to be
  ignored. `Workflow.save`, `to_dict` and the editor now write this form, one node per line;
  files with `edges` are read as before. Files saved by this version need this version to read.
- **The editor's panel sits beside the canvas.** One panel on the right lists the nodes, and shows
  the selected node's settings instead when one is selected, so the canvas is never covered. A
  host can draw it into an element of its own with `mount(element, {panel})`.
- **The editor connects and lays out nodes for you.** A node dropped onto the canvas is connected
  by the same rule as the short file. A workflow with no positions is laid out with dagre, Svelte
  Flow's own layout recipe, and **Tidy up** lays out the whole canvas again. The view fits the
  workflow each time one is opened, not only the first.
- **A workflow document with a key mozo does not read is refused,** naming the keys it takes:
  `{"workflow": {...}}` loaded as a workflow of no nodes, and `parameters` put on a node instead
  of under `data` was dropped. Either form, every level: the document, a node, an edge.
- **Opening a workflow in the editor is no longer reported as a change.** While a document waited
  for the node list to load, the editor could fire `"change"`, so a host showed "unsaved" on a
  workflow just opened. The canvas is now empty until the document is drawn. Every node is now one
  width, with its description cut to two lines (the whole of it is in the panel), so laid-out
  nodes no longer overlap.
- **A parameter may not be called `id`, `type`, `inputs`, `xy` or `data`,** since the short file
  writes parameters beside those. No built-in node has one.
- The editor's bundle ships the licences of everything compiled into it: Svelte, Svelte Flow and
  dagre beside Heroicons.

- **A node's description is its docstring's whole first paragraph, as plain text.** The catalogue
  took the first line only, so a sentence wrapped across two lines was cut in half (`track`'s was),
  and docstring markup such as ``` ``tracker_id`` ``` and `:func:` showed as written. Your own
  nodes get the same: write the description as the first paragraph.

# Mozo v1.12.0: Workflows Say What They Give

## What's New in v1.12.0

- **A workflow node can say it gives many values.** A node returning `-> list[Image]` is
  catalogued with `"many": true` on that output, and the editor draws its port with a dashed ring.
  `crop_around_detections` is the first: its output read as one image, though it gives one per
  detection. Connecting it to an image input works as before, and every node after it runs once
  per crop.
- **The editor opens workflows that have no node positions.** A workflow written in code or by
  hand stacked every node in one spot, with no edges drawn and nothing draggable. The editor now
  lays such nodes out top to bottom by their edges, and a saved workflow leaves out the empty
  `"position": {}` it used to write for them.

---

# Mozo v1.11.0: Docs That Come With It

## What's New in v1.11.0

- **The docs ship inside the package.** After `pip install mozo` every page is a markdown file in
  `mozo/docs/`, with an index at `mozo/docs/llms.txt`, so a coding agent in your project reads the
  docs for the version you installed, offline, rather than the website's. Text only: the website's
  images are left out of the wheel. The README says how to find the folder.
- **`MOZO_ENABLE` now holds workflows to the same list.** A workflow sent to `/workflow/run`,
  `/stream` or `/process` that names a model the server does not offer fails at that node with the
  same reason `/predict` gives, before anything is downloaded. Before, a server narrowed to
  permissive weights would still fetch and serve YOLO the moment a workflow named it. A node running
  a checkpoint of your own is not refused.
- **Documentation at [emrah.fr/mozo](https://emrah.fr/mozo/).** A page per model family, with
  examples run against the published weights; guides for Python, the HTTP server and workflows;
  deployment notes; and a reference for every public function, route and command. `docs/llms.txt`
  indexes it for agents. `tests/test_docs.py` fails when a family, public name, route or command has
  no entry, so the docs cannot quietly fall behind the code.

---

# Mozo v1.10.0: Text You Can Search

## What's New in v1.10.0

- **New family `granite_embedding_r2`** — English text embeddings, mozo's first text-only
  family: text in, one 384-d unit vector out, so documentation, notes or what EasyOCR read off
  an image can be searched by meaning. IBM's `small` model, Apache-2.0 code and weights, up to
  8,192 tokens a text; a longer text is refused rather than truncated. Served through
  `/encode` only — `/predict` answers 501 and points there — and from Python as
  `encode_text`. Bit-identical to `transformers` at every stage, and the same text gets the same
  vector on every Python from 3.10 to 3.14. Its vectors do not share a space with CLIP's or
  SigLIP 2's.
- **`Workflow.process(context=True)` also yields each item's Context**, as
  `(item, results, run)`: code that stores results itself can read `run.frame`, `run.time`
  and `run.frames` the way a node does. The editor's run counter now shows "42 of 315" for a
  video, because `/process` events carry `of`, `frame` and `time`.

---

# Mozo v1.1.1

## What's New in v1.1.1

- **`save_image` now writes PNG by default** (was JPEG). Annotated frames are thin lines and
  mask edges, which is what JPEG is worst at, and it makes the node agree with
  `mozo.image.encode_image`, which already defaulted to PNG. It is also the only common
  default that can carry an alpha channel, so `ben2`'s cut-outs save with their transparency
  intact. Pass `format=".jpg"` where size matters more than edges.
- **New family `ben2`** — background removal, returning an RGBA cut-out. An image port now
  carries three channels or four; a node is handed three unless it sets `alpha=True`.

---

# Mozo v1.1.0: Answers in Pixels

## What's New in v1.1.0

Fifteen families describe an image. The sixteenth rewrites it.

### 🩹 Moebius

Image inpainting — an image and a mask in, the masked thing gone. Two variants, `general` and
`places2`, at 226M parameters each.

```python
model = mozo.get_model("moebius/general")
clean = model.predict("street.jpg", mask=selection)          # or detections=found
```

It is the first family here whose `predict` returns an array of pixels rather than a PixelFlow
container, and **the first with no confidence number**, because it does not estimate anything:
it draws a sample. Run it again with another seed and you get a different, equally valid answer.
The paper reports it matching FLUX.1-Fill-dev — 11.9B parameters — across six benchmarks.

**Only the seam moves.** The adapter returns an array the same size as the input, with every
pixel the feathered edge does not reach identical to it byte for byte: about 8 px past the
selection at the default radius, and exactly the mask's edge at `feather=0`. Upstream returns
the decoder's reconstruction of the whole frame, which changes every pixel including the ones
nobody selected.

**It runs at 512×512 and cannot run at anything else.** That is a property of the weights rather
than a setting — the cross-attention's positional table is stored with a row per latent cell, so
a larger image has nowhere to put its positions. The frame is fitted and the answer composited
back at full resolution. A selection too thin to survive the fit is refused rather than returned
unchanged.

Bit-exact against upstream at every stage, under `torch.equal` with no tolerance: the
autoencoder's eight stages, both λ layers, the whole 1203-tensor UNet forward, and all nineteen
DDIM steps including the branch that fires once. The gate is falsifiable and has been falsified —
four perturbations, each caught at the stage its constant reaches. The most instructive is a
`GroupNorm` epsilon of 1e-6 against PyTorch's 1e-5, which moves the encoder trunk by 1.9e-03 and
would have been swallowed by any tolerance a reasonable person would have picked.

`torch-fp32` only, in two artifacts per variant — a denoiser and an autoencoder, with a `NOTICE`
each, because the autoencoder is a separate work under separate terms.

### ✏️ A task that returns a picture

`image_inpainting` is a new task type. `/predict` gains an arm that answers with a PNG and takes
a `seed`, the test UI gains a second pane showing the edited frame, and the workflow catalogue
gains an `Edit` category with a `moebius` node.

It also carried a fix the pose arm needed already: dispatch was on `task in BOXED`, which was
only ever "pose" by accident of that set having one member. With two members it would have
answered inpainting requests with pose JSON — and the coverage test could not have seen it,
because that test counted every `BOXED` member as served whether an arm existed or not. Both
dispatch by name now.

### ♾️ Workflows over a source of any size

`Workflow` runs once per item over anything iterable — a directory of a million photographs, or
a camera that never ends. Measured at ten million items from an infinite source with resident
memory identical at the first item and the last: 78.5 MiB throughout.

A failure goes to `on_error` rather than being swallowed or made fatal; one corrupt file in a
million should not end a six-hour run. An item whose graph failed part way is not yielded at all,
because a caller iterating results has no way to tell a partial answer from a whole one.

### 📈 The catalogue

80 variants across 16 families, 47 of them under permissive terms, and thirteen verification
gates.

---

# Mozo v1.0.2: Where the Joints Are

## What's New in v1.0.2

Mozo's fifteenth family, a second way to get joints, and graphs for the ten segmentation
variants that shipped without them.

### 🕺 ViTPose++

Top-down human pose estimation, Apache-2.0, in four sizes — `small` `base` `large` `huge`.

```python
people = mozo.get_model("rfdetr/medium").predict("crowd.jpg", threshold=0.5)
posed = mozo.get_model("vitpose/base").predict("crowd.jpg", detections=people)
posed[0].keypoints[0]      # KeyPoint(id=0, name='nose', x=..., y=..., confidence=...)
```

**It is the first family whose `predict` takes detections as input.** ViTPose has no detector:
it answers a question about a box someone else found. So it takes the full frame *and* the
boxes, and returns those same detections with seventeen COCO joints attached to each.

The full frame is not a convenience. The crop it works from is deliberately larger than the box
— widened to 3:4 and then padded by a further 1.25× — so a 50×140 person becomes roughly a
131×175 crop. A caller who cropped tightly has already thrown away pixels the model wants, and a
wrist just outside the box goes with them.

**It filters nothing.** Hand it detections and it poses all of them, so filtering to people is
the caller's call and stays visible in the caller's code. Hand it an empty set and you get an
empty set back; hand it nothing at all and Python raises, because there is nothing to infer.

Heatmaps are bit-identical to `transformers` on all four variants, and joints land within
4.3e-04 px end to end. `small` `base` and `large` also publish `onnx-fp32`; `huge` does not,
because its graph does not fit in one protobuf. There is no CoreML — it converts and is correct,
but it is not faster than torch on MPS.

### 🤸 RF-DETR keypoint-preview

A second answer to the same question, in one variant, at Roboflow's published operating point of
576×576. `torch-fp32` only.

Its class-id space is its own — background at 0, person at 1 — where the detection variants emit
COCO's sparse ids running to 90.

### 📦 A task that says it must be told where to look

`pose_estimation` is a task type whose models cannot be asked to go and find something. The
registry says so once, and `/predict` grows a repeatable `box` parameter that the test UI offers
for exactly those families. Nothing carries a per-family exception.

### 🎭 ONNX for the segmentation variants

The ten `seg-` variants YOLO11 and YOLO26 shipped in v1.0.1 published `torch-fp32` only. They
now publish `onnx-fp32` as well — a graph with two outputs, the rows and the prototypes their
mask coefficients belong to.

They were held back because the export gate could compare boxes, scores and class ids but not
masks, and publishing on three quarters of the answer would have reported an agreement it had
not checked. The gate now compares masks by per-detection overlap. Worst figures across the ten:
1.83e-03 px on boxes, 2.00e-04 on scores, 0.999779 IoU on masks.

Still no CoreML for those two families, and the mask branch does not change that — the `seg-`
checkpoints are the same backbone with a different head, so they inherit the same Metal compiler
abort their detection counterparts hit.

### 📖 Documentation

`docs/models.md` gains a ViTPose section covering the crop geometry, the DARK decoding and the
mixture-of-experts backbone, and states what each YOLO generation publishes.

---

# Mozo v1.0.1: Run a Graph, Not Just a Model

## What's New in v1.0.1

Until now mozo answered one question at a time: load a model, hand it an image, get a result.
A workflow is the other half — read an image, run a family, crop what it found, run a second
family on each crop, draw the answer, write it out — as a graph you can save, version, post over
HTTP, or run from the command line.

That is what the major version is for. Nothing about `get_model()` changes, and nothing in the
0.7.x catalogue moves; what changed is that the catalogue is now something you can compose.

> **PyPI goes 0.7.1 → 1.0.1.** v1.0.0 was tagged, but its release run failed the test gate
> before it published anything, so nothing reached PyPI. Everything below ships here for the
> first time.

### 🔀 Workflows

A workflow is a graph of nodes. 38 of them ship, across ten categories — reading an image,
seven detectors, segmentation, two classifiers, depth, thirteen transforms, three exposure
adjustments, eight annotators and a writer.

```python
from mozo.workflow import Workflow

flow = Workflow.from_dict(saved)
result = flow.run(image="street.jpg")
```

**Validation is construction.** `from_dict` refuses a graph rather than failing halfway through
one: an unknown node type, a wire between mismatched ports, an input nothing feeds, a cycle.
A workflow that builds is a workflow that can run.

**Batching lives in the node.** A list on an input port fans the node out over it, while
parameters stay shared — so `crop_around_detections` turns one image into one crop per detection
and everything downstream runs per crop, with nothing in the graph saying so.

**Four endpoints**, mounted on the same server that serves `/predict`, in the same process:
`/workflow/nodes` for the catalogue, `/workflow/validate`, `/workflow/run`, and `/workflow/stream`
for per-node progress. Models are shared with `/predict` rather than loaded twice — one
`ModelManager` for the process, so a family a workflow loads is a family `/predict` already has.

**An editor**, served at `/workflow`, and a command line:

```bash
mozo run flow.json --image street.jpg
```

### 🎭 Instance Segmentation on Two More Families

Ten new variants, all boxes-plus-masks, all sitting beside their detection counterparts under the
same `object_detection` task type because what they add is a field rather than a different
question:

- **YOLO26** `seg-nano` … `seg-xlarge`
- **YOLO11** `seg-nano` … `seg-xlarge`

A mask is a boolean at the source image's resolution, one per detection, aligned with the box
beside it. Both families publish `torch-fp32` for these.

### 📊 Model Count

**73 published variants** across fourteen families, up from 63. Of those, 36 are Apache-2.0,
30 are AGPL-3.0 (every YOLO variant), 4 are MIT (CLIP), 2 are CC-BY-NC-4.0 (Depth Anything
`base` and `large`) and 1 carries Meta's SAM License.

### ✅ Verification

**YOLO11 and YOLO26 are now checked against `ultralytics` itself**, stage by stage, rather than
only against themselves — `tools/verify/yolov11_reference.py` and its YOLO26 counterpart stand the
published implementation up beside the vendored one and compare preprocessing, every layer, the
head, the prototypes, the mask coefficients and the assembled masks. 335 comparisons across
YOLO11's ten variants, every one within tolerance, with three of the five segmentation variants
pixel-identical to the reference.

Both gates are falsifiable and have been made to fail on purpose: perturbing a constant moves the
stages that constant reaches and no others, with a control that provably moves nothing.

### 🐛 Fixes

- **One model manager per process.** The server built its own, so a family reached through
  `/predict` and through a workflow loaded twice, was resident twice, and `/models/loaded`
  reported only half of what was in memory.
- **The YOLO runtime seam states its assumption.** It takes the first output of a graph, which is
  right for every graph published today and wrong for a multi-output one; it now refuses rather
  than misreading.
- **The export gate refuses what it cannot check** instead of comparing a graph against the module
  on three quarters of the answer.
- **The suite runs without weights again.** Three workflow tests asked the manifest whether a model
  was published and then ran it, which is a different question: run through a workflow, a missing
  checkpoint surfaces as a failed node rather than an exception, so a machine without the weights
  failed instead of skipping.
- **Route introspection survives FastAPI 0.137**, which stopped copying an included router's routes
  into `app.routes` and started listing the router itself.

### 📖 Documentation

`docs/workflows.md` covers what a workflow is, how validation and batching behave, and what each
node category is for. `docs/models.md` gains the segmentation sections.

---

# Mozo v0.7.1: Decide What You Serve

## What's New in v0.7.1

A server no longer has to offer everything mozo publishes. `MOZO_ENABLE` narrows a deployment to
the models you choose, which matters because the weights are separate works with their own terms:
of the 63 published variants, 20 are AGPL-3.0, 2 are CC-BY-NC-4.0 and 1 carries Meta's SAM
License, and serving those over a network places obligations on you that serving the other 40
does not.

> **PyPI goes 0.6.0 → 0.7.1.** 0.7.0 was written up and committed but never tagged or
> released, so everything below ships here for the first time.

### 🔒 `MOZO_ENABLE`

One environment variable, alongside `MOZO_CACHE` and the rest. Name a family or a single variant,
comma-separated, mix the two freely:

```bash
MOZO_ENABLE=clip,siglip2/base-224 mozo start
```

Unset offers everything, so nothing changes for anyone who does not set it.

**An allow-list, not a deny-list.** A deny-list naming today's AGPL families would serve whatever
lands tomorrow, silently. An allow-list serves nothing it was not told to.

**A name that matches nothing warns rather than refusing to start.** An unrecognised token can
only ever subtract, so a typo yields a server missing a model, never one serving something
unsanctioned.

**A refusal never names a model the server declined to offer**, so a narrowed catalogue stays
narrowed at every wrong turn rather than answering it with a menu of what you configured it to
decline.

**The refusal costs nothing.** It is the first thing `/predict` and `/encode` do — before the
image decode, before the download — so a model excluded on licence grounds is never fetched, and
its weights never reach the disk.

**Server-side only.** `mozo.get_model()` is untouched.

For a permissively licensed demo, the README carries the line that offers the 40 Apache-2.0 and
MIT variants and nothing else. Depth Anything is named variant by variant because it is the one
family whose licence is not uniform — seven of its nine are Apache-2.0 and two are CC-BY-NC-4.0 —
which is why the variable takes `family/variant` at all.

### 🔢 SigLIP 2 Carries Five Variants

mozo registered fifteen fixed-resolution SigLIP 2 models and shipped weights for three, so twelve
of them answered a request with a 500. Now five are registered and five are published:
`base-224`, `base-256`, `so400m-384`, `so400m16-256` and `giant-384` — together 89% of the
downloads across those fifteen. The other ten need a checkpoint and no new code.

### 🐛 Fixes

**The test page said "Cannot reach the server" when a deployment offered no models.** It now says
which problem it actually is.

### 📊 Model Count

**Total Models:** 63 variants across 14 model families
**Growth:** +2 variants (SigLIP 2 3 → 5), no new families

Of the 63, **36 are Apache-2.0**, 20 are AGPL-3.0 (every YOLO variant), 4 are MIT (CLIP), 2 are
CC-BY-NC-4.0 (Depth Anything `base` and `large`), and 1 carries Meta's SAM License.

### ✅ Verification

**SigLIP 2: 160 comparisons across all five variants, every one bit-identical** — pixels, token
ids, image features, text features, logits and sigmoid, at exact equality with no tolerance.

### 📖 Documentation

The README's model catalogue is restructured into per-family tables — every variant, what it
returns, and what its weights are licensed under — and the licence section now says what to do
about the terms rather than only stating them.

---

# Mozo v0.6.0: Neither Box Nor Map

## What's New in v0.6.0

The first two families that answer with neither a box nor a map. Name the classes in words and
you get a score for each; ask for the vectors instead and you get the shared space the score was
computed in. No training, no labelled data, no fixed class list.

> **PyPI goes 0.4.0 → 0.6.0.** v0.5.0 was written up and committed but never tagged or
> released, so everything in its notes — Grounding DINO included — ships here for the first
> time.

### 🎉 New Model Families

**CLIP:** Zero-shot classification and the embeddings behind it, by OpenAI
- 4 variants: `base`, `base-16`, `large`, `large-336`, all Vision Transformers
- Phrases up to 77 tokens each; a score per phrase, or 512/768-d vectors
- MIT, code and weights

The five ResNet variants are not carried: they replace the Vision Transformer with a modified
ResNet and attention pooling, which is a second image tower rather than a second configuration.

**The two towers build and load separately.** An ingest job that only encodes images never
allocates the text tower, so a query service holds 63.4M parameters rather than 151.3M.

```python
model = mozo.get_model("clip/base")
scored = model.predict("aisle.jpg", ["a forklift", "a person", "an empty aisle"])
scored[0].class_name        # 'a forklift'
scored[0].confidence        # a cosine similarity -- see below
```

**Scores are cosine similarities, not probabilities.** They are not softmaxed, so they do not sum
to one, they may be negative, and adding a phrase does not move the others — which is exactly
what makes a threshold calibratable. They are also compressed: a good match sits far below 1.0,
so 0.31 does not mean "31% sure". Softmax it yourself if you want CLIP's published closed-set
behaviour.

**SigLIP 2:** The same question, answered as a probability, by Google
- 3 variants published: `base-224`, `so400m-384`, `giant-384`
- Phrases up to 64 tokens each; a probability per phrase, or 768/1152/1536-d vectors
- Multilingual
- Apache-2.0 throughout, ungated, on the code and on every checkpoint

**The score is what makes it worth having beside CLIP.** SigLIP was trained pair by pair with a
sigmoid loss and carries a learned bias as well as a temperature, so `predict` returns a
probability for one image and one phrase on its own. Asking about a single phrase is a well-posed
question, and every phrase can be near zero when none of them fits. A cosine similarity can
express neither. It is still not a calibrated class probability, and the docstring says so.

**No `transformers` dependency**, and none on `tokenizers`, `sentencepiece` or `safetensors`
either. The Gemma vocabulary ships in the wheel — Google publishes that same tokenizer ungated
under Apache-2.0 inside every SigLIP 2 repository, which the NOTICE explains, since it looks
alarming until checked.

### 🆕 New Endpoint: `/encode`

Some models represent an image and a phrase in one shared space, so a dot product between two
vectors says how well they match. That is what makes a corpus embedded once searchable by words
afterwards — but only through a vector database, which is yours. mozo produces the vectors and
stops there.

```bash
curl -X POST "http://localhost:8000/encode/clip/base" -F "file=@aisle.jpg"
curl -X POST "http://localhost:8000/encode/clip/base?text=a%20forklift&text=a%20person"
```

Send either images or phrases, not both: they are two towers and one call runs one of them.
Repeat the part or the parameter for a batch. The response carries `model`, `revision`, `dim` and
`embeddings`, and the revision is read off the loaded model rather than re-derived — a vector
stamped with the wrong revision is undetectable afterwards.

Refusals are answered before the image decode and before the download, so naming a family that
does not embed never costs a multi-gigabyte fetch.

### 🐛 Fixes

**An installed mozo raised the moment anything encoded a phrase.** SigLIP 2 reads its tokenizer
vocabulary from a file that was missing from the wheel. Every vendored package's LICENSE, NOTICE
and PROVENANCE were missing too, for all four YOLO extractions and SigLIP 2.

**CLIP rendered with no text box on the test page**, so the page could not drive it. The page had
kept a hand-written copy of which tasks take a prompt; it now reads that from `/models`.

### 📊 Model Count

**Total Models:** 61 variants across 14 model families
**Growth:** +7 variants (CLIP 4, SigLIP 2 3), +2 families

Of the 61, **34 are Apache-2.0**, 20 are AGPL-3.0 (every YOLO variant), 4 are MIT (CLIP), 2 are
CC-BY-NC-4.0 (Depth Anything `base` and `large`), and 1 carries Meta's SAM License. MIT is a
licence mozo had not shipped before.

### ✅ Verification

**CLIP: 104 comparisons across all four variants, every one bit-identical** against
`openai/CLIP` — the authors' own repository, which is what the published checkpoints reproduce.
Preprocessing, token ids, image features, text features, cosine similarities and
`logits_per_image`, at exact equality rather than a tolerance.

**SigLIP 2: 96 comparisons across the three published variants**, bit-identical at every stage —
pixels, token ids, image features, text features, logits and sigmoid.

### ⏱️ Benchmarks

SigLIP 2 against `transformers` — same checkpoint, same photograph, same device, so any gap is
the extraction rather than a comparison between models:

- **4% slower on CPU, 11% faster on MPS**, both from the same eager-attention choice, where the
  fused path is the slower one on Metal
- **Batching wins 1.6x at 224 and nothing at 384** — one image already saturates the device
- The text tower is a quarter of a classify call rather than a rounding error, because the
  context is always padded to 64

### 📖 Documentation

`docs/models.md` — detailed notes for all 14 families: what each one answers, what it costs, and
what its weights are licensed under.

---

# Mozo v0.5.0: Describe It In Words

## What's New in v0.5.0

### 🎉 New Model Family

**Grounding DINO:** Open-vocabulary detection by IDEA Research
- 2 variants: `tiny` (Swin-T, 48.4 box AP COCO zero-shot) and `base` (Swin-B, 56.7)
- Both checkpoints upstream publishes; there is no third
- Apache 2.0 code *and* weights

It answers the same question OWLv2 does, through the same endpoint and the same response shape,
so the two are substitutable. Reach for it when the prompt is a *description* rather than a noun:
OWLv2 embeds each phrase independently, while Grounding DINO fuses the text into the image
features six times over and lets the decoder attend back to the words, so `"the mug on the left"`
is read as a phrase rather than a bag of words.

```python
model = mozo.get_model("grounding_dino/tiny")
found = model.predict("kitchen.jpg", ["a kettle", "the mug on the left"])
found[0].class_name        # 'a kettle' -- the phrase you asked for
```

**No `transformers` dependency**, despite the BERT text encoder: the published checkpoint carries
its own fine-tuned BERT tower, and the WordPiece vocabulary ships in the wheel. Tokenizing a
prompt needs no network.

**Torch only, and that is structural.** The image is resized rather than letterboxed, so there is
no fixed input size and no graph to export. `runtime="auto"` will not offer one.

### 📊 Model Count

**Total Models:** 54 variants across 12 model families
**Growth:** +2 variants (Grounding DINO), +1 family

### ✅ Verification

138 comparisons against the authors' own implementation, both variants, exact equality with no
tolerance — preprocessing, token ids, the BERT tower, all three Swin levels, encoder memory and
each of the six decoder layers, not just the final boxes.

---

# Mozo v0.4.0: Real-Time Detection

## What's New in v0.4.0

### 🎉 New Model Family

**RF-DETR:** Real-time transformer-based object detection and instance segmentation by Roboflow
- 8 variants: nano, small, medium, large + seg-nano, seg-small, seg-medium, seg-large
- Pretrained COCO models with fine-tuned checkpoint support
- Apache 2.0 license

### 📊 Model Count

**Total Models:** 63 variants across 10 model families
**Growth:** +8 variants (RF-DETR), +1 family

### 🔧 Improvements

**Automatic device detection:** CPU/GPU (CUDA)/Apple MPS auto-selection integrated into
`get_model()` and ModelFactory

**Unified image loading:** all adapters accept both file paths and numpy arrays as input

**Simplified API:** a top-level `get_model()` for direct access with a shared ModelManager

### 🗑️ Removed

- **StabilityInpainting** adapter removed (cloud-based inpainting service discontinued)
- **Datamarkin** adapter removed (the hosted Vision Service is no longer available)
- **SAM3** adapter removed — it was never registered and therefore unreachable; it will
  return once implemented properly
