# Workflows

A model on its own is rarely the job. Detect, then blur the faces, then save — that is a workflow.

mozo runs them as a graph of nodes in a JSON file. Draw it in the browser at `/workflow`, keep it in
version control, and run it from Python, the command line, or over HTTP. Same models, same process,
nothing extra to install.

## Running one

```python
from mozo.workflow import Workflow

workflow = Workflow.load("blur_faces.json")
results = workflow.run(source="street.jpg")

results[workflow.terminals[0]]      # what came out of the end
```

```bash
mozo run blur_faces.json --file street.jpg
```

`run()` gives back every node's output keyed by node id — an intermediate result is often the one
worth looking at, and hiding it would mean running the whole thing again to see it. `stream()`
reports each node as it starts and finishes, which is what the editor draws.

Overrides address a parameter by name: `run(source=..., threshold=0.6)`. A name more than one node
uses is refused rather than guessed at; address one node's as `node.parameter` instead, which is
how SAM 3 and Grounding DINO side by side get different thresholds:

```python
workflow.run(source="street.jpg", **{"sam.threshold": 0.5, "dino.threshold": 0.3})
```

```bash
mozo run compare.json --file street.jpg --set sam.threshold=0.5 --set dino.threshold=0.3
```

Over HTTP the `inputs` object takes the same keys. Giving one parameter two ways in one run —
`threshold` and `dino.threshold` — is refused rather than one silently winning.

### Where the models come from

Model nodes load through the shared cache behind `mozo.get_model` unless you say otherwise. An
application embedding mozo passes its own as `models=` to `run`, `stream`, `process` or
`run_many`: a `ModelManager` — its device choice, its unloading, its lifetime — or any callable
shaped like `get_model`, which is how progress and cancel reach a load that happens part way into
a run:

```python
from mozo import ModelManager

models = ModelManager()
workflow.run(source="street.jpg", models=models)
workflow.run(source="street.jpg", models=lambda family, variant, **kw: models.get_model(
    family, variant, progress=job.advance, cancelled=job.cancelled, **kw))
```

It reaches every node through the node's arguments, so a pipeline's worker threads use it too.
A node asks for it by annotation, as it asks for `State`: `models: Models`, then
`models("rfdetr", variant)`. The editor never shows it.

`rfdetr_checkpoint` runs an RF-DETR you trained: choose the checkpoint file, the architecture it
was trained as, and its class names, one per line in class-id order — or leave those empty to use
the names the checkpoint carries.

### Before a run

Every run first checks the files the workflow names — sources, checkpoints of your own — and
refuses to start if any are missing, listing all of them at once. A workflow shared from another
machine usually names several; fix the paths in the file and run it again. mozo's own models are
names, not paths: they come from the cache or are downloaded, and a run says which it will
download and how much before it starts. `workflow.check()` answers both without running.

`check().models` lists every published model the workflow loads, cached or not — node, family,
variant, the weights licence and the bytes still to fetch, 0 once cached — so a host can show
what a workflow needs and under which terms before anything is fetched. `check().downloads` is the
ones with bytes left. Pass the device the models will run on, `check(device="cuda")`, and nothing
imports torch; the sizes are for the runtime mozo picks on that device, so a host loading another
runtime asks `mozo.weights.download_size` with it.

### Prompts

The text-prompted nodes — `sam3`, `owlv2`, `grounding_dino`, `clip`, `siglip2` — take a list of
prompts, one per line in the editor, each exactly as written: `"bolt, M8"` is one prompt. Each
result's `class_id` is the index of the prompt that found it — or, for CLIP and SigLIP 2, of the
label it chose — and its `class_name` is that prompt. That holds for every one of them, so results
read back the same way whichever model produced them. A list with an empty entry is refused rather
than shortened, since shortening it would move every later prompt's `class_id`. A workflow saved
before prompts were a list holds a comma-separated string, and it still means what it meant.

`workflow.prompt_parameters` names every parameter that takes prompts, as the `node.parameter`
keys an override takes, so a host gives one list to every prompted node without knowing what any
node calls it:

```python
prompts = ["hard hat", "vest", "boot"]
results = workflow.run(source="site.jpg",
                       **{key: prompts for key in workflow.prompt_parameters})
```

Whether every node gets the same list is the caller's choice; give each its own by key instead.
Write prompts as short noun phrases — `"hard hat"`, not `"a photo of a hard hat"`. The locating
models find nothing for a caption: on the test photograph SAM 3 and Grounding DINO each found six
or more people for `"person"` and none for `"a photo of a person"`, while CLIP ranked both the same.

### Annotating a list of images

Three graphs cover the common jobs:

| Job | Graph |
|---|---|
| Find objects by name | `read_media` → `sam3` |
| Find boxes, then outline them | `read_media` → `grounding_dino` → `edgetam_boxes` (or `sam2_boxes`, `sam3_boxes`) |
| Give existing boxes masks | `read_media` + `read_annotations` → `edgetam_boxes` |

`sam2_boxes`, `edgetam_boxes` and `sam3_boxes` add a mask to every box they are given, in one
call per image, and keep every row as it came: same order, same class, name, score and box. A row
whose mask comes back empty keeps its box with an empty mask. An image with no boxes never runs the
model. `sam3_boxes` after a `sam3` node on the same image reuses that node's image encode.

`read_annotations` reads back the file `save_annotations` writes, matching each line to the image
by its file name and checking the line was written for an image of that size. Feed it files — a
folder, or `run_many` over paths — not pixels in memory, which have no name to match. Names must be
unique within one file — the name has no extension, so `a.jpg` and `a.png` clash — and two lines
for one name are refused. To write the file from your own code,
use `mozo.workflow.annotation_line(detections, label=..., width=..., height=...)` — the function
`save_annotations` itself writes through — so your file and mozo's cannot drift apart.

Stop a run by closing its iterator, or leaving the `for` loop: no further item starts, a model
call already running finishes, and the run's threads exit right after it.

## What a workflow is

```json
{
  "name": "Detect street objects",
  "description": "Segment everything YOLO knows in one photo.",
  "nodes": [
    {"id": "load-1", "type": "read_media",
     "position": {"x": 0, "y": 0},
     "data": {"parameters": {"source": "street.jpg"}}},
    {"id": "detect-1", "type": "yolov26",
     "data": {"parameters": {"variant": "seg-nano", "threshold": 0.5}}}
  ],
  "edges": [
    {"source": "load-1", "sourceHandle": "image",
     "target": "detect-1", "targetHandle": "image"}
  ]
}
```

`name` and `description` are optional: they are what a person choosing a workflow reads, the
editor shows and edits them, and a file without them loads as it always did. They are written back
only when set, so saving an old file changes nothing in it.

A workflow either is valid or does not exist. Loading one checks that every connection names ports
that exist, that the types on both ends agree, that every input is fed exactly once, and that the
graph is acyclic — so a workflow that loaded will not fail for a structural reason halfway through a
run with a model already on the GPU. Every way a file can fail to be a workflow raises one error,
`mozo.workflow.WorkflowError` — a `ValueError` — with a message that says which: an unknown node
type names the ones there are. A missing file stays `FileNotFoundError`, and an override that names
nothing stays a `KeyError`: that is the caller's mistake, not the file's.

## Nodes

A node is an ordinary Python function. Its signature is its declaration:

```python
@node(category="Annotate")
def draw_boxes(image: Image, detections: Detections,
               thickness: int | None = None, color: Color | None = None) -> Image:
    """Draw a box around each detection."""
    return pf.annotate.box(image.copy(), detections, thickness=thickness, colors=_colors(color))
```

One rule splits the two kinds of argument: **an annotation that names a port type is an input;
anything else is a parameter.** Inputs are wired from other nodes; parameters are typed in. The
widget the editor draws comes from the annotation too — `int`, `float`, `str`, `bool`, `Color`,
`list[str]`, and `Literal[...]` for a choice. `int | None` means the parameter may be left unset, which is how a
thickness scales itself to the image.

The catalogue the editor reads is generated from these declarations, so it cannot drift from what
runs. Adding a node is adding a function.

A node that loads a published model says which on its variant parameter, `variant:
variants("edgetam") = "edgetam"`: the editor offers the variants as a dropdown, and `check()` reads
the family from it — which is how `edgetam_boxes` is known to download EdgeTAM. A choice of
architecture for a checkpoint of your own is `architectures("rfdetr")` instead: nothing is
downloaded for it.

A node whose results are numbered by a list of descriptions annotates that list `Prompts` instead
of `list[str]`, and reads it with `mozo.text.prompt_list`. The mark is a promise about the output:
**each result's `class_id` is the index into this list, and its `class_name` that entry.** Mark a
node only if it keeps it — that is what lets `prompt_parameters` hand it a host's class list and
read the answers back like any other prompted node's. The catalogue flags the parameter with
`"prompts": true`.

## Port types

A connection is only allowed between ports of the same type.

| | |
|---|---|
| `IMAGE` | `HxWx3` RGB `uint8` — mozo's image contract |
| `DETECTIONS` | PixelFlow `Detections`: boxes, masks, keypoints, read text |
| `CLASSIFICATIONS` | PixelFlow `Classifications` — scores with no boxes |
| `DEPTH` | a float depth map with a range |
| `EMBEDDING` | an `NxD` float array |

`CLASSIFICATIONS` is separate from `DETECTIONS` although both are PixelFlow types, because
`Classifications` has no boxes at all — sharing a port type would let the editor offer a connection
that fails the moment it runs. `DEPTH` is separate from `IMAGE` because a depth map is a
measurement: flattened to eight bits it is a picture, and the metric Depth Anything variants lose
the only thing that distinguishes them from the relative ones.

## Batching

A node written for one image runs over fifty without knowing. A list arriving on an input fans the
node out and collects the results, and that lives in the engine so every node gets it from one
implementation. `crop_around_detections` turns one image into one per detection, which every node
downstream then runs over.

Lists of different lengths are refused rather than padded.

## Over HTTP

```
GET  /workflow            the editor
GET  /workflow/nodes      the catalogue
POST /workflow/validate   build it, and say why not
POST /workflow/run        run it
POST /workflow/stream     run it, one server-sent event per node
```

`run` and `stream` take the document as a form field, an optional uploaded image, and `include`,
which is `terminals` by default. Sending every node's output means encoding every intermediate: on
one 1920×1281 photograph a five-node workflow takes 27 ms to run and 347 ms to encode. The editor
asks for `all`; a batch job over full-resolution frames should not have to.

## What is not there

- **Tracking.** Stateful across frames, and mozo has no concept of a video.
- **Zones**, and clicking for **SAM 2** / **EdgeTAM** / **SAM 3**. Both are pixel coordinates, and
  the editor has no way yet to express a click. Boxes already work: `sam2_boxes`, `edgetam_boxes`
  and `sam3_boxes` segment the boxes another node found. A tool that clicks calls the model
  directly.
