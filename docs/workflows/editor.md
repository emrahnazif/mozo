# Build a workflow in the editor

<!--
summary: Chain models and image operations into a workflow: draw it in mozo's browser editor, keep it as JSON, and embed the editor in your own app.
label: Editor
keywords: mozo, workflow, workflow editor, node graph, pipeline, computer vision pipeline
-->

A model on its own is rarely the job. Find the faces, blur them, save the picture: that is a
workflow. In mozo a workflow is a graph of nodes saved as a JSON file. You draw it in the browser,
keep it in version control, and run it from Python, the command line or over HTTP, with the same
models and nothing extra to install.

![The mozo workflow editor: a graph of nodes on a canvas, with a node's settings on the right](/docs/assets/workflow.png)

## Open the editor

```bash
mozo start
```

Then open `http://localhost:8000/workflow`. The panel on the right lists every node; drag one
onto the canvas and its inputs are connected to the latest node that gives what they take, the
same rule a workflow file follows (below). Drag a wire to connect something else. Click a node and
the panel shows its parameters instead; **← All nodes** goes back. The panel never covers the
canvas.

**Test** runs the graph on one image and shows every node's output; **Run** runs it over everything
its source produces, such as every frame of a video. **Tidy up** lays out every node again, top to
bottom along the connections. **Export** saves the workflow as a JSON file, and **Import** opens
one.

[Nodes](/docs/workflows/nodes.md) lists all 50 nodes with their inputs, outputs and parameters.

## What a workflow is

This workflow finds faces with Grounding DINO, blurs them, and saves the result:

```json
{
  "name": "Blur faces",
  "description": "Find every face and blur it.",
  "nodes": [
    {"id": "read", "type": "read_media", "source": "example.jpg"},
    {"id": "find", "type": "grounding_dino", "text": ["face"], "variant": "tiny", "threshold": 0.3},
    {"id": "blur", "type": "blur_regions"},
    {"id": "save", "type": "save_image", "path": "blurred"}
  ]
}
```

```bash
mozo run blur_faces.json
```

![The example photograph with four faces blurred; the fifth person, mostly hidden behind a cup, is not](/docs/assets/blur-faces.jpg)

One node per line: its `id`, its `type` (the node's name) and its parameters. There are no
connections in it, because **an input that is not written takes the latest earlier node in the list
with an output of its type.** `find` takes `read`'s image, and `blur` takes `find`'s detections and
`read`'s image, since `find` gives detections and no image. One output can feed any number of
nodes.

A node writes an input only when it wants something other than the latest, under `inputs`:

```json
{"id": "draw", "type": "draw_boxes", "inputs": {"image": "read"}}
```

The value is a node's id, or `node.port` when that node has two outputs of the type, such as
`flip.detections`. `null` means nothing is connected yet, which the editor writes for a half-built
graph and which does not run.

- **What is refused:** a parameter the node does not have, an input it does not take, and an input
  the rule cannot settle (nothing earlier gives the type, or the latest such node gives two). Each
  says which node and what to write.
- **`xy`** is where the editor draws the node, `[x, y]`. It is optional: the editor lays out nodes
  that have none, so a workflow written in code or by hand opens ready to edit.
- **`name` and `description`** are optional, and are what someone choosing between workflows reads.

Saving, from the editor or with `Workflow.save`, writes this form: each node after the nodes it
takes from, keeping the order otherwise, and only the inputs the rule would get wrong. Opening what
was saved gives the same graph. A file with `edges` is read the way mozo 1.12 and earlier wrote it,
each edge naming both ports, so older files keep working.

## Connections are typed

Every port has a type, and a connection is only allowed between ports of the same type. The editor
refuses a wrong connection before anything runs.

| Type | Carries |
|---|---|
| `image` | an `HxWx3` RGB `uint8` array (`HxWx4` where a node takes an alpha channel) |
| `detections` | a PixelFlow `Detections`: boxes, masks, keypoints, read text |
| `classifications` | a PixelFlow `Classifications`: scores with no boxes |
| `depth` | a float depth map with its range |

`classifications` is separate from `detections` because a `Classifications` has no boxes: one type
for both would let the editor offer a connection that fails as soon as it runs. `depth` is separate
from `image` because a depth map is a measurement, and flattening it to eight bits would throw the
metres away.

A workflow is either valid or it does not load. Loading checks that every connection names ports
that exist, that the types on both ends agree, that every input is fed exactly once, and that the
graph has no cycle. So a workflow that loaded does not fail halfway through for a structural reason,
with a model already on the GPU. Every way a file can fail raises one error,
`mozo.workflow.WorkflowError`, with a message that says what is wrong:

```text
find.detections is detections, but blur.image takes image
```

**One node runs over many images by itself.** When a list arrives on an input, the node runs once
per item and the results are collected. `crop_around_detections` turns one image into one per
detection, and every node after it then runs on each crop. Lists of different lengths are refused,
not padded.

## Put the editor in your own app

The workflow routes come from `workflow_router()`. `mozo start` mounts one whose model loads are held
to `MOZO_ENABLE`. Your own FastAPI application can mount its own, under any prefix, with your own way
of loading models:

```python
from fastapi import FastAPI
from mozo import ModelManager
from mozo.workflow.api import workflow_router

app = FastAPI()
manager = ModelManager()

app.include_router(workflow_router(models=manager.get_model, process=False), prefix="/mozo")
```

- `models` is how model nodes load their models: a `ModelManager`'s `get_model`, or any function
  shaped like it. Without it, nodes load through mozo's shared cache, a second copy beside yours.
- `process=False` leaves out whole-source runs, for an application that runs workflows as its own
  jobs. The editor then offers **Test** and not **Run**.

The editor is then at `/mozo/workflow`. It finds its files and its API from where it was served, so
the prefix needs no configuration. `/mozo/workflow?toolbar=0` hides its toolbar.

To put the editor inside a page of your own, mount it into an element:

```html
<div id="editor" style="height: 600px"></div>
<script type="module">
  import { mount } from "/mozo/workflow/assets/editor.js";

  const editor = mount(document.querySelector("#editor"), { document: saved, toolbar: false });
  editor.on("change", () => markUnsaved());
  saveButton.onclick = () => save(editor.getDocument());
  testButton.onclick = async () => show(await editor.test(file));
</script>
```

| Call | What it does |
|---|---|
| `mount(element, {document, toolbar, base, panel})` | Draws an editor filling `element`. `document` opens a workflow, `toolbar: false` leaves the buttons to you, `base` points at a router mounted elsewhere, `panel` draws the panel into an element of yours (below). |
| `getDocument()` / `setDocument(document)` | Reads the workflow on the canvas, in the short form, or replaces it with one in either form, fitting the view to it. Setting one is not a change. A connection to a missing node or port is left out, and the editor says which. |
| `on("change", listener)` | Calls `listener` with the document whenever the person changes it, once a drag ends rather than during it. Returns a function that stops listening. |
| `test(file)` | Tests on a file, or on the one chosen on the input node. Resolves to every node's output. |
| `destroy()` | Removes the editor from the page. |

To put the panel in a column of your own, beside a chat say, give `mount` the element:

```js
mount(document.querySelector("#canvas"), { toolbar: false, panel: document.querySelector("#side") });
```

The panel fills that element and the canvas fills the first, so neither covers the other.

The editor mounts in a shadow root, so its styles cannot reach your page and yours cannot reach it.
The panel gets a shadow root of its own, restyled by the same variables set on its element.
Restyle it through its CSS variables, set on the element you mounted it into: `--mz-color-bg`,
`--mz-color-text`, `--mz-font-sans`, `--mz-radius-md` and the rest in mozo's
`ui/src/styles/editor.css`. `color-scheme: dark` on that element turns it dark. Nodes your
application registers appear in its palette, because the palette is the node registry.

## What it cannot do yet

- **Zones**, and **clicks** for SAM 2, EdgeTAM, HQ-SAM and SAM 3. Both are pixel coordinates, and the editor
  has no way to draw one yet. Boxes work: `sam2_boxes`, `edgetam_boxes`, `sam_hq_boxes` and `sam3_boxes` add masks
  to the boxes another node found. For clicks, call the model from
  [Python](/docs/models/sam2.md).
