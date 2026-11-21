# Write your own workflow node

<!--
summary: Add a node to mozo's workflows by writing one Python function: its signature declares its ports and parameters, and the editor follows.
label: Write a node
keywords: mozo, workflow node, custom node, python, node decorator, workflow editor
-->

A node is an ordinary Python function with a decorator. Its signature is its declaration: the
annotations say which arguments are wired from other nodes and which are typed into the editor, and
the editor's palette, its forms and its type checks all follow from that. Adding a node is adding a
function.

## A first node

The built-in nodes can filter by class id but not by class name. This node fills the gap:

```python
from mozo.workflow import Detections, node


@node(category="Transform")
def keep_classes(detections: Detections, names: list[str] = ["person"]) -> Detections:
    """Keep only the detections whose class name is in the list."""
    kept = Detections()
    for detection in detections:
        if detection.class_name in names:
            kept.add_detection(detection)
    return kept
```

One rule splits the arguments: **an annotation that names a port type is an input; anything else is
a parameter.** `detections: Detections` is an input port, wired from another node. `names` is a
parameter, typed into the editor. The return annotation is the output port. The docstring's first
line is the description the editor shows, and `category` is the palette group it appears in.

Import the module before you load a workflow that uses the node, and it is registered: it appears in
`mozo.workflow.catalogue()`, in the editor's palette when the server imported it, and
`Workflow.load` accepts it. A node stays an ordinary function, so you can test it without a graph:

```python
import mozo
from my_nodes import keep_classes

found = mozo.get_model("rfdetr/nano").predict("example.jpg")
len(found), len(keep_classes(found, names=["cup"]))      # (13, 6)
```

## Ports and parameters

The port types are `Image`, `Detections`, `Classifications` and `Depth`, all from `mozo.workflow`.
A node with two outputs names them with `outputs=["image", "detections"]` in the decorator and
returns a tuple.

A node that turns one item into several, such as one crop per detection, returns a list and says
so with `-> list[Image]`. Every node after it then runs once per value. The catalogue marks that
output `"many": true`, so whoever builds a workflow can see that one item becomes many before
running it.

A parameter's annotation decides the editor's widget:

| Annotation | Widget |
|---|---|
| `int`, `float`, `str`, `bool` | a number, text or checkbox field |
| `int \| None` (or any `X \| None`) | the same, and it may be left unset |
| `Literal["a", "b"]` | a choice between those values |
| `list[str]` | a list, one entry per line |
| `Color` | a colour picker |
| `Source` | a file picker for the file a run reads: the one `--file` and `run(source=...)` replace |
| `FilePath` | a file picker for a file the workflow uses as saved, such as a checkpoint |
| `Prompts` | a list of phrases, with a promise about the results (below) |

All of these are importable from `mozo.workflow`, except `FilePath`, which is in
`mozo.workflow.node`.

**`Prompts` is a promise.** Annotate a list of phrases `Prompts` instead of `list[str]` only if
every result's `class_id` is the index into that list and its `class_name` is that phrase. That
promise is what lets `workflow.prompt_parameters` hand your node a caller's class list and read the
answers back like every built-in prompted node's.

## Load a model

A node that runs a model asks for `Models` by annotation and loads through it, so it uses whatever
models the run was given, a `ModelManager` of the caller's or mozo's shared cache. Mark its variant
parameter with `Weights` and the editor offers the family's variants as a choice, and
`workflow.check()` knows which model the node downloads:

```python
from typing import Annotated, Literal

from mozo import MODEL_REGISTRY
from mozo.workflow import Detections, Image, Weights, node
from mozo.workflow.node import Models

RFDETR = Annotated[Literal[tuple(MODEL_REGISTRY["rfdetr"]["variants"])], Weights("rfdetr")]


@node(category="Detect", exclusive=True)
def people(image: Image, models: Models, variant: RFDETR = "nano",
           threshold: float = 0.5) -> Detections:
    """RF-DETR, keeping only people."""
    found = models("rfdetr", variant).predict(image, threshold=threshold)
    return found.filter_by_class_id([1])
```

`exclusive=True` lets only one item inside the node at a time, for a node that holds a model or a
device. The editor never shows `models`: there is nothing for a person to type.

## Facts about the run

Some nodes need more than their inputs. Two more annotations, both from `mozo.workflow.node`, ask
the run for them:

- **`Context`** carries facts about the run and this item's place in it: `run.index`, `run.frame`,
  `run.time`, `run.fps`, `run.width`, `run.height`, `run.frames`. It is read-only. Values between
  nodes travel on wires, never here.
- **`State`** is a dict the node keeps for one run: a video writer's open file, a tracker, a running
  total. Each run gets a fresh one, so two runs never see each other's. Open resources on the first
  item, and register how to release them with `state.on_close(...)`.

```python
from mozo.workflow import Detections, node
from mozo.workflow.node import Context, State


@node(category="Track", ordered=True)
def count_tracks(detections: Detections, run: Context, state: State) -> Detections:
    """Count the distinct tracker ids seen so far, and log each new one with its time."""
    seen = state.setdefault("seen", set())
    for detection in detections:
        if detection.tracker_id is not None and detection.tracker_id not in seen:
            seen.add(detection.tracker_id)
            print(f"new object {detection.tracker_id} at {run.time:.2f}s (frame {run.frame})")
    return detections
```

After a `track` node on the street video from [Videos and folders](/docs/workflows/video.md), it
printed `new object 1 at 0.17s (frame 4)` and nine more lines like it.

`ordered=True` is for a node whose calls are a sequence rather than a set: a video writer, a
tracker, a running total. The node then sees one item at a time, in order, however many workers the
run uses. Asking for `State` implies `exclusive`.

## Ship nodes with your application

Nodes register when their module is imported. In an application that mounts the workflow routes
itself, import your node modules before you mount them, and they appear in the editor's palette
beside the built-in ones: the palette is the registry. [The editor page](/docs/workflows/editor.md#put-the-editor-in-your-own-app)
shows the mounting.

Two nodes cannot share a name: registering a second `keep_classes` raises an error rather than
replacing the first.
