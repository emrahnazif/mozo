# Workflow API reference: mozo.workflow

<!--
summary: Every public name in mozo.workflow: the Workflow class and its run methods, results and errors, the node decorator, port and parameter types.
label: Workflow API
keywords: mozo, workflow api, reference, workflow, node decorator, pipeline
-->

Everything `from mozo.workflow import ...` gives you, plus the node-author types in
`mozo.workflow.node` and the router in `mozo.workflow.api`. The guides are
[Run a workflow](/docs/workflows/python.md) and [Write a node](/docs/workflows/writing-nodes.md).

## Workflow

A loaded workflow: a validated graph of nodes, ready to run. Build it with `Workflow.load` or
`Workflow.from_dict`; a document that is not a valid workflow raises `WorkflowError`.

### Workflow.load

```python
Workflow.load(path)
```

Read and validate a workflow file. Raises `FileNotFoundError` for a missing file and
`WorkflowError` for a file that is not a valid workflow.

### Workflow.from_dict

```python
Workflow.from_dict(data)
```

Validate a workflow document already parsed from JSON, in either form: the short form (nodes only,
inputs written where the latest earlier output of their type is not the one wanted) or the full
form with `edges`. [What a workflow is](/docs/workflows/editor.md#what-a-workflow-is) describes
both.

### Workflow.to_dict

```python
workflow.to_dict()
```

The workflow as a short document, the inverse of `from_dict`: each node after the nodes it takes
from, and only the inputs the rule would get wrong. `workflow.connections` has every edge, for code
that wants them.

### Workflow.save

```python
workflow.save(path)
```

Write the workflow to a JSON file in the short form, one node per line, so a text edit can
replace one node.

### Workflow.run

```python
workflow.run(*, models=None, **overrides)
```

Run the workflow once and return `{node id: output}` for every node. A node that failed is missing,
with everything after it; use `stream` to see why.

| Parameter | Description |
|---|---|
| `models` | Where model nodes get their models: a `ModelManager`, or any callable shaped like `get_model`. `None` is the shared cache behind `mozo.get_model`. The same on every run method. |
| `**overrides` | Parameter values to use instead of the saved ones, by name, or as `"node.parameter"` for one node's. |

Raises `KeyError` if an override names no parameter, or a name more than one node has; the message
lists the keys that would say which.

### Workflow.stream

```python
workflow.stream(*, models=None, **overrides)
```

Run the workflow and yield an [Event](#event) before and after each node. A failure yields one
`"failed"` event and ends the stream. Overrides are checked when `stream` is called, before the
first event.

### Workflow.run_many

```python
workflow.run_many(items, *, over=None, on_error=None, workers=2, model_workers=1,
                  stats=None, models=None, **overrides)
```

Run the workflow once per item and yield `(item, results)` in the order the items arrived. `items`
is read lazily and may be a generator that never ends.

| Parameter | Default | Description |
|---|---|---|
| `items` | | What to run on: paths, bytes or arrays, or any iterable of them. |
| `over` | `None` | The parameter each item is bound to. `None` is the workflow's file input, `file_parameter`. |
| `on_error` | `None` | Called as `on_error(item, event)` with the failing event; that item is skipped and the run continues. `None` raises and stops the run. |
| `workers` | `2` | Items in flight at once, as a staged pipeline. `1` runs strictly one after another. |
| `model_workers` | `1` | Items inside one model node at once. More can help a small model on a large device; it multiplies activation memory. |
| `stats` | `None` | A dict to fill with per-node timing: `elapsed_s`, and per stage its `width`, `busy_s`, `saturation` and queue sizes. For tuning `workers`. |

An item whose graph failed part way is never yielded with partial results. Stop the run by closing
the iterator, or leaving the `for` loop: no item starts a node after that, and a call already
running finishes.

### Workflow.process

```python
workflow.process(*, workers=2, on_error=None, model_workers=1, stats=None, models=None,
                 context=False, **overrides)
```

Run once per item the workflow's source produces, such as every frame of a video, and yield
`(item, results)` as each finishes. With `context=True`, yield `(item, results, run)`, where `run`
is the item's [Context](#context). The other arguments are as for `run_many`.

### Workflow.check

```python
workflow.check(*, device=None, **overrides)
```

What the workflow needs that is not here, without running anything. Returns a [Preflight](#preflight).
With `device` given, nothing imports torch: sizes are for the runtime mozo picks on that device.

### Workflow.parameter

```python
workflow.parameter(key)
```

Which node's parameter an override key sets, as `(node id, parameter)`. Raises `KeyError` as `run`
does.

### Workflow properties

| Property | Description |
|---|---|
| `terminals` | Ids of the nodes nothing reads from, where a run's answers end up. |
| `source` | The id of the node a run's items come from, or `None`. |
| `file_parameter` | The parameter a file given to the run binds to, as a `"node.parameter"` key. |
| `parameters` | Every parameter, as `{name: [node id, ...]}`. |
| `prompt_parameters` | Every parameter that takes phrases, as `"node.parameter"` keys. |

## Event

One thing that happened during a run, yielded by `stream`. Fields: `node` (the node id), `status`
(`"running"`, `"completed"` or `"failed"`), `output` (the node's output once completed) and `error`
(the message, when failed).

## Preflight

What `check` returns. `missing` is `(node id, parameter, path)` for every file the workflow names
that is not there; a run refuses to start while it is not empty. `models` is a [ModelUse](#modeluse)
for every published model the workflow loads, cached or not. `downloads` is the ones with bytes
still to download.

## ModelUse

One published model a node loads: `node`, `family`, `variant`, `license`, and `size`, the bytes
still to download (0 once cached).

## WorkflowError

Raised for any document that is not a valid workflow: an unknown node type, a missing port, a type
mismatch between connected ports, an input fed twice or not at all, or a cycle. A subclass of
`ValueError`. The message says which, for example
`find.detections is detections, but blur.image takes image`.

## node

```python
@node(*, category, outputs=None, ordered=False, exclusive=False, alpha=False)
```

Register the decorated function as a node, and return the function unchanged.

| Parameter | Description |
|---|---|
| `category` | The palette group, such as `"Transform"` or `"Detect"`. |
| `outputs` | Names for the output ports, in order, for a node that returns a tuple. |
| `ordered` | The node's calls are a sequence: one item at a time, in arrival order. A video writer, a tracker. |
| `exclusive` | One item inside the node at a time, for a node holding a model or a device. Implied by `ordered`, and by asking for `State`. |
| `alpha` | The node accepts an image with an alpha channel. Others are handed three channels. |

Raises `ValueError` if a node of that name is already registered.

## catalogue

```python
mozo.workflow.catalogue()
```

Every registered node as the editor reads it: name, category, description, inputs, outputs, and
parameters with their kinds and defaults. `GET /workflow/nodes` serves the same list.

## get

```python
mozo.workflow.get(name)
```

The [NodeSpec](#nodespec) of one registered node. Raises a `KeyError` naming the nodes there are if
there is none.

## names

```python
mozo.workflow.names()
```

Every registered node name, in registration order.

## annotation_line

```python
mozo.workflow.annotation_line(detections, *, label, width, height, index=0, time=None, frame=None)
```

One line of an annotations file, exactly as the `save_annotations` node writes it, so a file you
write and one mozo writes cannot drift apart.

## NodeSpec

Everything about one kind of node, read from its function: `name`, `category`, `description`,
`inputs`, `outputs`, `parameters`, and its flags. Returned by `get`.

## Port types

The annotations that make an argument an input port, or a return value an output port:

| Name | Wire name | Carries |
|---|---|---|
| `Image` | `image` | an `HxWx3` RGB `uint8` array, `HxWx4` for a node that sets `alpha` |
| `Detections` | `detections` | a PixelFlow `Detections` |
| `Classifications` | `classifications` | a PixelFlow `Classifications` |
| `Depth` | `depth` | a float depth map with its range |
| `Embedding` | `embedding` | an `NxD` float array. No built-in node produces one yet. |
| `PortType` | | the enum of the wire names above |

## Parameter types

| Name | Marks a parameter as |
|---|---|
| `Color` | a colour, given as a hex string such as `"#FF0000"`. |
| `Source` | the file a run reads, the one `--file` and `run(source=...)` replace. |
| `Prompts` | a list of phrases, where each result's `class_id` indexes the list. |
| `Weights` | a choice of variant of a published family: `Annotated[Literal[...], Weights("rfdetr")]`. |

`FilePath`, in `mozo.workflow.node`, marks a file the workflow uses as saved, such as a checkpoint.

## Context

```python
from mozo.workflow.node import Context
```

Facts about the run and the item's place in it, read-only, for a node that annotates an argument
with it or a `process(context=True)` caller.

| Member | Description |
|---|---|
| `index` | The item's place in this run, from 0. |
| `frame` | Which frame of the file it is, from 0, or `None` for an image or a folder. |
| `time` | That frame's time in the file, in seconds. |
| `fps` | The rate the source yields, after `stride` or `rate`, or `None`. |
| `width`, `height` | The source's frame size. |
| `frames` | How many items the source expects, or `None` for a live stream. |
| `is_live` | Whether the source is a live stream. |
| `name` | The source's name. |
| `label` | A name for this item, for a file named after it: a folder's items by their files, a video's frames by the source and their place in it. |

## State

```python
from mozo.workflow.node import State
```

A dict a node keeps for one run, asked for by annotation. Each run gets a fresh one.
`state.on_close(function)` registers how to release something the node opened.

## Models

```python
from mozo.workflow.node import Models
```

Asked for by annotation: call it as `models(family, variant, **kwargs)` to load a model the way the
run was told to.

## workflow_router

```python
from mozo.workflow.api import workflow_router

workflow_router(*, models=None, process=True)
```

The workflow routes and the editor, as a FastAPI `APIRouter` at `/workflow`, to mount under any
prefix. `models` is as for `Workflow.run`. `process=False` leaves out `/process`, and the editor
then offers Test alone. [The editor page](/docs/workflows/editor.md#put-the-editor-in-your-own-app)
shows it in use.
