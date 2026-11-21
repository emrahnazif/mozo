# Run a workflow from Python or the command line

<!--
summary: Run a mozo workflow file from Python, the command line or HTTP: override parameters, run it over many images, check it first, and use your own models.
label: Run a workflow
keywords: mozo, workflow, run workflow, batch processing, command line, python
-->

A workflow file runs without the editor. These examples use `blur_faces.json` from
[the editor page](/docs/workflows/editor.md#what-a-workflow-is), which finds faces with Grounding
DINO, blurs them and saves the picture.

## Run it once

```python
from mozo.workflow import Workflow

workflow = Workflow.load("blur_faces.json")
results = workflow.run(source="example.jpg")

len(results["find"])        # 5: the faces Grounding DINO found
```

`run()` returns every node's output, keyed by node id, not only the last one: an intermediate result
is often the one worth looking at. `workflow.terminals` names the nodes at the ends. A node that
failed is missing from the results, with everything after it.

From the command line:

```bash
mozo run blur_faces.json --file example.jpg
```

```text
Blur faces -- Find every face and blur it.
  1 item in 2.8s (0/s)
```

`--file` runs on another image or video than the one saved in the workflow. `--test` runs one item
and describes what every node produced.

## Change parameters for one run

Name a parameter to use another value than the one saved in the file:

```python
results = workflow.run(source="example.jpg", threshold=0.5)
```

```bash
mozo run blur_faces.json --file example.jpg --set threshold=0.5
```

When two nodes have a parameter of the same name, a bare name is refused rather than guessed at.
Address one node's parameter as `node.parameter` instead, which is how two models side by side get
different thresholds:

```python
workflow.run(source="example.jpg", **{"find.threshold": 0.5})
```

```bash
mozo run blur_faces.json --set find.threshold=0.5
```

A name that matches nothing raises a `KeyError` listing the parameters there are. Giving one
parameter two ways in one run, `threshold` and `find.threshold`, is refused too.

`workflow.prompt_parameters` names every parameter that takes phrases, as `node.parameter` keys
(here `('find.text',)`). Use it to give your own class list to every prompted node without knowing
what each node calls it:

```python
prompts = ["face", "licence plate"]
results = workflow.run(source="example.jpg",
                       **{key: prompts for key in workflow.prompt_parameters})
```

## Run it over many images

`run_many` runs the workflow once per item and yields `(item, results)` as each one finishes:

```python
from pathlib import Path

for path, results in workflow.run_many(sorted(Path("photos").glob("*.jpg"))):
    print(path, len(results["find"]))
```

Items are read lazily, so a folder of a million images costs no more memory than a handful. Each
item is bound to the workflow's file input (`read.source` here); pass `over=` to bind another
parameter. By default two items are in flight at once, as a pipeline: the model works on one image
while the next is already being decoded. `workers=1` runs them strictly one after another.

A failure stops the run. To skip the item instead, pass `on_error`:

```python
def skip(item, event):
    print(f"{item}: {event.node} failed: {event.error}")

for path, results in workflow.run_many(paths, on_error=skip):
    ...
```

To stop early, leave the `for` loop. No further item starts, a model call already running finishes,
and the run's threads exit.

For a video, or anything else whose source decides how many items there are, use `process()`:
[Videos and folders](/docs/workflows/video.md) covers it.

## Check it before it runs

```python
check = workflow.check(device="cuda")
check.missing       # files the workflow names that are not there
check.models        # every model it loads: node, family, variant, licence, bytes still to download
check.downloads     # the models with bytes still to download
```

Every run does this first. A run refuses to start if a file the workflow names is missing, listing
all of them at once, and it logs which models it will download and how much. `check()` answers
both without running anything, and with `device` given it does not even import torch: a host can
show what a workflow needs, and under which licences, before anything is fetched.

`stream()` runs the workflow and reports each node as it starts and finishes, which is how the
editor draws progress:

```python
for event in workflow.stream(source="example.jpg"):
    print(event.node, event.status)      # read running, read completed, find running, ...
```

A failure yields one `failed` event naming the node and the error, and ends the stream.

## Use your own models

Model nodes load through the shared cache behind `mozo.get_model` unless you say otherwise. Pass
`models=` to any run method to use your own `ModelManager`, with its device choice and lifetime:

```python
from mozo import ModelManager

models = ModelManager()
workflow.run(source="example.jpg", models=models)
```

Any function shaped like `get_model` works too, which is how download progress and cancellation
reach a model that loads part way into a run:

```python
workflow.run(source="example.jpg", models=lambda family, variant, **kwargs: models.get_model(
    family, variant, progress=show_progress, cancelled=job.cancelled, **kwargs))
```

Over HTTP, a running server runs workflows at `POST /workflow/run`, with the document as the form
field `workflow`, overrides as a JSON object in `inputs`, and an optional `file`:

```bash
curl -X POST "http://localhost:8000/workflow/run" \
  -F "workflow=<blur_faces.json" -F 'inputs={"threshold": 0.4}' -F "file=@example.jpg"
```

```json
{"results": {"save": null}, "terminals": ["save"]}
```

[HTTP API](/docs/reference/http-api.md) lists the other workflow routes: validate, stream and
process.
