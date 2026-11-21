# HTTP API reference

<!--
summary: Every route mozo's server offers, with its parameters, what it returns, and its errors: predict, encode, prepare, models and workflows.
label: HTTP API
keywords: mozo, http api, rest api, reference, predict, encode
-->

Every route `mozo start` serves. The guide is [HTTP server](/docs/server.md); each model's page
says which of these parameters it takes. A running server also serves this as an OpenAPI document
at `/docs`.

Errors are JSON, `{"detail": "..."}`, with a message that says what to do:

| Status | Meaning |
|---|---|
| `400` | The request is wrong: not an image, a required prompt missing, a parameter out of range or of the wrong kind for this model. |
| `404` | No such model on this server, or no image held for a `?sha256=`. The detail lists what is served. |
| `500` | The model failed to load or to run. |
| `501` | The model has no such route, such as `/predict` on a text-embedding model or `/encode` on a detector. |

## POST /predict/{family}/{variant}

Run a model on one image. The image is a multipart upload named `file`; everything else is a query
parameter. Repeat a parameter to give several values.

| Parameter | For | Description |
|---|---|---|
| `file` | all | The image. Leave it out only when naming a prepared image by `sha256`. |
| `threshold` | detectors, prompted models | Confidence floor. Omitted, the family's own default applies. |
| `labels` | detectors | Comma-separated class names replacing the model's own, such as `hardhat,vest`. |
| `text` | prompted models | A phrase to look for. Repeat it for several: `?text=car&text=person`. Not comma-separated, so a phrase may contain a comma. |
| `point` | SAM 2, EdgeTAM, SAM 3 | A click as `x,y` in the image's pixels. Each needs a `label`. On SAM 3, a click or `text`, not both. |
| `label` | SAM 2, EdgeTAM, SAM 3 | `1` to include the matching point, `0` to exclude it. One per point, in order. No default. |
| `box` | SAM 2, EdgeTAM, SAM 3, ViTPose, Moebius | A box as `x1,y1,x2,y2`. The segmenters segment it, ViTPose finds the joints of the person in it, Moebius removes it. |
| `name` | SAM 2, EdgeTAM, SAM 3 | What to call what you pointed at. Omitted, `class_name` is `null`. |
| `multimask` | SAM 2, EdgeTAM, SAM 3 | `true` (default) for three candidate masks ranked by predicted IoU, `false` for one. |
| `seed` | Moebius | Which sample to draw. Default `0`. |
| `sha256` | SAM 2, EdgeTAM, SAM 3, OWLv2 | The sha256 of an image this server holds from `/prepare`. Send it instead of `file`. |

Responses by task:

| Models | Response |
|---|---|
| Detection, segmentation, keypoints, detection from words, click segmentation, OCR | `application/json`: a list of detections with every PixelFlow field. |
| CLIP, SigLIP 2 | `application/json`: a list of `{class_id, class_name, confidence, metadata}`. |
| Depth Anything V2 | `image/png`, 16-bit. Headers `x-depth-unit` (`metres` or `none`), `x-depth-min` and `x-depth-max`: `depth = min + png / 65535 * (max - min)`. |
| BEN2 | `image/png`: the photograph as RGBA, background transparent. |
| Moebius | `image/png`: the photograph with every box painted out. |

A prepared image not held by this server (another worker, a restart, or dropped to stay within
`MOZO_PREPARED_MB`) answers `404`. Send the file again with the same `sha256` and it is kept for the
requests after.

## POST /encode/{family}/{variant}

Return vectors instead of an answer. For CLIP, SigLIP 2 and Granite Embedding R2.

| Parameter | Description |
|---|---|
| `file` | An image to embed. Repeat for several. CLIP and SigLIP 2. |
| `text` | A phrase or text to embed, as a query parameter. Repeat for several. |

Images or text in one call, not both. Returns:

```json
{"model": "clip/base", "revision": "2026-08-23", "dim": 512, "embeddings": [[-0.00398, -0.00317, ...]]}
```

`embeddings` has one row per input, in order, each L2-normalised. Keep `model` and `revision` with
stored vectors: a vector only compares with vectors from the same weights.

## POST /prepare/{family}/{variant}

Encode an image ahead of its prompts and keep it, for SAM 2, EdgeTAM, SAM 3 and OWLv2. Send the
image as `file`. Returns `{"sha256": "..."}`, the hash of the file's bytes; later `/predict`
requests send `?sha256=` instead of the file. Memory for kept images is limited by
`MOZO_PREPARED_MB`, 512 MB by default, and the least recently used is dropped first.

## GET /

Health: `{"status": "ok", "version": "1.11.0", "loaded_models": [...]}`.

## GET /models

Every model this server offers, as `{family: entry}`, without loading anything. An entry has:

| Field | Description |
|---|---|
| `task_type` | Such as `object_detection` or `promptable_segmentation`. |
| `description` | What the family does. |
| `variants` | The variants this server offers. |
| `licenses` | `{variant: licence}`. |
| `prompted` | Whether `/predict` needs `text`. |
| `boxed` | Whether it takes boxes as input rather than finding them (ViTPose, Moebius). |
| `clicks` | Whether it takes `point` and `label`. |
| `prepares` | Whether `/prepare` and `sha256` work for it. |
| `encodes` | What `/encode` takes for it: `["image", "text"]`, `["text"]`, or `[]`. |

## GET /models/loaded

The models in memory now: `{"models": ["rfdetr/nano", ...]}`.

## GET /test-ui

A browser page to try any model on an image.

## GET /static/example.jpg

The photograph these docs use in their examples, five people around a table. The test page loads
it, and it is a quick way to get an image to try a route on.

## GET /docs

The generated OpenAPI reference for every route.

## GET /workflow

The workflow editor. See [Build a workflow in the editor](/docs/workflows/editor.md).

## GET /workflow/nodes

Every node a workflow can use, as `{"nodes": [...], "process": true}`. `process` says whether
`/workflow/process` is offered.

## POST /workflow/validate

Check a workflow without running it. Form field `workflow`, the document as JSON. Returns
`{"valid": true, "order": [...], "terminals": [...]}`, or `{"valid": false, "error": "..."}` with
the reason.

## POST /workflow/run

Run a workflow once.

| Form field | Default | Description |
|---|---|---|
| `workflow` | | The workflow document, as JSON. |
| `inputs` | `{}` | Parameter overrides, as a JSON object, keyed as `run()` takes them. |
| `include` | `terminals` | `terminals` for the end nodes' outputs, `all` for every node's. |
| `file` | | An image or video to run on, instead of the one the workflow names. |

Returns `{"results": {node id: output}, "terminals": [...]}`. Images come back encoded, so ask for
`all` only when you need the intermediates: encoding every node's output costs more than running a
small workflow.

## POST /workflow/stream

As `/workflow/run`, but answers with server-sent events: one as each node starts and another as it
finishes, with its output.

## POST /workflow/process

Run a workflow over everything its source produces, such as every frame of a video, reporting
progress as server-sent events. Form fields `workflow`, `inputs` and `file` as for `run`, plus
`preview`, a node id to send a thumbnail of with each item. Each event carries `item`, the number
done, `of`, how many the source expects, and `frame` and `time`, where the item sits in the file.
Not offered when the router was mounted with `process=False`.
