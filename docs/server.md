# Serve models over HTTP

<!--
summary: Start mozo's HTTP server, send it images from any language, read what comes back, and check which models it serves.
label: HTTP server
keywords: mozo, http api, model server, fastapi, rest api, curl
-->

Run mozo as a server when the caller is not Python, runs on another machine, or when several
programs should share one set of loaded models. Every model is available at the same kind of URL,
and the server loads a model the first time someone asks for it.

## Start the server

```bash
mozo start
```

```text
Starting Mozo server on 0.0.0.0:8000...
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

It starts in a second whatever it serves, because it loads nothing until a request needs it. The
first request for a model downloads and loads it, which can take minutes for a large model on an
empty cache; every request after that is fast.

| Option | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Address to listen on. |
| `--port` | `8000` | Port to listen on. |
| `--workers` | `1` | Worker processes. Each loads its own copy of every model it serves. |
| `--reload` | off | Restart when the code changes, for development. Forces one worker. |

The server has no authentication and listens on every interface by default. Put it behind
something you control before it faces a network you do not. [Production](/docs/deployment/production.md)
covers workers, memory and choosing what an instance serves.

## Send a request

A model answers at `POST /predict/{family}/{variant}`, with the image as a multipart upload named
`file`:

```bash
curl -X POST "http://localhost:8000/predict/rfdetr/medium" -F "file=@example.jpg"
```

Options and prompts are query parameters. Repeat a parameter to pass several values:

```bash
curl -X POST "http://localhost:8000/predict/owlv2/base-ensemble?text=a%20laptop&text=a%20cup&threshold=0.3" \
  -F "file=@example.jpg"
```

The same request from Python, with `requests`:

```python
import requests

with open("example.jpg", "rb") as image:
    response = requests.post(
        "http://localhost:8000/predict/owlv2/base-ensemble",
        params={"text": ["a laptop", "a cup"], "threshold": 0.3},
        files={"file": image},
    )
response.raise_for_status()

for detection in response.json()[:3]:
    print(detection["class_name"], detection["confidence"], detection["bbox"])
```

```text
a laptop 0.717 [761.75, 782.39, 1355.03, 1148.92]
a cup 0.494 [933.21, 481.59, 1044.93, 603.8]
a cup 0.473 [416.91, 507.51, 550.1, 663.35]
```

Each family's page lists the parameters it takes, and [HTTP API](/docs/reference/http-api.md) has
every route and parameter in one place. The server also documents itself at
`http://localhost:8000/docs`.

The embedding families answer at a second route, `POST /encode/{family}/{variant}`, which returns
vectors instead of an answer: an image as `file`, or phrases as `text`, but not both in one call.
DINOv2 takes images only, Granite Embedding R2 text only, and `/predict` sends both to `/encode`.

```bash
curl -X POST "http://localhost:8000/encode/granite_embedding_r2/small?text=how%20do%20I%20count%20people%3F"
```

## What comes back

| Models | Response |
|---|---|
| Detection, segmentation, keypoints, detection from words, click segmentation, OCR | JSON: a list of detections |
| CLIP, SigLIP 2 on `/predict` | JSON: a list of `{class_id, class_name, confidence}`, one per phrase |
| Depth Anything V2 | a 16-bit PNG, with the depth range in `x-depth-unit`, `x-depth-min` and `x-depth-max` headers |
| BEN2 | an RGBA PNG: the photograph with its background made transparent |
| Moebius | a PNG: the photograph with the boxed region painted out |
| `/encode` | JSON: `{model, revision, dim, embeddings}` |

A detection carries every PixelFlow field, including the ones that are `null`:

```json
{"inference_id": null, "bbox": [0.61, 122.33, 761.01, 1268.59], "masks": null, "segments": null,
 "keypoints": null, "class_id": 1, "class_name": "person", "text": null, "labels": null,
 "confidence": 0.97, "tracker_id": null, "metadata": null, "zones": [], "zone_names": [],
 "line_crossings": [], "first_seen_time": null, "total_time": 0.0}
```

`masks: null` beside a filled `segments` is a difference worth seeing, so nothing is left out.

`/encode` returns the model and the revision of its weights with every response. Keep them with
the vectors you store: a vector only compares with vectors from the same weights.

## Check what it serves

```bash
curl http://localhost:8000/
```

```json
{"status": "ok", "version": "1.12.0", "loaded_models": ["owlv2/base-ensemble"]}
```

| Route | Answers with |
|---|---|
| `GET /` | Health, the mozo version and the models loaded now. |
| `GET /models` | Every model this server offers: task, variants, licence per variant, and which prompts it takes. Loads nothing. |
| `GET /models/loaded` | The models in memory now. |
| `GET /docs` | The generated OpenAPI reference. |
| `GET /test-ui` | A page to try any model from the browser. |
| `GET /workflow` | The workflow editor. |

By default the server offers every model. `MOZO_ENABLE` limits it to a list of families or
variants, for example to serve only permissively licensed weights:
[Configuration](/docs/deployment/configuration.md) has the details.

For an interactive client that clicks on one image many times, `POST /prepare/{family}/{variant}`
encodes the image once and later clicks name it by its sha256 instead of uploading it again.
[SAM 2](/docs/models/sam2.md#prepare-an-image-before-the-first-prompt) explains it.

## Errors

Errors are JSON with a `detail` that says what to do:

```json
{"detail": "owlv2 is prompted: pass ?text=... naming what to look for, and give every one a concept. Repeat it for several: ?text=car&text=person."}
```

| Status | Meaning |
|---|---|
| `400` | The request is wrong: the file is not an image, a required prompt is missing, or a parameter is out of range. |
| `404` | No such model on this server, or an image `?sha256=` names is not held. The detail lists what is served. |
| `500` | The model failed to load or to run. |
| `501` | The model has no such route, for example `/predict` on a text-embedding model. |
