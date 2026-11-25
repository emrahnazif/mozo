# Configure the cache, downloads and served models

<!--
summary: Every setting mozo reads: where weights are cached, offline mode, a download mirror, which models a server offers, and memory for prepared images.
label: Configuration
keywords: mozo, configuration, environment variables, offline, air-gapped, model cache
-->

Mozo needs no configuration: weights download on first use into `~/.cache/mozo`, and the server
offers every model. These settings change that. All are environment variables, and the weight
settings can also be set in code.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MOZO_CACHE` | `~/.cache/mozo` | Where downloaded weights are kept. |
| `MOZO_OFFLINE` | unset | `1` refuses every download. A missing file raises an error naming its path, URL and sha256, so it can be placed by hand. |
| `MOZO_BASE_URL` | mozo's own server | Download weights from a mirror instead. A `file://` URL to a copy of the weights tree works. |
| `MOZO_ENABLE` | every model | Which models the server offers: families (`clip`) and single variants (`clip/base`), comma-separated. |
| `MOZO_PREPARED_MB` | `512` | Memory the server may use for images kept by `/prepare`. The least recently used goes first. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` under `mozo start` | Torch's own: on Apple silicon, run operations `mps` lacks on the CPU instead of failing. |

`MOZO_ENABLE` and `MOZO_PREPARED_MB` are read by the server once, when it starts.

## Set it in code

An application that embeds mozo can set the weight settings itself. Each wins over its environment
variable, and applies to the next download:

```python
import mozo

mozo.configure(cache="~/MyApp/weights", offline=False)
```

`mozo.configure(cache=None)` goes back to the environment variable. The same module answers what
the cache holds, without importing torch or touching the network:

```python
from mozo import weights

weights.is_cached("sam2", "large", device="mps")       # would a load need the network?
weights.download_size("sam2", "large", device="mps")   # bytes a load would still download
weights.cached()                                       # every file: model, key, size, state
weights.remove("sam3", "sam3")                         # delete it, returns the bytes freed
```

[Weights API](/docs/reference/weights.md) has every function.

## Choose what a server offers

`MOZO_ENABLE` is an allow-list. Unset, the server offers all 93 models. Set, it offers only what it
names:

```bash
MOZO_ENABLE=rfdetr,clip/base,granite_embedding_r2 mozo start
```

A family name offers all its variants; `family/variant` offers one. A name that matches nothing is
logged and ignored, never fatal: the setting can only take models away. A request for a model the
server does not offer answers `404`, and `GET /models` lists only what it offers.

It is an allow-list rather than a list of exclusions because the two fail in opposite directions
when you upgrade. A list of exclusions would start serving whatever family the upgrade added; an
allow-list serves nothing it was not told to. [Licences](/docs/deployment/weights-and-licences.md)
has the list that offers only permissively licensed weights.

Workflows are held to the same list. A workflow sent to `/workflow/run` that names a model the
server does not offer fails at that node, with the same reason `/predict` gives, before any weights
are downloaded. A node that runs a checkpoint of your own is not refused: the list decides which of
mozo's published weights a server hands out.

`MOZO_ENABLE` is a server setting. It does not affect `mozo.get_model`; importing mozo is not a
deployment.

## Run without internet access

Download on a connected machine, then carry the cache across:

```python
from mozo import weights

for model in ["rfdetr/medium", "sam2/large"]:
    family, variant = model.split("/")
    weights.fetch(family, variant, device="cuda")
```

Copy the cache folder to the same path on the offline machine, or point `MOZO_CACHE` at it, and set
`MOZO_OFFLINE=1` so a missing file fails at once instead of trying the network:

```bash
MOZO_CACHE=/data/mozo-cache MOZO_OFFLINE=1 mozo start
```

To serve weights to several machines from one place, put a copy of the weights tree on a file share
or an internal web server and point `MOZO_BASE_URL` at it, such as `file:///mnt/weights` or
`https://weights.internal/mozo`. The tree mirrors mozo's own: one folder per family, variant and
revision, holding the files the manifest names, such as `rfdetr/medium/2026-08-18/torch-fp32.pth`.
Every file is still checked against the sha256 in the package, so a mirror cannot serve different
bytes.
