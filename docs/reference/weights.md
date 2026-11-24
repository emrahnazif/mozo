# Weights API reference: mozo.weights

<!--
summary: Every function in mozo.weights: download ahead of time, check the cache, read licences and revisions, remove files, and watch progress.
label: Weights API
keywords: mozo, weights, model cache, download, licences, offline
-->

`mozo.weights` answers questions about a model's files: is it downloaded, how big is it, which
licence and revision is it, and it downloads or deletes them. None of these functions load a model,
and the ones that only read the manifest import no torch and touch no network.

Most take the same arguments:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `family` | `str` | | Model family, such as `"sam2"`. |
| `variant` | `str` | | Variant within it, such as `"large"`. |
| `runtime` | `str` | `"auto"` | `"auto"` for what a load on `device` would pick, or a runtime by name, such as `"onnx-fp32"`. |
| `device` | `str` | `None` | The device a load would run on. With `"auto"` it decides the runtime; `None` asks torch for this machine's best. |
| `revision` | `str` | `None` | A published revision to pin. `None` is the latest. |

## configure

```python
weights.configure(*, cache=..., offline=..., base_url=...)
```

Set where weights are cached, whether they may be downloaded, and from where. Each wins over its
environment variable (`MOZO_CACHE`, `MOZO_OFFLINE`, `MOZO_BASE_URL`). An argument left out is left
as it is, and `None` clears it back to the environment. The settings are process-wide and apply to
the next download.

| Parameter | Description |
|---|---|
| `cache` | The cache folder. |
| `offline` | `True` refuses downloads: a missing file raises with its path, URL and hash, so you can place it by hand. |
| `base_url` | A mirror to download from instead of mozo's. A `file://` URL pointing at a copy of the weights tree works. |

```python
from mozo import weights

weights.configure(cache="~/MyApp/weights", offline=False)
```

## fetch

```python
weights.fetch(family, variant, runtime="auto", *, device=None, revision=None,
              progress=None, cancelled=None)
```

Download everything a load of this model needs, without loading it. Afterwards `is_cached` is true
and the load downloads nothing. `progress` and `cancelled` are as for [watching](#watching).

Returns `{artifact key: local path}` for every file the load will read. Raises `DownloadCancelled`
if `cancelled` said stop (files already finished stay cached), and `WeightsError` if the model is
not published, a download fails, or mozo is offline and a file is missing.

## is_cached

```python
weights.is_cached(family, variant, runtime="auto", *, device=None, revision=None)
```

Whether loading this model would need nothing from the network.

## download_size

```python
weights.download_size(family, variant, runtime="auto", *, device=None, revision=None)
```

Bytes a load would still have to download, from the manifest's sizes: 0 once cached. Counts every
file the load needs, licence and NOTICE included, so you can say "this will download 898 MB" before
starting.

## model_size

```python
weights.model_size(family, variant, runtime="auto", *, device=None, revision=None)
```

Bytes of the weights a load runs from, cached or not: an estimate of what the model occupies once
loaded.

## license_of

```python
weights.license_of(family, variant, *, revision=None)
```

The licence the weights are published under, such as `"Apache-2.0"`, `"AGPL-3.0"` or
`"SAM License"`. From the manifest, so it costs no download. The LICENSE file published beside the
weights stays the terms themselves; this names them.

```python
weights.license_of("yolov8", "nano")      # 'AGPL-3.0'
```

## revision_of

```python
weights.revision_of(family, variant, *, revision=None)
```

The revision a load would use, such as `"2026-08-18"`, without downloading anything. Record it with
anything you store that depends on the exact weights, such as embeddings.

## cached

```python
weights.cached()
```

Everything in the cache, one record per file, in path order:
`{family, variant, revision, key, runtime, size, path, state}`. `state` is `"complete"` for a
verified file, `"unpacked"` for an extracted CoreML package, `"built"` for the TensorRT engines
built from a graph on this machine, `"partial"` for an interrupted download, and `"unlisted"` for a
file this version of mozo does not know, such as an old revision.

## remove

```python
weights.remove(family, variant, revision=None)
```

Delete a model's cached files and nothing else. `revision=None` removes every cached revision of
the variant. Returns the bytes freed. A model already loaded keeps working; the next load downloads
again. Raises `WeightsError` if this process is downloading one of the files right now.

```python
weights.remove("sam3", "sam3")      # 3449283301
```

## watching

```python
with weights.watching(progress=None, cancelled=None, *, total=0):
    ...
```

Report, and allow cancelling, every download made inside the block on this thread.

| Parameter | Description |
|---|---|
| `progress` | Called as `progress(done_bytes, total_bytes)`, counted across every file in the block: at most every 0.1 s and once as each file completes. |
| `cancelled` | Checked once per chunk. Returning `True` raises `DownloadCancelled`, deletes the partial file, and caches nothing. |
| `total` | Bytes the block is expected to download, for example from `download_size`. |

```python
with weights.watching(progress=print):
    mozo.get_model("sam2/large")
```

## resolve

```python
weights.resolve(family, variant, key="torch-fp32", *, revision=None)
```

A local path to one published file, downloaded and verified if needed, with its LICENSE and NOTICE
beside it. Raises `WeightsError` if it is not published, fails to download or verify, or is missing
while offline.

## parts

```python
weights.parts(family, variant, runtime, *, revision=None)
```

Every file that makes up one runtime, downloaded and verified, as `{part name: path}`. Most runtimes
are one file, keyed `""`. SAM 2's ONNX graphs are two, `{"decoder": ..., "encoder": ...}`.

## artifacts

```python
weights.artifacts(family, variant, *, revision=None)
```

The artifact keys a revision publishes, such as `["coreml-fp32", "labels", "onnx-fp32",
"torch-fp32"]`, without LICENSE and NOTICE. A split runtime lists its parts.

## companions

```python
weights.companions(family, variant, *, revision=None)
```

The files that travel with every artifact: `["LICENSE"]`, and `"NOTICE"` where one is published.

## manifest

```python
weights.manifest()
```

The manifest bundled with this version of mozo, as a dict: every model, revision, file, size, hash
and licence.

## runtime_of

```python
weights.runtime_of(key)      # runtime_of("onnx-fp32-encoder") == "onnx-fp32"
```

The runtime an artifact key belongs to. A key is `<framework>-<precision>`, with `-<part>` for a
split runtime.

## framework_of

```python
weights.framework_of(key)    # framework_of("onnx-fp32-encoder") == "onnx"
```

The framework an artifact key names, or the whole key when it names none, such as `"LICENSE"`.

## part_of

```python
weights.part_of(key)         # part_of("onnx-fp32-encoder") == "encoder"
```

The part an artifact key names within its runtime, or `""` when the key is the whole runtime.

## cache_dir

```python
weights.cache_dir()
```

The cache folder: what `configure` set, else `$MOZO_CACHE`, else `~/.cache/mozo`.

## WeightsError

Raised when a model cannot be resolved, downloaded or verified. The message names the file, the
URL and the expected hash where they apply.

## DownloadCancelled

Raised when a `cancelled()` callback said stop. The partial file is already gone.
