# Python API reference: mozo

<!--
summary: Every public name in the mozo package: get_model, unload, clear, configure, ModelManager, the model registry, image helpers and model objects.
label: Python API
keywords: mozo, python api, reference, get_model, modelmanager, load_image
-->

The names `import mozo` gives you, plus `mozo.image` and the methods every loaded model has.
Downloading and caching weights is in [Weights API](/docs/reference/weights.md), and workflows are
in [Workflow API](/docs/reference/workflow-api.md). Each model's own `predict` arguments are on its
[model page](/docs/models/all-models.md).

## get_model

```python
mozo.get_model(identifier, variant=None, device=None, *, progress=None, cancelled=None)
```

Load a model from the shared cache, building it on first use, and return it.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `identifier` | `str` | | `"family/variant"`, or the family alone when `variant` is given. |
| `variant` | `str` | `None` | The variant, if `identifier` did not carry it. |
| `device` | `str` | `None` | `"cuda"`, `"mps"` or `"cpu"`. `None` takes the best available. |
| `progress` | callable | `None` | Called as `progress(done_bytes, total_bytes)` while weights download. |
| `cancelled` | callable | `None` | Checked once per downloaded chunk. Returning `True` raises `mozo.weights.DownloadCancelled`, and nothing is cached. |

Returns the model object, shared with every other caller. Asking again returns the same object.

Raises `ValueError` if the family is unknown, the variant is not one it publishes, or `identifier`
has no `/` and no `variant` is given.

```python
model = mozo.get_model("rfdetr/medium")
model = mozo.get_model("rfdetr", "medium", device="cpu")
```

`get_model` does not take `runtime`, `checkpoint_path` or `labels`; use a
[ModelManager](#modelmanager) for those.

## unload

```python
mozo.unload(identifier, variant=None, device=None)
```

Release a model `get_model` loaded and give its accelerator memory back. Takes what `get_model`
took, because that is the model's identity. Returns `True` if it was loaded.

Nothing is freed while anything else still holds the model object; a warning says so.

## clear

```python
mozo.clear()
```

Release every model `get_model` loaded. Returns the list of ids released.

## configure

```python
mozo.configure(*, cache=..., offline=..., base_url=...)
```

Set where weights are cached, whether they may be downloaded, and from where, in code. The same
function as `mozo.weights.configure`; [Weights API](/docs/reference/weights.md#configure) has the
details.

## get_model_info

```python
mozo.get_model_info(family, variant=None)
```

Return one family's catalogue entry, and check that `variant` is one it publishes when given. Raises
`ValueError` naming what is available if not. Loads nothing.

```python
mozo.get_model_info("rfdetr", "nano")["task_type"]      # 'object_detection'
```

## MODEL_REGISTRY

```python
mozo.MODEL_REGISTRY
```

Every family, as `{family: entry}`. An entry has `task_type`, `description` and `variants`, plus
`adapter_class` and `module`, which name the code that runs it. Reading it imports nothing heavy
and loads nothing.

```python
mozo.MODEL_REGISTRY["sam2"]["variants"]      # ['tiny', 'small', 'base_plus', 'large']
```

## \_\_version\_\_

```python
mozo.__version__      # '1.12.0'
```

## ModelManager

```python
mozo.ModelManager()
```

A thread-safe cache of loaded models. `mozo.get_model` uses one shared `ModelManager`; build your
own for models with a separate lifetime, or to pass arguments `get_model` does not take.

### ModelManager.get_model

```python
models.get_model(family, variant, device=None, *, progress=None, cancelled=None, **kwargs)
```

Return a loaded model, building it on first use. `device`, `progress` and `cancelled` are as for
[get_model](#get_model). The keyword arguments go to the model:

| Argument | Description |
|---|---|
| `runtime` | `"auto"` (the default), or an artifact by name such as `"onnx-fp32"` or `"coreml-fp32"`. Raises if the variant does not publish it. |
| `checkpoint_path` | A checkpoint of your own, for the families that take one. See [Python](/docs/python.md#your-own-checkpoints). |
| `labels` | Class names, replacing the checkpoint's. |
| `revision` | A published revision to pin, instead of the latest. |

All of them are part of the model's identity: two checkpoints under one variant name are two
models. If another thread is already loading the same model, this waits for that load instead of
starting a second one, and a load never waits behind a different model's.

### ModelManager.unload

```python
models.unload(family, variant, device=None, **kwargs)
```

Release one model, taking exactly what `get_model` took. Returns whether it was loaded. A load of it
still under way is waited for, then released.

### ModelManager.clear

```python
models.clear()
```

Release every model, waiting for any load under way. Returns the ids released.

### ModelManager.loaded

```python
models.loaded()
```

The ids of the loaded models, in the order they were first asked for.

### ModelManager.resident

```python
models.resident()
```

One record per loaded model: `{id, family, variant, device, runtime, last_used, size}`. `device` and
`runtime` are what the model is actually using. `last_used` is a `time.time()` timestamp. `size` is
the bytes of the weights it was loaded from, an estimate of its memory that leaves out activations,
or `None` when unknown.

## load_image

```python
from mozo.image import load_image

load_image(image)
```

Decode a file path, encoded image bytes, or an array into mozo's image form: an `HxWx3` RGB `uint8`
array. Every model's `predict` does this itself; call it once when you pass one image to several
models.

A path or bytes are decoded, so the result is certainly RGB. An array is checked for shape and
dtype and otherwise trusted: nothing in an array says whether it is RGB or BGR. An array from
`cv2.imread` is BGR, so pass the path instead. Raises `ValueError` for a file that is not an image
and for an array of the wrong shape or dtype.

## encode_image

```python
from mozo.image import encode_image

encode_image(image, extension=".png")
```

Encode an `HxWx3` RGB or `HxWx4` RGBA `uint8` array as image bytes, the reverse of `load_image`.
`extension` is the format, with or without the dot. PNG by default, because annotated images are
thin lines and mask edges, which JPEG blurs. An RGBA image needs a format that keeps transparency:
`.png`, `.webp`, `.tif` or `.tiff`. Raises `ValueError` otherwise.

## Generation

```python
mozo.Generation(text, finish_reason, prompt_tokens, generated_tokens)
```

What a model that answers in words returns from `predict` (Qwen3-VL). A frozen dataclass:

| Field | Description |
|---|---|
| `text` | The answer, with the model's control tokens removed. |
| `finish_reason` | `"stop"` if the model ended the answer, `"length"` if it reached `max_new_tokens` first and the text is cut off. |
| `prompt_tokens` | Tokens read, the image's included. |
| `generated_tokens` | Tokens written, the stopping token included. |

`str(g)` is the text, and `g.to_dict()` the four fields, which is what `/predict` returns as JSON.
There is no confidence number: the text is the model's continuation, not a checked fact.

## Model objects

What `get_model` returns depends on the family; each model page lists its methods and their
arguments. These are shared:

| Member | Description |
|---|---|
| `predict(image, ...)` | Run the model on one image. Every family except Granite Embedding R2. Qwen3-VL's takes a prompt and returns a `mozo.Generation`: `text`, `finish_reason` (`"stop"` or `"length"`), `prompt_tokens`, `generated_tokens`, and `to_dict()`. |
| `encode_image(image)` | Vectors for one image or a list of them, `(N, D)`. CLIP and SigLIP 2. |
| `encode_text(text)` | Text vectors, `(N, D)`. CLIP, SigLIP 2 and Granite Embedding R2. |
| `prepare(image)` | Run the image encode before the first prompt. SAM 2, EdgeTAM, SAM 3 and OWLv2. |
| `cutout(image, ...)` | The photograph with its background made transparent. BEN2. |
| `device` | The device it runs on. |
| `runtime` | The artifact it runs from, such as `"torch-fp32"` or `"coreml-fp32"`. |
| `revision` | The revision of the weights, on CLIP, SigLIP 2, Granite Embedding R2 and Qwen3-VL. Store it with the vectors. |
| `unit` | `"metres"` or `None`. Depth Anything V2. |

`predict` may be called from several threads at once. That is safe on CPU, CUDA and ONNX Runtime;
on Apple silicon (`mps`) and CoreML it is tested but not documented as thread-safe by PyTorch or
Apple.
