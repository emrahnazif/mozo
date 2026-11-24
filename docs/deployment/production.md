# Run mozo's server in production

<!--
summary: What to arrange before mozo's server faces real traffic: workers and memory, loading models ahead of time, concurrency, and access control.
label: Production
keywords: mozo, deployment, production, gpu memory, workers, model serving
-->

`mozo start` is a real server, FastAPI on uvicorn, safe to call from many clients at once, with
every weight file verified by hash. Four things about running it are yours to decide: how many
workers, which models it holds in memory, when they load, and who may reach it.

## One worker, unless you have measured

```bash
mozo start --workers 4
```

Workers are separate processes and share nothing, so **each loads its own copy of every model it
serves**. Four workers serving SAM 3 hold four copies of 3.4 GB. One worker already serves
concurrent requests in parallel (below), so prefer one unless you have measured that more helps on
your machine.

## Memory is yours to manage

Nothing is unloaded automatically. A model stays in memory from its first request until you release
it, because no automatic rule fits every machine: a limit by model count cannot tell a 100 MB model
from a 3.4 GB one, and the same number cannot suit a laptop and an 80 GB accelerator.

So decide what an instance serves instead of letting callers decide. `MOZO_ENABLE` limits the server
to the models you name; [Configuration](/docs/deployment/configuration.md#choose-what-a-server-offers)
has the details. `GET /models/loaded` shows what is in memory now. In Python,
`ModelManager.resident()` reports each model's device, last use and estimated size, and `unload()`
and `clear()` release models and give accelerator memory back.

## Load models before the first request

The server starts in a second, because it loads nothing until a request needs it. The first request
for a model then downloads and loads it: seconds from the cache, minutes for a large model on an
empty one. Download weights when you build the machine or the image, so the first request only
loads:

```python
from mozo import weights

weights.fetch("rfdetr", "medium", device="cuda")
```

Then send each model one request after the server starts, if the first real request should not wait
for the load either.

A [TensorRT runtime](/docs/deployment/runtimes.md#tensorrt) has one more first-time cost: its first
load on a machine builds an engine for that GPU, which takes about a minute. Load the model once on
the machine itself, or in the image if it is built on the same GPU model, so that is paid before
traffic arrives.

A model is built once, however many requests for it arrive at the same moment, and no request waits
behind a different model's load, even one whose weights are still downloading.

## Concurrent requests

One model serves concurrent requests in parallel: `predict` is not serialised, and in parallel it
runs 1.2 to 2.7 times faster than one call after another. That is safe on CPU, CUDA and ONNX Runtime.
On Apple silicon (`mps`) and CoreML it was tested, 6,000 concurrent calls with no error and results
identical to sequential ones, but PyTorch and Apple do not document it as thread-safe.

## Put it behind something

**There is no authentication and no rate limiting**, and by default the server listens on every
interface (`0.0.0.0`). Before it faces a network you do not control, put it behind a reverse proxy
or gateway that authenticates callers, limits request sizes and rates, and terminates TLS. To keep
it on one machine, bind it there:

```bash
mozo start --host 127.0.0.1
```

Check the licences of the models it serves before you expose it.
[Licences](/docs/deployment/weights-and-licences.md) covers what serving AGPL-3.0 and SAM License
weights over a network asks of you, and how to serve only permissive ones.
