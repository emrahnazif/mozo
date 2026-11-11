"""One loaded model, called from several threads at once, answers as it does one call at a time.

The server shares one model object across its threadpool, a workflow's model stage can run
``model_workers`` threads, and a desktop app sends SAM 2 a click on every mouse move. Nothing
serialises those calls, deliberately: in parallel they are faster (the README has the numbers).

So this holds the families whose predict touches shared state -- SAM 2's and EdgeTAM's feature
caches, YOLO and RF-DETR emptying the MPS cache after every call -- on every runtime this machine
can execute, to the results the same calls give sequentially. Real models from the local cache,
offline; a model not downloaded here is skipped.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from conftest import executable_runtimes

#: The families the thread-safety audit found shared state in, one small variant each, and
#: whether a call needs a click.
FAMILIES = {("sam2", "tiny"): True, ("edgetam", "edgetam"): True,
            ("yolov8", "nano"): False, ("rfdetr", "nano"): False}
THREADS, CALLS = 4, 8


def _cases():
    from mozo.runtimes import get_default_device

    device = get_default_device()
    for family, variant, runtime in executable_runtimes(FAMILIES):
        yield pytest.param(family, variant, device, runtime,
                           id=f"{family}/{variant}/{device}/{runtime}")


@pytest.mark.parametrize("family, variant, device, runtime", list(_cases()))
def test_concurrent_calls_answer_as_sequential_ones(family, variant, device, runtime, image):
    from mozo import weights
    from mozo.manager import ModelManager

    weights.configure(offline=True)
    if not weights.is_cached(family, variant, runtime, device=device):
        pytest.skip(f"{family}/{variant} {runtime} is not downloaded here")

    height, width = image.shape[:2]
    rng = np.random.default_rng(0)
    prompts = ([{"points": [[rng.uniform(0, width), rng.uniform(0, height)]], "labels": [1]}
                for _ in range(CALLS)] if FAMILIES[family, variant] else [{}] * CALLS)

    models = ModelManager()

    def call(i: int):
        # Through the manager each time -- a lock-free cache hit -- so no local holds the model
        # and clear() below can free it.
        found = models.get_model(family, variant, device=device, runtime=runtime).predict(
            image, **prompts[i])
        return sorted((row.get("class_id"), tuple(np.round(row["bbox"], 1)))
                      for row in found.to_dict())

    try:
        expected = [call(i) for i in range(CALLS)]
        with ThreadPoolExecutor(THREADS) as pool:
            answers = list(pool.map(call, list(range(CALLS)) * THREADS))
        assert answers == expected * THREADS
    finally:
        models.clear()  # hand the accelerator memory back before the next case
