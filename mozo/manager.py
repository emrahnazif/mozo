"""Load each model once, and keep it.

Loading costs seconds and hundreds of megabytes, so a server that loads per request is unusable.
This holds a model after its first use and hands the same object to every later caller. Nothing
is evicted.

An earlier version bounded the cache by model count. That was the wrong question twice over: a
count cannot tell 0.10 GB from 1.34 GB, and mozo publishes both, so "two models" meant anywhere
from 0.20 GB to 2.68 GB. The same number could not be right for a 6 GB laptop and an 80 GB
accelerator, and a hundred 100 MB models are no problem at all while two large ones may already
be too many. Measured, it also broke the obvious deployment -- detection, segmentation and depth
from one instance, 0.60 GB between them -- into an eviction on every request: 762 ms each,
silently, instead of free.

Memory is therefore the caller's, and the lever is which models you ask for. For a separate
lifetime build a separate :class:`ModelManager` and drop it; for no cache at all, import an
adapter directly.

    >>> from mozo.manager import ModelManager
    >>> models = ModelManager()
    >>> models.get_model("rfdetr", "nano")     # doctest: +SKIP
"""

from __future__ import annotations

__all__ = ["ModelManager"]

import logging
from collections.abc import Callable
from contextlib import nullcontext
from importlib import import_module
from threading import Lock
from typing import Any

from . import weights
from .registry import get_model_info


_log = logging.getLogger(__name__)


def _build(family: str, variant: str, **kwargs: Any) -> Any:
    """Import the family's adapter class and instantiate it.

    The registry says which class to import and nothing more. In particular the *variant* is not
    checked here: RF-DETR accepts an unpublished variant name when you bring your own checkpoint,
    because there the variant names an architecture rather than a published model, and only the
    adapter knows that. The registry's own message names the families that do exist.
    """
    info = get_model_info(family)
    adapter = getattr(import_module(info["module"]), info["adapter_class"])
    return adapter(variant=variant, **kwargs)


def _load_total(family: str, variant: str, device: str | None, kwargs: dict[str, Any]) -> int:
    """Bytes a build with these adapter arguments will download, for a progress bar's total.

    Zero when there is no plan to make -- a checkpoint of your own downloads nothing, and a
    variant mozo does not publish has nothing to plan from. The total then grows as files
    arrive, and the build itself is what says, better, if something is wrong.
    """
    if "checkpoint_path" in kwargs:
        return 0
    try:
        return weights.download_size(family, variant, kwargs.get("runtime", "auto"),
                                     device=device, revision=kwargs.get("revision"))
    except Exception:  # noqa: BLE001 - see the docstring
        return 0


class ModelManager:
    """A thread-safe cache of loaded models, one entry per distinct request.

    Examples:
        >>> models = ModelManager()
        >>> models.loaded()
        []
    """

    def __init__(self) -> None:
        #: model id -> model, in the order they were first asked for.
        self._models: dict[str, Any] = {}
        #: model id -> the lock held across that model's build, so two threads cannot build one
        #: model twice while two different models build at once: SAM 3's 3.4 GB download no
        #: longer holds up a model that is already on disk. Reads of ``_models`` are not locked;
        #: a single dict lookup needs no help to be atomic.
        self._building: dict[str, Lock] = {}
        #: Guards ``_building`` itself, for the instant it takes to find or add one lock.
        self._building_lock = Lock()

    def get_model(
        self, family: str, variant: str, device: str | None = None, *,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return a loaded model, building it on first use.

        Args:
            family: Model family, e.g. ``"rfdetr"``.
            variant: Variant within that family, e.g. ``"nano"``.
            device: Where to run. ``None`` lets the adapter pick the best this machine has.
            progress: Called as ``progress(done_bytes, total_bytes)`` while the build downloads
                weights. See :func:`mozo.weights.watching`.
            cancelled: Checked once per downloaded chunk; true raises
                :class:`~mozo.weights.DownloadCancelled` and nothing is cached. It stops a
                download, not a build: torch loading weights cannot be interrupted.
            **kwargs: Passed to the adapter -- ``checkpoint_path``, ``labels``, ``revision``,
                ``runtime``. Part of the cache key, so two checkpoints under one variant name
                are two entries rather than one.

        Raises:
            ValueError: If the family is not registered, or the adapter rejects the variant.
            mozo.weights.DownloadCancelled: If *cancelled* said stop.
        """
        # Everything that changes what gets built belongs in the identity. A checkpoint of your
        # own, a pinned revision, a different runtime -- each is a different model wearing the
        # same name, and handing back the first build for the second request is the kind of wrong
        # answer that never raises. RF-DETR accepts unpublished variant names precisely so that
        # two people's "my-training" can be two different models.
        extra = "".join(f"|{key}={value!r}" for key, value in sorted(kwargs.items()))
        model_id = f"{family}/{variant}" + (f"@{device}" if device else "") + extra

        resident = self._models.get(model_id)
        if resident is not None:
            return resident

        # One lock per model, held across its build and nowhere else. A request for a model already
        # in memory is answered above without any lock; a second request for one being built
        # waits for that build rather than starting another; a request for a different model
        # builds alongside it.
        with self._building_lock:
            building = self._building.setdefault(model_id, Lock())
        with building:
            resident = self._models.get(model_id)
            if resident is not None:
                return resident  # another thread loaded it while this one queued

            _log.info("[mozo] loading %s", model_id)
            watched = progress is not None or cancelled is not None
            with (weights.watching(progress, cancelled,
                                   total=_load_total(family, variant, device, kwargs))
                  if watched else nullcontext()):
                model = _build(family, variant, device=device, **kwargs)
            self._models[model_id] = model
            return model

    def loaded(self) -> list[str]:
        """Resident model ids, in the order they were first asked for."""
        return list(self._models)
