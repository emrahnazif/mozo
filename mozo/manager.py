"""Load each model once, and keep it until told otherwise.

Loading costs seconds and hundreds of megabytes, so a server that loads per request is unusable.
This holds a model after its first use and hands the same object to every later caller. Nothing
is evicted automatically; :meth:`ModelManager.unload` and :meth:`ModelManager.clear` release
models when the caller decides to.

An earlier version bounded the cache by model count. That was the wrong question twice over: a
count cannot tell 0.10 GB from 1.34 GB, and mozo publishes both, so "two models" meant anywhere
from 0.20 GB to 2.68 GB. The same number could not be right for a 6 GB laptop and an 80 GB
accelerator, and a hundred 100 MB models are no problem at all while two large ones may already
be too many. Measured, it also broke the obvious deployment -- detection, segmentation and depth
from one instance, 0.60 GB between them -- into an eviction on every request: 762 ms each,
silently, instead of free.

Memory is therefore the caller's, and the levers are which models you ask for and which you
unload. :meth:`ModelManager.resident` reports each model's device, when it was last used and an
estimate of its size, which is what a caller deciding what to unload needs. For a separate
lifetime build a separate :class:`ModelManager` and drop it; for no cache at all, import an
adapter directly.

    >>> from mozo.manager import ModelManager
    >>> models = ModelManager()
    >>> models.get_model("rfdetr", "nano")     # doctest: +SKIP
    >>> models.unload("rfdetr", "nano")        # doctest: +SKIP
    True
"""

from __future__ import annotations

__all__ = ["ModelManager"]

import logging
import time
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from threading import Event, Lock
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


def _model_id(family: str, variant: str, device: str | None, kwargs: dict[str, Any]) -> str:
    """The cache key: everything that changes what gets built.

    A checkpoint of your own, a pinned revision, a different runtime -- each is a different model
    wearing the same name, and handing back the first build for the second request is the kind of
    wrong answer that never raises. RF-DETR accepts unpublished variant names precisely so that
    two people's "my-training" can be two different models.

    Loading and unloading both key through here. The manager this one replaced composed the key
    one way to load and another to unload, so a model loaded onto an explicit device could never
    be released.
    """
    extra = "".join(f"|{key}={value!r}" for key, value in sorted(kwargs.items()))
    return f"{family}/{variant}" + (f"@{device}" if device else "") + extra


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


def _free_accelerator_memory() -> None:
    """:func:`mozo.runtimes.free_device_memory`, imported when needed: it brings numpy."""
    from .runtimes import free_device_memory

    free_device_memory()


def _estimated_size(model: Any, family: str, variant: str, kwargs: dict[str, Any]) -> int | None:
    """The bytes of the weights a just-built model was loaded from, or ``None`` if unknown.

    Its checkpoint file's size if it was given one; otherwise :func:`mozo.weights.model_size`
    for the runtime it actually chose. Worked out once, at build: it cannot change afterwards.
    """
    checkpoint = kwargs.get("checkpoint_path")
    if checkpoint:
        path = Path(checkpoint)
        return path.stat().st_size if path.is_file() else None
    runtime = getattr(model, "runtime", None)
    if runtime is None:
        return None
    try:
        return weights.model_size(family, variant, runtime,
                                  device=getattr(model, "device", None) or "cpu",
                                  revision=kwargs.get("revision"))
    except Exception:  # noqa: BLE001 - an estimate that cannot be made is simply absent
        return None


@dataclass
class _Resident:
    """One loaded model, and what :meth:`ModelManager.resident` reports about it."""

    model: Any
    family: str
    variant: str
    device: str | None
    size: int | None
    last_used: float = field(default_factory=time.time)


@dataclass
class _Build:
    """A model being built: who is waiting for it, and when it is over.

    Exists only while the build runs, so the manager holds nothing for a model it is not
    building or holding. Everyone who asks for the model meanwhile waits on *finished* rather
    than starting a second build; their progress callbacks join *followers*, so they hear the
    download the first caller started.
    """

    finished: Event = field(default_factory=Event)
    followers: list[Callable[[int, int], None]] = field(default_factory=list)

    def report(self, done: int, total: int) -> None:
        for follower in list(self.followers):
            follower(done, total)


class ModelManager:
    """A thread-safe cache of loaded models, one entry per distinct request.

    Examples:
        >>> models = ModelManager()
        >>> models.loaded()
        []
    """

    def __init__(self) -> None:
        #: model id -> what is loaded under it, in the order they were first asked for.
        self._models: dict[str, _Resident] = {}
        #: model id -> its build, while one is running and only then. Different models build at
        #: once -- SAM 3's 3.4 GB download holds up nothing else -- and one model builds once.
        self._builds: dict[str, _Build] = {}
        #: Guards both dicts for the instant it takes to look or change. Never held across a
        #: build. Reads of a resident model take no lock: one dict lookup needs no help.
        self._lock = Lock()

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
            progress: Called as ``progress(done_bytes, total_bytes)`` while weights download.
                See :func:`mozo.weights.watching`. A caller arriving while another's build is
                under way hears that build's download, if that build is being watched.
            cancelled: Checked once per downloaded chunk; true raises
                :class:`~mozo.weights.DownloadCancelled` and nothing is cached. It stops a
                download, not a build: torch loading weights cannot be interrupted. A caller
                waiting on another's build stops waiting instead, and leaves that build be.
            **kwargs: Passed to the adapter -- ``checkpoint_path``, ``labels``, ``revision``,
                ``runtime``. Part of the cache key, so two checkpoints under one variant name
                are two entries rather than one.

        Raises:
            ValueError: If the family is not registered, or the adapter rejects the variant.
            mozo.weights.DownloadCancelled: If *cancelled* said stop.
        """
        model_id = _model_id(family, variant, device, kwargs)
        while True:
            resident = self._models.get(model_id)
            if resident is not None:
                resident.last_used = time.time()
                return resident.model

            with self._lock:
                if model_id in self._models:
                    continue
                build = self._builds.get(model_id)
                mine = build is None
                if mine:
                    build = self._builds[model_id] = _Build()
                if progress is not None:
                    build.followers.append(progress)

            if mine:
                return self._run_build(model_id, build, family, variant, device, kwargs,
                                       cancelled)
            # Another caller is building it. Once that ends the model is resident, or that build
            # failed -- its caller cancelled, say -- and the next pass builds it here instead.
            self._wait(build, progress, cancelled, model_id)

    def _run_build(self, model_id: str, build: _Build, family: str, variant: str,
               device: str | None, kwargs: dict[str, Any],
               cancelled: Callable[[], bool] | None) -> Any:
        """Run one build, publish its model, and let everyone waiting on it go."""
        try:
            _log.info("[mozo] loading %s", model_id)
            watched = bool(build.followers) or cancelled is not None
            with (weights.watching(build.report, cancelled,
                                   total=_load_total(family, variant, device, kwargs))
                  if watched else nullcontext()):
                model = _build(family, variant, device=device, **kwargs)
            size = _estimated_size(model, family, variant, kwargs)
            with self._lock:
                self._models[model_id] = _Resident(model, family, variant, device, size)
            return model
        finally:
            with self._lock:
                del self._builds[model_id]
            build.finished.set()

    def _wait(self, build: _Build, progress: Callable[[int, int], None] | None,
              cancelled: Callable[[], bool] | None, model_id: str) -> None:
        """Wait for another caller's build to end, or for *cancelled* to say stop waiting."""
        try:
            while not build.finished.wait(0.05):
                if cancelled is not None and cancelled():
                    raise weights.DownloadCancelled(f"stopped waiting for {model_id} to load")
        finally:
            with self._lock:
                if progress in build.followers:
                    build.followers.remove(progress)

    def unload(self, family: str, variant: str, device: str | None = None, **kwargs: Any) -> bool:
        """Release one model and give its accelerator memory back. Returns whether it was loaded.

        Takes exactly what :meth:`get_model` took, because the arguments are the model's
        identity: ``unload("rfdetr", "nano", device="cpu")`` releases what
        ``get_model("rfdetr", "nano", device="cpu")`` loaded, and nothing else. A build of it
        still under way is waited for, then released.

        Nothing is freed while anything else still holds the model object; a warning says so.
        The next :meth:`get_model` loads it again.
        """
        model_id = _model_id(family, variant, device, kwargs)
        return bool(self._release([model_id]))

    def clear(self) -> list[str]:
        """Release every model, waiting for any build under way. Returns the ids released."""
        return self._release(None)

    def _release(self, ids: list[str] | None) -> list[str]:
        """Release *ids*, or every model for ``None``, once their builds end. Returns the released.

        The models leave the cache as weak references and nothing else, so no local here can
        keep one alive through the check that warns about a reference held somewhere else.
        """
        with self._lock:
            builds = [b for i, b in self._builds.items() if ids is None or i in ids]
        for build in builds:
            build.finished.wait()
        with self._lock:
            chosen = list(self._models) if ids is None else [i for i in ids if i in self._models]
            watched = [(i, _weak(self._models.pop(i).model)) for i in chosen]
        if not watched:
            return []
        _free_accelerator_memory()
        for model_id, ref in watched:
            if ref is not None and ref() is not None:
                _log.warning("[mozo] unloaded %s, but something else still holds it, so its "
                             "memory is not freed until that reference goes", model_id)
        return chosen

    def loaded(self) -> list[str]:
        """Resident model ids, in the order they were first asked for."""
        return list(self._models)

    def resident(self) -> list[dict[str, Any]]:
        """Each resident model, with what deciding what to unload needs.

        Records are ``{id, family, variant, device, runtime, last_used, size}``, in the order the
        models were first asked for. ``device`` and ``runtime`` are what the model is actually
        using. ``last_used`` is a :func:`time.time` timestamp of the last :meth:`get_model` that
        returned it. ``size`` estimates its memory as the bytes of the weights it was loaded
        from, from the manifest -- or the checkpoint file's size -- and is ``None`` when neither
        is known. It is an estimate, not a measurement: activations come on top, and nothing
        cheap measures memory the same way on CUDA, Apple silicon, CoreML and ONNX Runtime.
        """
        return [{
            "id": model_id, "family": r.family, "variant": r.variant,
            "device": getattr(r.model, "device", r.device),
            "runtime": getattr(r.model, "runtime", None),
            "last_used": r.last_used, "size": r.size,
        } for model_id, r in list(self._models.items())]


def _weak(model: Any) -> weakref.ref | None:
    """A weak reference to *model*, or ``None`` for an object that cannot have one."""
    try:
        return weakref.ref(model)
    except TypeError:
        return None
