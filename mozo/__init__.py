"""mozo -- computer vision models that run from a pip install.

Eighty-one published variants across seventeen families, served over HTTP or called from Python. No
Docker, no cluster, no configuration: weights are fetched and verified on first use and cached
under ``~/.cache/mozo``.

Server:

    $ mozo start
    $ curl -X POST http://localhost:8000/predict/rfdetr/nano -F file=@image.jpg

Python:

    >>> from mozo import get_model
    >>> model = get_model("rfdetr/nano")             # doctest: +SKIP
    >>> detections = model.predict("image.jpg")      # doctest: +SKIP

:func:`get_model` shares one cache, so asking twice loads once and the model stays for the life
of the process. Build a :class:`~mozo.manager.ModelManager` of your own for a separate cache and
a separate lifetime, or import an adapter directly for no cache at all.

See https://github.com/emrahnazif/mozo.
"""

from __future__ import annotations

# Single source of truth for the package version.
# pyproject.toml reads this via [tool.setuptools.dynamic] version.attr
__version__ = "1.3.0"

__all__ = ["MODEL_REGISTRY", "ModelManager", "__version__", "clear", "configure", "get_model",
           "get_model_info", "unload"]

import logging as _logging

# A library reports what it is doing and leaves where that goes to whoever embeds it. Every
# module logs under ``mozo``; this handler keeps Python's last-resort printer from writing those
# records to stderr when the host has configured nothing. The command line attaches a handler
# of its own, which is why ``mozo start`` and ``mozo run`` still print.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from mozo.manager import ModelManager
from mozo.registry import MODEL_REGISTRY, get_model_info
from mozo.weights import configure

#: The cache behind :func:`get_model`. A dict and a lock; it loads nothing until asked.
_shared = ModelManager()


def get_model(identifier: str, variant: str | None = None, device: str | None = None, *,
              progress=None, cancelled=None):
    """Load a model from a shared cache.

    Args:
        identifier: ``"family/variant"``, or just the family when *variant* is given.
        variant: The variant, if *identifier* did not carry it.
        device: ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` to take the best available.
        progress: ``progress(done_bytes, total_bytes)`` while weights download.
        cancelled: Checked once per downloaded chunk; true stops the download. See
            :meth:`~mozo.manager.ModelManager.get_model`.

    Returns:
        The family's predictor, with a ``predict()`` method, shared with every other caller and
        safe to call from several threads at once -- see
        :meth:`~mozo.manager.ModelManager.get_model` for the one caveat, on Apple silicon.

    Examples:
        >>> get_model("rfdetr/nano")                    # doctest: +SKIP
        >>> get_model("rfdetr", "nano", device="cpu")   # doctest: +SKIP
    """
    family, variant = _split(identifier, variant)
    return _shared.get_model(family, variant, device=device,
                             progress=progress, cancelled=cancelled)


def unload(identifier: str, variant: str | None = None, device: str | None = None) -> bool:
    """Release a model :func:`get_model` loaded, and give its accelerator memory back.

    Takes what :func:`get_model` took, because that is the model's identity. Returns whether it
    was loaded. See :meth:`~mozo.manager.ModelManager.unload`.

    Examples:
        >>> unload("rfdetr/nano")                       # doctest: +SKIP
        True
    """
    family, variant = _split(identifier, variant)
    return _shared.unload(family, variant, device=device)


def clear() -> list[str]:
    """Release every model :func:`get_model` loaded. Returns the ids released."""
    return _shared.clear()


def _split(identifier: str, variant: str | None) -> tuple[str, str]:
    """``("rfdetr/nano", None)`` or ``("rfdetr", "nano")`` to ``("rfdetr", "nano")``."""
    if variant is not None:
        return identifier, variant
    if "/" not in identifier:
        raise ValueError(
            f"Expected 'family/variant' or a separate variant argument, got {identifier!r}")
    family, variant = identifier.split("/", 1)
    return family, variant
