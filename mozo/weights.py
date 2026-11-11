"""Turn a model name into a verified local file.

Every published artifact mozo can run is listed in ``manifest.json``, which ships inside the
package. Resolution therefore needs no network and no configuration: the manifest that came with
this version of mozo is the definitive answer to what bytes a given model refers to.

    >>> from mozo.weights import resolve
    >>> resolve("rfdetr", "small")                      # doctest: +SKIP
    PosixPath('/Users/you/.cache/mozo/rfdetr/small/2026-08-18/torch-fp32.pth')

A model absent from the manifest has no weights mozo publishes -- either because its licence
forbids redistribution, or because it is only ever run against checkpoints you bring yourself.
That is not an error condition to work around; pass the file to the adapter directly.

Environment:
    ``MOZO_CACHE``     Where downloads live. Default ``~/.cache/mozo``.
    ``MOZO_BASE_URL``  Serve artifacts from a mirror instead of the manifest's ``base_url``.
                       A ``file://`` URL pointing at a ``weights/`` tree works, which is how an
                       air-gapped host can be fed from removable media.
    ``MOZO_OFFLINE``   Set to ``1`` to refuse downloads. Missing files raise an error naming the
                       exact path, URL and hash, so they can be placed by hand.

Each of the three can also be set in code with :func:`configure`, which wins over the
environment -- for an application that keeps weights inside its own folder.

What is in the cache is answerable without the network or torch: :func:`is_cached`,
:func:`download_size`, :func:`cached`, and :func:`remove` to give the space back.
"""

from __future__ import annotations

__all__ = [
    "DownloadCancelled",
    "WeightsError",
    "artifacts",
    "cache_dir",
    "cached",
    "companions",
    "configure",
    "download_size",
    "fetch",
    "framework_of",
    "is_cached",
    "license_of",
    "manifest",
    "model_size",
    "part_of",
    "parts",
    "remove",
    "resolve",
    "revision_of",
    "runtime_of",
    "watching",
]

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Sits next to this module and ships in the wheel.
_MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"

_DOWNLOAD_CHUNK_BYTES = 1 << 16
#: How often a watching caller hears about progress, at most. Its callback runs on the
#: downloading thread, and once per chunk would be some 52,000 calls for SAM 3's 3.4 GB.
_PROGRESS_SECONDS = 0.1
_TIMEOUT_SECONDS = 60.0

#: Fetched alongside every artifact, so a cached model is never separated from its terms.
_LICENCE_KEY = "LICENSE"
_NOTICE_KEY = "NOTICE"
# Artifacts that ship *with* whichever one you asked for, rather than being a choice. The licence
# is always published; a NOTICE only where the upstream terms ask for attribution to travel with
# the copy, which for CC-BY-NC weights they do.
_ACCOMPANYING = (_LICENCE_KEY, _NOTICE_KEY)
#: Of those, the ones every revision must publish. A model may have nothing to attribute, but it
#: always has terms.
_REQUIRED = frozenset({_LICENCE_KEY})

#: How the cache names what is not a finished entry yet, and what a CoreML archive unpacks to.
#: Stated here because this module owns the cache's layout; :func:`_fetch`,
#: :func:`mozo.runtimes._unpacked` and :func:`cached` all read them from here.
_PARTIAL_SUFFIX = ".part"
_UNPACKED_SUFFIX = ".mlpackage"
_STAGING_SUFFIX = ".unpacking"

_manifest: dict[str, Any] | None = None

#: What :func:`configure` set. A key that is absent falls through to the environment, then to
#: the default, so clearing a setting is removing it rather than writing a second default here.
_settings: dict[str, Any] = {}

#: "Not passed" for :func:`configure`, where ``None`` already means "clear this setting".
_UNSET: Any = object()


_log = logging.getLogger(__name__)


class WeightsError(RuntimeError):
    """Raised when a model cannot be resolved, downloaded, or verified."""


class DownloadCancelled(WeightsError):
    """Raised when a caller's ``cancelled()`` said stop. The partial file is already gone."""


@dataclass
class _Watch:
    """One caller's interest in the downloads made on its behalf: where to report, when to stop.

    ``done`` accumulates across every file fetched under it, so a load that brings an encoder, a
    decoder and a licence reports one rising count rather than three that each restart at zero.
    """

    progress: Callable[[int, int], None] | None
    cancelled: Callable[[], bool] | None
    total: int
    done: int = 0


#: The watch in effect for this thread or task. A context variable rather than an argument
#: threaded through thirteen adapters: the download happens far below the call that cares about
#: it, and everything between them is the same whether anyone is watching or not. Worker threads
#: start with an empty context, so a pool that loads on someone's behalf has to copy it across.
_watch: ContextVar[_Watch | None] = ContextVar("mozo_weights_watch", default=None)

#: Files being downloaded by this process right now, so :func:`remove` can refuse rather than
#: delete a directory out from under a transfer. Another process sharing the cache is not seen.
_downloading: set[Path] = set()
_downloading_lock = threading.Lock()


@contextmanager
def watching(
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    *, total: int = 0,
) -> Iterator[None]:
    """Report and allow cancelling every download made inside this block, in this thread.

    Args:
        progress: Called as ``progress(done_bytes, total_bytes)``, with *done* counted across all
            files fetched in the block: at most every 0.1 s, and once as each file completes, so
            the last call is ``(total, total)``. It runs on the downloading thread.
        cancelled: Checked once per chunk. When it returns true the download raises
            :class:`DownloadCancelled`, its partial file is deleted, and nothing is cached.
        total: Bytes the block is expected to fetch, e.g. from :func:`download_size`. Grows if
            more arrives, so *done* never exceeds it.

    Examples:
        >>> with watching(progress=print):                          # doctest: +SKIP
        ...     mozo.get_model("sam2/large")
    """
    token = _watch.set(_Watch(progress, cancelled, total))
    try:
        yield
    finally:
        _watch.reset(token)


#: Framework prefixes that name an execution path. A revision also publishes data artifacts --
#: the licence, the label vocabulary -- and those are not things a model can be run as.
_RUNNABLE = ("torch", "onnx", "coreml", "tensorrt")


def runtime_of(key: str) -> str:
    """Return the runtime an artifact key belongs to.

    An artifact key is ``<framework>-<precision>`` with an optional ``-<part>`` on the end. Most
    families publish a runtime as one file and the key *is* the runtime; SAM 2 publishes its
    graphs split across an encoder and a decoder, because the expensive half depends only on the
    image and exporting them together would forfeit the reuse that makes clicking cheap.

    Examples:
        >>> runtime_of("torch-fp32")
        'torch-fp32'
        >>> runtime_of("onnx-fp32-encoder")
        'onnx-fp32'
    """
    return "-".join(key.split("-")[:2])


def part_of(key: str) -> str:
    """Return the part an artifact key names within its runtime, or ``""`` if it is the whole.

    Examples:
        >>> part_of("onnx-fp32-encoder")
        'encoder'
        >>> part_of("torch-fp32")
        ''
    """
    return "-".join(key.split("-")[2:])


def framework_of(key: str) -> str:
    """Return the framework an artifact key names, or the whole key if it names no runtime.

    Examples:
        >>> framework_of("onnx-fp32-encoder")
        'onnx'
        >>> framework_of("LICENSE")
        'LICENSE'
    """
    return key.split("-")[0]


def manifest() -> dict[str, Any]:
    """Return the bundled manifest, read once and kept for the life of the process."""
    global _manifest
    if _manifest is None:
        _manifest = json.loads(_MANIFEST_PATH.read_text())
    return _manifest


def configure(
    *, cache: str | Path | None = _UNSET, offline: bool | None = _UNSET,
    base_url: str | None = _UNSET,
) -> None:
    """Set where weights live, whether they may be fetched, and from where -- in code.

    Each wins over its environment variable (``MOZO_CACHE``, ``MOZO_OFFLINE``,
    ``MOZO_BASE_URL``), which stays the default when nothing is set here. An argument left out is
    left as it is; ``None`` clears it back to the environment. The settings are process-wide, as
    the environment is, and are read on every call, so a change applies to the next download.

    Args:
        cache: The cache root, e.g. a folder inside an application's own workspace.
        offline: ``True`` refuses downloads; a missing file raises with its path, URL and hash.
        base_url: A mirror to fetch from instead of the manifest's ``base_url``. A ``file://``
            URL pointing at a ``weights/`` tree works.

    Examples:
        >>> configure(cache="/tmp/my-app/weights")           # doctest: +SKIP
        >>> configure(cache=None)                           # doctest: +SKIP
    """
    for name, value in (("cache", cache), ("offline", offline), ("base_url", base_url)):
        if value is None:
            _settings.pop(name, None)
        elif value is not _UNSET:
            _settings[name] = value


def cache_dir() -> Path:
    """Return the download cache root: what :func:`configure` set, else ``$MOZO_CACHE``, else
    ``~/.cache/mozo``."""
    override = str(_settings.get("cache") or os.environ.get("MOZO_CACHE", "")).strip()
    base = Path(override) if override else Path.home() / ".cache" / "mozo"
    return base.expanduser()


def _base_url() -> str:
    """Return the URL artifacts are served from, honouring a mirror set in code or in
    ``MOZO_BASE_URL``."""
    override = (_settings.get("base_url") or os.environ.get("MOZO_BASE_URL", "")).strip()
    return (override or manifest()["base_url"]).rstrip("/")


def _lookup(family: str, variant: str, revision: str | None) -> tuple[str, dict[str, Any]]:
    """Return the resolved revision name and its manifest entry.

    Raises:
        WeightsError: If the model is not published, or the requested revision does not exist.
    """
    model_id = f"{family}/{variant}"
    model = manifest()["models"].get(model_id)
    if model is None:
        raise WeightsError(
            f"mozo publishes no weights for {model_id!r}. Either its licence does not permit "
            f"redistribution, or it runs only against checkpoints you supply -- pass the "
            f"checkpoint path to the adapter instead."
        )

    name = revision or model["latest"]
    entry = model["revisions"].get(name)
    if entry is None:
        available = ", ".join(sorted(model["revisions"]))
        raise WeightsError(f"{model_id}: no revision {name!r}. Published revisions: {available}")
    return name, entry


def _composing(entry: dict[str, Any], runtime: str) -> list[str]:
    """Return the artifact keys that compose *runtime*, in a stable order.

    One definition of "these files are that runtime", used both to fetch them and to explain
    why a runtime name is not itself a file.
    """
    return sorted(k for k in entry["artifacts"] if runtime_of(k) == runtime)


def _artifact(
    entry: dict[str, Any], model_id: str, revision: str, key: str
) -> dict[str, Any]:
    """Return one artifact record from a revision.

    Raises:
        WeightsError: If the revision does not publish that artifact.
    """
    artifact = entry["artifacts"].get(key)
    if artifact is None:
        # A runtime split across parts is the trap worth naming: ``select_runtime`` hands back
        # ``onnx-fp32``, every adapter passes that straight to ``resolve``, and for SAM 2 there
        # is no such file -- only an encoder and a decoder. Saying "not published" there would
        # contradict the list the caller was just given to choose from.
        composed = _composing(entry, key)
        if composed:
            raise WeightsError(
                f"{model_id} publishes {key!r} as {len(composed)} files, not one: "
                f"{', '.join(composed)}. Ask for one of those by name, or use "
                f"mozo.weights.parts() with {key!r} to get them all."
            )
        available = ", ".join(sorted(k for k in entry["artifacts"] if k not in _ACCOMPANYING))
        raise WeightsError(
            f"{model_id} revision {revision} does not publish {key!r}. Available: {available}"
        )
    return artifact


def _fetch(url: str, target: Path, *, size: int, sha256: str) -> None:
    """Download *url* to *target*, verifying size and hash before it takes its final name.

    The transfer lands on a sibling ``.part`` file and is hashed as it streams, so an interrupted
    or corrupted download can never be mistaken for a complete cache entry. The partial file's
    name is this writer's own, because two writers can want one file -- ``rfdetr/nano`` on two
    devices are two models sharing one checkpoint, and two processes can share one cache. Both
    produce the same verified bytes, so whichever renames second replaces it with itself.

    Progress goes to the :func:`watching` caller when there is one, and to the log otherwise,
    which is what the command line prints.

    Raises:
        DownloadCancelled: If the watching caller's ``cancelled()`` returned true.
        WeightsError: If the transfer fails, or the bytes do not match the manifest.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}{_PARTIAL_SUFFIX}")
    watch = _watch.get()
    if watch is not None:
        # Only ever raised, so *done* -- which a size check below holds to *size* -- stays under it.
        watch.total = max(watch.total, watch.done + size)
    report = watch.progress if watch is not None else None
    digest = hashlib.sha256()
    written = 0
    last_report = time.monotonic()

    with _downloading_lock:
        _downloading.add(target)
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response, partial.open("wb") as out:
            for chunk in iter(lambda: response.read(_DOWNLOAD_CHUNK_BYTES), b""):
                if watch is not None and watch.cancelled is not None and watch.cancelled():
                    raise DownloadCancelled(f"download of {target.name} was cancelled")
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if watch is not None:
                    watch.done += len(chunk)
                now = time.monotonic()
                if report is not None and now - last_report >= _PROGRESS_SECONDS:
                    report(watch.done, watch.total)
                    last_report = now
                elif watch is None and size and now - last_report > 1.0:
                    _log.info("  %s  %5.1f%%", target.name, 100 * written / size)
                    last_report = now
        if report is not None:
            report(watch.done, watch.total)  # every file ends on an exact count
    except BaseException as error:
        # Failed, cancelled, interrupted, or a callback that raised: nothing half-written stays.
        partial.unlink(missing_ok=True)
        if isinstance(error, OSError):
            raise WeightsError(f"failed to download {url}: {error}") from error
        raise
    finally:
        with _downloading_lock:
            _downloading.discard(target)

    if written != size or digest.hexdigest() != sha256:
        partial.unlink(missing_ok=True)
        raise WeightsError(
            f"{target.name} does not match the manifest "
            f"(expected {size} bytes / {sha256}, got {written} bytes / {digest.hexdigest()}). "
            f"The mirror at {url} may be stale or corrupt."
        )
    partial.replace(target)


def _offline() -> bool:
    """Is fetching forbidden? What :func:`configure` set wins; otherwise ``MOZO_OFFLINE`` unset,
    empty or ``0`` all mean no."""
    if "offline" in _settings:
        return bool(_settings["offline"])
    return os.environ.get("MOZO_OFFLINE", "").strip() not in ("", "0")


def _model_dir(family: str, variant: str) -> Path:
    """Where one model's revisions live in the cache. Every path into the cache starts here."""
    return cache_dir() / family / variant


def _target(artifact: dict[str, Any], revision_dir: Path) -> Path:
    """Where *artifact* lives in the cache: its file name, inside its revision's directory."""
    return revision_dir / Path(artifact["path"]).name


def _obtain(artifact: dict[str, Any], revision_dir: Path) -> Path:
    """Return the local path for *artifact*, downloading it if the cache does not already hold it.

    Raises:
        WeightsError: If the file is absent and offline mode forbids fetching it.
    """
    target = _target(artifact, revision_dir)
    if target.is_file():
        return target

    url = f"{_base_url()}/{artifact['path']}"
    if _offline():
        raise WeightsError(
            f"mozo is offline (MOZO_OFFLINE, or configure(offline=True)) and {target} is not "
            f"cached.\n"
            f"  fetch:  {url}\n"
            f"  place:  {target}\n"
            f"  sha256: {artifact['sha256']}"
        )

    _log.info("[mozo] downloading %s (%.1f MB)", target.name, artifact["size"] / 1e6)
    _fetch(url, target, size=artifact["size"], sha256=artifact["sha256"])
    return target


def revision_of(family: str, variant: str, *, revision: str | None = None) -> str:
    """Return the revision name a call would resolve to, without downloading anything.

    ``latest`` is a pointer, so "which weights is this" has an answer only once it is followed.
    Callers that record where a result came from -- an embedding written to a vector index, say,
    which is only comparable against others from the same weights -- need that answer without
    paying for the bytes.

    Raises:
        WeightsError: If the model or revision is not published.

    Examples:
        >>> revision_of("rfdetr", "small")          # doctest: +SKIP
        '2026-08-18'
    """
    name, _ = _lookup(family, variant, revision)
    return name


def license_of(family: str, variant: str, *, revision: str | None = None) -> str:
    """Return the licence a revision's weights are published under, e.g. ``"Apache-2.0"``.

    Answered from the manifest, so it costs no download and no torch import: a caller can show
    the terms, or filter on them, before deciding to fetch anything. The identifier is derived
    from the LICENSE file published beside the weights when the manifest is generated, and that
    file remains the terms themselves; this only names them. It is per revision because terms
    can change between releases.

    Only weights mozo publishes have one. A checkpoint you supply is under whatever terms you
    got it under, and mozo cannot know them.

    Raises:
        WeightsError: If the model or revision is not published.

    Examples:
        >>> license_of("yolov8", "nano")
        'AGPL-3.0'
        >>> license_of("rfdetr", "small")
        'Apache-2.0'
    """
    _, entry = _lookup(family, variant, revision)
    return entry["license"]


def artifacts(family: str, variant: str, *, revision: str | None = None) -> list[str]:
    """Return the artifact keys a revision publishes, e.g. ``["onnx-fp32", "torch-fp32"]``.

    ``LICENSE`` and ``NOTICE`` are omitted -- they ship with every artifact rather than being
    ones you choose.
    These are files. For a family that splits a runtime across several of them this lists the
    parts -- ``onnx-fp32-encoder`` and ``onnx-fp32-decoder`` rather than ``onnx-fp32`` -- and a
    part is not something ``runtime=`` accepts. :func:`mozo.runtimes.runnable` turns this into
    the list a caller may choose from.

    Raises:
        WeightsError: If the model or revision is not published.

    Examples:
        >>> artifacts("rfdetr", "small")  # doctest: +SKIP
        ['torch-fp32']
    """
    _, entry = _lookup(family, variant, revision)
    return sorted(k for k in entry["artifacts"] if k not in _ACCOMPANYING)


def companions(family: str, variant: str, *, revision: str | None = None) -> list[str]:
    """Return the accompanying artifact keys a revision publishes.

    The complement of :func:`artifacts`, which filters these out because they ship with whatever
    you asked for rather than being a thing you can run. A caller that needs to check a model's
    terms travelled with it asks here, instead of walking the manifest -- its layout is this
    module's to know.

    Raises:
        WeightsError: If the model or revision is not published.

    Examples:
        >>> companions("rfdetr", "small")  # doctest: +SKIP
        ['LICENSE']
    """
    _, entry = _lookup(family, variant, revision)
    return sorted(k for k in entry["artifacts"] if k in _ACCOMPANYING)


def parts(
    family: str,
    variant: str,
    runtime: str,
    *,
    revision: str | None = None,
) -> dict[str, Path]:
    """Return every artifact that composes one runtime, downloaded and verified.

    Most families publish a runtime as a single file, and this returns one entry keyed ``""``.
    SAM 2 splits its graphs into an encoder and a decoder -- the expensive half depends only on
    the image, and exporting them as one graph would forfeit the reuse that makes a second click
    cheap -- so this returns one entry per part, keyed by the part's name.

    Callers that already know they want one specific file should use :func:`resolve` with its
    exact key. This is for the case the caller does *not* know: it was handed a runtime name by
    :func:`mozo.runtimes.select_runtime` and needs whatever that runtime is made of.

    Args:
        family: Model family, e.g. ``"sam2"``.
        variant: Variant within it, e.g. ``"large"``.
        runtime: A runtime name as :func:`mozo.runtimes.runnable` reports it, e.g.
            ``"onnx-fp32"``. Not an artifact key with a part on the end.
        revision: Published revision to pin. Defaults to the manifest's ``latest``.

    Returns:
        Part name to local path, sorted by part name. A single-file runtime is ``{"": path}``;
        SAM 2's ONNX is ``{"decoder": ..., "encoder": ...}``.

    Raises:
        WeightsError: If the model or revision is not published, if *runtime* is not one this
            revision can be executed as, or if any part fails to download or verify. A part that
            cannot be obtained does not stop the others being tried, so an offline caller is told
            about every file it has to place rather than one per run.

    Examples:
        >>> parts("sam2", "large", "onnx-fp32")           # doctest: +SKIP
        {'decoder': PosixPath(...), 'encoder': PosixPath(...)}
        >>> parts("rfdetr", "small", "torch-fp32")        # doctest: +SKIP
        {'': PosixPath(...)}
    """
    from .runtimes import runnable

    _, entry = _lookup(family, variant, revision)
    # Through ``runnable`` rather than by excluding the licence: ``labels`` is neither a
    # companion nor something a model can be executed as, and answering with it here would
    # contradict the list a caller was given to choose from.
    available = runnable(sorted(entry["artifacts"]))
    if runtime not in available:
        raise WeightsError(
            f"{family}/{variant} publishes no {runtime!r}. "
            f"Available runtimes: {', '.join(available) or 'none'}"
        )

    keys = _composing(entry, runtime)

    # Offline, none of these touches the network, so collecting every complaint costs nothing
    # and spares a caller placing files by hand from learning about them one run at a time.
    # A real download is the opposite trade: keys are sorted, so the small part is tried first,
    # and carrying on after it fails would start SAM 2's 852 MB encoder to produce an error
    # already known -- two more minutes of timeouts for an answer nobody will use.
    if _offline():
        missing = []
        for key in keys:
            try:
                resolve(family, variant, key, revision=revision)
            except WeightsError as error:
                missing.append(str(error))
        if missing:
            raise WeightsError(
                f"{runtime} is {len(keys)} files and {len(missing)} are not cached:\n\n"
                + "\n\n".join(missing)
            )

    return {part_of(key): resolve(family, variant, key, revision=revision) for key in keys}


def resolve(
    family: str,
    variant: str,
    key: str = "torch-fp32",
    *,
    revision: str | None = None,
) -> Path:
    """Return a local path to one published artifact, downloading and verifying it if needed.

    The artifact's licence is fetched into the same directory, along with its NOTICE where one
    is published, so a cached model always sits beside the terms it was published under and the
    attribution those terms require.

    Args:
        family: Model family, e.g. ``"rfdetr"``.
        variant: Variant within that family, e.g. ``"small"``.
        key: Which artifact, as :func:`artifacts` reports it. Runnable ones are named
            ``<runtime>-<precision>``; ``labels`` and ``LICENSE`` are not, which is why this is
            one opaque string rather than two halves that would have to be rejoined.
        revision: Published revision to pin. Defaults to the manifest's ``latest``.

    Returns:
        Path to the artifact inside the cache. Its revision is part of the path, so a pinned
        revision is honoured even when another is already cached.

    Raises:
        WeightsError: If the model, revision, or artifact is not published; if the download
            fails or does not match the manifest; or if it is absent while mozo is offline.

    Examples:
        >>> resolve("rfdetr", "small")                            # doctest: +SKIP
        >>> resolve("rfdetr", "small", "onnx-fp32")               # doctest: +SKIP
        >>> resolve("rfdetr", "small", revision="2026-08-18")     # doctest: +SKIP
    """
    revision_name, entry = _lookup(family, variant, revision)
    model_id = f"{family}/{variant}"

    artifact = _artifact(entry, model_id, revision_name, key)

    revision_dir = _model_dir(family, variant) / revision_name
    path = _obtain(artifact, revision_dir)
    for companion in _ACCOMPANYING:
        # The licence is required of every revision and _artifact says so if it is missing; a
        # NOTICE is published only where the terms ask for attribution to travel with the copy.
        published = entry["artifacts"].get(companion)
        if published is None and companion not in _REQUIRED:
            continue
        _obtain(_artifact(entry, model_id, revision_name, companion), revision_dir)
    return path


# -- The cache, as an inventory ---------------------------------------------------------------
#
# Everything below answers from the manifest and the disk. None of it touches the network, and
# none of it imports torch as long as a device or an explicit runtime is given: ``auto`` with
# neither has to ask torch which device this machine has, exactly as a load would.


def _planned(
    family: str, variant: str, runtime: str, device: str | None, revision: str | None,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Return where a load of this model would cache its files, and which files those are.

    The runtime is chosen as the adapter chooses it -- the same :func:`select_runtime`, the same
    frameworks from :data:`mozo.registry.EXECUTES` -- then expanded to every part it is made of,
    plus what :data:`mozo.registry.NEEDS` says that runtime brings with it, plus the licence and
    NOTICE every download carries. A key the revision does not publish is skipped, as the adapter
    skips it.

    Raises:
        WeightsError: If the model or revision is not published.
        mozo.runtimes.RuntimeError_: If *runtime* is not published, or not one this family runs.
    """
    from .registry import EXECUTES, NEEDS
    from .runtimes import get_default_device, select_runtime

    name, entry = _lookup(family, variant, revision)
    published = entry["artifacts"]
    if runtime == "auto" and device is None:
        device = get_default_device()  # the one path here that imports torch
    chosen = select_runtime(device or "cpu", sorted(published), runtime,
                            executes=EXECUTES.get(family))

    brings = NEEDS.get(family, {})
    keys = [*_composing(entry, chosen),
            *brings.get(framework_of(chosen), ()), *brings.get("*", ()), *_ACCOMPANYING]
    files = {key: published[key] for key in dict.fromkeys(keys) if key in published}
    return _model_dir(family, variant) / name, files


def _missing(directory: Path, files: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Of a :func:`_planned` load's files, the ones the cache does not hold yet."""
    return [artifact for artifact in files.values() if not _target(artifact, directory).is_file()]


def is_cached(
    family: str, variant: str, runtime: str = "auto", *,
    device: str | None = None, revision: str | None = None,
) -> bool:
    """Would loading this model need nothing from the network?

    Args:
        family: Model family, e.g. ``"sam2"``.
        variant: Variant within it, e.g. ``"large"``.
        runtime: ``"auto"`` for whatever a load on *device* would pick, or a runtime by name.
        device: The device a load would run on. With ``"auto"`` it decides the runtime; leaving
            it out asks torch for this machine's best, as a load would.
        revision: Published revision to pin. Defaults to the manifest's ``latest``.

    Raises:
        WeightsError: If the model or revision is not published.
        mozo.runtimes.RuntimeError_: If *runtime* is not one this model can run.

    Examples:
        >>> is_cached("sam2", "large", device="mps")            # doctest: +SKIP
        False
    """
    return not _missing(*_planned(family, variant, runtime, device, revision))


def download_size(
    family: str, variant: str, runtime: str = "auto", *,
    device: str | None = None, revision: str | None = None,
) -> int:
    """Bytes a load of this model would still have to fetch, from the manifest's sizes.

    Zero once :func:`is_cached` is true. Counts every file the load needs that the cache does not
    already hold -- each part of a split runtime, a labels file, the licence and NOTICE -- so a
    host can say "this will download 852 MB" before starting. Arguments as for :func:`is_cached`.

    Examples:
        >>> download_size("sam2", "large", device="mps")        # doctest: +SKIP
        898465192
    """
    return sum(a["size"] for a in _missing(*_planned(family, variant, runtime, device, revision)))


def model_size(
    family: str, variant: str, runtime: str = "auto", *,
    device: str | None = None, revision: str | None = None,
) -> int:
    """Bytes of the weights a load of this model runs from, cached or not.

    Every part of the runtime and what it brings with it, but not the licence or NOTICE: an
    estimate of what the model occupies once loaded, which is what a host deciding what to
    unload weighs. Arguments as for :func:`is_cached`.

    Examples:
        >>> model_size("rfdetr", "large", "torch-fp32")         # doctest: +SKIP
        135957704
    """
    _, files = _planned(family, variant, runtime, device, revision)
    return sum(a["size"] for key, a in files.items() if key not in _ACCOMPANYING)


def fetch(
    family: str, variant: str, runtime: str = "auto", *,
    device: str | None = None, revision: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """Download everything a load of this model needs, without loading it.

    For a host that downloads in one job and loads in another -- the network on an I/O lane, the
    model on the GPU lane. Afterwards :func:`is_cached` is true and the load fetches nothing.
    Arguments as for :func:`is_cached`, plus *progress* and *cancelled* as for :func:`watching`;
    *total* is :func:`download_size`, so *done* ends exactly there.

    Returns:
        Artifact key to local path, for every file the load will read.

    Raises:
        DownloadCancelled: If *cancelled* returned true. Files finished before it stay cached;
            the one in flight leaves nothing.
        WeightsError: If the model or revision is not published, a download fails, or mozo is
            offline and something is missing.

    Examples:
        >>> fetch("sam2", "large", device="mps", progress=print)    # doctest: +SKIP
    """
    directory, files = _planned(family, variant, runtime, device, revision)
    with watching(progress, cancelled, total=sum(a["size"] for a in _missing(directory, files))):
        return {key: _obtain(artifact, directory) for key, artifact in files.items()}


def _size(path: Path) -> int:
    """Bytes on disk under *path*: a file's size, or everything inside a directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def cached() -> list[dict[str, Any]]:
    """Everything in the cache, one record per file or unpacked directory, in path order.

    Each record is ``{family, variant, revision, key, runtime, size, path, state}``:

    - ``key`` is the manifest's artifact key (``"torch-fp32"``, ``"LICENSE"``), or ``None`` for
      something the manifest does not list.
    - ``runtime`` is the runtime that key belongs to, or ``None`` for a licence, NOTICE or labels.
    - ``state`` is ``"complete"`` for a verified file the manifest lists -- a file only takes its
      final name after its size and hash match; ``"unpacked"`` for the ``.mlpackage`` a CoreML
      archive was extracted to; ``"partial"`` for an interrupted download or extraction; and
      ``"unlisted"`` for anything this version's manifest does not know, such as a revision an
      upgrade has moved past.

    Grouping is the caller's: by ``(family, variant, revision)`` for a model, by ``runtime`` for
    what one load uses, by ``state`` for what can go.
    """
    from .runtimes import runnable

    root = cache_dir()
    records = []
    for revision_dir in sorted(p for p in root.glob("*/*/*") if p.is_dir()):
        family, variant, revision = revision_dir.relative_to(root).parts
        try:
            _, entry = _lookup(family, variant, revision)
            by_name = {Path(a["path"]).name: key for key, a in entry["artifacts"].items()}
        except WeightsError:
            by_name = {}  # a model or revision this version's manifest does not list
        # An unpacked or half-unpacked CoreML package shares its archive's stem.
        by_stem = {Path(name).stem: key for name, key in by_name.items()}

        for path in sorted(revision_dir.iterdir()):
            if path.name.startswith("."):
                continue
            if path.name.endswith(_PARTIAL_SUFFIX):
                # "<file>.<writer>.part"; older releases wrote "<file>.part".
                stem = path.name[: -len(_PARTIAL_SUFFIX)]
                key, state = by_name.get(stem) or by_name.get(stem.rsplit(".", 1)[0]), "partial"
            elif path.suffix == _STAGING_SUFFIX:
                key, state = by_stem.get(path.stem), "partial"
            elif path.suffix == _UNPACKED_SUFFIX:
                key = by_stem.get(path.stem)
                state = "unpacked" if key else "unlisted"
            else:
                key = by_name.get(path.name)
                state = "complete" if key else "unlisted"
            records.append({
                "family": family, "variant": variant, "revision": revision, "key": key,
                "runtime": next(iter(runnable([key])), None) if key else None,
                "size": _size(path), "path": path, "state": state,
            })
    return records


def remove(family: str, variant: str, revision: str | None = None) -> int:
    """Delete a model's cached files, and nothing else. Returns the bytes freed.

    Everything under that model's revision directory goes -- every runtime, its unpacked CoreML
    package, any partial download, the licence and NOTICE -- and so does the directory, and the
    variant's and family's once they are empty. Nothing outside them is touched.

    A model already loaded keeps working: its weights are in memory. The next load of it
    downloads again.

    Args:
        family: Model family, e.g. ``"sam3"``.
        variant: Variant within it.
        revision: One revision to remove. ``None`` removes every cached revision of the variant,
            including ones the manifest no longer lists.

    Raises:
        ValueError: If a name is empty or would reach outside the cache.
        WeightsError: If this process is downloading one of its files right now.

    Examples:
        >>> remove("sam3", "sam3")                              # doctest: +SKIP
        3449283301
    """
    names = (family, variant) if revision is None else (family, variant, revision)
    # Each must be a single path component, so nothing joined from them can leave the cache.
    if any(n in ("", ".", "..") or Path(n).name != n for n in names):
        raise ValueError(f"not a model in the cache: {'/'.join(names)!r}")

    variant_dir = _model_dir(family, variant)
    doomed = [variant_dir / revision] if revision else sorted(variant_dir.glob("*"))
    with _downloading_lock:
        busy = [t for t in _downloading if any(t.is_relative_to(d) for d in doomed)]
    if busy:
        raise WeightsError(
            f"{family}/{variant} is downloading ({', '.join(t.name for t in busy)}). "
            "Cancel it, or wait for it to finish, before removing it."
        )
    freed = 0
    for directory in doomed:
        if directory.is_dir():
            freed += _size(directory)
            shutil.rmtree(directory)
    for directory in (variant_dir, variant_dir.parent):
        # rmdir rather than rmtree: it refuses a directory that still holds anything.
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return freed
