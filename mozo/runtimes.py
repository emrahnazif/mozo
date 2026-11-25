"""Run a published artifact, whatever it happens to be.

A model is an architecture plus weights, but at inference time it is just a function from an
input array to a tuple of output arrays. That function can come from several places -- a torch
module, an ONNX graph, a CoreML package -- and which one a machine should use depends on what
mozo publishes for that model and what the machine can actually execute.

This module owns the artifact-shaped half of that: everything here works from a file and knows
nothing about any particular model. Torch runners are built by the adapter that owns the
architecture, because only it knows how to turn a checkpoint into a module.

    >>> from mozo.runtimes import OnnxRunner
    >>> run = OnnxRunner(path, device="cuda")            # doctest: +SKIP
    >>> boxes, logits = run(batch)                       # doctest: +SKIP

Pre- and post-processing stay with the model, not here: they are model maths and must be
identical whichever runtime produced the numbers in between.
"""

from __future__ import annotations

import functools

__all__ = [
    "CoreMLRunner",
    "OnnxRunner",
    "RuntimeError_",
    "TensorRTRunner",
    "coreml_functions",
    "executable",
    "get_default_device",
    "make_runner",
    "part_of",
    "providers_for",
    "runnable",
    "runtime_of",
    "select_runtime",
]

import logging
import shutil
import zipfile
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .weights import (_ENGINES_SUFFIX, _PARTIAL_SUFFIX, _RUNNABLE, _STAGING_SUFFIX,
                      _UNPACKED_SUFFIX, framework_of, part_of, runtime_of)

#: Cached: the answer cannot change within a process, and ``torch.cuda.is_available()`` costs
#: real time the first time it is asked.
_default_device: str | None = None


_log = logging.getLogger(__name__)


def free_device_memory() -> None:
    """Collect, then hand the accelerators' cached blocks back to the driver.

    Dropping the last reference to a model frees its tensors into torch's caching allocator, not
    to the device: another process -- a training run about to start -- sees no more free memory
    until the cache is emptied. torch is only consulted if something already imported it; a
    process that never loaded a model has nothing to give back.
    """
    import gc
    import sys

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_default_device() -> str:
    """Return the best device this machine has: ``cuda``, else ``mps``, else ``cpu``.

    Which artifact to run and what to run it on are one question -- :func:`select_runtime` takes
    the answer as its first argument -- so they are answered in one place.

    Examples:
        >>> get_default_device() in {"cuda", "mps", "cpu"}
        True
    """
    global _default_device
    if _default_device is None:
        # torch is an unconditional dependency, and every caller of this is an adapter that has
        # already imported it. The module that must answer without torch is mozo.registry, and
        # it earns that by importing nothing at all.
        import torch

        if torch.cuda.is_available():
            _default_device = "cuda"
        elif torch.backends.mps.is_available():
            _default_device = "mps"
        else:
            _default_device = "cpu"
    return _default_device


#: Preferred artifact per device, best first.
#:
#: Torch leads everywhere, because it is the only runtime measured on every device and the one
#: every model publishes. It is tempting to assume ONNX Runtime wins on CPU -- it is the usual
#: claim -- but on Apple silicon, RF-DETR small measures p50 136 ms on torch against 184 ms
#: through ONNX Runtime on the same CPU, and 106 ms on torch MPS against 183 ms for ONNX, which
#: has no MPS path at all and quietly runs on the CPU. Defaulting to a guess would have shipped
#: that regression silently to everyone on a Mac.
#:
#: One fp16 entry appears here, ``torch-fp16`` on CUDA, and it only exists for families that
#: derive it (:data:`mozo.registry.DERIVES`), each measured first. OWLv2's is 4.4x faster on an
#: RTX 3090 (5.5x for ``large``) at unchanged COCO AP, changing 0.4-1.4% of detections across a
#: 0.3 threshold against fp32. Elsewhere, where fp16 was tried for a whole-file runtime, it
#: measured worse on accuracy and no better on speed; and ``coreml-fp16``, which SAM 2's three
#: smaller variants do publish, has never been measured against torch MPS for a promptable model.
#: This table's whole claim is that its entries were measured, so an unmeasured one does not join
#: it. ``tensorrt-fp16`` was measured and is deliberately absent: 2.3-4.8% of OWLv2's confident
#: detections change (TensorRT's fp16 kernels are less exact than torch's on it), and Grounding
#: DINO's runs every image at one square size, so a user has to ask for it.
#:
#: What that absence buys is narrower than it looks. Entries are matched by exact membership
#: (``if key in usable``), so leaving a runtime out means ``auto`` will not *prefer* it -- but
#: the last-resort branch below still takes whatever is executable when nothing preferred is
#: published, and a variant shipping only fp16 would get it. Absence is a preference, not a veto.
#:
#: ``tensorrt-fp32`` leads on CUDA, so a family that derives it (:data:`mozo.registry.DERIVES`)
#: opts into ``auto`` choosing it there; each entry records the measurement that justified it --
#: exact against fp32, and faster. It is executable only with TensorRT 11 or later installed
#: (:func:`executable`), so a machine without it stays on torch.
#:
#: CoreML leads on Apple silicon because it was measured to: RF-DETR nano runs 10.8 ms through
#: CoreML against 53.3 ms on torch MPS, five times faster, at a worst output delta of 0.001.
#: That is the one entry here that overrides torch, and it is the one with the widest margin.
#:
#: It is not always that wide, and the second measurement is worth stating beside the first so
#: the entry is not read as a promise. SAM 3's vision encoder runs 840 ms through CoreML against
#: 1098 ms on torch MPS -- 1.3x, not 5x -- and 4488 ms on torch CPU. Both models prefer CoreML
#: here; only one of them dramatically. A family whose margin came out under 1.0 would belong in
#: ``EXECUTES`` or unpublished, not in a third preference table.
_PREFERENCE: dict[str, tuple[str, ...]] = {
    "cuda": ("tensorrt-fp32", "torch-fp16", "torch-fp32", "onnx-fp32"),
    "cpu": ("torch-fp32", "onnx-fp32"),
    "mps": ("coreml-fp32", "torch-fp32", "onnx-fp32"),
}


#: What each runtime needs importable before it can execute anything. torch is a core
#: dependency and always present; the rest are the caller's to install. Which frameworks *name*
#: a runtime is a fact about artifact keys and lives in :mod:`mozo.weights`; what it takes to
#: execute one is this module's business.
_REQUIRES = {"onnx": "onnxruntime", "coreml": "coremltools"}

#: Frameworks whose availability is more than an importable module, checked by a function
#: instead of :data:`_REQUIRES`. TensorRT goes by version: a TensorRT 10 machine has the module,
#: and ``auto`` choosing a runtime the runner then refuses would break the default path.
_AVAILABLE = {"tensorrt": lambda: _tensorrt_usable()}

#: Runtimes that run on one kind of device only, by runtime name or by framework. TensorRT is
#: NVIDIA's. ``torch-fp16`` is offered on CUDA alone because that is the only device it was
#: measured on: half precision on a CPU is slower than fp32, and on MPS it is unmeasured.
#: ``torch-bf16`` likewise: Qwen3-VL in bfloat16 on a CPU measured 2.7x slower than fp32 *and*
#: changed the answer from its second token.
_DEVICES: dict[str, tuple[str, ...]] = {
    "torch-fp16": ("cuda",), "torch-bf16": ("cuda",), "tensorrt": ("cuda",)}


def _devices_for(runtime: str) -> tuple[str, ...] | None:
    """The device kinds *runtime* runs on, or ``None`` if it runs on any."""
    return _DEVICES.get(runtime) or _DEVICES.get(framework_of(runtime))


def _runs_on(runtime: str, device: str | None) -> bool:
    kinds = _devices_for(runtime)
    return kinds is None or device is None or device.split(":")[0] in kinds


def runnable(published: list[str], derives: dict[str, str] | None = None) -> list[str]:
    """Return the runtimes these artifacts can be executed as, each named once.

    Keys that only differ by their part collapse into the runtime they compose, so a caller
    choosing a runtime sees ``onnx-fp32`` rather than an encoder and a decoder it would have to
    know to rejoin. :func:`mozo.weights.parts` is what turns the answer back into files.

    Args:
        published: Artifact keys, as :func:`mozo.weights.artifacts` lists them.
        derives: The family's entry in :data:`mozo.registry.DERIVES`: runtimes built at load
            time from another artifact. Each is offered only when its source is published.

    Examples:
        >>> runnable(["labels", "onnx-fp16", "torch-fp32"])
        ['onnx-fp16', 'torch-fp32']
        >>> runnable(["onnx-fp32-decoder", "onnx-fp32-encoder", "torch-fp32"])
        ['onnx-fp32', 'torch-fp32']
        >>> runnable(["torch-fp32"], {"torch-fp16": "torch-fp32", "tensorrt-fp16": "onnx-fp16"})
        ['torch-fp32', 'torch-fp16']
    """
    offered = list(dict.fromkeys(
        runtime_of(key) for key in published if framework_of(key) in _RUNNABLE
    ))
    return offered + [runtime for runtime, source in (derives or {}).items()
                      if source in offered and runtime not in offered]


def executable(published: list[str], derives: dict[str, str] | None = None,
               device: str | None = None) -> list[str]:
    """Return the runtimes this machine can actually run, not merely the ones published.

    Publishing a CoreML artifact does not put ``coremltools`` on the machine reading the
    manifest, and ``auto`` choosing a runtime whose library is missing would make the default
    path fail on a working install. Asking for one by name still raises -- with instructions --
    because that is a request that cannot be honoured rather than a preference to work around.

    Given *device*, runtimes that cannot run there are dropped first, so checking a library is
    only paid for where it could matter: a CPU load never imports TensorRT.
    """
    from importlib.util import find_spec

    def installed(framework: str) -> bool:
        if (check := _AVAILABLE.get(framework)) is not None:
            return check()
        module = _REQUIRES.get(framework)
        return module is None or find_spec(module) is not None

    return [runtime for runtime in runnable(published, derives)
            if (device is None or _runs_on(runtime, device)) and installed(framework_of(runtime))]


#: ONNX Runtime reports input types as strings; these are the ones an artifact can declare.
_ORT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16}


class RuntimeError_(RuntimeError):
    """Raised when an artifact cannot be executed on the requested device."""


def providers_for(device: str) -> list[str]:
    """Return the ONNX Runtime execution providers to try for *device*, best first.

    The CPU provider is always last, because a session that cannot place a node anywhere else
    must still run rather than fail.
    """
    if device.startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # Everything else, Apple silicon included, runs on the CPU provider. Not
    # CoreMLExecutionProvider: it claims partial support and then fails -- on RF-DETR it placed
    # 608 of 1415 nodes, split the graph into 189 partitions, and errored at run time. Reaching
    # Apple's accelerators needs a real CoreML artifact, not this provider.
    return ["CPUExecutionProvider"]


def select_runtime(
    device: str | None,
    published: list[str],
    requested: str = "auto",
    executes: tuple[str, ...] | None = None,
    derives: dict[str, str] | None = None,
) -> str:
    """Choose which published artifact to run on *device*.

    Args:
        device: Where the model will run -- ``cpu``, ``cuda``, ``mps``. ``None`` only with a
            runtime named in *requested*, to ask whether it exists without saying where.
        published: Artifact keys the model publishes, from :func:`mozo.weights.artifacts`.
        requested: An explicit runtime such as ``"onnx-fp16"``, or ``"auto"`` to take the best
            published option for the device.
        executes: Frameworks the *caller* can run, e.g. ``("torch",)``. A published artifact is
            no use to an adapter that has no code to execute it, and that is a fact about the
            adapter rather than about the machine -- so it belongs here, beside the two filters
            that already narrow the choice, and not in a check the caller makes after this
            function has already answered. ``None`` means the caller can run anything published.
        derives: Runtimes the family builds at load time from another published artifact, from
            :data:`mozo.registry.DERIVES`. Offered where their source is published.

    Returns:
        A runtime name, which is an artifact key for a family that publishes its runtime as one
        file and the shared prefix of several for a family that splits it. Either way it is what
        :func:`mozo.weights.parts` takes.

    Raises:
        RuntimeError_: If *requested* is not published, does not run on *device*, or nothing
            suitable exists.

    Examples:
        >>> select_runtime("cpu", ["onnx-fp32", "torch-fp32"])
        'torch-fp32'
        >>> select_runtime("cpu", ["onnx-fp32"])
        'onnx-fp32'
        >>> select_runtime("cpu", ["onnx-fp32", "torch-fp32"], requested="onnx-fp32")
        'onnx-fp32'
        >>> select_runtime("cpu", ["onnx-fp32-decoder", "onnx-fp32-encoder", "torch-fp32"])
        'torch-fp32'
        >>> select_runtime("cpu", ["onnx-fp32", "torch-fp32"], executes=("torch",))
        'torch-fp32'
    """
    offered = runnable(published, derives)
    supported = ([key for key in offered if framework_of(key) in executes]
                 if executes is not None else offered)

    if requested != "auto":
        if requested not in offered:
            raise RuntimeError_(
                f"{requested!r} is not published for this model. "
                f"Available: {', '.join(offered)}"
            )
        if requested not in supported:
            raise RuntimeError_(
                f"{requested!r} is published for this model, but this family can only execute "
                f"{', '.join(executes)}. Available: {', '.join(supported) or 'nothing'}"
            )
        if not _runs_on(requested, device):
            raise RuntimeError_(
                f"{requested!r} runs only on {' or '.join(_devices_for(requested))}, and this "
                f"model is on {device!r}. Available here: "
                f"{', '.join(k for k in supported if _runs_on(k, device)) or 'nothing'}"
            )
        return requested

    usable = [key for key in executable(published, derives, device) if key in supported]
    for key in _PREFERENCE.get(device.split(":")[0], _PREFERENCE["cpu"]):
        if key in usable:
            return key

    # Nothing preferred is available, but something executable is -- take it rather than refuse.
    if usable:
        return usable[0]
    raise RuntimeError_(
        "nothing this model publishes can run here. Published: "
        f"{', '.join(offered) or 'nothing runnable'}"
    )


def make_runner(path: str | Path, key: str, device: str = "cpu"):
    """Return a runner for one artifact, chosen by its key.

    Args:
        path: The artifact on disk.
        key: Its artifact key, e.g. ``"onnx-fp32"`` or ``"coreml-fp32"``.
        device: Device the caller wants, where the runtime can be pointed at one.

    Raises:
        RuntimeError_: If nothing here can execute that kind of artifact. Torch artifacts are
            excluded deliberately: building a module from a checkpoint needs the architecture,
            which only the family's adapter has.
    """
    kind = key.split("-")[0]
    if kind == "onnx":
        return OnnxRunner(path, device=device)
    if kind == "coreml":
        return CoreMLRunner(path)
    raise RuntimeError_(f"no runner for {key!r}; torch artifacts are built by their adapter")


def _unpacked(archive: Path) -> Path:
    """Return the ``.mlpackage`` directory for *archive*, extracting it beside itself once.

    A CoreML package is a directory, and an artifact is a single verified file, so it travels
    zipped. Extraction lands next to the archive in the cache and is skipped when it is already
    there -- the cache entry is immutable, so one unpack per revision is all it can need.
    """
    if archive.is_dir():
        return archive
    unpacked = archive.with_suffix(_UNPACKED_SUFFIX)
    if not unpacked.is_dir():
        staging = archive.with_suffix(_STAGING_SUFFIX)
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
        staging.replace(unpacked)
    return unpacked


#: Each package compiled once per process, by its unpacked path. CoreML compiles a whole package
#: to load any function of it, so without this every function loaded the package's every function
#: again: about 0.8 s of each load, and a temporary copy of all its weights.
_compiled: dict[Path, str] = {}
_compiling = Lock()


def _compile(package: Path) -> str:
    """The compiled ``.mlmodelc`` for *package*, compiled the first time it is asked for."""
    import coremltools as ct

    with _compiling:
        if package not in _compiled:
            _compiled[package] = ct.utils.compile_model(str(package))
        return _compiled[package]


def coreml_functions(path: str | Path) -> dict[str, tuple]:
    """Each function of a multifunction CoreML package, with its first input's shape.

    Read from the package's specification alone, so nothing is compiled or loaded. Empty for a
    package with one function. A package can hold one graph at several fixed shapes, sharing its
    weights; the shapes are what a caller chooses between, so they are what this returns.
    """
    import coremltools as ct

    spec = ct.utils.load_spec(str(_unpacked(Path(path))))
    return {function.name: tuple(function.input[0].type.multiArrayType.shape)
            for function in spec.description.functions}


class CoreMLRunner:
    """A CoreML package loaded for inference, callable on a batch.

    CoreML picks its own hardware and offers no execution-provider knob: it schedules across the
    ANE, the GPU and the CPU as it sees fit. On RF-DETR the ANE contributes nothing measurable
    (compute units ALL and CPU_AND_GPU both run at 10.8 ms, CPU-only at 42 ms), so no attempt is
    made to steer it -- the default is already what the measurements were taken with.

    That the default is safe is a property of what mozo publishes, not of CoreML. The ANE is not
    always neutral: SAM 3's vision encoder at fp16 runs 750 ms on CPU_AND_GPU, 1826 ms on ALL,
    and did not finish a single pass in two minutes on CPU_AND_NE -- offered the ANE, it takes
    segments and loses badly. Every CoreML artifact published here is fp32, which the ANE cannot
    execute at all, so every one of them is scheduled the same way however this is set. A family
    that wanted fp16 would need this class to grow a compute-units argument first; see
    ``tools/export/sam3.py`` for why none does.

    Args:
        path: The package, or the zip it travels as.
        function: Which function of a multifunction package to run, by name; ``None`` for the
            default. :func:`coreml_functions` lists them.

    Attributes:
        inputs: Input tensor names, in the order the package declares them.
        outputs: Output tensor names, in the order the package declares them.
        input_shape: The first input's declared shape, as :class:`OnnxRunner` reports it -- so an
            adapter can ask any runner what size to letterbox to without knowing which it holds.

    Examples:
        >>> run = CoreMLRunner("model.mlpackage")             # doctest: +SKIP
        >>> boxes, logits = run(batch)                        # doctest: +SKIP
    """

    def __init__(self, path: str | Path, function: str | None = None) -> None:
        try:
            import coremltools as ct
        except ImportError as error:  # pragma: no cover - depends on the install
            raise RuntimeError_(
                "coremltools is not installed, so CoreML artifacts cannot be run."
            ) from error

        package = _unpacked(Path(path))
        description = ct.utils.load_spec(str(package)).description
        if description.functions:
            # A multifunction package declares its inputs per function, not at the top.
            function = function or description.defaultFunctionName
            description = next(f for f in description.functions if f.name == function)
        self._model = ct.models.CompiledMLModel(_compile(package), function_name=function)
        self.inputs: list[str] = [i.name for i in description.input]
        self.outputs: list[str] = [o.name for o in description.output]
        self.input_shape: tuple = tuple(description.input[0].type.multiArrayType.shape)

    def __call__(self, batch: np.ndarray, **extra: Any) -> tuple[np.ndarray, ...]:
        """Run the package and return its outputs in declaration order."""
        feed: dict[str, Any] = {self.inputs[0]: batch}
        feed.update(extra)
        got = self._model.predict(feed)
        return tuple(np.asarray(got[name]) for name in self.outputs)


class OnnxRunner:
    """An ONNX graph loaded into an inference session, callable on a batch.

    ONNX Runtime falls back to the CPU provider silently when the one you asked for is missing
    -- a GPU box with the CPU-only ``onnxruntime`` wheel installed will happily serve at a
    fraction of the speed and say nothing. :attr:`provider` records what actually got used, and
    a fallback prints a line rather than passing unnoticed.

    Args:
        path: Path to the ``.onnx`` file.
        device: Device the caller wants, used to order execution providers.

    Attributes:
        provider: The execution provider the session actually placed nodes on.
        inputs: Input tensor names, in the order the graph declares them.
        outputs: Output tensor names, in the order the graph declares them.
        input_dtype: What the graph wants fed to it. An fp16 artifact declares fp16 inputs, and
            callers should not have to know that -- :meth:`__call__` casts.
        input_shape: The first input's declared shape. A graph exported at a fixed side fixes the
            size its caller must letterbox to, and asking it beats assuming.

    Examples:
        >>> run = OnnxRunner("model.onnx", device="cpu")  # doctest: +SKIP
        >>> outputs = run(np.zeros((1, 3, 512, 512), dtype="float32"))  # doctest: +SKIP
    """

    def __init__(self, path: str | Path, device: str = "cpu") -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - depends on the install
            raise RuntimeError_(
                "onnxruntime is not installed, so ONNX artifacts cannot be run. "
                "Install it, or select a torch runtime."
            ) from error

        wanted = providers_for(device)
        available = ort.get_available_providers()
        usable = [p for p in wanted if p in available]

        self._session = ort.InferenceSession(str(path), providers=usable)
        self.provider: str = self._session.get_providers()[0]
        self.inputs: list[str] = [i.name for i in self._session.get_inputs()]
        self.outputs: list[str] = [o.name for o in self._session.get_outputs()]
        self.input_dtype = _ORT_DTYPES.get(self._session.get_inputs()[0].type, np.float32)
        self.input_shape: tuple = tuple(self._session.get_inputs()[0].shape)

        if self.provider != wanted[0]:
            _log.warning(
                "[mozo] %s: asked for %s, running on %s. Latency will not reflect %s.",
                Path(path).name, wanted[0], self.provider, device,
            )

    def __call__(self, batch: np.ndarray, **extra: Any) -> tuple[np.ndarray, ...]:
        """Run the graph and return its outputs in declaration order.

        Args:
            batch: The first input, typically an ``NCHW`` float32 array.
            **extra: Any further inputs the graph declares, by name.

        Returns:
            One array per graph output, in the order :attr:`outputs` lists them.
        """
        if batch.dtype != self.input_dtype:
            batch = batch.astype(self.input_dtype)
        feed: dict[str, Any] = {self.inputs[0]: batch}
        feed.update(extra)
        return tuple(self._session.run(None, feed))


def _tensorrt_install_hint() -> str:
    """The pip line for the TensorRT build matching this torch's CUDA, which it must match."""
    import torch

    major = (torch.version.cuda or "12").split(".")[0]
    return f"pip install tensorrt-cu{major}"


#: The oldest TensorRT mozo builds engines with. 10.13 miscompiles Grounding DINO's position
#: encoding, and 11 is the first to build strongly typed engines only.
_TENSORRT_MIN = 11


@functools.lru_cache(maxsize=1)
def _tensorrt_usable() -> bool:
    """Whether TensorRT can run here, asked once per process: importing it takes a second, and
    :func:`executable` asks on every model load whose family derives a TensorRT runtime."""
    return _tensorrt_unusable() is None


def _tensorrt_unusable() -> str | None:
    """Why TensorRT cannot run here, or None when it can: missing, or older than 11."""
    try:
        import tensorrt
    except ImportError:
        return "TensorRT is not installed"
    if int(tensorrt.__version__.split(".")[0]) < _TENSORRT_MIN:
        return f"mozo needs TensorRT {_TENSORRT_MIN} or later; {tensorrt.__version__} is installed"
    return None


class TensorRTRunner:
    """A TensorRT engine built from an ONNX graph, callable on CUDA tensors.

    An engine is specific to the GPU and the TensorRT version that built it, so it cannot be
    published. The first load on a machine builds it from a graph and keeps it, named for the
    graph, the GPU and the TensorRT version; every later load deserialises it in about a second.
    Kept under a model's cache directory, ``weights.remove`` deletes it with the model and
    ``weights.cached`` reports it as ``"built"``.

    **TensorRT 11 or later, strongly typed: precision comes from the graph.** TensorRT 11 builds
    no other kind of engine -- its fp16 switch is gone -- so an fp16 engine is built from a graph
    whose own types are fp16, such as one exported under torch's fp16 autocast. That makes the
    engine's precision the graph author's decision rather than TensorRT's, op by op. TensorRT 10
    is refused: it miscompiles Grounding DINO's position encoding (measured with 10.13), and its
    builder-chosen fp16 is the other kind of engine.

    **fp32 is IEEE fp32.** TensorRT computes fp32 matrix products in TF32 unless told not to,
    which torch does not do by default; that alone moved OWLv2's scores by up to 0.017 (p99) on
    an fp32 graph, where with TF32 off the engine matched torch exactly.

    **It takes and returns torch tensors on the device**, unlike the other runners here, which
    speak numpy. Those runtimes copy to the host by nature; this one would pay a device round
    trip per call for nothing, and every caller already holds torch tensors.

    **Thread safety.** One execution context, enqueued under a lock on a stream of its own, and the
    caller's stream waits on it. Calls from several threads are serialised on the GPU, which is
    what one context can do, and each gets its own outputs.

    Args:
        path: The ``.onnx`` graph to build from.
        device: A CUDA device, ``"cuda"`` or ``"cuda:N"``.
        engines: Directory the engine is kept in. Defaults to ``<graph>.engines/`` beside it.
        export: Writes the graph to *path* when called with it. Given, the graph need not exist:
            it is written only if an engine has to be built, and deleted once it has been, since
            it can always be written again and an engine is all a load needs.
        shapes: ``{input: (min, preferred, max)}`` for every input with a dynamic dimension --
            the range the engine accepts, and the shape it is tuned for.
        probe: Inputs to run a freshly built engine on once, refusing it if any output comes back
            NaN. Zeros by default, which suits an image graph; a graph with masks needs real
            inputs, because all-zero masks hide everything and NaN is then what torch gives too.

    Attributes:
        inputs: Input tensor names, in the order the engine declares them.
        outputs: Output tensor names, in the order the engine declares them.
        input_shape: The first input's shape.
        engine_path: Where the engine is kept.
        built: Whether this load built the engine, rather than finding it kept.

    Examples:
        >>> run = TensorRTRunner("model-fp16.onnx", device="cuda")        # doctest: +SKIP
        >>> (patches,) = run(pixels)                                       # doctest: +SKIP
    """

    def __init__(self, path: str | Path, device: str = "cuda", *, engines: str | Path | None = None,
                 export=None, shapes: dict[str, tuple] | None = None,
                 probe: tuple | None = None) -> None:
        if (problem := _tensorrt_unusable()) is not None:
            raise RuntimeError_(
                f"{problem}, so tensorrt runtimes cannot be run. TensorRT 10 miscompiles some "
                "models (measured with 10.13: Grounding DINO's boxes off by up to the whole "
                "image). Install the build that matches your torch's CUDA, 12 or 13: "
                f"{_tensorrt_install_hint()}"
            )
        import tensorrt as trt
        import torch

        self._torch, self._trt = torch, trt
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError_(f"TensorRT runs only on CUDA, not on {device!r}.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        graph = Path(path)
        self._logger = trt.Logger(trt.Logger.ERROR)
        with torch.cuda.device(self.device):
            self.engine_path = self._engine_path(graph, Path(engines) if engines else None)
            self.built = not self.engine_path.is_file()
            # A fresh build is deserialised from the bytes in hand rather than read back.
            blob = (self._build(graph, export, shapes or {}) if self.built
                    else self.engine_path.read_bytes())
            engine = trt.Runtime(self._logger).deserialize_cuda_engine(blob)
            del blob
            if engine is None:
                raise RuntimeError_(
                    f"{self.engine_path} could not be loaded. Delete it and load again to "
                    "rebuild it."
                )
            self._engine = engine
            self._context = engine.create_execution_context()
            self._stream = torch.cuda.Stream(self.device)

        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inputs = [n for n in names
                       if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if n not in self.inputs]
        self._dtypes = {n: torch.from_numpy(np.empty(0, trt.nptype(engine.get_tensor_dtype(n)))
                                            ).dtype for n in names}
        self.input_shape = tuple(engine.get_tensor_shape(self.inputs[0]))
        self._lock = Lock()

        if self.built:
            self._check(probe)

    def _engine_path(self, graph: Path, engines: Path | None) -> Path:
        import re

        props = self._torch.cuda.get_device_properties(self.device)
        gpu = re.sub(r"[^a-z0-9]+", "-", props.name.lower()).strip("-")
        name = (f"{graph.stem}-{gpu}-sm{props.major}{props.minor}"
                f"-trt{self._trt.__version__}.engine")
        return (engines or graph.with_suffix(_ENGINES_SUFFIX)) / name

    def _build(self, graph: Path, export, shapes: dict):
        """Build the engine on a thread of its own, wait for it, and return the serialised engine.

        TensorRT's builder switches the C locale of the thread that runs it to ASCII and does not
        switch it back (measured with 10.13: ``locale.getpreferredencoding()`` reads
        ``ANSI_X3.4-1968`` afterwards). On the caller's thread that would leave every later
        ``open()`` without an explicit encoding unable to read UTF-8 -- in a server, for the life
        of the process. The change is per thread, so a thread that ends takes it with it.
        """
        from threading import Thread

        exported = export is not None and not graph.is_file()
        if exported:
            graph.parent.mkdir(parents=True, exist_ok=True)
            _log.warning("[mozo] exporting %s to build a TensorRT engine from.", graph.name)
            export(graph)

        result: list = []

        def build() -> None:
            try:
                with self._torch.cuda.device(self.device):
                    result.append(self._build_here(graph, shapes))
            except BaseException as error:  # handed back to the caller below
                result.append(error)

        worker = Thread(target=build, name="mozo-tensorrt-build")
        worker.start()
        worker.join()
        if exported:
            # The graph and the weights an exporter may have written beside it: an engine is all
            # a later load needs, and the graph can always be written again.
            for leftover in graph.parent.glob(f"{graph.name}*"):
                leftover.unlink(missing_ok=True)
        if isinstance(result[0], BaseException):
            raise result[0]
        return result[0]

    def _build_here(self, graph: Path, shapes: dict):
        import os
        import time
        import uuid

        trt = self._trt
        _log.warning(
            "[mozo] building a TensorRT engine from %s for %s. This happens once per GPU and "
            "TensorRT version and takes a few minutes; later loads reuse it.",
            graph.name, self._torch.cuda.get_device_name(self.device),
        )
        start = time.perf_counter()
        builder = trt.Builder(self._logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, self._logger)
        if not parser.parse_from_file(str(graph)):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError_(f"TensorRT could not parse {graph.name}: {errors}")
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        if shapes:
            profile = builder.create_optimization_profile()
            for name, (low, preferred, high) in shapes.items():
                profile.set_shape(name, low, preferred, high)
            config.add_optimization_profile(profile)
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError_(f"TensorRT could not build an engine from {graph.name}.")
        # Written under a private name and moved into place, so a crash or a second process
        # building at the same moment never leaves a half-written engine where a load finds it.
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)
        staging = self.engine_path.with_name(
            f"{self.engine_path.name}.{uuid.uuid4().hex}{_PARTIAL_SUFFIX}")
        staging.write_bytes(memoryview(blob))
        os.replace(staging, self.engine_path)
        _log.warning("[mozo] built %s in %.0f s.", self.engine_path.name,
                     time.perf_counter() - start)
        return blob

    def _check(self, probe: tuple | None) -> None:
        """Run a freshly built engine once and refuse it if anything comes back NaN.

        NaN, not non-finite: ``-inf`` is a legitimate output -- Grounding DINO masks the padding
        of its logits with it -- and NaN never is.
        """
        torch = self._torch
        if probe is None:
            # Zeros at the shape the engine was optimised for: a fair test of an image graph,
            # not of one whose masks would then hide every token.
            probe = [torch.zeros(tuple(self._engine.get_tensor_profile_shape(name, 0)[1])
                                 if -1 in self._engine.get_tensor_shape(name)
                                 else tuple(self._engine.get_tensor_shape(name)),
                                 dtype=self._dtypes[name], device=self.device)
                     for name in self.inputs]
        bad = [name for name, out in zip(self.outputs, self(*probe))
               if out.is_floating_point() and torch.isnan(out).any()]
        if bad:
            self.engine_path.unlink(missing_ok=True)
            raise RuntimeError_(
                f"the TensorRT engine built from this graph returned NaN in {', '.join(bad)} "
                "on a test input, so it was deleted. Run this model on torch instead."
            )

    def __call__(self, *inputs) -> tuple:
        """Run the engine on torch tensors and return its outputs in declaration order.

        Inputs are moved to the device and cast to what the engine declares. Outputs are fresh
        tensors on the device, ready on the caller's current stream.
        """
        torch = self._torch
        with self._lock, torch.cuda.device(self.device):
            caller = torch.cuda.current_stream()
            self._stream.wait_stream(caller)
            held = []
            for name, value in zip(self.inputs, inputs):
                value = value.to(self.device, self._dtypes[name]).contiguous()
                self._context.set_input_shape(name, tuple(value.shape))
                self._context.set_tensor_address(name, value.data_ptr())
                held.append(value)
            outputs = []
            for name in self.outputs:
                out = torch.empty(tuple(self._context.get_tensor_shape(name)),
                                  dtype=self._dtypes[name], device=self.device)
                self._context.set_tensor_address(name, out.data_ptr())
                outputs.append(out)
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError_("TensorRT failed to run the engine.")
            caller.wait_stream(self._stream)
            # Tell the allocator these were used on our stream, so their memory is not handed
            # to someone else before the engine is done with it.
            for tensor in (*held, *outputs):
                tensor.record_stream(self._stream)
        return tuple(outputs)
