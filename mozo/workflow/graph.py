"""A workflow: nodes, the connections between them, and what happens when you run it.

A workflow either is valid or does not exist. Constructing one checks that every connection names
ports that exist, that the types on both ends agree, that every input is fed exactly once, and that
the graph is acyclic. Anything wrong raises here, naming the node -- so ``/workflow/validate`` is
just construction, and a workflow that loaded will not fail for a structural reason halfway through
a run with a model already on the GPU.

Running is one loop, in :meth:`Workflow.stream`. :meth:`Workflow.run` drains it. The implementation
this replaces had two loops that had drifted apart, one of them with error handling the other
lacked.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Optional

from . import registry
from .node import Connection, Context, Models, NodeSpec, Parameter, State, WorkflowError

__all__ = ["Event", "ModelUse", "Preflight", "Workflow", "WorkflowError"]

_log = logging.getLogger(__name__)


class ModelUse(NamedTuple):
    """One published model a workflow's node will load.

    Attributes:
        node: The node's id.
        family: The model family, e.g. ``"edgetam"``.
        variant: The variant, after the run's overrides and the node's own default.
        license: The weights licence, by name, from the manifest -- ``"Apache-2.0"``, ``"SAM
            License"``. Shown before anything is fetched, because a workflow shared by someone
            else can name weights whose terms the person running it has not seen.
        size: Bytes still to download, 0 once cached; None where the manifest could not say,
            in which case loading it will say why.
    """

    node: str
    family: str
    variant: str
    license: Optional[str]
    size: Optional[int]


@dataclass(frozen=True)
class Preflight:
    """What a workflow needs before it can run, found without running it.

    Attributes:
        missing: ``(node id, parameter, path)`` for every file or folder the workflow names that
            is not here. A run refuses to start while this is not empty: each of these would fail
            on its own, one run at a time, and a workflow shared from another machine usually has
            several.
        models: A :class:`ModelUse` for every published model the workflow's nodes load, cached
            or not, in run order -- what a host lists, with licences and what is left to fetch,
            before offering to download or remove anything.
    """

    missing: tuple = ()
    models: tuple = ()

    @property
    def downloads(self) -> tuple:
        """``(node id, family, variant, bytes)`` for every model the run will have to download
        first: the :attr:`models` with bytes still to fetch. Information, not a problem -- a run
        downloads and carries on."""
        return tuple((use.node, use.family, use.variant, use.size)
                     for use in self.models if use.size)

def _quietly(call, *args, **kwargs):
    """``call``'s answer, or None where the manifest cannot give one -- an unpublished variant,
    a runtime this device cannot run. Planning is information: the load says why, not this."""
    try:
        return call(*args, **kwargs)
    except Exception:   # noqa: BLE001
        return None


#: A source that yielded nothing is not a source that yielded None.
_NOTHING = object()


def _forget(states: list) -> None:
    """End the run for every node that kept something, releasing what each of them opened.

    Every node is closed even where one of them raises, and the first failure is re-raised after:
    a run whose video writer was never released because an unrelated sink threw has produced a file
    that will not play, which is a worse answer than the error that caused it.
    """
    failures = []
    for state in states:
        try:
            state.close()
        except Exception as error:  # noqa: BLE001 -- this node's failure, not the other nodes'
            failures.append(error)
    if failures:
        raise failures[0]


def _closing(events: Iterator, states: list) -> Iterator:
    """*events*, with the run's state released when it ends -- or when the caller walks away."""
    try:
        yield from events
    finally:
        _forget(states)


@dataclass(frozen=True)
class Step:
    """One node in one workflow: which kind it is, and what its parameters are set to."""

    id: str
    spec: NodeSpec
    parameters: dict
    #: Where the editor drew it. Meaningless to execution, carried so that loading and saving a
    #: workflow does not silently rearrange someone's canvas.
    position: dict = field(default_factory=dict)

    def arguments(self, settings: dict) -> dict:
        """This node's saved parameters, with the run's overrides on top.

        The one statement of how a parameter is settled, because both engines settle them: the
        serial loop calls this per item and the pipeline calls it once at build time. Written twice
        it would be a rule that can differ by worker count, which is the single thing running
        concurrently is not allowed to change -- and this module has already paid for two loops
        that drifted apart once.
        """
        return {**self.parameters, **settings.get(self.id, {})}

    def call(self, arguments: dict) -> tuple:
        """Run the node, as ``(what it produced, the Event that says why it did not)``.

        The failure message is what a caller finally reads -- ``deliver`` puts it in the
        ``RuntimeError`` -- so it is built once here rather than once per engine. Written twice,
        a run's failures would read differently depending on how many workers were asked for,
        which is the single thing ``workers`` is not allowed to change.
        """
        try:
            return self.spec(**arguments), None
        except Exception as error:      # noqa: BLE001 -- one item's failure, not the run's
            return None, Event(self.id, "failed", error=f"{self.spec.name}: {error}")

    def first(self, arguments: dict) -> tuple:
        """A source's first item, as this node's output, for a run of one item.

        :meth:`Workflow.run` and :meth:`Workflow.stream` are one pass over one item, so a source
        that would yield two hundred thousand frames is asked for one and closed. That is what
        makes a workflow with a video in it something the editor can show: without it the source
        returned its generator, the generator travelled down the graph as though it were an image,
        and the failure surfaced two nodes later reading ``OpenCV(-5:Bad argument)`` -- an error
        naming the wrong node and blaming the wrong library.

        Closed in a ``finally`` rather than left to be collected, because the thing it holds is an
        open file handle on a video, and a preview must not keep one.
        """
        produced = None
        try:
            produced = self.spec.run(**arguments)
            item = next(produced, _NOTHING)
            if item is _NOTHING:
                raise ValueError("produced nothing to run on")
            return {self.spec.outputs[0].name: item}, None
        except Exception as error:      # noqa: BLE001 -- this node's failure, reported as one
            return None, Event(self.id, "failed", error=f"{self.spec.name}: {error}")
        finally:
            if produced is not None:
                produced.close()

    def to_dict(self) -> dict:
        """Serialise to the editor's node format."""
        return {
            "id": self.id,
            "type": self.spec.name,
            "position": self.position,
            "data": {"parameters": self.parameters},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Step:
        """Read one node, resolving its type against the registry.

        Owns both halves of its wire format, the way :class:`~mozo.workflow.node.Connection` owns
        both halves of an edge's. Adding a field to the editor's node record is then one change
        here rather than two literals in the graph that have to be kept in step.
        """
        if not isinstance(data, dict):
            raise WorkflowError(f"a node is an object, not {data!r}")
        if "id" not in data or "type" not in data:
            raise WorkflowError(f"node needs an id and a type: {data}")
        try:
            spec = registry.get(data["type"])
        except KeyError as error:
            # The registry's message, which names every node it knows. ``registry.get`` keeps its
            # own KeyError for its own callers; here an unknown type is a fault in the file.
            raise WorkflowError(error.args[0]) from None
        parameters = (data.get("data") or {}).get("parameters", {})
        if not isinstance(parameters, dict):
            raise WorkflowError(f"{data['id']}: parameters are an object, not {parameters!r}")
        return cls(
            id=data["id"],
            spec=spec,
            parameters=dict(parameters),
            position=dict(data.get("position") or {}),
        )


@dataclass(frozen=True)
class Event:
    """One thing that happened during a run."""

    node: str
    status: str  # "running" | "completed" | "failed"
    output: Any = None
    error: Optional[str] = None


#: Why a workflow cannot be run over a source it does not have. One sentence, because both
#: :meth:`Workflow.process` and the HTTP endpoint refuse for it and a caller reading one should
#: not be told a different remedy than a caller reading the other.
NO_SOURCE = ("this workflow has no source, so there is nothing for a run to be a pass over. "
             "Give it one -- read_media -- or bring the items yourself with "
             "run_many(items, over=...).")


def _named(item) -> str:
    """*item* in a failure message, short enough to read.

    ``run_many`` brings paths, which name themselves. :meth:`Workflow.process` brings whatever the
    source yielded, and for a video or a folder that is the frame -- so the message used to be the
    ``repr`` of a 720x1280 array, printed once per failure, burying the reason it was raised for.
    """
    if isinstance(item, (str, bytes, Path)):
        return repr(item)
    shape = getattr(item, "shape", None)
    return f"a {shape[1]}x{shape[0]} image" if shape and len(shape) == 3 else type(item).__name__


def deliver(item, outcome, failure: Optional[Event], on_error: Optional[Any]) -> Iterator:
    """Hand one finished item to the caller, the one way both engines hand it over.

    Serial and pipelined must agree here or ``workers`` would change what a failure looks like.
    It lives beside :class:`Event` rather than in either engine, so the serial path does not have
    to import the concurrent one to give an answer back.
    """
    if failure is None:
        yield item, outcome
    elif on_error is None:
        raise RuntimeError(f"{_named(item)}: {failure.error}")
    else:
        on_error(item, failure)


class Workflow:
    """A directed acyclic graph of nodes.

    Args:
        steps: The nodes.
        connections: The edges between them.
        name: What the workflow is called, for a person choosing one. Optional; a file without
            one is a workflow without one, and nothing invents it from the file name.
        description: What it does, in a sentence or two. Optional, likewise.

    Raises:
        WorkflowError: If the nodes and edges cannot make a workflow -- two nodes sharing an id,
            an edge to a node or port that is not there, ports of different types, an input left
            unconnected, a cycle.

    Examples:
        >>> workflow = Workflow.load("blur_faces.json")     # doctest: +SKIP
        >>> results = workflow.run(image="street.jpg")      # doctest: +SKIP
    """

    def __init__(self, steps: list, connections: list, *, name: Optional[str] = None,
                 description: Optional[str] = None) -> None:
        if not isinstance(name, (str, type(None))):
            raise WorkflowError(f"a workflow's name is text, not {name!r}")
        if not isinstance(description, (str, type(None))):
            raise WorkflowError(f"a workflow's description is text, not {description!r}")
        #: What the workflow is called, or None. Carried through :meth:`to_dict` and :meth:`save`.
        self.name = name
        #: What it does, or None.
        self.description = description
        self.steps = {step.id: step for step in steps}
        if len(self.steps) != len(steps):
            raise WorkflowError("two nodes share an id")
        self.connections = list(connections)

        # Settled at construction, because validity depends on them: checking the wiring is what
        # produces the inbound map, and computing the order is how a cycle is found. Everything
        # else about the graph -- :attr:`terminals`, :attr:`parameters` -- is a question you can
        # ask of it, and stays a property.
        #: Node id -> ``{input port: the wire feeding it}``.
        self.incoming = self._check_wiring()
        #: Execution order.
        self.order = self._sorted()
        #: How many inputs each wire feeds, so a value can be dropped once nothing else wants it.
        self.readers: dict = {}
        for wires in self.incoming.values():
            for wire in wires.values():
                self.readers[wire] = self.readers.get(wire, 0) + 1

    # --- Loading and saving ---

    @classmethod
    def load(cls, path: str | Path) -> Workflow:
        """Read a workflow from a JSON file.

        Raises:
            FileNotFoundError: If there is no file at *path*.
            WorkflowError: If there is, and it is not JSON or not a workflow.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no workflow at {path}")
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise WorkflowError(f"{path} is not JSON: {error}") from None
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Workflow:
        """Read a workflow from the editor's ``{name, description, nodes, edges}`` format.

        ``name`` and ``description`` are optional, so a file written before they existed loads as
        it always did.

        Raises:
            WorkflowError: If *data* is not a workflow. One error for every way it can fail to be,
                with a message that says which.
        """
        if not isinstance(data, dict):
            raise WorkflowError(f"a workflow is a JSON object, not {type(data).__name__}")
        nodes, edges = data.get("nodes", []), data.get("edges", [])
        for key, value in (("nodes", nodes), ("edges", edges)):
            if not isinstance(value, list):
                raise WorkflowError(f"a workflow's {key} are a list, not {value!r}")
        return cls(
            [Step.from_dict(node) for node in nodes],
            [Connection.from_dict(edge) for edge in edges],
            name=data.get("name"),
            description=data.get("description"),
        )

    def to_dict(self) -> dict:
        """Write the workflow back out in the format it was read from.

        ``name`` and ``description`` are written only when set, so a file without them comes back
        exactly as it went in.
        """
        described = {"name": self.name, "description": self.description}
        return {
            **{key: value for key, value in described.items() if value is not None},
            "nodes": [step.to_dict() for step in self.steps.values()],
            "edges": [connection.to_dict() for connection in self.connections],
        }

    def save(self, path: str | Path) -> None:
        """Write the workflow to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # --- Shape ---

    @property
    def terminals(self) -> tuple:
        """Ids of the nodes nothing reads from -- where a run's answers end up."""
        feeding = {connection.source for connection in self.connections}
        return tuple(step_id for step_id in self.order if step_id not in feeding)

    @property
    def source(self) -> Optional[str]:
        """The node this run's items come from, or None where the caller supplies them.

        One at most, for now. Two sources is two runs interleaved -- each with its own rate, its
        own size, its own count -- and :class:`~mozo.workflow.node.Context` would have to answer
        "which one" before it could answer anything. That is the multi-camera design, and refusing
        it here says so plainly rather than picking one source's facts and calling them the run's.
        """
        found = [step_id for step_id, step in self.steps.items() if step.spec.produces_many]
        if len(found) > 1:
            raise WorkflowError(
                f"{len(found)} sources in one workflow: {found}. A run is one pass over one "
                f"source, so two would be two runs and the facts of neither.")
        return found[0] if found else None

    @property
    def file_parameter(self) -> str:
        """The parameter a file chosen somewhere else is the value of, as a ``"node.parameter"`` key.

        A question about the graph's shape, like :attr:`source` and :attr:`parameters`, and asked
        here so that every way in gets the same answer. The three that need it -- the HTTP upload,
        the command line's ``--file``, and :meth:`run_many`'s default -- each used to name a
        parameter as a string literal instead, which meant each knew one node's vocabulary. Renaming
        that parameter broke them one at a time, and the command line was still broken after the
        other two were fixed.

        Derived from what the parameter *is* rather than what it is called:
        :data:`~mozo.workflow.node.Source` is the annotation for a value a person cannot type,
        which is why it puts a file picker on the node. An input node added tomorrow, calling its
        parameter anything at all, is reachable from all three without any of them changing.

        Only a :data:`~mozo.workflow.node.Source`: a :data:`~mozo.workflow.node.FilePath` -- a
        checkpoint of your own, annotations to read back -- is a fixed path the workflow was saved
        with, not the file a run is given.

        A key that names its node, because the annotation has already said which node is meant: a
        bare name would be refused as ambiguous whenever another node had a parameter of the same
        name, and two input nodes sharing one would count as one.

        Raises:
            ValueError: If there is not exactly one. Both other answers are wrong in a way that
                would be silent -- nowhere to put the file means it was chosen for nothing, and
                two places means choosing one of them on the caller's behalf.
        """
        keys = [f"{step_id}.{parameter.name}" for step_id, parameter in self._files()
                if parameter.kind == "source"]
        if not keys:
            raise ValueError(
                "no node in this workflow reads a file, so there is nothing for one to be. "
                "Add an input node, or set the value by name.")
        if len(keys) > 1:
            raise ValueError(
                f"{len(keys)} parameters take a file: {keys}. One file cannot say which it is "
                f"-- set each by its key instead.")
        return keys[0]

    @property
    def parameters(self) -> dict:
        """Every parameter in the workflow, as ``name -> [node id, ...]``.

        The bare names :meth:`run` accepts as overrides, and what it consults to reject an
        ambiguous one. Any of them is also reachable as ``"node.parameter"``.
        """
        found: dict = {}
        for step in self.steps.values():
            for parameter in step.spec.parameters:
                found.setdefault(parameter.name, []).append(step.id)
        return found

    @property
    def prompt_parameters(self) -> tuple:
        """Every :data:`~mozo.workflow.node.Prompts` parameter, as ``"node.parameter"`` keys in run
        order -- what :meth:`run` takes, so a host reaches them without knowing their names::

            workflow.run(image=path, **{key: prompts for key in workflow.prompt_parameters})
        """
        return tuple(f"{step_id}.{parameter.name}" for step_id in self.order
                     for parameter in self.steps[step_id].spec.parameters if parameter.prompts)

    def parameter(self, key: str) -> tuple[str, Parameter]:
        """Which node's parameter an override *key* sets, as ``(node id, parameter)``.

        A key is a parameter's name, when one node has it, or ``"node.parameter"`` for any one --
        which is how two nodes with a parameter of the same name, SAM 3 beside Grounding DINO, are
        each given their own value. Split on the last dot: a parameter's name is a Python
        identifier and never has one, so a node id may.

        Raises:
            KeyError: If *key* names no parameter, or a name more than one node has. The message
                lists what there is, or the keys that would say which.
        """
        step_id, dot, name = key.rpartition(".")
        if not dot:
            owners = self.parameters.get(key)
            if not owners:
                raise KeyError(
                    f"no parameter {key!r} in this workflow. It has: {sorted(self.parameters)}")
            if len(owners) > 1:
                raise KeyError(
                    f"{len(owners)} nodes have a parameter {key!r}: {owners}. Say which with "
                    + " or ".join(repr(f"{owner}.{key}") for owner in owners) + ".")
            step_id = owners[0]
        if step_id not in self.steps:
            raise KeyError(f"no node {step_id!r} in this workflow, for {key!r}. "
                           f"It has: {sorted(self.steps)}")
        step = self.steps[step_id]
        try:
            return step_id, step.spec.parameter(name)
        except KeyError as error:
            raise KeyError(f"{step_id}: {error.args[0]}") from None

    # --- Running ---

    def run(self, *, models=None, **overrides) -> dict:
        """Run the workflow and return every node's output, keyed by node id.

        Args:
            models: Where the model nodes get their models: a
                :class:`~mozo.manager.ModelManager`, or any callable shaped like ``get_model``.
                ``None`` is the shared cache behind :func:`mozo.get_model`. See
                :class:`~mozo.workflow.node.Models`. The same for every run method.
            **overrides: Parameter values to use instead of the ones saved in the workflow, by
                parameter name, or as ``"node.parameter"`` for one node's -- see :meth:`parameter`.
                A name that more than one node uses is refused rather than guessed at, naming the
                keys that would say which. From Python a dotted key goes in by unpacking:
                ``run(**{"find.text": ["person"]})``.

        Returns:
            ``{node id: output}``. Every node, not only the terminal ones -- an intermediate result
            is often the one worth looking at, and hiding it would mean re-running to see it.
            :attr:`terminals` names the ends.

            A node that failed is absent, along with everything downstream of it. Use
            :meth:`stream` where the reason matters: it reports the failure and names the node.
        """
        settings = self._resolve(overrides)
        states = self._remember(settings, models)
        try:
            return self._drain(settings)[0]
        finally:
            _forget(states)

    def run_many(self, items, *, over: Optional[str] = None, on_error=None, workers: int = 2,
                 model_workers: int = 1, stats: Optional[dict] = None, models=None,
                 **overrides) -> Iterator[tuple]:
        """Run the workflow once per item, yielding ``(item, results)`` as each finishes.

        A generator rather than a list, which is the whole reason this works at any size: what a
        run holds is bounded by how many items may be alive at once, not by how many there are, so
        a directory of a million costs what a handful costs. ``items`` is consumed lazily, so it
        may be a generator itself and may never end -- though above ``workers=1`` it is read ahead
        of the results by a bounded amount, since a stage with nothing queued has nothing to
        overlap.

        The overrides are settled **once**, before the first item, because their shape does not
        change -- only the one value does. Resolving per item would rebuild :attr:`parameters` a
        million times for a million items, which measured at a tenth of the whole run.

        Args:
            items: What to run on, one at a time. Anything :meth:`run` accepts for *over* -- paths,
                bytes, arrays -- and any iterable of them.
            over: The parameter each item is bound to. Left unset it is
                :attr:`file_parameter` -- the one that takes a file, whatever the node calls it --
                so a batch of paths needs no name at all and a node named differently still works.
            on_error: Called as ``on_error(item, event)`` with the failing :class:`Event`, which
                skips that item and continues. Left unset, a failure raises and the run stops.
                A corrupt file in a million should not end a six-hour run, but silently dropping
                it is worse than stopping, so the caller says where the failure goes. Raising from
                inside the callback stops the run, whatever *workers* is.
            workers: How many items may be in flight at once. **Two by default**, which runs the
                workflow as a staged pipeline: every node gets its own queue and its own workers,
                so the model works on one item while decoding is already on the next and drawing is
                still finishing the last. See :mod:`mozo.workflow.pipeline`.

                Two rather than more because two was the peak on every real workflow measured --
                four was equal or slower, and on a graph with two model stages it fell from 1.91x
                to 1.40x. Two rather than one because one leaves the bottleneck waiting for its
                own inputs. ``workers=1`` still runs the plain serial loop, unchanged, for a
                caller who wants no threads at all.

                The threads are made per call, so ``run_many`` over a single item pays for a pool
                it cannot use -- measured at 18 us serial against 436 us at two workers on a
                three-node graph of trivial nodes. Against real node work that is noise; against a
                camera loop calling ``run_many([frame])`` per frame it is the whole cost, and such
                a caller wants one ``run_many`` over the frames instead.

                **A node holding a model is pinned to one item regardless**, because a second
                concurrent inference doubles activation memory, and running out of memory ends a
                run rather than slowing it.

                This changes how long a run takes and nothing else: same items, same results, same
                order, and failures reach *on_error* the same way.

        Yields:
            ``(item, results)`` in the order *items* arrived, where *results* is exactly what
            :meth:`run` returns. An item whose graph failed part way is not yielded at all, rather
            than yielded with the nodes that did complete: a caller iterating results would have
            no way to tell a partial answer from a whole one, and half a workflow's output is the
            kind of wrong that looks right.

            **Stopping is closing the iterator** -- ``close()``, or leaving a ``for`` loop over it.
            No item starts a node after that. A call already running finishes -- a model mid-way
            through an image cannot be interrupted -- and the pipeline's threads exit as soon as
            it does. ``close()`` does not wait for them: they are gone within the length of the
            longest call that was in flight, not necessarily by the time it returns.

        Raises:
            KeyError: On the first item, if *over* or an override names no parameter or an
                ambiguous one, or if *over* is also given as an override -- every item would then
                run on that fixed value instead of itself. A workflow that cannot be run does not
                become a million failed items.

        Examples:
            >>> for path, results in workflow.run_many(paths):   # doctest: +SKIP
            ...     save(results["annotate"], path)
        """
        over = self.file_parameter if over is None else over
        if over in overrides:
            raise KeyError(
                f"{over!r} is both the parameter each item binds to and an override, so every "
                f"item would run on {overrides[over]!r} rather than on itself. Pass one or "
                f"the other.")
        # Refuses an override reaching the same parameter under another key, too.
        settings = self._resolve({over: None, **overrides})
        # Carried as the node and the parameter: two nodes' settings may both hold that name.
        step_id, parameter = self.parameter(over)
        over = (step_id, parameter.name)
        states = self._remember(settings, models)
        try:
            yield from self._many(items, over, settings, on_error, workers, model_workers, stats)
        finally:
            # Reached however the run ended: the last item, an exception, or a caller who stopped
            # iterating -- the case that matters most, because that is the one where a half-written
            # video would otherwise be left with no moov atom and no way to play it.
            _forget(states)

    def _many(self, items, over: Optional[tuple], settings: dict, on_error, workers: int,
              model_workers: int, stats: Optional[dict], driving: Optional[str] = None,
              run: Optional[Context] = None) -> Iterator[tuple]:
        """One pass over *items*, however they were come by.

        The two ways in meet here. :meth:`run_many` brings the items and binds each to a parameter
        named by *over*; :meth:`process` asks the source for them and names it in *driving*. From
        this point down there is no difference worth keeping: an item is a value that has to reach
        the graph, and the only question is which port it arrives on.
        """
        folded, folded_results = self._fold(settings, over)
        if workers > 1:
            # A staged pipeline: one queue and one worker set per node. Serial below is unchanged,
            # so asking for no workers runs exactly the code that ran before this existed. Imported
            # here rather than at module scope because ``pipeline`` imports from this module --
            # deferring the one arrow back is what keeps the pair acyclic.
            from .pipeline import run_pipelined
            yield from run_pipelined(self, items, over, settings, workers, on_error,
                                     model_workers, stats, folded, folded_results, driving, run)
            return
        if stats is not None:
            raise ValueError(
                "stats needs workers > 1: there are no stages to report on a serial run, and "
                "filling it with nothing would read as a run that did no work.")

        # The one leaf that changes per item. Everything else in ``settings`` is already final,
        # and ``_steps`` only reads it, so one dict is safe to carry across the whole run. This is
        # also why it cannot be shared by two items at once, and why the pipeline above binds the
        # value to each item's parcel instead.
        bound = settings[over[0]] if over is not None else None
        port = self.steps[driving].spec.outputs[0].name if driving else None

        for index, item in enumerate(items):
            if bound is not None:
                bound[over[1]] = item
            # The driving source is folded too, only per item rather than once: what it produced
            # is settled for this item exactly as a constant is settled for the run, so both reach
            # the graph by the same road and ``_steps`` does not have to know which is which.
            settled = {**folded, driving: {port: item}} if driving else folded
            reported = {**folded_results, driving: item} if driving else folded_results
            results, failure = {}, None
            for event in self._steps(settings, settled, reported,
                                     run.at(index) if run is not None else None):
                if event.status == "completed":
                    results[event.node] = event.output
                elif event.status == "failed":
                    failure = event
            yield from deliver(item, results, failure, on_error)

    def process(self, *, workers: int = 2, on_error=None, model_workers: int = 1,
                stats: Optional[dict] = None, models=None, **overrides) -> Iterator[tuple]:
        """Run once per item the source yields, yielding ``(item, results)`` as each finishes.

        A run is one pass over whatever the workflow's source produces: one image or every
        frame of a video from :func:`~mozo.workflow.nodes.io.read_media`, an unbounded stream from
        a camera. How many
        items there are stops being something the caller has to know and becomes something the
        workflow says -- which is the difference between this and :meth:`run_many`, where the
        caller brings the items and the workflow is only told what to do with each.

        The source is asked for its facts before its first item, and they are settled from then on:
        every node that asked for a :class:`~mozo.workflow.node.Context` reads the same rate, size
        and count, and reads this item's index off the one the engine already assigned.

        Args:
            workers: As :meth:`run_many` means it.
            on_error: As :meth:`run_many` means it.

        Yields:
            ``(item, results)`` in the order the source produced them.

        Raises:
            ValueError: If the workflow has no source, naming what it would need. A workflow whose
                items come from the caller is run with :meth:`run_many` instead.
        """
        source_id = self.source
        if source_id is None:
            raise ValueError(NO_SOURCE)

        settings = self._resolve(overrides)
        states = self._remember(settings, models)
        try:
            step = self.steps[source_id]
            arguments = step.arguments(settings)
            # Empty: the source names the run, on the next line but one. Seeding this from the
            # first parameter guessed at which one was the name from dictionary order, and the
            # declare overwrote it anyway.
            run = Context()
            if step.spec.context:
                arguments[step.spec.context] = run
            # ``spec.run`` rather than the spec itself: a source is not called per item, so the
            # batching that ``NodeSpec.__call__`` exists for has nothing to fan out over.
            produced = step.spec.run(**arguments)
            first = next(produced, _NOTHING)
            # Settled here, after the source has had its chance to declare and before any node can
            # read: everything before this point is the source describing the run, everything
            # after is the run.
            run.seal()
            items = () if first is _NOTHING else chain((first,), produced)
            yield from self._many(items, None, settings, on_error, workers, model_workers, stats,
                                  driving=source_id, run=run)
        finally:
            _forget(states)

    def stream(self, *, models=None, **overrides) -> Iterator[Event]:
        """Run the workflow, reporting each node as it starts and as it finishes.

        Returns:
            An iterator of :class:`Event` -- one per node before it runs and another after, so a
            caller can show progress. A failure yields one ``"failed"`` event and stops:
            everything after it depended on the value that was not produced.

        Raises:
            KeyError: If an override names no parameter, or an ambiguous one. Raised here rather
                than on the first step, so that a caller who got a workflow back knows it will run
                -- the same reason construction is what validates the document.
        """
        settings = self._resolve(overrides)
        states = self._remember(settings, models)
        # Not a generator function: the overrides are refused here, before the caller has an
        # iterator, and a generator would defer that to the first ``next``. The wrapper is what
        # releases what the nodes opened -- including when the caller stops reading part way,
        # which reaches the generator as GeneratorExit and runs its ``finally`` all the same.
        return _closing(self._steps(settings), states)

    def _files(self) -> Iterator[tuple]:
        """``(node id, parameter)`` for every parameter that names a file, in run order.

        A file parameter is a :data:`~mozo.workflow.node.Source` -- the file a run is given -- or
        a :data:`~mozo.workflow.node.FilePath`, a fixed one, whatever either is called. One
        definition, so the input ``--file`` binds to and the files checked before a run cannot
        disagree about which they are.
        """
        for step_id in self.order:
            for parameter in self.steps[step_id].spec.parameters:
                if parameter.kind in ("source", "path"):
                    yield step_id, parameter

    def check(self, *, device: Optional[str] = None, **overrides) -> Preflight:
        """What this workflow needs that is not here, without running anything.

        Every file parameter -- the ones the editor offers a file picker for, such as a source to
        read or a checkpoint of your own -- is checked for its file, after *overrides* are
        applied. Every node that loads a published model -- the ones whose variant is a
        :class:`~mozo.workflow.node.Weights` choice -- is listed with its licence and what is
        still to download. Every run does this first: missing files stop it, downloads are logged.

        Args:
            device: The device the models will run on, as the host chose it. Sizes are for the
                runtime mozo picks on it -- what :func:`mozo.get_model` loads -- and naming it
                keeps this from importing torch to ask which device there is. A host loading a
                different runtime asks :func:`mozo.weights.download_size` with that runtime.
            **overrides: Parameter values, as :meth:`run` takes them. ``device`` is therefore
                not a name a node parameter may have.
        """
        return self._inspect(self._resolve(overrides), device=device)

    def _inspect(self, settings: dict, plan: bool = True,
                 device: Optional[str] = None) -> Preflight:
        from mozo import weights

        missing, models = [], []
        for step_id, parameter in self._files():
            value = self.steps[step_id].arguments(settings).get(parameter.name)
            if isinstance(value, (str, Path)) and value and not Path(value).exists():
                missing.append((step_id, parameter.name, str(value)))
        # The node says which model it loads, on its variant parameter. One loading only a
        # checkpoint of your own says nothing here -- it names a file, checked as one above.
        planned = [self.steps[step_id] for step_id in self.order
                   if plan and self.steps[step_id].spec.weights is not None]
        if planned and device is None:
            # Asked once, as a load would ask it, rather than once per model.
            from mozo.runtimes import get_default_device
            device = get_default_device()
        for step in planned:
            chosen = step.spec.weights
            # Left out of the file, a parameter takes the node's own default -- as the run will.
            variant = step.arguments(settings).get(chosen.name, chosen.default)
            models.append(ModelUse(
                step.id, chosen.family, variant,
                _quietly(weights.license_of, chosen.family, variant),
                _quietly(weights.download_size, chosen.family, variant, device=device)))
        return Preflight(tuple(missing), tuple(models))

    def _preflight(self, settings: dict) -> None:
        """Refuse to start a run that cannot finish, naming everything it lacks at once.

        Download sizes are worked out only when someone will read them: a library caller who
        configured no logging pays for none of it on every run.
        """
        found = self._inspect(settings, plan=_log.isEnabledFor(logging.INFO))
        if found.missing:
            labels = [f"{node} ({self.steps[node].spec.name})" for node, _, _ in found.missing]
            width = max(map(len, labels))
            lines = [f"  {label:<{width}}  {name}: {path}"
                     for label, (_, name, path) in zip(labels, found.missing)]
            files = "1 file that is" if len(lines) == 1 else f"{len(lines)} files that are"
            raise FileNotFoundError(
                f"this workflow refers to {files} not here:\n" + "\n".join(lines) +
                "\nFix the paths in the workflow file and run it again.")
        for node, family, variant, size in found.downloads:
            _log.info("[mozo] %s will download %s/%s (%.1f MB)", node, family, variant, size / 1e6)

    def _remember(self, settings: dict, models=None) -> list:
        """Check the run can start, then give every node what it supplies: a place to keep
        things, and its models. Every run method comes through here before its first node.

        Each node that keeps something gets a place to keep it, for this run and no other. Each
        node that loads a model gets the run's :class:`~mozo.workflow.node.Models`, one for the
        whole run, built from *models* as :meth:`run` describes.

        Written into *settings* rather than handed to the executor, because that is already the
        one channel by which a node's arguments are settled once and read per item -- both engines
        go through :meth:`Step.arguments`, so neither has to learn what a state is. The pipeline
        reads it once at build time and the serial loop reads it per item, and they get the same
        object either way, which is the whole requirement.

        Returns:
            What was made, for the caller to :func:`_forget` when the run ends.
        """
        self._preflight(settings)
        states, supplied = [], Models(models)
        for step_id, step in self.steps.items():
            if step.spec.state:
                state = State()
                settings.setdefault(step_id, {})[step.spec.state] = state
                states.append(state)
            if step.spec.models:
                settings.setdefault(step_id, {})[step.spec.models] = supplied
        return states

    def _fold(self, settings: dict, over: Optional[tuple] = None) -> tuple:
        """Run the nodes whose value cannot change, once, before the run begins.

        A node with no inputs and settled parameters answers the same thing every time it is
        asked. Asked per item it is asked once per item: a workflow matching every frame of a
        two-hour video against one reference image decoded that reference two hundred thousand
        times, which is not a slow run, it is the same run with a file read into it repeatedly.
        Measured on a fifty-item run before this existed: fifty calls, forty-nine of them for a
        value already held.

        Two no-input nodes are **not** constant, and both exclusions are the whole subtlety here:

        * the node the items bind to, *over*'s ``(node id, parameter)``. Its parameter is what changes per item, so
          folding it would run the whole workflow on whichever item happened to be first.
        * a source, which is not one value but many, and drives the run rather than feeding it.

        A node that raises is left in the graph rather than folded, so that it fails where the
        engines already know how to report it -- naming the node, stopping the item, reaching
        ``on_error``. Folding its failure would turn a workflow's own error into this method's.

        Returns:
            ``({node id: {port: value}}, {node id: result})`` -- the wires for the engines to read
            instead of running the node, and the results to report as though it had run, because
            from the outside it did.
        """
        wires_by_node, results_by_node = {}, {}
        for step_id in self.order:
            step = self.steps[step_id]
            if self.incoming[step_id] or step.spec.produces_many:
                continue
            if over is not None and step_id == over[0]:
                continue
            wires, failure = step.call(step.arguments(settings))
            if failure is not None:
                continue
            wires_by_node[step_id] = wires
            results_by_node[step_id] = step.spec.result(wires)
        return wires_by_node, results_by_node

    def _drain(self, settings: dict) -> tuple:
        """One run, as ``(what completed, the failure that stopped it or None)``.

        The one place a run is turned into an answer. :meth:`run` wants the first half and
        :meth:`run_many` wants both, which is one loop with two callers rather than the two loops
        this module's docstring describes having already had to merge once.
        """
        results, failure = {}, None
        for event in self._steps(settings):
            if event.status == "completed":
                results[event.node] = event.output
            elif event.status == "failed":
                failure = event
        return results, failure

    def _steps(self, settings: dict, folded: Optional[dict] = None,
               folded_results: Optional[dict] = None,
               context: Optional[Context] = None) -> Iterator[Event]:
        """Run each node in order, reporting as it goes.

        *folded* is what :meth:`_fold` already ran, as ``{node id: {port: value}}``. Those nodes
        are reported as completed and then skipped: a caller reads a run's results and has no
        business knowing which values were computed for it and which were computed once for every
        item, only that every node it drew produced what it produced.

        Left unset by :meth:`run` and :meth:`stream`, and set only by :meth:`run_many`. A constant
        is constant *across items*, so a run of one item has nothing to save and nothing to fold:
        folding there would buy nothing and would report the node as finishing without having been
        seen to start, which is the one thing a progress stream promises not to do.
        """
        folded = folded or {}
        folded_results = folded_results or {}
        #: ``(node id, output port) -> value``. Keyed by port because a node may produce several
        #: things, and a connection already says which one it wants.
        produced: dict = {}
        #: What is still to read each wire. A value is dropped when this reaches zero: it has been
        #: handed to every node that wanted it and yielded to the caller, so holding it only keeps
        #: an image alive for the rest of the run. On a five-node chain over one 4K photograph that
        #: is the difference between 182 MB and 70 MB of peak resident memory.
        waiting = dict(self.readers)

        for step_id, result in folded_results.items():
            yield Event(step_id, "completed", output=result)

        for step_id in self.order:
            if step_id in folded:
                continue
            step = self.steps[step_id]
            yield Event(step_id, "running")

            arguments = step.arguments(settings)

            if step.spec.produces_many:
                # A source in a one-item run. It gets an unsealed context to declare into and is
                # then asked for a single item, so the nodes after it read the real rate and size
                # of the video rather than nothing -- a preview of a workflow is still that
                # workflow, and a sink in it must open itself the way it would on the whole run.
                context = Context()
                if step.spec.context:
                    arguments[step.spec.context] = context
                wires, failure = step.first(arguments)
                context.seal()
                if failure is not None:
                    yield failure
                    return
                produced.update({(step_id, port): value for port, value in wires.items()})
                yield Event(step_id, "completed", output=step.spec.result(wires))
                continue

            if step.spec.context:
                arguments[step.spec.context] = context if context is not None else Context().seal()
            for port, wire in self.incoming[step_id].items():
                if wire[0] in folded:
                    # Held for the whole run rather than dropped when its last reader has taken
                    # it: the readers counted in ``self.readers`` are this item's, and a constant
                    # has every later item still to feed.
                    arguments[port] = folded[wire[0]][wire[1]]
                    continue
                arguments[port] = produced[wire]
                waiting[wire] -= 1
                if not waiting[wire]:
                    del produced[wire]

            wires, failure = step.call(arguments)
            if failure is not None:
                yield failure
                return

            produced.update({(step_id, port): value for port, value in wires.items()})
            yield Event(step_id, "completed", output=step.spec.result(wires))

    def _resolve(self, overrides: dict) -> dict:
        """Turn ``{key: value}`` into ``{node id: {parameter: value}}``, or explain why not.

        Two keys reaching one parameter -- ``text`` and ``find.text`` when only ``find`` has a
        ``text`` -- are refused rather than one silently winning, since which of two values a run
        used is not something a caller should have to find out from the result.
        """
        settings: dict = {}
        given: dict = {}
        for key, value in overrides.items():
            step_id, parameter = self.parameter(key)
            earlier = given.setdefault((step_id, parameter.name), key)
            if earlier != key:
                raise KeyError(f"{earlier!r} and {key!r} both set {step_id}.{parameter.name}. "
                               f"Give one.")
            settings.setdefault(step_id, {})[parameter.name] = value
        return settings

    # --- Validation ---

    def _check_wiring(self) -> dict:
        """Every connection names real ports of agreeing types, and every input is fed once.

        Returns what it had to work out to check that: which node feeds each input. Running reads
        it instead of scanning the edges again for every node.
        """
        fed: dict = {step_id: {} for step_id in self.steps}
        for connection in self.connections:
            for end in (connection.source, connection.target):
                if end not in self.steps:
                    raise WorkflowError(f"edge names a node that is not here: {end!r}")

            source, target = self.steps[connection.source], self.steps[connection.target]
            produced = {port.name: port for port in source.spec.outputs}
            if not produced:
                raise WorkflowError(f"{source.id} ({source.spec.name}) produces nothing to connect")
            if connection.source_output not in produced:
                raise WorkflowError(
                    f"{source.id} ({source.spec.name}) has no output "
                    f"{connection.source_output!r}. It produces {sorted(produced)}.")

            out = produced[connection.source_output]
            try:
                port = target.spec.port(connection.target_input)
            except KeyError as error:
                raise WorkflowError(f"{target.id}: {error.args[0]}") from None
            if port.type is not out.type:
                raise WorkflowError(
                    f"{source.id}.{out.name} is {out.type.value}, "
                    f"but {target.id}.{port.name} takes {port.type.value}")

            if port.name in fed[target.id]:
                raise WorkflowError(f"{target.id}.{port.name} is fed by two connections")
            fed[target.id][port.name] = (connection.source, connection.source_output)

        for step in self.steps.values():
            for port in step.spec.inputs:
                if port.name not in fed[step.id]:
                    raise WorkflowError(
                        f"{step.id} ({step.spec.name}) has nothing connected to its "
                        f"{port.name!r} input")
        return fed

    def _sorted(self) -> tuple:
        """Kahn's algorithm. Finding a cycle is the same work as finding the order."""
        waiting = {step_id: 0 for step_id in self.steps}
        downstream: dict = {step_id: [] for step_id in self.steps}
        for connection in self.connections:
            downstream[connection.source].append(connection.target)
            waiting[connection.target] += 1

        ready = [step_id for step_id, count in waiting.items() if count == 0]
        order = []
        while ready:
            step_id = ready.pop(0)
            order.append(step_id)
            for follower in downstream[step_id]:
                waiting[follower] -= 1
                if waiting[follower] == 0:
                    ready.append(follower)

        if len(order) != len(self.steps):
            raise WorkflowError(
                "the workflow has a cycle, through: "
                + ", ".join(sorted(set(self.steps) - set(order))))
        return tuple(order)
