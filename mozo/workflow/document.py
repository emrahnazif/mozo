"""The short form of a workflow document: one node per line, connections written only where needed.

    {"name": "People outlined",
     "nodes": [
      {"id": "image", "type": "read_media"},
      {"id": "find", "type": "grounding_dino", "text": ["person"], "threshold": 0.3},
      {"id": "outline", "type": "edgetam_boxes"},
      {"id": "draw", "type": "draw_masks", "inputs": {"image": "image"}},
      {"id": "keep", "type": "save_image"}
     ]}

**The rule:** an input that is not written takes the latest earlier node in the list with an output
of its type. ``outline`` above takes ``find``'s detections and ``image``'s picture without saying
so. ``draw`` writes its image input because the latest image before it is not the one it wants.
One output can feed any number of nodes.

Written for whoever builds a workflow as text, a small language model above all. In the full
form, a chain of five nodes is four edges of four fields each, re-sent on every attempt, and a
wrong handle is the most common way to fail. Here the same chain is five lines with no edges, and
changing one threshold means editing one line.

A document is in the full form when it has ``edges``, and in this form when it does not. The full
form is still read exactly as before, so every file written by an earlier version keeps its
meaning. :func:`write` is what :meth:`~mozo.workflow.Workflow.to_dict` returns. It writes only the
inputs the rule would get wrong, so reading what it wrote gives back the same graph. The editor
holds the same rule in ``ui/src/lib/api.js``, and ``tests/workflow/test_editor_conversion.py``
checks that the two agree.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from .node import RESERVED, Connection, Port, WorkflowError


def is_short(data: dict) -> bool:
    """Is *data* written in the short form? Only a full-form document has ``edges``."""
    return "edges" not in data


def read(nodes: list) -> tuple[list, list]:
    """The steps and connections of a short document's *nodes*.

    Raises:
        WorkflowError: For a node that is not an object, an unknown type or parameter, an
            ``inputs`` naming nothing that is here, or an input the rule cannot settle.
    """
    from .graph import Step     # graph imports this module

    steps, wanted = [], []
    for node in nodes:
        if not isinstance(node, dict):
            raise WorkflowError(f"a node is an object, not {node!r}")
        if "data" in node:
            # A full-form node in a document without edges -- one node, written by hand before
            # this form existed. Read as it always was; its inputs follow the rule.
            step = Step.from_dict({key: value for key, value in node.items() if key != "inputs"})
            steps.append(step)
            wanted.append(node.get("inputs") or {})
            continue
        if "id" not in node or "type" not in node:
            raise WorkflowError(f"node needs an id and a type: {node}")
        parameters = {key: value for key, value in node.items() if key not in RESERVED}
        step = Step.from_dict({"id": node["id"], "type": node["type"],
                               "position": _xy(node["id"], node.get("xy")),
                               "data": {"parameters": parameters}})
        known = {parameter.name for parameter in step.spec.parameters}
        unknown = sorted(set(parameters) - known)
        if unknown:
            raise WorkflowError(
                f"{step.id} ({step.spec.name}) has no parameter {', '.join(map(repr, unknown))}. "
                f"It has: {sorted(known) or 'none'}")
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise WorkflowError(f"{step.id}: inputs are an object of port: node, not {inputs!r}")
        for name in inputs:
            try:
                step.spec.port(name)
            except KeyError as error:
                raise WorkflowError(f"{step.id}: {error.args[0]}") from None
        steps.append(step)
        wanted.append(inputs)

    by_id = {step.id: step for step in steps}
    connections = []
    for index, (step, inputs) in enumerate(zip(steps, wanted)):
        for port in step.spec.inputs:
            if port.name in inputs and inputs[port.name] is None:
                continue     # written as unconnected: the wiring check says it needs feeding
            if port.name in inputs:
                source, output = _named(by_id, step, port, inputs[port.name])
            else:
                found = _latest(steps[:index], port)
                if isinstance(found, str):
                    raise WorkflowError(f"{step.id}.{port.name}: {found}")
                source, output = found
            connections.append(Connection(source, output, step.id, port.name))
    return steps, connections


def write(workflow) -> dict:
    """*workflow* as a short document: dependencies first, and only the inputs the rule misses."""
    order = _saved_order(workflow)
    steps = workflow.steps
    nodes = []
    for index, step_id in enumerate(order):
        step = steps[step_id]
        earlier = [steps[other] for other in order[:index]]
        inputs = {}
        for port in step.spec.inputs:
            source, output = workflow.incoming[step_id][port.name]
            if _latest(earlier, port) != (source, output):
                inputs[port.name] = (source if _of(steps[source], port) == [output]
                                     else f"{source}.{output}")
        known = {parameter.name for parameter in step.spec.parameters}
        node = {"id": step.id, "type": step.spec.name}
        if inputs:
            node["inputs"] = inputs
        # A parameter the node does not have can only come from a full-form file, which never
        # checked. Written here it would be refused on the way back in, and it meant nothing.
        node.update((key, value) for key, value in step.parameters.items() if key in known)
        if step.position:
            # Half up, as the editor's Math.round does; Python's round() goes to the even one.
            node["xy"] = [math.floor(step.position[axis] + 0.5) for axis in ("x", "y")]
        nodes.append(node)

    described = {"name": workflow.name, "description": workflow.description}
    return {**{key: value for key, value in described.items() if value is not None},
            "nodes": nodes}


def dumps(document: dict) -> str:
    """*document* as JSON text with one node per line, so a text edit can replace one node."""
    head = [f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},"
            for key, value in document.items() if key != "nodes"]
    nodes = [f"    {json.dumps(node, ensure_ascii=False)}" for node in document.get("nodes", [])]
    body = ['  "nodes": [', ",\n".join(nodes), "  ]"] if nodes else ['  "nodes": []']
    return "\n".join(["{", *head, *body, "}"]) + "\n"


def _of(step, port: Port) -> list:
    """The names of *step*'s outputs that could feed *port*."""
    return [output.name for output in step.spec.outputs if output.type is port.type]


def _latest(earlier: list, port: Port) -> tuple | str:
    """``(node, output)`` the rule gives *port* after *earlier*, or why it gives nothing."""
    for step in reversed(earlier):
        names = _of(step, port)
        if len(names) == 1:
            return step.id, names[0]
        if names:
            return (f"{step.id}, the latest node before it giving {port.type.value}, gives "
                    f"{len(names)}: {names}. Say which with \"inputs\": "
                    f"{{\"{port.name}\": \"{step.id}.{names[0]}\"}}")
    return (f"takes {port.type.value}, and no node before it gives one. Put one earlier, or say "
            f"where it comes from with \"inputs\"")


def _named(by_id: dict, step, port: Port, reference) -> tuple:
    """``(node, output)`` an ``inputs`` entry names: ``"node"`` or ``"node.output"``."""
    if not isinstance(reference, str):
        raise WorkflowError(f"{step.id}.{port.name}: inputs name a node, not {reference!r}")
    if reference in by_id:
        names = _of(by_id[reference], port)
        if len(names) != 1:
            given = [output.name for output in by_id[reference].spec.outputs]
            raise WorkflowError(
                f"{step.id}.{port.name} takes {port.type.value}, and {reference} gives "
                + (f"{len(names)} of those: {names}. Name one, \"{reference}.{names[0]}\""
                   if names else f"none: it gives {given or 'nothing'}"))
        return reference, names[0]
    source, _, output = reference.rpartition(".")
    if source not in by_id:
        raise WorkflowError(
            f"{step.id}.{port.name} takes from {reference!r}, which is not a node here")
    return source, output     # a wrong port or type is reported by the wiring check, with the rest


def _xy(node: str, xy) -> Optional[dict]:
    """A short node's ``xy`` as the position the editor reads, or None where it has none."""
    if xy is None:
        return None
    if (not isinstance(xy, list) or len(xy) != 2
            or not all(isinstance(axis, (int, float)) and not isinstance(axis, bool) for axis in xy)):
        raise WorkflowError(f"{node}: xy is where the editor draws it, [x, y], not {xy!r}")
    return {"x": xy[0], "y": xy[1]}


def _saved_order(workflow) -> list:
    """Every node after the nodes it takes from, keeping the workflow's own order otherwise.

    Not :attr:`~mozo.workflow.Workflow.order`, which is free to put an independent node anywhere.
    Here the order is what a file shows, so a node keeps its place unless a connection moves it,
    and a small change to the canvas stays a small change to the file.
    """
    needs = {step_id: {source for source, _ in wires.values()}
             for step_id, wires in workflow.incoming.items()}
    waiting, placed, order = list(workflow.steps), set(), []
    while waiting:
        step_id = next(step_id for step_id in waiting if needs[step_id] <= placed)
        waiting.remove(step_id)
        placed.add(step_id)
        order.append(step_id)
    return order
