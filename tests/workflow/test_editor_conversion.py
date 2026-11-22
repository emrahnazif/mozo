"""The editor's canvas-to-document conversion, run outside a browser.

`toDocument` and `fromDocument` are the only pieces of the editor that decide what mozo receives,
and they are pure functions -- no DOM, no fetch. So they can be exercised with node, and the
document they produce can be fed straight to the server. That covers everything about the editor
except the clicking.

The editor holds its own copy of the short form's rule (mozo/workflow/document.py), so the same
documents go through both here and the results are compared: two copies of a rule are only safe
while something checks they agree.

Skipped where node is absent; the editor cannot be built there either.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mozo.workflow import Workflow, catalogue
from mozo.workflow.document import dumps

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "ui" / "src" / "lib" / "api.js"


def full(nodes: list, edges: list) -> dict:
    """A full-form document: ``nodes`` as ``(id, type, parameters, (x, y) or None)``."""
    return {
        "nodes": [{"id": id_, "type": kind, "data": {"parameters": parameters},
                   **({"position": {"x": at[0], "y": at[1]}} if at else {})}
                  for id_, kind, parameters, at in nodes],
        "edges": [{"source": s, "sourceHandle": o, "target": t, "targetHandle": i}
                  for s, o, t, i in edges],
    }


#: The same documents through Python and JS, in the full form the editor used to write. Every node
#: is placed, because the editor gives a node with no position one and Python leaves it without.
SHARED = {
    "chain": full(
        [("read", "read_media", {"source": "a.jpg"}, (10, 20)),
         ("find", "grounding_dino", {"text": ["face"], "threshold": 0.3}, (10.5, 120.5)),
         ("blur", "blur_regions", {}, (0, 200)), ("save", "save_image", {}, (0, 300))],
        [("read", "image", "find", "image"), ("read", "image", "blur", "image"),
         ("find", "detections", "blur", "detections"), ("blur", "image", "save", "image")]),
    "draw on the original": full(
        [("save", "save_image", {}, (0, 400)), ("read", "read_media", {}, (0, 0)),
         ("gray", "to_grayscale", {}, (0, 100)), ("find", "yolov26", {"variant": "nano"}, (0, 200)),
         ("draw", "draw_boxes", {}, (0, 300))],
        [("read", "image", "gray", "image"), ("gray", "image", "find", "image"),
         ("read", "image", "draw", "image"), ("find", "detections", "draw", "detections"),
         ("draw", "image", "save", "image")]),
    "two outputs": full(
        [("read", "read_media", {}, (0, 0)), ("find", "yolov26", {}, (0, 100)),
         ("flip", "flip_horizontal_with_detections", {}, (0, 200)), ("draw", "draw_boxes", {}, (0, 300))],
        [("read", "image", "find", "image"), ("read", "image", "flip", "image"),
         ("find", "detections", "flip", "detections"), ("read", "image", "draw", "image"),
         ("flip", "detections", "draw", "detections")]),
}

#: Short documents, read by both: the connections each side finds must be the same.
SHORT = {
    "chain": {"nodes": [{"id": "read", "type": "read_media"},
                        {"id": "find", "type": "grounding_dino", "text": ["person"]},
                        {"id": "draw", "type": "draw_boxes"}, {"id": "save", "type": "save_image"}]},
    "written inputs": {"nodes": [
        {"id": "read", "type": "read_media"}, {"id": "gray", "type": "to_grayscale"},
        {"id": "find", "type": "yolov26"},
        {"id": "draw", "type": "draw_boxes", "inputs": {"image": "read"}},
        {"id": "flip", "type": "flip_horizontal_with_detections"},
        {"id": "boxes", "type": "draw_boxes", "inputs": {"image": "read.image", "detections": "flip"}}]},
}

SCRIPT = """
import { toDocument, fromDocument, fileNameFor, drawable, serialise, connectDropped } from '%(api)s';

const catalogue = %(catalogue)s;
const shared = %(shared)s;
const short = %(short)s;

const nodes = [
  { id: 'load-1', type: 'default', position: { x: 10, y: 20 },
    data: { nodeType: 'read_media', parameters: { source: 'a.jpg' } } },
  { id: 'det-1', type: 'default', position: { x: 200, y: 20 },
    data: { nodeType: 'yolov26', parameters: { variant: 'seg-nano', stale: 1 } } },
];
const edges = [
  { source: 'load-1', sourceHandle: 'image', target: 'det-1', targetHandle: 'image' },
];

const document = toDocument(nodes, edges, {}, catalogue);
const back = fromDocument(document, catalogue);
const again = toDocument(back.nodes, back.edges, back.about, catalogue);

const about = { name: '  Blur faces ', description: 'Find faces, blur them.' };
const named = toDocument(nodes, edges, about, catalogue);
const namedBack = fromDocument(named, catalogue);

// Half-built: the draw node's detections input has nothing connected yet.
const half = toDocument([...nodes, { id: 'draw', type: 'default', position: { x: 0, y: 200 },
                                     data: { nodeType: 'draw_boxes', parameters: {} } }],
                        [...edges, { source: 'load-1', sourceHandle: 'image', target: 'draw', targetHandle: 'image' }],
                        {}, catalogue);
const halfBack = fromDocument(half, catalogue);

// Written by code: no positions, a branch -- read feeds both find and blur.
const loose = fromDocument(short.chain, catalogue);
const mixed = fromDocument({ nodes: [{ id: 'kept', type: 'read_media', xy: [40, 0] },
                                     { id: 'new', type: 'save_image' }] }, catalogue);

const wires = canvas => canvas.edges.map(e => [e.source, e.sourceHandle, e.target, e.targetHandle]);
const dropped = { id: 'boxes', type: 'default', position: { x: 0, y: 0 },
                  data: { nodeType: 'draw_boxes', parameters: {} } };

console.log(JSON.stringify({
  shared: Object.fromEntries(Object.entries(shared).map(([name, doc]) => {
    const canvas = fromDocument(doc, catalogue);
    return [name, toDocument(canvas.nodes, canvas.edges, canvas.about, catalogue)];
  })),
  short: Object.fromEntries(Object.entries(short).map(([name, doc]) => [name, wires(fromDocument(doc, catalogue))])),
  text: serialise({ name: 'Ünï', nodes: [{ id: 'a', type: 'read_media', xy: [1, 2] }] }),
  loose: Object.fromEntries(loose.nodes.map(n => [n.id, n.position])),
  looseWires: wires(loose),
  mixed: Object.fromEntries(mixed.nodes.map(n => [n.id, n.position])),
  dropped: wires({ edges: connectDropped(dropped, [...back.nodes, dropped], back.edges, catalogue) }),
  half, halfWires: wires(halfBack),
  uncatalogued: toDocument(nodes, edges, {}, {}),
  document,
  stable: JSON.stringify(document) === JSON.stringify(again),
  parameters: back.nodes[1].data.parameters,
  edgeIds: back.edges.map(e => e.id),
  named,
  namedAgain: toDocument(namedBack.nodes, namedBack.edges, namedBack.about, catalogue),
  fileNames: [fileNameFor(about), fileNameFor({ name: '' }), fileNameFor({ name: 'Ünïcode / ok?' })],
  drawable: drawable([
    ...edges,
    { source: 'load-1', sourceHandle: 'picture', target: 'det-1', targetHandle: 'image' },
    { source: 'ghost', sourceHandle: 'image', target: 'det-1', targetHandle: 'image' },
  ], back.nodes, catalogue),
}));
"""


@pytest.fixture(scope="module")
def converted() -> dict:
    """Run the conversion in node and hand back what it said."""
    if not shutil.which("node"):
        pytest.skip("node is not installed, so the editor cannot be built here either")
    if not API.is_file():
        pytest.skip("the editor source is not present")

    script = ROOT / "ui" / ".conversion-check.mjs"
    script.write_text(SCRIPT % {
        "api": API, "catalogue": json.dumps({entry["name"]: entry for entry in catalogue()}),
        "shared": json.dumps(SHARED), "short": json.dumps(SHORT)})
    try:
        finished = subprocess.run(["node", str(script)], capture_output=True, text=True)
    finally:
        script.unlink(missing_ok=True)
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout)


def wires(workflow: Workflow) -> list:
    return sorted([c.source, c.source_output, c.target, c.target_input] for c in workflow.connections)


class TestPythonAndTheEditorAgree:
    """The rule is written twice, so the two copies are run on the same documents."""

    @pytest.mark.parametrize("name", SHARED)
    def test_they_write_the_same_short_document(self, converted, name):
        assert converted["shared"][name] == Workflow.from_dict(SHARED[name]).to_dict()

    @pytest.mark.parametrize("name", SHORT)
    def test_they_read_the_same_connections(self, converted, name):
        assert sorted(converted["short"][name]) == wires(Workflow.from_dict(SHORT[name]))

    def test_they_write_the_same_text(self, converted):
        assert converted["text"] == dumps({"name": "Ünï", "nodes": [
            {"id": "a", "type": "read_media", "xy": [1, 2]}]})


class TestTheConversion:
    """What the canvas becomes, and what it becomes again."""

    def test_a_canvas_becomes_a_short_document(self, converted):
        assert converted["document"] == {"nodes": [
            {"id": "load-1", "type": "read_media", "source": "a.jpg", "xy": [10, 20]},
            {"id": "det-1", "type": "yolov26", "variant": "seg-nano", "xy": [200, 20]}]}

    def test_loading_a_document_and_saving_it_gives_the_same_document(self, converted):
        assert converted["stable"], "a save-load-save cycle changed the file"

    def test_a_parameter_the_node_does_not_have_is_dropped_and_the_rest_survive(self, converted):
        assert converted["parameters"] == {"variant": "seg-nano"}

    def test_an_input_with_nothing_connected_stays_unconnected(self, converted):
        """Left unwritten, the rule would connect it the next time the document is opened."""
        assert converted["half"]["nodes"][2]["inputs"] == {"detections": None}
        assert ["load-1", "image", "draw", "image"] in converted["halfWires"]
        assert not [w for w in converted["halfWires"] if w[2:] == ["draw", "detections"]]

    def test_without_a_catalogue_every_connection_is_written_out(self, converted):
        """The rule needs the catalogue; without one, a connection left unwritten would be lost."""
        assert converted["uncatalogued"]["nodes"][1]["inputs"] == {"image": "load-1.image"}

    def test_a_short_document_is_connected_by_the_rule(self, converted):
        assert sorted(converted["looseWires"]) == sorted([
            ["read", "image", "find", "image"], ["read", "image", "draw", "image"],
            ["find", "detections", "draw", "detections"], ["draw", "image", "save", "image"]])

    def test_nodes_without_a_position_are_laid_out_top_to_bottom(self, converted):
        """A document written by code has none, and Svelte Flow cannot draw or drag such a node."""
        ys = [converted["loose"][node]["y"] for node in ("read", "find", "draw", "save")]
        assert ys == sorted(ys) and len(set(ys)) == 4

    def test_a_placed_node_keeps_its_position_and_the_rest_go_beside_it(self, converted):
        assert converted["mixed"]["kept"] == {"x": 40, "y": 0}
        assert converted["mixed"]["new"]["x"] > 40

    def test_a_dropped_node_is_connected_by_the_rule(self, converted):
        assert sorted(converted["dropped"]) == [
            ["det-1", "detections", "boxes", "detections"], ["load-1", "image", "boxes", "image"]]

    def test_every_edge_gets_its_own_id(self, converted):
        ids = converted["edgeIds"]
        assert len(set(ids)) == len(ids) and all(ids)

    def test_a_name_and_a_description_survive_and_lead_the_file(self, converted):
        named = converted["named"]
        assert list(named)[:2] == ["name", "description"]
        assert (named["name"], named["description"]) == ("Blur faces", "Find faces, blur them.")
        assert converted["namedAgain"] == named

    def test_an_edge_to_a_port_or_node_that_is_not_there_is_left_out(self, converted):
        """Drawn, it would stop every store on the page from updating -- see ``drawable``."""
        split = converted["drawable"]
        assert [edge["sourceHandle"] for edge in split["drawn"]] == ["image"]
        assert [(edge["source"], edge["sourceHandle"]) for edge in split["left"]] == [
            ("load-1", "picture"), ("ghost", "image")]

    def test_the_exported_file_is_named_after_the_workflow(self, converted):
        assert converted["fileNames"] == ["blur-faces.json", "workflow.json", "n-code-ok.json"]


class TestTheServerAcceptsIt:
    """The document the editor emits, taken at its word by the server."""

    def test_its_name_reaches_the_server(self, converted):
        assert Workflow.from_dict(converted["named"]).name == "Blur faces"

    def test_it_is_a_valid_workflow(self, converted, client):
        answer = client.post("/workflow/validate",
                             data={"workflow": json.dumps(converted["document"])}).json()
        assert answer["valid"], answer
        assert answer["order"] == ["load-1", "det-1"]
        assert answer["terminals"] == ["det-1"]

    def test_a_half_built_one_is_refused_naming_the_input(self, converted, client):
        answer = client.post("/workflow/validate",
                             data={"workflow": json.dumps(converted["half"])}).json()
        assert not answer["valid"]
        assert "detections" in json.dumps(answer)
