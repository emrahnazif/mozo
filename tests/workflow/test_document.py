"""The short workflow document: one node per line, connections written only where the rule misses."""

from __future__ import annotations

import re

import pytest

import workflow_nodes  # noqa: F401  (registers make, brighten, ...)
from conftest import document
from mozo.workflow import Image, Workflow, WorkflowError, node
from mozo.workflow.document import dumps


@node(category="Test", outputs=["image", "image_2"])
def twins(image: Image) -> tuple[Image, Image]:
    """Two images from one, for an input the rule cannot settle alone."""
    return image, image


def wires(workflow: Workflow) -> set:
    """Every connection, as ``(source, output, target, input)``."""
    return {(c.source, c.source_output, c.target, c.target_input) for c in workflow.connections}


def short(*nodes: dict, **about) -> dict:
    return {**about, "nodes": list(nodes)}


class TestTheRule:
    """An input not written takes the latest earlier output of its type."""

    def test_a_chain_needs_no_connections(self):
        workflow = Workflow.from_dict(short(
            {"id": "a", "type": "make"}, {"id": "b", "type": "brighten", "by": 2},
            {"id": "c", "type": "measure"}))
        assert wires(workflow) == {("a", "image", "b", "image"), ("b", "image", "c", "image")}
        assert workflow.steps["b"].parameters == {"by": 2}

    def test_each_input_takes_the_latest_of_its_own_type(self):
        """``brighten`` gives an image, not detections, so ``detect``'s boxes still reach ``split``."""
        workflow = Workflow.from_dict(short(
            {"id": "a", "type": "make"}, {"id": "d", "type": "detect"},
            {"id": "c", "type": "combine", "inputs": {"left": "a"}}))
        assert ("a", "image", "d", "image") in wires(workflow)
        assert ("a", "image", "c", "right") in wires(workflow)

    def test_one_output_feeds_any_number_of_nodes(self):
        workflow = Workflow.from_dict(short(
            {"id": "a", "type": "make"}, {"id": "d", "type": "detect"},
            {"id": "m", "type": "measure"}))
        assert {("a", "image", "d", "image"), ("a", "image", "m", "image")} <= wires(workflow)

    def test_an_input_written_by_port(self):
        workflow = Workflow.from_dict(short(
            {"id": "a", "type": "make"}, {"id": "t", "type": "twins"},
            {"id": "b", "type": "brighten", "inputs": {"image": "t.image_2"}}))
        assert ("t", "image_2", "b", "image") in wires(workflow)

    def test_a_node_giving_two_of_the_type_must_be_named(self):
        with pytest.raises(WorkflowError, match=r'b.image: t, the latest node .* "t.image"'):
            Workflow.from_dict(short({"id": "a", "type": "make"}, {"id": "t", "type": "twins"},
                                     {"id": "b", "type": "brighten"}))

    def test_nothing_earlier_of_the_type_says_so(self):
        with pytest.raises(WorkflowError, match="b.image: takes image, and no node before it"):
            Workflow.from_dict(short({"id": "b", "type": "brighten"}))


class TestWhatIsRefused:
    """Typos are the failure this form exists for, so none is ignored."""

    def test_an_unknown_parameter(self):
        with pytest.raises(WorkflowError, match=r"b \(brighten\) has no parameter 'bye'. It has: \['by'\]"):
            Workflow.from_dict(short({"id": "a", "type": "make"},
                                     {"id": "b", "type": "brighten", "bye": 2}))

    def test_an_unknown_input(self):
        with pytest.raises(WorkflowError, match="has no input 'picture'"):
            Workflow.from_dict(short({"id": "a", "type": "make"},
                                     {"id": "b", "type": "brighten", "inputs": {"picture": "a"}}))

    def test_an_input_from_a_node_that_is_not_here(self):
        with pytest.raises(WorkflowError, match="takes from 'ghost', which is not a node here"):
            Workflow.from_dict(short({"id": "a", "type": "make"},
                                     {"id": "b", "type": "brighten", "inputs": {"image": "ghost"}}))

    def test_an_input_from_a_node_giving_none_of_the_type(self):
        with pytest.raises(WorkflowError, match="takes image, and d gives none"):
            Workflow.from_dict(short({"id": "a", "type": "make"}, {"id": "d", "type": "detect"},
                                     {"id": "b", "type": "brighten", "inputs": {"image": "d"}}))

    def test_an_xy_that_is_not_a_pair_of_numbers(self):
        with pytest.raises(WorkflowError, match=r"xy is where the editor draws it, \[x, y\]"):
            Workflow.from_dict(short({"id": "a", "type": "make", "xy": {"x": 1, "y": 2}}))

    def test_a_node_may_not_have_a_parameter_called_like_a_field(self):
        with pytest.raises(TypeError, match="may not be called 'inputs'"):
            @node(category="Test")
            def clash(image: Image, inputs: int = 0) -> Image:
                """Clash."""


class TestSaving:
    """What is written is what is read back, in a stable order."""

    def test_only_what_the_rule_misses_is_written(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("brighten", {}), "c": ("combine", {})},
            [("a", "image", "b", "image"), ("a", "image", "c", "left"),
             ("b", "image", "c", "right")]))
        assert workflow.to_dict()["nodes"] == [
            {"id": "a", "type": "make"}, {"id": "b", "type": "brighten"},
            {"id": "c", "type": "combine", "inputs": {"left": "a"}}]

    def test_a_node_comes_after_what_it_takes_from_and_keeps_its_place_otherwise(self):
        workflow = Workflow.from_dict(document(
            {"m": ("measure", {}), "x": ("make", {}), "a": ("make", {})},
            [("a", "image", "m", "image")]))
        assert [n["id"] for n in workflow.to_dict()["nodes"]] == ["x", "a", "m"]

    @pytest.mark.parametrize("edges", [
        [("a", "image", "b", "image"), ("a", "image", "c", "left"), ("b", "image", "c", "right")],
        [("a", "image", "b", "image"), ("b", "image", "c", "left"), ("a", "image", "c", "right")],
    ])
    def test_reading_what_was_saved_gives_the_same_graph(self, edges):
        original = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("brighten", {}), "c": ("combine", {})}, edges))
        again = Workflow.from_dict(original.to_dict())
        assert wires(again) == wires(original)
        assert again.to_dict() == original.to_dict()

    def test_a_second_output_of_the_same_type_is_written_by_port(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "t": ("twins", {}), "b": ("brighten", {})},
            [("a", "image", "t", "image"), ("t", "image_2", "b", "image")]))
        assert workflow.to_dict()["nodes"][2]["inputs"] == {"image": "t.image_2"}

    def test_the_position_is_saved_as_whole_pixels(self):
        workflow = Workflow.from_dict(short({"id": "a", "type": "make", "xy": [10.4, 20.6]}))
        assert workflow.to_dict()["nodes"][0]["xy"] == [10, 21]

    def test_a_parameter_the_node_does_not_have_is_dropped(self):
        """Only a full-form file could carry one, and written short it would be refused."""
        workflow = Workflow.from_dict(document({"a": ("make", {"width": 3, "stale": 1})}))
        assert workflow.to_dict()["nodes"] == [{"id": "a", "type": "make", "width": 3}]

    def test_one_node_per_line(self):
        text = dumps({"name": "Ünï", "nodes": [{"id": "a", "type": "make"}]})
        assert text == '{\n  "name": "Ünï",\n  "nodes": [\n    {"id": "a", "type": "make"}\n  ]\n}\n'


class TestTheFullForm:
    """Every file written before keeps its meaning."""

    @pytest.mark.parametrize("data, message", [
        ({"name": "Squares", "workflow": {"type": "find"}},
         "a workflow has no field 'workflow'. It has: name, description, nodes, edges"),
        ({"nodes": [{"id": "a", "type": "make", "parameters": {}}], "edges": []},
         "node a has no field 'parameters'. It has: id, type, position, data"),
        ({"nodes": [{"id": "a", "type": "make", "data": {"parameter": {}}}], "edges": []},
         "node a's data has no field 'parameter'. It has: parameters"),
        ({"nodes": [{"id": "a", "type": "make"}, {"id": "b", "type": "measure"}],
          "edges": [{"source": "a", "sourceHandle": "image", "target": "b", "targetHandle": "image",
                     "label": "x"}]},
         "an edge has no field 'label'"),
    ])
    def test_a_key_mozo_does_not_read_is_refused_naming_the_ones_it_does(self, data, message):
        """Ignored, a misplaced key loaded as a workflow missing what it meant to say."""
        with pytest.raises(WorkflowError, match=re.escape(message)):
            Workflow.from_dict(data)

    def test_a_document_with_edges_is_read_as_before(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("brighten", {"by": 2})}, [("a", "image", "b", "image")]))
        assert wires(workflow) == {("a", "image", "b", "image")}

    def test_a_full_form_node_without_edges_still_loads(self):
        workflow = Workflow.from_dict({"nodes": [
            {"id": "a", "type": "make", "position": {"x": 1, "y": 2}, "data": {"parameters": {"width": 4}}}]})
        assert workflow.steps["a"].parameters == {"width": 4}
        assert workflow.to_dict()["nodes"] == [{"id": "a", "type": "make", "width": 4, "xy": [1, 2]}]
