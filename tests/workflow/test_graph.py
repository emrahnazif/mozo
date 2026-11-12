"""Building a workflow, and what running one does.

The rule these tests hold is that a workflow either is valid or does not exist. Every structural
mistake is refused by the constructor, naming the node -- so nothing here checks that a bad graph
fails *during* a run, because a bad graph never gets that far.
"""

from __future__ import annotations

import json

import pytest

from conftest import FIXTURE, document
from workflow_nodes import RECORD
from mozo.workflow import Workflow, WorkflowError


@pytest.fixture(autouse=True)
def record():
    """Clear the log of which nodes ran, before each test, and hand it over."""
    RECORD.clear()
    return RECORD


class TestOrder:
    """Nodes run after whatever feeds them."""

    def test_a_chain_runs_from_the_source_down(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("brighten", {}), "c": ("measure", {})},
            [("a", "image", "b", "image"), ("b", "image", "c", "image")]))
        assert workflow.order == ("a", "b", "c")

    def test_a_diamond_puts_the_join_last(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "l": ("brighten", {}), "r": ("widen", {}), "j": ("combine", {})},
            [("a", "image", "l", "image"), ("a", "image", "r", "image"),
             ("l", "image", "j", "left"), ("r", "image", "j", "right")]))
        assert workflow.order[0] == "a"
        assert workflow.order[-1] == "j"

    def test_the_order_does_not_depend_on_how_the_edges_were_written(self):
        edges = [("b", "image", "c", "image"), ("a", "image", "b", "image")]
        workflow = Workflow.from_dict(document(
            {"c": ("measure", {}), "b": ("brighten", {}), "a": ("make", {})}, edges))
        assert workflow.order == ("a", "b", "c")

    def test_a_cycle_is_refused_and_the_message_names_it(self):
        with pytest.raises(WorkflowError, match="cycle"):
            Workflow.from_dict(document(
                {"a": ("brighten", {}), "b": ("brighten", {})},
                [("a", "image", "b", "image"), ("b", "image", "a", "image")]))

    def test_terminals_are_the_nodes_nothing_reads_from(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("brighten", {}), "c": ("detect", {})},
            [("a", "image", "b", "image"), ("a", "image", "c", "image")]))
        assert set(workflow.terminals) == {"b", "c"}


class TestWiring:
    """Every connection names ports that exist, of types that agree."""

    def test_an_edge_to_a_node_that_is_not_there_is_refused(self):
        with pytest.raises(WorkflowError, match="not here"):
            Workflow.from_dict(document({"a": ("make", {})}, [("a", "image", "ghost", "image")]))

    def test_an_output_the_source_does_not_have_is_refused(self):
        with pytest.raises(WorkflowError, match="has no output 'picture'"):
            Workflow.from_dict(document(
                {"a": ("make", {}), "b": ("brighten", {})}, [("a", "picture", "b", "image")]))

    def test_an_input_the_target_does_not_have_is_refused(self):
        with pytest.raises(WorkflowError, match="has no input"):
            Workflow.from_dict(document(
                {"a": ("make", {}), "b": ("brighten", {})}, [("a", "image", "b", "picture")]))

    def test_connecting_types_that_disagree_is_refused(self):
        with pytest.raises(WorkflowError, match="detections.*takes image"):
            Workflow.from_dict(document(
                {"a": ("make", {}), "d": ("detect", {}), "b": ("brighten", {})},
                [("a", "image", "d", "image"), ("d", "detections", "b", "image")]))

    def test_a_node_that_produces_nothing_cannot_be_a_source(self):
        with pytest.raises(WorkflowError, match="produces nothing"):
            Workflow.from_dict(document(
                {"a": ("make", {}), "m": ("measure", {}), "b": ("brighten", {})},
                [("a", "image", "m", "image"), ("m", "image", "b", "image")]))

    def test_feeding_one_input_twice_is_refused(self):
        with pytest.raises(WorkflowError, match="fed by two"):
            Workflow.from_dict(document(
                {"a": ("make", {}), "b": ("make", {}), "c": ("brighten", {})},
                [("a", "image", "c", "image"), ("b", "image", "c", "image")]))

    def test_an_input_with_nothing_connected_is_refused(self):
        with pytest.raises(WorkflowError, match="nothing connected to its 'image'"):
            Workflow.from_dict(document({"b": ("brighten", {})}, []))

    def test_two_nodes_sharing_an_id_is_refused(self):
        twice = document({"a": ("make", {})}, [])
        twice["nodes"].append({"id": "a", "type": "make", "data": {"parameters": {}}})
        with pytest.raises(WorkflowError, match="share an id"):
            Workflow.from_dict(twice)

    def test_an_edge_missing_a_handle_is_refused_rather_than_guessed(self):
        with pytest.raises(WorkflowError, match="missing 'targetHandle'"):
            Workflow.from_dict({
                "nodes": document({"a": ("make", {}), "b": ("brighten", {})}, [])["nodes"],
                "edges": [{"source": "a", "sourceHandle": "image", "target": "b"}],
            })

    def test_an_unknown_node_type_names_the_ones_there_are(self):
        with pytest.raises(WorkflowError, match="unknown node 'invent'"):
            Workflow.from_dict(document({"a": ("invent", {})}, []))


class TestRunning:
    """What a run produces."""

    def test_every_node_s_output_comes_back_keyed_by_its_id(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {"width": 3}), "b": ("brighten", {"by": 5})},
            [("a", "image", "b", "image")]))
        results = workflow.run()
        assert set(results) == {"a", "b"}
        assert results["a"][0, 0, 0] == 3
        assert results["b"][0, 0, 0] == 8

    def test_a_node_that_produces_nothing_comes_back_as_none(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "m": ("measure", {})}, [("a", "image", "m", "image")]))
        assert workflow.run()["m"] is None

    @pytest.mark.parametrize("saved, overrides, used", [
        ({"width": 7}, {}, 7),          # what the workflow saved
        ({}, {}, 2),                    # nothing saved: the node's own default
        ({"width": 2}, {"width": 9}, 9),  # saved, then overridden at run time
    ])
    def test_where_a_parameter_s_value_comes_from(self, record, saved, overrides, used):
        Workflow.from_dict(document({"a": ("make", saved)}, [])).run(**overrides)
        assert record == [("make", used)]

    def test_a_node_runs_after_the_one_feeding_it(self, record):
        Workflow.from_dict(document(
            {"a": ("make", {"width": 4}), "m": ("measure", {})},
            [("a", "image", "m", "image")])).run()
        assert record == [("make", 4), ("measure", 4)]


class TestWhichParameterTakesAFile:
    """One derivation of it, because three copies of the name is how one of them went stale.

    The HTTP upload, the command line's ``--file`` and :meth:`run_many`'s default all need the
    same answer. Each used to hold it as a string literal, so renaming the parameter fixed one
    caller at a time and left the command line raising ``KeyError`` with nothing to notice.
    """

    def test_it_is_read_off_the_annotation_not_the_name(self):
        """``Source`` is what declares a value a person cannot type. The name is incidental."""
        made = Workflow.from_dict(document({"a": ("read_media", {})}))
        assert made.file_parameter == "a.source"

    def test_two_inputs_sharing_a_name_are_two_not_one(self):
        """Counted by name, two input nodes that both call it ``source`` were one parameter, and
        the refusal below was skipped for an ambiguity error with the wrong advice."""
        made = Workflow.from_dict(document(
            {"a": ("read_media", {}), "b": ("read_media", {"source": str(FIXTURE)})}))
        with pytest.raises(ValueError, match=r"2 parameters take a file: \['a.source', 'b.source'\]"):
            made.file_parameter

    def test_a_workflow_with_no_file_to_read_says_so(self):
        made = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("brighten", {})}, [("a", "image", "b", "image")]))
        with pytest.raises(ValueError, match="nothing for one to be"):
            made.file_parameter

    def test_two_of_them_is_refused_rather_than_chosen_between(self):
        """One file cannot say which it is, and picking for the caller would be silent."""
        made = Workflow.from_dict(document({"a": ("read_media", {}), "b": ("choose", {})}))
        with pytest.raises(ValueError, match="take a file"):
            made.file_parameter

    def test_run_many_binds_to_it_without_being_told(self, image):
        made = Workflow.from_dict(document({"a": ("read_media", {})}))
        got = list(made.run_many([str(FIXTURE)] * 2))
        assert [item for item, _ in got] == [str(FIXTURE)] * 2
        assert all(results["a"].shape == image.shape for _, results in got)


class TestWhatAFailureSays:
    """The item in a failure message, short enough to read."""

    def test_it_names_the_item_rather_than_printing_it(self):
        """``run_many`` brings paths, which name themselves. :meth:`Workflow.process` brings
        whatever the source yielded, and for a video or a folder that is the frame -- so this
        used to be the repr of a 720x1280 array with the reason buried underneath it."""
        made = Workflow.from_dict(document(
            {"a": ("emit", {"count": 2}), "b": ("explode", {})}, [("a", "image", "b", "image")]))
        with pytest.raises(RuntimeError, match=r"^a \d+x\d+ image: "):
            list(made.process())

    def test_a_path_still_names_itself(self):
        made = Workflow.from_dict(document(
            {"a": ("read_media", {}), "b": ("explode", {})}, [("a", "image", "b", "image")]))
        with pytest.raises(RuntimeError, match=r"^'.*example\.jpg': "):
            list(made.run_many([str(FIXTURE)]))


class TestOverrides:
    """Running the same workflow on something else."""

    def test_a_parameter_that_is_not_there_names_the_ones_that_are(self):
        workflow = Workflow.from_dict(document({"a": ("make", {})}, []))
        with pytest.raises(KeyError, match="no parameter 'height'"):
            workflow.run(height=1)

    def test_an_ambiguous_override_is_refused_rather_than_guessed(self):
        workflow = Workflow.from_dict(document({"a": ("make", {}), "b": ("make", {})}, []))
        with pytest.raises(KeyError, match="2 nodes have a parameter 'width'"):
            workflow.run(width=1)

    def test_the_parameters_of_a_workflow_are_reported_with_their_owners(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("brighten", {})}, [("a", "image", "b", "image")]))
        assert workflow.parameters == {"width": ["a"], "by": ["b"]}


class TestBatching:
    """A node written for one image runs over a list of them."""

    def test_a_list_arriving_on_an_input_fans_the_node_out(self, record):
        Workflow.from_dict(document(
            {"a": ("several", {"count": 3}), "m": ("measure", {})},
            [("a", "image", "m", "image")])).run()
        assert record == [("measure", 1), ("measure", 2), ("measure", 3)]

    def test_the_outputs_come_back_as_a_list_in_order(self):
        results = Workflow.from_dict(document(
            {"a": ("several", {"count": 2}), "b": ("brighten", {"by": 10})},
            [("a", "image", "b", "image")])).run()
        assert [image[0, 0, 0] for image in results["b"]] == [11, 12]

    def test_a_parameter_is_shared_across_the_batch(self, record):
        Workflow.from_dict(document(
            {"a": ("several", {"count": 2}), "b": ("brighten", {"by": 4})},
            [("a", "image", "b", "image")])).run()
        assert record == [("brighten", 1), ("brighten", 2)]

    def test_a_scalar_input_is_shared_across_a_batched_one(self):
        results = Workflow.from_dict(document(
            {"a": ("several", {"count": 2}), "b": ("make", {"width": 1}), "j": ("combine", {})},
            [("a", "image", "j", "left"), ("b", "image", "j", "right")])).run()
        assert [image.shape[1] for image in results["j"]] == [2, 3]

    def test_batches_of_different_lengths_are_refused_rather_than_padded(self):
        events = list(Workflow.from_dict(document(
            {"a": ("several", {"count": 2}), "b": ("several", {"count": 3}),
             "j": ("combine", {})},
            [("a", "image", "j", "left"), ("b", "image", "j", "right")])).stream())
        assert events[-1].status == "failed"
        assert "different lengths" in events[-1].error

    def test_a_batch_stays_a_batch_through_a_chain(self):
        results = Workflow.from_dict(document(
            {"a": ("several", {"count": 3}), "b": ("brighten", {}), "c": ("widen", {})},
            [("a", "image", "b", "image"), ("b", "image", "c", "image")])).run()
        assert len(results["c"]) == 3


class TestSeveralOutputs:
    """Wiring a node that produces more than one thing."""

    def test_each_output_can_feed_a_different_node(self):
        results = Workflow.from_dict(document(
            {"a": ("make", {"width": 2}), "s": ("split", {}),
             "m": ("measure", {}), "b": ("brighten", {})},
            [("a", "image", "s", "image"),
             ("s", "image", "b", "image"),
             ("s", "image", "m", "image")])).run()
        assert results["b"][0, 0, 0] == 4      # make 2, split +1, brighten +1
        assert results["m"] is None

    def test_the_caller_gets_both_as_a_tuple(self):
        image, detections = Workflow.from_dict(document(
            {"a": ("make", {"width": 3}), "s": ("split", {})},
            [("a", "image", "s", "image")])).run()["s"]
        assert image[0, 0, 0] == 4
        assert len(detections) == 3

    def test_an_output_the_node_does_not_have_names_the_ones_it_does(self):
        with pytest.raises(ValueError, match=r"has no output 'depth'.*\['detections', 'image'\]"):
            Workflow.from_dict(document(
                {"a": ("make", {}), "s": ("split", {}), "b": ("brighten", {})},
                [("a", "image", "s", "image"), ("s", "depth", "b", "image")]))

    def test_the_type_of_the_named_output_is_what_is_checked(self):
        with pytest.raises(WorkflowError, match="detections.*takes image"):
            Workflow.from_dict(document(
                {"a": ("make", {}), "s": ("split", {}), "b": ("brighten", {})},
                [("a", "image", "s", "image"), ("s", "detections", "b", "image")]))

    def test_a_batch_flows_through_each_output_independently(self):
        results = Workflow.from_dict(document(
            {"a": ("several", {"count": 3}), "s": ("split", {}), "b": ("brighten", {})},
            [("a", "image", "s", "image"), ("s", "image", "b", "image")])).run()
        assert len(results["b"]) == 3


class TestStreaming:
    """Progress, and what happens when a node fails."""

    def test_each_node_is_reported_starting_and_finishing(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("brighten", {})}, [("a", "image", "b", "image")]))
        assert [(event.node, event.status) for event in workflow.stream()] == [
            ("a", "running"), ("a", "completed"), ("b", "running"), ("b", "completed")]

    def test_a_failure_is_reported_once_and_names_the_node(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "x": ("explode", {})}, [("a", "image", "x", "image")]))
        events = list(workflow.stream())
        assert events[-1].status == "failed"
        assert events[-1].node == "x"
        assert "explode: as promised" in events[-1].error

    def test_nothing_downstream_of_a_failure_runs(self, record):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "x": ("explode", {}), "m": ("measure", {})},
            [("a", "image", "x", "image"), ("x", "image", "m", "image")]))
        list(workflow.stream())
        assert ("measure", 2) not in record

    def test_run_reports_only_what_completed(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "x": ("explode", {})}, [("a", "image", "x", "image")]))
        assert set(workflow.run()) == {"a"}


#: How far ahead of what it has yielded ``run_many`` may read its source. Nothing today, and the
#: bound rather than the constant because a pooled executor reads ``K`` items ahead by definition
#: -- the property worth holding is that read-ahead stays bounded, not that it stays zero.
READ_AHEAD = 0


class TestRunningOverManyItems:
    """One run per item, at any size, and what happens when one of them fails."""

    @pytest.fixture
    def counting(self):
        """A chain whose answer says which item produced it."""
        return Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("brighten", {})}, [("a", "image", "b", "image")]))

    @pytest.fixture
    def failing(self):
        """A chain whose second node always raises."""
        return Workflow.from_dict(document(
            {"a": ("make", {}), "x": ("explode", {})}, [("a", "image", "x", "image")]))

    def test_each_item_gets_its_own_run(self, counting, record):
        """Serial only: at the default two workers, item 2 starts before item 1 has finished."""
        answers = list(counting.run_many([1, 2, 3], over="width", workers=1))
        assert [item for item, _ in answers] == [1, 2, 3]
        assert [int(results["b"][0, 0, 0]) for _, results in answers] == [2, 3, 4]
        assert record == [("make", 1), ("brighten", 1), ("make", 2), ("brighten", 2),
                          ("make", 3), ("brighten", 3)]

    def test_the_default_binds_each_item_to_the_node_that_reads_a_file(self):
        """The documented default, on the one shipped node that has an ``image`` parameter."""
        workflow = Workflow.from_dict(document({"a": ("read_media", {})}))
        answers = list(workflow.run_many([str(FIXTURE), str(FIXTURE)]))
        assert [item for item, _ in answers] == [str(FIXTURE), str(FIXTURE)]
        assert answers[0][1]["a"].shape == answers[1][1]["a"].shape

    def test_the_source_is_not_read_further_ahead_than_the_run_has_got(self):
        """The property the whole method exists for: a million items cost no more than one.

        Serial reads nothing ahead at all. The pipelined default reads ahead by a bounded amount
        instead -- bounded, so a million items still cost what the items in flight cost, but not
        zero. ``tests/workflow/test_pipeline.py`` states that bound.
        """
        reached = []

        def counted():
            for width in range(1, 5):
                reached.append(width)
                yield width

        workflow = Workflow.from_dict(document({"a": ("make", {})}))
        answers = workflow.run_many(counted(), over="width", workers=1)
        next(answers)
        assert len(reached) <= 1 + READ_AHEAD, "the source was drained ahead of the run"
        next(answers)
        assert len(reached) <= 2 + READ_AHEAD

    def test_other_overrides_are_passed_through_to_every_item(self, counting):
        answers = list(counting.run_many([1, 2], over="width", by=10))
        assert [int(results["b"][0, 0, 0]) for _, results in answers] == [11, 12]

    def test_a_failing_item_stops_the_run_when_nothing_is_there_to_catch_it(self, failing):
        with pytest.raises(RuntimeError, match="as promised"):
            list(failing.run_many([1, 2], over="width"))

    def test_a_failing_item_is_handed_over_and_the_rest_continue(self, failing):
        failures = []
        answers = list(failing.run_many(
            [1, 2, 3], over="width", on_error=lambda item, event: failures.append((item, event))))

        assert answers == [], "an item whose graph failed must not come back as a partial answer"
        assert [item for item, _ in failures] == [1, 2, 3]
        assert failures[0][1].node == "x"
        assert "as promised" in failures[0][1].error

    def test_raising_from_the_callback_is_how_a_failure_budget_is_spelled(self, failing):
        seen = []

        def give_up(item, event):
            seen.append(item)
            if len(seen) == 2:
                raise RuntimeError("too many failures")

        with pytest.raises(RuntimeError, match="too many failures"):
            list(failing.run_many(range(1, 100), over="width", on_error=give_up))
        assert seen == [1, 2]

    def test_a_workflow_that_cannot_be_run_says_so_before_the_first_item(self):
        workflow = Workflow.from_dict(document({"a": ("make", {})}))
        answers = workflow.run_many([1, 2], over="nonesuch")
        with pytest.raises(KeyError, match="no parameter"):
            next(answers)

    def test_binding_the_items_to_a_parameter_that_is_also_fixed_is_refused(self, counting):
        """Every item would run on the fixed value, which is a silent no-op rather than an error."""
        answers = counting.run_many([1, 2], over="width", width=9)
        with pytest.raises(KeyError, match="rather than on itself"):
            next(answers)


class TestTheFileFormat:
    """Reading and writing the editor's document."""

    def test_a_workflow_survives_a_round_trip(self, tmp_path):
        saved = document(
            {"a": ("make", {"width": 3}), "b": ("brighten", {"by": 2})},
            [("a", "image", "b", "image")])
        saved["nodes"][0]["position"] = {"x": 10, "y": 20}

        path = tmp_path / "w.json"
        Workflow.from_dict(saved).save(path)
        again = Workflow.load(path)

        assert again.to_dict() == Workflow.from_dict(saved).to_dict()
        assert again.steps["a"].position == {"x": 10, "y": 20}
        assert again.steps["a"].parameters == {"width": 3}

    def test_the_saved_file_is_json_with_nodes_and_edges(self, tmp_path):
        path = tmp_path / "w.json"
        Workflow.from_dict(document({"a": ("make", {})}, [])).save(path)
        assert set(json.loads(path.read_text())) == {"nodes", "edges"}

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Workflow.load(tmp_path / "nothing.json")

    def test_a_node_without_an_id_or_a_type_is_refused(self):
        with pytest.raises(WorkflowError, match="needs an id and a type"):
            Workflow.from_dict({"nodes": [{"type": "make"}], "edges": []})


    def test_a_name_and_a_description_survive_a_round_trip(self, tmp_path):
        saved = {"name": "Blur faces", "description": "Find faces and blur them.",
                 **document({"a": ("make", {})}, [])}
        saved["nodes"][0]["position"] = {"x": 0, "y": 0}
        path = tmp_path / "w.json"
        Workflow.from_dict(saved).save(path)
        again = Workflow.load(path)

        assert (again.name, again.description) == ("Blur faces", "Find faces and blur them.")
        assert again.to_dict() == saved
        assert list(json.loads(path.read_text()))[:2] == ["name", "description"]

    def test_a_file_without_them_loads_and_saves_as_it_was(self, tmp_path):
        """Written only when set, so an old file loaded and saved shows no change in a diff."""
        path = tmp_path / "w.json"
        Workflow.from_dict(document({"a": ("make", {"width": 3})}, [])).save(path)
        before = path.read_text()

        again = Workflow.load(path)
        assert again.name is None and again.description is None
        again.save(path)
        assert path.read_text() == before

    def test_a_name_that_is_not_text_is_refused(self):
        with pytest.raises(WorkflowError, match="name is text"):
            Workflow.from_dict({"name": 3, **document({"a": ("make", {})}, [])})


#: The ways a document can fail to be a workflow that ``TestWiring`` does not already cover -- its
#: refusals are asserted as ``WorkflowError`` there.
NOT_A_WORKFLOW = [
    (["nodes"], "a JSON object"),
    ({"nodes": {}}, "nodes are a list"),
    ({"nodes": [], "edges": "a-b"}, "edges are a list"),
    ({"nodes": ["make"]}, "a node is an object"),
    ({"nodes": [{"id": "a", "type": "make", "data": {"parameters": [1]}}]}, "parameters are"),
    ({**document({"a": ("make", {})}), "edges": ["a->b"]}, "an edge is an object"),
]


class TestAFileThatIsNotAWorkflow:
    """One error for every way a document can be invalid, so a caller catches one thing."""

    @pytest.mark.parametrize("data, says", NOT_A_WORKFLOW, ids=lambda v: v if isinstance(v, str) else "")
    def test_it_raises_one_error_that_says_which(self, data, says):
        with pytest.raises(WorkflowError, match=says):
            Workflow.from_dict(data)

    def test_the_error_is_a_value_error_so_existing_handlers_still_catch_it(self):
        with pytest.raises(ValueError):
            Workflow.from_dict(document({"a": ("invent", {})}))

    def test_a_file_that_is_not_json_names_itself(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{nodes")
        with pytest.raises(WorkflowError, match="broken.json is not JSON"):
            Workflow.load(path)

    def test_two_sources_are_a_fault_in_the_file(self):
        both = Workflow.from_dict(document({"a": ("emit", {}), "b": ("emit", {})}))
        with pytest.raises(WorkflowError, match="2 sources"):
            both.source

    def test_an_override_that_names_nothing_is_the_caller_s_mistake_not_the_file_s(self):
        """Kept a ``KeyError``: the file is fine, and saying otherwise would send someone to fix
        a file that has nothing wrong with it."""
        workflow = Workflow.from_dict(document({"a": ("make", {})}))
        with pytest.raises(KeyError) as raised:
            workflow.run(height=1)
        assert not isinstance(raised.value, WorkflowError)


class TestOverridesForOneNode:
    """``node.parameter`` sets one node's parameter, so two nodes sharing a name can differ."""

    @pytest.fixture
    def twins(self):
        """Two nodes with a parameter of the same name, side by side."""
        return Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("make", {}), "c": ("combine", {})},
            [("a", "image", "c", "left"), ("b", "image", "c", "right")]))

    def test_each_node_gets_its_own_value(self, twins):
        joined = twins.run(**{"a.width": 2, "b.width": 3})["c"]
        assert joined[0].tolist() == [[2, 2, 2]] * 2 + [[3, 3, 3]] * 3

    def test_an_ambiguous_name_says_how_to_say_which(self, twins):
        with pytest.raises(KeyError, match="Say which with 'a.width' or 'b.width'"):
            twins.run(width=1)

    def test_a_node_that_is_not_there_is_named(self, twins):
        with pytest.raises(KeyError, match="no node 'z' in this workflow"):
            twins.run(**{"z.width": 1})

    def test_a_parameter_the_node_does_not_have_is_named(self, twins):
        with pytest.raises(KeyError, match=r"a: make has no parameter 'height'"):
            twins.run(**{"a.height": 1})

    def test_two_keys_for_one_parameter_are_refused_rather_than_one_winning(self):
        workflow = Workflow.from_dict(document({"a": ("make", {})}))
        with pytest.raises(KeyError, match="'width' and 'a.width' both set a.width"):
            workflow.run(**{"width": 2, "a.width": 3})

    def test_a_node_id_may_have_a_dot_in_it(self):
        workflow = Workflow.from_dict(document({"make.v2": ("make", {})}))
        assert workflow.run(**{"make.v2.width": 4})["make.v2"].shape[1] == 4

    def test_check_takes_them_too(self, twins):
        assert twins.check(**{"a.width": 2}).missing == ()

    @pytest.mark.parametrize("workers", [1, 2])
    def test_run_many_binds_the_items_to_one_node_and_leaves_its_twin_alone(self, twins, workers):
        """The engines used to find the bound node by looking for the parameter's name among the
        overrides. With ``b.width`` set, both nodes' settings hold a ``width``, and a lookup by name
        could bind the item to the wrong one or drop the other's value."""
        answers = dict(twins.run_many([1, 2], over="a.width", workers=workers, **{"b.width": 3}))
        assert [answers[item]["c"].shape[1] for item in (1, 2)] == [4, 5]
        assert all(int(answers[item]["c"][0, -1, 0]) == 3 for item in (1, 2))

    def test_run_many_refuses_an_item_parameter_that_is_also_an_override(self, twins):
        with pytest.raises(KeyError, match="both the parameter each item binds to"):
            list(twins.run_many([1], over="a.width", **{"a.width": 2, "b.width": 1}))


class TestPromptParameters:
    """Every parameter that takes prompts, found by what it is rather than what it is called."""

    @pytest.fixture
    def side_by_side(self):
        return Workflow.from_dict(document(
            {"a": ("make", {}), "find": ("locate", {}), "pick": ("choose_between", {})},
            [("a", "image", "find", "image"), ("a", "image", "pick", "image")]))

    def test_each_is_named_by_the_key_an_override_takes(self, side_by_side):
        assert side_by_side.prompt_parameters == ("find.text", "pick.classes")

    def test_a_workflow_without_any_has_none(self):
        assert Workflow.from_dict(document({"a": ("make", {})})).prompt_parameters == ()

    def test_one_list_given_once_reaches_every_node_and_numbers_its_answers(self, side_by_side):
        prompts = ["hard hat", "vest", "boot"]
        results = side_by_side.run(**{key: prompts for key in side_by_side.prompt_parameters})
        for answer in (results["find"], results["pick"]):
            assert sorted((row.class_id, row.class_name) for row in answer) == list(enumerate(prompts))

    def test_the_catalogue_says_which_parameters_take_prompts(self):
        from mozo.workflow import registry

        [text] = registry.get("locate").to_dict()["parameters"]
        assert text["prompts"] is True and text["kind"] == "list"
        assert "prompts" not in registry.get("make").to_dict()["parameters"][0]
