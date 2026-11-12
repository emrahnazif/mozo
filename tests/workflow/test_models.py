"""Where a workflow run's models come from, and a model of your own as a node.

A host embedding mozo runs workflows with its own :class:`~mozo.manager.ModelManager` -- its
device choice, its unloading, its lifetime -- or with a callable that passes progress and cancel
to a load part way into a run. The default stays the shared cache behind :func:`mozo.get_model`,
so a workflow that says nothing runs as it always has.
"""

from __future__ import annotations

import threading
from typing import Annotated

import pixelflow as pf
import pytest

import mozo
from conftest import FIXTURE, document
from mozo.manager import ModelManager
from mozo.workflow import Workflow
from mozo.workflow.node import Models, NodeSpec, Weights
from mozo.workflow.nodes.model import variants


class Recorder:
    """A provider that records what it was asked for and hands back a model that finds nothing."""

    def __init__(self):
        self.asked, self.threads = [], set()

    def get_model(self, family, variant, **kwargs):
        self.asked.append((family, variant, kwargs))
        self.threads.add(threading.current_thread().name)
        return self

    def predict(self, image, **kwargs):
        return pf.Detections()


def _detect(node: str = "rfdetr", **parameters) -> Workflow:
    return Workflow.from_dict(document(
        {"load": ("read_media", {"source": str(FIXTURE)}), "detect": (node, parameters)},
        [("load", "image", "detect", "image")]))


@pytest.fixture
def shared(monkeypatch):
    """A fresh shared cache, so a test can see whether anything reached it."""
    fresh = ModelManager()
    monkeypatch.setattr(mozo, "_shared", fresh)
    return fresh


class TestWhereModelsComeFrom:
    def test_a_callers_provider_is_asked_and_the_shared_cache_is_not(self, shared):
        recorder = Recorder()
        results = _detect(variant="small").run(models=recorder)
        assert "detect" in results
        assert recorder.asked == [("rfdetr", "small", {})]
        assert shared.loaded() == []

    def test_a_model_manager_of_the_callers_own_is_used(self, shared, monkeypatch):
        monkeypatch.setattr("mozo.manager._build", lambda family, variant, **kw: Recorder())
        own = ModelManager()
        _detect().run(models=own)
        assert own.loaded() == ["rfdetr/nano"]
        assert shared.loaded() == []

    def test_a_plain_callable_works_too(self, shared):
        """How a host passes progress and cancel through to a load inside a run."""
        recorder = Recorder()
        _detect().run(models=lambda family, variant, **kw: recorder.get_model(family, variant, **kw))
        assert recorder.asked == [("rfdetr", "nano", {})]

    def test_saying_nothing_is_the_shared_cache_as_before(self, shared, monkeypatch):
        monkeypatch.setattr("mozo.manager._build", lambda family, variant, **kw: Recorder())
        _detect().run()
        assert shared.loaded() == ["rfdetr/nano"]

    def test_it_reaches_a_pipelines_worker_threads(self, shared):
        """Supplied through the node's arguments, not a thread-local, so workers see it."""
        recorder = Recorder()
        done = list(_detect().run_many([str(FIXTURE)] * 6, over="source", workers=2,
                                       models=recorder))
        assert len(done) == 6
        assert len(recorder.asked) == 6
        assert threading.current_thread().name not in recorder.threads
        assert shared.loaded() == []

    def test_every_run_method_takes_it(self, shared):
        recorder = Recorder()
        workflow = _detect()
        list(workflow.stream(models=recorder))
        list(workflow.process(models=recorder, workers=1))
        assert [family for family, _, _ in recorder.asked] == ["rfdetr", "rfdetr"]
        assert shared.loaded() == []

    def test_the_editor_never_sees_it(self):
        from mozo.workflow.registry import catalogue

        for entry in catalogue():
            assert "models" not in {p["name"] for p in entry["parameters"]}, entry["name"]


class TestACheckpointOfYourOwn:
    def test_it_asks_for_the_architecture_with_the_checkpoint_and_labels(self, tmp_path):
        recorder = Recorder()
        checkpoint = tmp_path / "hardhats.pth"
        checkpoint.write_bytes(b"weights")  # the run checks the file is there before it starts
        _detect("rfdetr_checkpoint", checkpoint=str(checkpoint), architecture="small",
                labels="hardhat, vest").run(models=recorder)
        assert recorder.asked == [("rfdetr", "small", {
            "checkpoint_path": str(checkpoint), "labels": ["hardhat", "vest"]})]

    def test_without_labels_the_checkpoints_own_names_are_used(self, tmp_path):
        recorder = Recorder()
        (tmp_path / "x.pth").write_bytes(b"weights")
        _detect("rfdetr_checkpoint", checkpoint=str(tmp_path / "x.pth")).run(models=recorder)
        assert recorder.asked == [("rfdetr", "nano", {"checkpoint_path": str(tmp_path / "x.pth")})]

    def test_no_checkpoint_chosen_says_so(self):
        events = list(_detect("rfdetr_checkpoint").stream(models=Recorder()))
        [failed] = [e for e in events if e.status == "failed"]
        assert failed.node == "detect" and "choose the checkpoint" in failed.error

    def test_a_real_checkpoint_runs_and_answers_with_its_labels(self, shared):
        """End to end: RF-DETR nano's published weights, run as though they were your own."""
        from mozo import weights

        weights.configure(offline=True)
        if not weights.is_cached("rfdetr", "nano", "torch-fp32"):
            pytest.skip("rfdetr/nano is not downloaded here")
        checkpoint = weights.resolve("rfdetr", "nano", "torch-fp32")
        names = [f"thing-{i}" for i in range(91)]  # COCO ids run to 90
        own = ModelManager()
        results = _detect("rfdetr_checkpoint", checkpoint=str(checkpoint),
                          labels=", ".join(names), threshold=0.3).run(models=own)
        found = results["detect"].to_dict()
        assert found, "the fixture photograph has people in it"
        assert {row["class_name"] for row in found} <= set(names)
        assert shared.loaded() == []


class TestBeforeARun:
    """A workflow from another machine names files that are not here. Say so once, all of them."""

    def _shared(self):
        return Workflow.from_dict(document({
            "load": ("read_media", {"source": "/nowhere/frames"}),
            "helmets": ("rfdetr_checkpoint", {"checkpoint": "/nowhere/hardhat.pth"}),
            "vests": ("rfdetr_checkpoint", {"checkpoint": "/nowhere/vest.pth"}),
        }, [("load", "image", "helmets", "image"), ("load", "image", "vests", "image")]))

    def test_check_lists_every_missing_file_without_running(self):
        found = self._shared().check()
        assert found.missing == (("load", "source", "/nowhere/frames"),
                                 ("helmets", "checkpoint", "/nowhere/hardhat.pth"),
                                 ("vests", "checkpoint", "/nowhere/vest.pth"))

    @pytest.mark.parametrize("start", [
        lambda workflow: workflow.run(models=Recorder()),
        lambda workflow: list(workflow.stream(models=Recorder())),
        lambda workflow: list(workflow.process(models=Recorder())),
    ], ids=["run", "stream", "process"])
    def test_a_run_refuses_to_start_and_names_them_all(self, start):
        with pytest.raises(FileNotFoundError) as error:
            start(self._shared())
        message = str(error.value)
        assert "3 files that are not here" in message
        for name in ("/nowhere/frames", "/nowhere/hardhat.pth", "/nowhere/vest.pth", "helmets"):
            assert name in message

    def test_what_is_checked_is_what_the_run_would_use(self, tmp_path):
        """A value supplied at run time replaces the saved one before anything is checked."""
        (tmp_path / "hardhat.pth").write_bytes(b"w")
        found = self._shared().check(source=str(FIXTURE))
        assert [node for node, _, _ in found.missing] == ["helmets", "vests"]

    def test_models_still_to_download_are_noted_not_refused(self, tmp_path, caplog):
        from mozo import weights

        weights.configure(cache=tmp_path / "empty")
        workflow = _detect(variant="medium")
        [(node, family, variant, size)] = workflow.check().downloads
        assert (node, family, variant) == ("detect", "rfdetr", "medium")
        assert size == weights.download_size("rfdetr", "medium")

        with caplog.at_level("INFO", logger="mozo.workflow.graph"):
            workflow.run(models=Recorder())  # notes it, then runs
        assert "detect will download rfdetr/medium" in caplog.text

    def test_a_model_already_downloaded_is_not_noted(self, monkeypatch):
        from mozo import weights

        monkeypatch.setattr(weights, "download_size", lambda family, variant: 0)
        from mozo.workflow import Preflight

        assert _detect().check() == Preflight(missing=(), downloads=())


def _model_nodes():
    from mozo.workflow import registry

    return [registry.get(name) for name in registry.names() if registry.get(name).models]


class TestEveryModelNodeDeclaresItsModel:
    def test_every_declared_family_is_published(self):
        from mozo.registry import MODEL_REGISTRY

        declared = {spec.name: spec.weights.family for spec in _model_nodes() if spec.weights}
        assert declared and set(declared.values()) <= set(MODEL_REGISTRY)
        assert declared["edgetam_boxes"] == "edgetam" and declared["sam3_boxes"] == "sam3"

    def test_only_a_checkpoint_of_your_own_declares_none(self):
        assert [spec.name for spec in _model_nodes() if not spec.weights] == ["rfdetr_checkpoint"]

    def test_no_node_takes_a_parameter_called_device(self):
        """``check(device=)`` sits beside the parameter overrides, so a node parameter of that
        name could never be overridden there."""
        from mozo.workflow import registry

        assert not [name for name in registry.names()
                    if any(p.name == "device" for p in registry.get(name).parameters)]

    @pytest.mark.parametrize("spec", [s for s in _model_nodes() if s.weights], ids=lambda s: s.name)
    def test_the_node_asks_for_the_model_it_declares(self, spec):
        """Run the node and record what it asks the run's models for. The declaration is a
        second statement of the call in the node's body, and this is what keeps the two equal."""
        import numpy as np

        from mozo.workflow.node import PortType

        asked = []

        class Asked(Exception):
            pass

        def models(family, variant, **kwargs):
            asked.append((family, variant))
            raise Asked

        one = pf.Detections()
        one.add_detection(pf.Detection(bbox=[1, 1, 6, 6], confidence=0.9, class_id=0))
        given = {PortType.IMAGE: np.zeros((8, 8, 3), dtype=np.uint8), PortType.DETECTIONS: one}
        arguments = {port.name: given[port.type] for port in spec.inputs}
        arguments |= {p.name: p.default for p in spec.parameters}
        arguments[spec.models] = models
        with pytest.raises(Asked):
            spec.run(**arguments)
        assert asked == [(spec.weights.family, spec.weights.default)]


class TestTheWeightsMark:
    def test_two_variant_choices_on_one_node_are_refused(self):
        def both(models: Models, a: variants("sam2") = "tiny",
                 b: variants("edgetam") = "edgetam") -> None:
            pass

        with pytest.raises(TypeError, match="each choose a model's variant"):
            NodeSpec.from_function(both, "Test")

    def test_the_mark_belongs_on_a_choice(self):
        def loose(variant: Annotated[str, Weights("sam2")] = "tiny") -> None:
            pass

        with pytest.raises(TypeError, match="must be a Literal"):
            NodeSpec.from_function(loose, "Test")
