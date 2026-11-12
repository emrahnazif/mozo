"""Workflows that annotate a list of images: detect, segment the boxes, read existing ones.

An application turns each terminal ``Detections`` into annotations, mapping ``class_id`` back to
its class and each mask to an outline. So what these tests hold is the contract it relies on:
rows in order with every field kept, prompts that are never split, and per-item boxes read back
from the file ``save_annotations`` writes -- written here by ``save_annotations`` itself, so the
two cannot drift apart unnoticed.
"""

from __future__ import annotations

import shutil

import numpy as np
import pixelflow as pf
import pytest

from conftest import FIXTURE, document
from mozo.workflow import Workflow


def _found(boxes, scores, class_ids, names):
    """Detections as a detector would return them, each row with its own name."""
    made = pf.detections.from_arrays(np.asarray(boxes, dtype=np.float32), np.asarray(scores),
                                     class_ids=np.asarray(class_ids))
    for row, name in zip(made, names):
        row.class_name = name
    return made


def _rows(found):
    """What each row says, for comparing two Detections field by field."""
    return [(r.class_id, r.class_name, r.confidence, r.bbox) for r in found]


class Segmenter:
    """Stands in for SAM 2 / EdgeTAM: one mask per box, filling exactly that box."""

    def __init__(self):
        self.calls = []

    def predict(self, image, boxes=None, multimask_output=True, **kwargs):
        self.calls.append({"boxes": np.asarray(boxes), "multimask_output": multimask_output})
        height, width = image.shape[:2]
        masks = np.zeros((len(boxes), height, width), dtype=bool)
        for index, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=int)):
            masks[index, y1:y2, x1:x2] = True
        # Numbered by prompt, as the adapter numbers them; the box is the mask's own bounds.
        return pf.detections.from_arrays(np.asarray(boxes, dtype=np.float32),
                                         np.full(len(boxes), 0.99),
                                         class_ids=np.arange(len(boxes)), masks=masks)


class Detector:
    """Stands in for Grounding DINO: the given answers, one per image, in turn."""

    def __init__(self, *answers):
        self.answers = list(answers)

    def predict(self, image, text=None, **kwargs):
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class Provider:
    """The run's models: hands out the stand-ins, and remembers which families were asked for."""

    def __init__(self, **models):
        self.models, self.asked = models, []

    def __call__(self, family, variant, **kwargs):
        self.asked.append(family)
        return self.models[family]


def _on_cpu():
    """A provider of real models from the local cache, on the CPU."""
    from mozo.manager import ModelManager

    models = ModelManager()
    return lambda family, variant, **kw: models.get_model(
        family, variant, device="cpu", runtime="torch-fp32", **kw)


def _require(*models):
    from mozo import weights

    weights.configure(offline=True)
    for family, variant in models:
        if not weights.is_cached(family, variant, "torch-fp32"):
            pytest.skip(f"{family}/{variant} is not downloaded here")


def _graph(nodes, edges=()):
    return Workflow.from_dict(document(nodes, list(edges)))


THREE = _found([[10, 20, 60, 80], [100, 30, 180, 90], [200, 200, 260, 300]],
               [0.91, 0.42, 0.67], [2, 0, 1], ["vest", "hardhat", "boot"])

SEGMENTED = [("load", "image", "segment", "image"), ("detect", "detections", "segment", "detections")]


def _detect_then_segment(segmenter="edgetam_boxes", detector=("grounding_dino", {"text": ["vest"]}),
                         source=FIXTURE):
    return _graph({"load": ("read_media", {"source": str(source)}), "detect": detector,
                   "segment": (segmenter, {})},
                  [("load", "image", "detect", "image"), *SEGMENTED])


def _read_back(path, source=FIXTURE):
    return _graph({"load": ("read_media", {"source": str(source)} if source else {}),
                   "boxes": ("read_annotations", {"path": str(path)})},
                  [("load", "image", "boxes", "image")])


def _save(source, path, *answers):
    """Write *path* with save_annotations: each image read from *source* gets the next answer."""
    workflow = _graph({"load": ("read_media", {"source": str(source)}),
                       "detect": ("grounding_dino", {}),
                       "save": ("save_annotations", {"path": str(path)})},
                      [("load", "image", "detect", "image"), ("load", "image", "save", "image"),
                       ("detect", "detections", "save", "detections")])
    list(workflow.process(models=Provider(grounding_dino=Detector(*answers)), workers=1))


class TestSegmentingBoxes:
    @pytest.mark.parametrize("node, family", [("edgetam_boxes", "edgetam"), ("sam2_boxes", "sam2")])
    def test_every_row_comes_back_in_order_with_its_fields_and_a_mask(self, node, family):
        out = _detect_then_segment(node).run(
            models=Provider(grounding_dino=Detector(THREE), **{family: Segmenter()}))["segment"]

        assert _rows(out) == _rows(THREE)  # the detector's boxes, not the masks' bounds
        for given, got in zip(THREE, out):
            [mask] = got.masks
            x1, y1, x2, y2 = (int(v) for v in given.bbox)
            assert mask.any() and not mask[:y1].any() and not mask[y2:].any()
            assert not mask[:, :x1].any() and not mask[:, x2:].any()

    def test_all_of_an_images_boxes_go_in_one_call_with_one_mask_each(self):
        segmenter = Segmenter()
        _detect_then_segment().run(models=Provider(grounding_dino=Detector(THREE), edgetam=segmenter))
        [call] = segmenter.calls
        assert call["boxes"].shape == (3, 4) and call["multimask_output"] is False

    def test_an_image_with_no_boxes_never_asks_for_the_model(self):
        models = Provider(grounding_dino=Detector(pf.Detections()), edgetam=Segmenter())
        assert len(_detect_then_segment().run(models=models)["segment"]) == 0
        assert models.asked == ["grounding_dino"]

    def test_a_row_whose_mask_comes_back_empty_is_kept(self):
        class Empty(Segmenter):
            def predict(self, image, boxes=None, **kwargs):
                found = super().predict(image, boxes=boxes, **kwargs)
                found[1].masks = [np.zeros(image.shape[:2], dtype=bool)]
                return found

        out = _detect_then_segment().run(
            models=Provider(grounding_dino=Detector(THREE), edgetam=Empty()))["segment"]
        assert len(out) == 3 and out[1].class_name == "hardhat" and not out[1].masks[0].any()

    def test_a_model_that_drops_a_box_is_refused_rather_than_misassigned(self):
        class Short(Segmenter):
            def predict(self, image, boxes=None, **kwargs):
                return super().predict(image, boxes=boxes[:2], **kwargs)

        events = list(_detect_then_segment().stream(
            models=Provider(grounding_dino=Detector(THREE), edgetam=Short())))
        [failed] = [e for e in events if e.status == "failed"]
        assert "2 masks for 3 boxes" in failed.error


def test_real_models_detect_then_segment_keeping_every_detected_row():
    """RF-DETR finds the boxes and EdgeTAM outlines them, offline from the local cache."""
    _require(("rfdetr", "nano"), ("edgetam", "edgetam"))
    results = _detect_then_segment(detector=("rfdetr", {"variant": "nano"})).run(models=_on_cpu())

    assert len(results["detect"]) > 0, "the fixture photograph has people in it"
    assert _rows(results["segment"]) == _rows(results["detect"])
    assert all(row.masks and row.masks[0].any() for row in results["segment"])


# --- Reading existing annotations back ---------------------------------------------------------

BOXES = {
    "a": _found([[10, 10, 50, 50]], [0.9], [0], ["vest"]),
    "b": _found([[20, 20, 80, 90], [300, 300, 400, 420]], [0.8, 0.7], [1, 0], ["hardhat", "vest"]),
    "c": _found([[5, 5, 15, 15], [30, 30, 60, 60], [70, 70, 90, 95]], [0.6, 0.5, 0.4],
                [2, 2, 1], ["boot", "boot", "hardhat"]),
}


@pytest.fixture
def boxed_folder(tmp_path):
    """Three photographs, and the file save_annotations writes giving each its own boxes."""
    folder = tmp_path / "photos"
    folder.mkdir()
    for stem in BOXES:
        shutil.copy(FIXTURE, folder / f"{stem}{FIXTURE.suffix}")
    annotations = tmp_path / "boxes.jsonl"
    _save(folder, annotations, *BOXES.values())
    return folder, annotations


def _complete_objects(folder, annotations):
    return _graph({"load": ("read_media", {"source": str(folder)}),
                   "detect": ("read_annotations", {"path": str(annotations)}),
                   "segment": ("edgetam_boxes", {})},
                  [("load", "image", "detect", "image"), *SEGMENTED])


class TestReadingAnnotations:
    @pytest.mark.parametrize("workers", [1, 2])
    def test_each_image_gets_its_own_boxes_back_with_masks_in_order(self, boxed_folder, workers):
        folder, annotations = boxed_folder
        done = list(_complete_objects(folder, annotations).process(
            models=Provider(edgetam=Segmenter()), workers=workers))

        assert [_rows(results["segment"]) for _, results in done] == \
            [_rows(given) for given in BOXES.values()]
        assert all(row.masks[0].any() for _, results in done for row in results["segment"])

    def test_what_save_annotations_writes_reads_back_the_same_masks_included(self, tmp_path):
        saved = tmp_path / "out.jsonl"
        segmented = _detect_then_segment().run(
            models=Provider(grounding_dino=Detector(THREE), edgetam=Segmenter()))["segment"]
        _save(FIXTURE, saved, segmented)

        back = _read_back(saved).run()["boxes"]
        assert _rows(back) == _rows(THREE)
        assert all(np.array_equal(b.masks[0], s.masks[0]) for b, s in zip(back, segmented))

    def test_run_many_over_file_paths_matches_each_file_by_its_name(self, boxed_folder):
        """How an application hands over its own list: paths, each named by its file."""
        folder, annotations = boxed_folder
        paths = [str(folder / f"{stem}{FIXTURE.suffix}") for stem in ("c", "a")]
        done = list(_complete_objects(folder, annotations).run_many(
            paths, models=Provider(edgetam=Segmenter())))
        assert [len(results["segment"]) for _, results in done] == [3, 1]

    def test_pixels_in_memory_have_no_name_and_are_refused(self, boxed_folder):
        from mozo.image import load_image

        _, annotations = boxed_folder
        events = list(_read_back(annotations, source=None).stream(source=load_image(str(FIXTURE))))
        [failed] = [e for e in events if e.status == "failed"]
        assert "pass files, not pixels" in failed.error

    def test_an_image_the_file_has_no_line_for_is_refused_by_name(self, boxed_folder):
        folder, annotations = boxed_folder
        annotations.write_text(annotations.read_text().splitlines(keepends=True)[1])  # only "b"
        events = list(_complete_objects(folder, annotations).stream(
            models=Provider(edgetam=Segmenter())))
        [failed] = [e for e in events if e.status == "failed"]  # the first image is "a"
        assert failed.node == "detect" and "has no line for 'a'" in failed.error

    def test_a_line_written_for_another_size_is_refused(self, tmp_path):
        import cv2

        from mozo.image import load_image

        smaller = tmp_path / f"{FIXTURE.stem}.png"  # the same name, a smaller picture
        cv2.imwrite(str(smaller), cv2.cvtColor(load_image(str(FIXTURE))[:10, :12], cv2.COLOR_RGB2BGR))
        annotations = tmp_path / "boxes.jsonl"
        _save(smaller, annotations, THREE)

        [failed] = [e for e in _read_back(annotations).stream() if e.status == "failed"]
        assert "written for a 12x10 image" in failed.error

    def test_no_file_chosen_says_so(self):
        [failed] = [e for e in _read_back("").stream() if e.status == "failed"]
        assert "choose the annotations file" in failed.error

    def test_the_check_before_a_run_lists_a_missing_annotations_file(self, boxed_folder):
        folder, _ = boxed_folder
        found = _complete_objects(folder, folder / "nowhere.jsonl").check()
        assert ("detect", "path", str(folder / "nowhere.jsonl")) in found.missing

    def test_the_file_a_run_reads_is_still_the_sources(self, boxed_folder):
        """An annotations file is a fixed path: --file and the editor's upload bind to the source."""
        assert _complete_objects(*boxed_folder).file_parameter == "source"


def _line(detections, **extra):
    """``annotation_line`` for the fixture photograph, at its own size."""
    from mozo.image import load_image
    from mozo.workflow import annotation_line

    height, width = load_image(str(FIXTURE)).shape[:2]
    return annotation_line(detections, label=FIXTURE.stem, width=width, height=height, **extra)


class TestWritingAnnotations:
    """``annotation_line`` is the format's one definition: the node writes through it, and so can
    an application handing a workflow the boxes it already has."""

    def test_what_it_writes_reads_back_the_same(self, tmp_path):
        annotations = tmp_path / "boxes.jsonl"
        annotations.write_text(_line(THREE))
        assert _rows(_read_back(annotations).run()["boxes"]) == _rows(THREE)

    def test_save_annotations_writes_exactly_what_it_returns(self, tmp_path):
        saved = tmp_path / "out.jsonl"
        _save(FIXTURE, saved, THREE)
        assert saved.read_text() == _line(THREE)


# --- Prompts as a list -------------------------------------------------------------------------

TEXT_NODES = ["sam3", "owlv2", "grounding_dino", "clip", "siglip2"]


class Prompted:
    """Stands in for a text-prompted model: records the prompts it was given."""

    def __init__(self):
        self.asked = []

    def predict(self, image, text=None, **kwargs):
        self.asked.append(list(text))
        return pf.Detections()


def _prompted(node, text):
    return _graph({"load": ("read_media", {"source": str(FIXTURE)}), "ask": (node, {"text": text})},
                  [("load", "image", "ask", "image")])


class TestPromptsAsAList:
    @pytest.mark.parametrize("node", TEXT_NODES)
    def test_a_list_is_taken_as_given_never_split(self, node):
        model = Prompted()
        _prompted(node, ["car, parked", "person"]).run(models=lambda family, variant, **kw: model)
        assert model.asked == [["car, parked", "person"]]

    @pytest.mark.parametrize("node", TEXT_NODES)
    def test_a_string_saved_before_still_means_what_it_meant(self, node):
        model = Prompted()
        _prompted(node, "car, person").run(models=lambda family, variant, **kw: model)
        assert model.asked == [["car", "person"]]

    def test_the_editor_is_told_it_is_a_list(self):
        from mozo.workflow.registry import catalogue

        kinds = {entry["name"]: {p["name"]: p["kind"] for p in entry["parameters"]}
                 for entry in catalogue()}
        assert all(kinds[node]["text"] == "list" for node in TEXT_NODES)
        assert kinds["rfdetr_checkpoint"]["labels"] == "list"
        assert kinds["rfdetr_checkpoint"]["checkpoint"] == kinds["read_annotations"]["path"] == "path"

    def test_a_real_sam3_answers_each_prompt_with_its_own_index(self):
        """The contract each docstring states, held on the model that has it: class_id is the
        prompt's index and class_name the prompt -- with a comma inside one of them."""
        _require(("sam3", "sam3"))
        found = _prompted("sam3", ["laptop, open", "person"]).run(models=_on_cpu())["ask"]
        assert {(r.class_id, r.class_name) for r in found} <= {(0, "laptop, open"), (1, "person")}
        assert any(r.class_id == 1 for r in found), "the fixture photograph has people in it"

    def test_checkpoint_labels_are_a_list_too(self, tmp_path):
        asked = []
        checkpoint = tmp_path / "mine.pth"
        checkpoint.write_bytes(b"w")
        _graph({"load": ("read_media", {"source": str(FIXTURE)}),
                "detect": ("rfdetr_checkpoint", {"checkpoint": str(checkpoint),
                                                 "labels": ["bolt, M8", "nut"]})},
               [("load", "image", "detect", "image")]).run(
            models=lambda family, variant, **kw: asked.append(kw) or Prompted())
        assert asked[0]["labels"] == ["bolt, M8", "nut"]


# --- Cancelling a run --------------------------------------------------------------------------

def test_closing_a_run_starts_no_further_item_and_its_threads_exit():
    """How an application cancels a job: it stops iterating. What run_many's docstring promises."""
    import threading
    import time

    call_seconds = 0.3
    started, lock = [], threading.Lock()

    class Slow:
        def predict(self, image, **kwargs):
            with lock:
                started.append(time.monotonic())
            time.sleep(call_seconds)
            return pf.Detections()

    workflow = _graph({"load": ("read_media", {}), "detect": ("rfdetr", {})},
                      [("load", "image", "detect", "image")])
    run = workflow.run_many([str(FIXTURE)] * 40, over="source", workers=2,
                            models=lambda family, variant, **kw: Slow())
    next(run)
    run.close()
    at_close = len(started)

    def pipeline_threads():
        return [t for t in threading.enumerate() if t.name.startswith(("stage-", "pipeline-"))]

    # Gone within the longest call in flight -- close() itself does not wait for them.
    for thread in pipeline_threads():
        thread.join(call_seconds * 5)
    assert pipeline_threads() == []
    assert len(started) == at_close, "an item started a model call after the run was closed"
    assert at_close < 40
