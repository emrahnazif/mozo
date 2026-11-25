"""The sam2_track node: boxes from the first frame that has any start objects, which keep their id
and the class of the row that started them.

Run through its NodeSpec, the path the engine takes, with a stand-in tracker: what is under test is
the node's bookkeeping, not SAM 2 (``tests/families/test_sam2.py`` and the gate hold the model).
"""

from __future__ import annotations

import numpy as np
import pixelflow as pf

from mozo.workflow import get
from mozo.workflow.node import Models, State

IMAGE = np.zeros((40, 60, 3), dtype=np.uint8)


def _rows(*boxes, names=("person", "car", "dog")):
    found = pf.Detections()
    for index, box in enumerate(boxes):
        found.add_detection(pf.Detection(bbox=list(box), class_id=index + 5,
                                         class_name=names[index], confidence=0.9))
    return found


class FakeTracker:
    """Starts one object per box, then reports every object as visible, ids in order."""

    def __init__(self):
        self.calls = []
        self.objects = 0

    def update(self, image, boxes=None, **kwargs):
        self.calls.append(None if boxes is None else np.asarray(boxes))
        self.objects += 0 if boxes is None else len(boxes)
        found = pf.Detections()
        for index in range(self.objects):
            found.add_detection(pf.Detection(bbox=[1, 1, 5, 5], class_id=index,
                                             confidence=0.8, tracker_id=index))
        return found


class FakeModel:
    def __init__(self):
        self.trackers = []

    def tracker(self):
        self.trackers.append(FakeTracker())
        return self.trackers[-1]


def _run(frames, model=None):
    model = model or FakeModel()
    models, state = Models(lambda family, variant, **kw: model), State()
    node = get("sam2_track")
    return [node.run(image=IMAGE, detections=f, models=models, state=state) for f in frames], model


def test_frames_before_the_first_detection_pass_through_empty_and_load_nothing():
    out, model = _run([_rows(), _rows()])
    assert [len(o) for o in out] == [0, 0] and model.trackers == []


def test_the_first_detections_start_the_objects_and_later_ones_are_not_read():
    out, model = _run([_rows(), _rows((1, 2, 10, 12), (20, 2, 30, 12)), _rows((3, 3, 9, 9))])
    [tracker] = model.trackers
    assert tracker.calls[0].tolist() == [[1, 2, 10, 12], [20, 2, 30, 12]]
    assert tracker.calls[1] is None  # the third frame's detection did not become an object
    assert [len(o) for o in out] == [0, 2, 2]


def test_each_object_carries_the_class_of_the_row_that_started_it():
    out, _ = _run([_rows((1, 2, 10, 12), (20, 2, 30, 12)), _rows()])
    for frame in out:
        assert [(r.tracker_id, r.class_id, r.class_name) for r in frame] == [
            (0, 5, "person"), (1, 6, "car")]


def test_a_run_starts_from_nothing():
    model = FakeModel()
    _run([_rows((1, 2, 10, 12))], model)
    _run([_rows((1, 2, 10, 12))], model)
    assert len(model.trackers) == 2


def test_it_is_ordered():
    assert get("sam2_track").ordered
