"""The sam3_track node: one SAM 3 tracker per run, fed the phrases on every frame.

Run through its NodeSpec, with a stand-in tracker: what is under test is the node's bookkeeping,
not SAM 3 (``tests/families/test_sam3_video.py`` and the gate hold the model).
"""

from __future__ import annotations

import numpy as np
import pixelflow as pf

from mozo.workflow import get
from mozo.workflow.node import Models, State

IMAGE = np.zeros((40, 60, 3), dtype=np.uint8)


class FakeTracker:
    def __init__(self):
        self.calls = []

    def update(self, image, text=None, **kwargs):
        self.calls.append(text)
        found = pf.Detections()
        found.add_detection(pf.Detection(bbox=[1, 1, 5, 5], class_id=0, class_name=text[0],
                                         confidence=0.8, tracker_id=len(self.calls) - 1))
        return found


class FakeModel:
    def __init__(self):
        self.trackers = []

    def tracker(self):
        self.trackers.append(FakeTracker())
        return self.trackers[-1]


def _run(frames, text, model=None):
    model = model or FakeModel()
    models, state = Models(lambda family, variant, **kw: model), State()
    node = get("sam3_track")
    return [node.run(image=IMAGE, models=models, state=state, text=text)
            for _ in range(frames)], model


def test_one_tracker_follows_the_whole_run_with_the_phrases_every_frame():
    out, model = _run(3, ["person", "dog"])
    [tracker] = model.trackers
    assert tracker.calls == [["person", "dog"]] * 3
    assert [frame[0].tracker_id for frame in out] == [0, 1, 2]


def test_a_run_starts_from_nothing():
    model = FakeModel()
    _run(1, ["person"], model)
    _run(1, ["person"], model)
    assert len(model.trackers) == 2


def test_it_is_an_ordered_track_node():
    spec = get("sam3_track")
    assert spec.ordered and spec.category == "Track"
