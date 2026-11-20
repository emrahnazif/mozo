"""The track node: the same object keeps its id from frame to frame, within one run and only it.

Run for real through its NodeSpec, the path the engine takes, on boxes moving across frames: a
tracker needs no weights, so this always runs.
"""

from __future__ import annotations

import pixelflow as pf
import pytest

from mozo.workflow import get
from mozo.workflow.node import Context, State


def _frame(*boxes) -> "pf.Detections":
    found = pf.Detections()
    for box in boxes:
        found.add_detection(pf.Detection(bbox=list(box), class_id=0, confidence=0.9))
    return found


def _run(frames, state=None, fps=10.0, first=0, **settings) -> list:
    """Each frame's tracker ids, the frames given in order to one run of the node (from frame `first`)."""
    run = Context(fps=fps).seal()
    state = State() if state is None else state
    return [[d.tracker_id for d in get("track").run(detections=f, run=run.at(first + i), state=state, **settings)]
            for i, f in enumerate(frames)]


def _moving(steps=6):
    """Two objects crossing the frame side by side, a few pixels a frame."""
    return [_frame((10 + 4 * i, 10, 40 + 4 * i, 40), (10 + 4 * i, 100, 40 + 4 * i, 130)) for i in range(steps)]


@pytest.mark.parametrize("tracker", ["bytetrack", "sort", "ocsort", "cbiou"])
def test_each_object_keeps_one_id(tracker):
    ids = _run(_moving(), tracker=tracker)
    confirmed = [frame for frame in ids if None not in frame]
    assert confirmed, f"{tracker} never confirmed the tracks"
    assert all(frame == confirmed[0] for frame in confirmed)  # the same two ids throughout
    assert len(set(confirmed[0])) == 2  # and two objects are two ids


def test_a_run_starts_from_nothing():
    """The tracker lives in the run's State: a second run (a new State) doesn't continue the first's ids."""
    assert _run(_moving())[-1] == _run(_moving())[-1]


def test_one_state_carries_the_tracks_across_calls():
    state = State()
    first = _run(_moving(4), state=state)
    later = _run([_frame((26, 10, 56, 40), (26, 100, 56, 130))], state=state, first=4)
    assert later[0] == first[-1]


def test_an_object_gone_longer_than_lost_after_comes_back_as_a_new_one():
    """Half a second unseen at 10 fps: past a 0.2 s lost_after, a new id; within the default 1 s, the same."""
    frames = _moving(4) + [_frame()] * 5 + [_frame((26, 10, 56, 40))] * 3
    kept = _run(frames)
    assert kept[-1] == [kept[3][0]]
    ids = _run(frames, lost_after=0.2)
    before = {i for frame in ids[:4] for i in frame if i is not None}
    after = {i for frame in ids[-3:] for i in frame if i is not None}
    assert after and not after & before
