"""Following objects from frame to frame: which detection in this frame is which one in the last.

Every node here is one :mod:`pixelflow.tracker` tracker, which owns the tracking. What this module
contributes is the declaration, and the one thing a workflow node can't get from a signature: the
tracker itself, which carries its tracks from frame to frame, kept in the node's
:class:`~mozo.workflow.node.State` for the run and dropped after. So a workflow run twice tracks
from nothing the second time, and two cameras never share a track.

The tracker is told the run's rate and each frame's time, both from the run's
:class:`~mozo.workflow.node.Context`: a strided read of a video is a slower sequence, and a tracker
that assumed the file's own rate would think everything moved several times faster. A run with no
rate (an image, a camera) tracks a frame at a time.
"""

from __future__ import annotations

from typing import Literal

import pixelflow as pf

from ..node import Context, Detections, State
from ..registry import node

#: The trackers on offer, by what the editor calls them. BoT-SORT is left out until a node can take
#: the frame it needs for its camera-motion compensation; the others look at the boxes alone.
TRACKERS = {"bytetrack": pf.tracker.ByteTrack, "sort": pf.tracker.SORT, "ocsort": pf.tracker.OCSORT,
            "cbiou": pf.tracker.CBIoU}
Tracker = Literal[tuple(TRACKERS)]


@node(category="Track", ordered=True)
def track(detections: Detections, run: Context, state: State, tracker: Tracker = "bytetrack",
          lost_after: float = 1.0) -> Detections:
    """Give each object the same ``tracker_id`` from frame to frame, so it's counted once however
    many frames it's in (a car that passed, not a car per frame).

    Args:
        tracker: ByteTrack, the default, keeps objects through a frame or two of low confidence;
            SORT is the simplest and fastest; OC-SORT recovers objects after an occlusion; C-BIoU
            follows fast or irregular motion.
        lost_after: Seconds an object may go unseen before its track ends; one seen again later gets
            a new id. Seconds rather than frames, so it means the same at any rate a video is read
            at (pixelflow's ``lost_track_buffer`` is this time, counted in frames at 30 fps).

    Returns:
        The detections, in the same order, each with the ``tracker_id`` of its track, or ``None``
        while a new track isn't confirmed yet.
    """
    if "tracker" not in state:
        state["tracker"] = TRACKERS[tracker](lost_track_buffer=round(lost_after * 30), frame_rate=run.fps)
    return state["tracker"].update(detections, timestamp=run.time)
