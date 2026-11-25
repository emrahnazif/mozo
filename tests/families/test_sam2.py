"""SAM 2: what is true of this family and not of the others.

The contract every promptable family shares -- ranking, naming, batching, the server's prompt
parsing -- is in ``test_promptable.py``, checked against all of them. What is left here is SAM 2's
own: four variants where the others have one, and several published runtimes where the others
publish only torch.
"""

from __future__ import annotations

import pytest

from mozo.registry import get_model_info
from mozo.vendors.sam2_deploy.config import SPECS

FAMILY = "sam2"


def test_registry_agrees_with_the_adapter():
    """The variant list is written twice -- here and in the adapter -- so that answering "what
    exists" needs no torch import. This is what holds the two copies in step."""
    from mozo.adapters.sam2 import Sam2Predictor

    entry = get_model_info(FAMILY)
    assert entry["adapter_class"] == "Sam2Predictor"
    assert entry["module"] == "mozo.adapters.sam2"
    assert entry["task_type"] == "promptable_segmentation"
    assert set(entry["variants"]) == set(Sam2Predictor.VARIANTS)


def test_the_adapter_publishes_every_variant_the_vendor_can_build():
    """The vendor knows four geometries; a fifth published with no geometry, or a geometry with
    nothing published, is a mismatch nothing else would report."""
    from mozo.adapters.sam2 import Sam2Predictor

    assert set(Sam2Predictor.VARIANTS) == set(SPECS)


def test_a_graph_runtime_is_refused_rather_than_quietly_replaced():
    """SAM 2 publishes ONNX and CoreML, and no promptable adapter can execute either yet -- such
    a model exports as several graphs. Answering the request with torch would look like it had
    been honoured, so it raises."""
    from mozo.adapters.sam2 import Sam2Predictor
    from mozo.runtimes import RuntimeError_

    with pytest.raises(RuntimeError_, match="can only execute torch"):
        Sam2Predictor("tiny", device="cpu", runtime="onnx-fp32")


def test_auto_never_chooses_a_runtime_the_adapter_would_refuse():
    """The point of declaring the capability to ``select_runtime`` rather than checking it
    afterwards. ``_PREFERENCE`` leads with CoreML on Apple silicon and SAM 2 publishes
    ``coreml-fp16``; the day someone measures it and adds it to that table, ``auto`` on a Mac
    would otherwise start choosing an artifact this adapter refuses -- breaking the *default* path."""
    from mozo.adapters.sam2 import Sam2Predictor
    from mozo.runtimes import select_runtime
    from mozo.weights import artifacts

    for variant in Sam2Predictor.VARIANTS:
        published = artifacts(FAMILY, variant)
        for device in ("cpu", "cuda", "mps"):
            chosen = select_runtime(device, published, "auto", executes=Sam2Predictor.EXECUTES)
            assert chosen.startswith("torch"), f"{variant} on {device} chose {chosen}"


# --- tracking a video ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from mozo.vendors.sam2_deploy.video import split_prompts  # noqa: E402


class TestPromptsStartObjects:
    """``update`` reads prompts by ``predict``'s rules; each prompt is one object."""

    def test_one_box_is_one_object(self):
        [one] = split_prompts(boxes=[1, 2, 3, 4])
        assert one["box"].tolist() == [1, 2, 3, 4] and one["points"] is None

    def test_a_batch_of_boxes_is_one_object_each(self):
        assert len(split_prompts(boxes=[[1, 2, 3, 4], [5, 6, 7, 8]])) == 2

    def test_clicks_alone_are_one_object_and_a_batch_of_click_sets_one_each(self):
        assert len(split_prompts(points=[[1, 2], [3, 4]], labels=[1, 0])) == 1
        assert len(split_prompts(points=[[[1, 2]], [[3, 4]]], labels=[[1], [1]])) == 2

    def test_a_box_with_clicks_is_one_object(self):
        [one] = split_prompts(points=[[1, 2]], labels=[0], boxes=[0, 0, 9, 9])
        assert one["box"] is not None and one["labels"].tolist() == [0]

    def test_nothing_is_no_objects(self):
        assert split_prompts() == []

    def test_points_without_labels_are_refused(self):
        with pytest.raises(ValueError, match="points and labels"):
            split_prompts(points=[[1, 2]])

    def test_box_and_point_set_counts_must_agree(self):
        with pytest.raises(ValueError, match="one point set per box"):
            split_prompts(points=[[1, 2]], labels=[1], boxes=[[0, 0, 5, 5], [1, 1, 6, 6]])


@pytest.fixture(scope="module")
def tiny():
    from mozo.adapters.sam2 import Sam2Predictor
    from mozo.weights import WeightsError

    try:
        return Sam2Predictor("tiny", device="cpu")
    except (WeightsError, FileNotFoundError) as error:
        pytest.skip(f"sam2/tiny weights unavailable: {error}")


@pytest.fixture(scope="module")
def frames(image):
    """A few frames of a scene moving right: the fixture photograph, shifted a little each time."""
    photo = image[::4, ::4]
    return [np.ascontiguousarray(np.roll(photo, 6 * i, axis=1)) for i in range(4)]


def _box(frame):
    height, width = frame.shape[:2]
    return [0.55 * width, 0.42 * height, 0.72 * width, 0.78 * height]


class TestTracker:
    def test_the_whole_checkpoint_loads_strictly(self, tiny):
        """The tracker's weights are built and loaded, not filtered out: nothing is left over."""
        network = tiny._segmenter.network
        assert sum(p.numel() for p in network.memory_attention.parameters()) > 0
        assert network.memory_encoder.out_proj.weight.shape[0] == 64

    def test_an_object_keeps_its_id_and_name_frame_to_frame(self, tiny, frames):
        tracker = tiny.tracker()
        first = tracker.update(frames[0], boxes=_box(frames[0]), name="person")
        later = [tracker.update(frame) for frame in frames[1:]]
        for found in [first, *later]:
            [row] = list(found)
            assert (row.tracker_id, row.class_id, row.class_name) == (0, 0, "person")
            assert 0.5 < row.confidence <= 1.0
            assert row.masks[0].shape == frames[0].shape[:2] and row.masks[0].any()

    def test_an_object_started_later_gets_the_next_id(self, tiny, frames):
        tracker = tiny.tracker()
        tracker.update(frames[0], boxes=_box(frames[0]))
        height, width = frames[1].shape[:2]
        found = tracker.update(frames[1], points=[[0.3 * width, 0.5 * height]], labels=[1])
        assert sorted(row.tracker_id for row in found) == [0, 1]

    def test_the_first_frame_needs_a_prompt(self, tiny, frames):
        with pytest.raises(ValueError, match="first frame needs a prompt"):
            tiny.tracker().update(frames[0])

    def test_every_frame_must_be_the_size_of_the_first(self, tiny, frames):
        tracker = tiny.tracker()
        tracker.update(frames[0], boxes=_box(frames[0]))
        with pytest.raises(ValueError, match="size of the first"):
            tracker.update(frames[1][:-8])

    def test_a_name_per_prompt_or_none(self, tiny, frames):
        with pytest.raises(ValueError, match="names for"):
            tiny.tracker().update(frames[0], boxes=[_box(frames[0])] * 2, name=["a", "b", "c"])

    def test_two_trackers_on_one_model_do_not_share_objects(self, tiny, frames):
        a, b = tiny.tracker(), tiny.tracker()
        a.update(frames[0], boxes=[_box(frames[0])] * 2)
        found = b.update(frames[0], boxes=_box(frames[0]))
        assert [row.tracker_id for row in found] == [0]

    def test_memory_stays_bounded_however_long_the_stream(self, tiny, frames):
        """Only what a later frame can read is kept: the prompted frame and the last 15."""
        session = tiny._segmenter.tracker()
        small = [np.ascontiguousarray(f[::2, ::2]) for f in frames]
        session.update(small[0], split_prompts(boxes=_box(small[0])))
        for i in range(1, 18):
            session.update(small[i % len(small)])
        [output] = session.outputs
        assert list(output["cond_frame_outputs"]) == [0]
        assert sorted(output["non_cond_frame_outputs"]) == list(range(3, 18))
