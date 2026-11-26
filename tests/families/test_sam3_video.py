"""SAM 3's video tracker, checked without weights.

The numbers are guarded by ``tools/verify/sam3_video.py``, which runs Meta's model beside mozo's.
Everything here runs in seconds and holds the pieces a refactor breaks quietly: which frames a
frame may read, what the small operations do to a mask, what the session refuses, and how the
adapter turns a frame's result into PixelFlow rows.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mozo.vendors.sam3_deploy import checkpoint as loader
from mozo.vendors.sam3_deploy.config import TRACKER, VIDEO
from mozo.vendors.sam3_deploy.tracker import group as group_module
from mozo.vendors.sam3_deploy.tracker.group import COND_KEPT, RECENT, ObjectGroup, non_overlapping
from mozo.vendors.sam3_deploy.tracker.memory import axial_rotary
from mozo.vendors.sam3_deploy.tracker.network import TrackerNet, closest_cond_frames, sine_1d
from mozo.vendors.sam3_deploy.tracker.ops import fill_holes, load_frame, mask_iou, nms
from mozo.vendors.sam3_deploy.tracker.session import Frame, VideoTracker, split_prompts


# --- the checkpoint ----------------------------------------------------------------------------

@pytest.mark.parametrize("meta,ours", [
    ("transformer.encoder.layers.2.self_attn.out_proj.weight",
     "memory_attention.layers.2.self_attn.o_proj.weight"),
    ("transformer.encoder.layers.0.cross_attn_image.k_proj.bias",
     "memory_attention.layers.0.cross_attn_image.k_proj.bias"),
    ("transformer.encoder.layers.3.norm2.weight", "memory_attention.layers.3.layer_norm2.weight"),
    ("transformer.encoder.norm.bias", "memory_attention.layer_norm.bias"),
    ("maskmem_backbone.mask_downsampler.encoder.6.weight",
     "memory_encoder.mask_downsampler.layers.2.conv.weight"),
    ("maskmem_backbone.mask_downsampler.encoder.10.bias",
     "memory_encoder.mask_downsampler.layers.3.layer_norm.bias"),
    ("maskmem_backbone.mask_downsampler.encoder.12.weight",
     "memory_encoder.mask_downsampler.final_conv.weight"),
    ("maskmem_backbone.fuser.layers.1.gamma", "memory_encoder.memory_fuser.1.scale"),
    ("maskmem_backbone.fuser.layers.0.pwconv2.weight",
     "memory_encoder.memory_fuser.0.pointwise_conv2.weight"),
    ("maskmem_backbone.out_proj.weight", "memory_encoder.projection.weight"),
    ("obj_ptr_proj.layers.2.weight", "obj_ptr_proj.layers.2.weight"),
    ("maskmem_tpos_enc", "maskmem_tpos_enc"),
])
def test_the_tracker_rename_table_only_renames(meta, ours):
    assert loader.rename(meta, loader.TRACKER_RULES) == ours


def test_the_click_head_and_the_tracker_split_the_tracker_section_between_them():
    """Every ``tracker.`` tensor goes to exactly one of the two, so a strict load of both leaves
    nothing over -- the video machinery is no longer filtered out."""
    keys = ["sam_prompt_encoder.not_a_point_embed.weight", "sam_mask_decoder.conv_s0.weight",
            "no_mem_embed", "no_mem_pos_enc", "transformer.encoder.norm.weight",
            "maskmem_backbone.out_proj.bias", "obj_ptr_tpos_proj.weight", "mask_downsample.bias"]
    checkpoint = {f"tracker.{k}": torch.zeros(1) for k in keys}
    click = loader.click_state_dict(checkpoint)
    tracker = loader.tracker_state_dict(checkpoint)
    assert len(click) + len(tracker) == len(keys)
    assert "no_mem_embed" in click and "no_mem_pos_enc" in tracker
    for key in ("tracker.transformer.", "tracker.maskmem_backbone.", "tracker.obj_ptr_proj."):
        assert not loader._skipped(f"{key}weight")


# --- what a frame reads ------------------------------------------------------------------------

def test_the_four_closest_conditioning_frames_are_read_in_upstreams_order():
    cond = {t: t for t in (0, 16, 32, 48, 64)}
    selected, unselected = closest_cond_frames(70, cond, 4)
    assert list(selected) == [64, 48, 32, 16], "closest before first, then by distance"
    assert list(unselected) == [0]
    few = {t: t for t in (32, 0)}
    assert list(closest_cond_frames(40, few, 4)[0]) == [32, 0], "four or fewer: as stored"


def _net() -> TrackerNet:
    class Click:
        image_size, grid = 1008, 72
    return TrackerNet(Click())


def _scored(**scores):
    return {int(t[1:]): {"eff_iou_score": torch.tensor(s)} for t, s in scores.items()}


def test_memory_selection_skips_frames_the_object_was_not_in():
    net = _net()
    outputs = _scored(f1=0.5, f2=0.0, f3=0.5, f4=0.0)
    assert net.select_frames(outputs, 5, 16) == [1, 3, 4], "the previous frame is always read"
    assert net.select_frames(outputs, 4, 16) == [1, 3]
    assert net.select_frames(outputs, 0, 16) == []


def test_memory_selection_never_reads_frame_zero_and_stops_at_its_reach():
    net = _net()
    outputs = _scored(**{f"f{t}": 1.0 for t in range(0, 40)})
    chosen = net.select_frames(outputs, 40, 16)
    assert chosen == list(range(25, 40)), "fifteen: the reach less the frame itself"
    assert 0 not in net.select_frames(outputs, 3, 16)


def test_a_frames_memory_score_averages_over_the_batch_as_upstream_does():
    scores = torch.tensor([[2.0], [-1.0]])
    iou = torch.tensor([0.5, 1.0])
    visible = torch.tensor([[2.0], [-1.0]]).sigmoid() * 2 - 1
    visible[1] = 0
    # (B, 1) times (B,) broadcasts to (B, B): the mean is mean(visibility) * mean(iou).
    assert torch.isclose(TrackerNet.memory_score(scores, iou), (visible * iou).mean())


def test_pointer_distances_are_encoded_sines_then_cosines():
    out = sine_1d(torch.tensor([0.0, 1.0]), 8)
    assert out.shape == (2, 8)
    assert torch.equal(out[0], torch.tensor([0.0] * 4 + [1.0] * 4))


def test_the_rotary_table_covers_the_grid_x_then_y():
    table = axial_rotary(256, 72, 10000.0)
    assert table.shape == (72 * 72, 128) and table.dtype == torch.complex64
    assert torch.allclose(table.abs(), torch.ones(()))


# --- the small operations ----------------------------------------------------------------------

def test_holes_are_filled_and_specks_removed():
    logits = torch.full((1, 1, 32, 32), -5.0)
    logits[0, 0, 4:20, 4:20] = 5.0      # an object
    logits[0, 0, 10, 10] = -5.0         # a one-pixel hole in it
    logits[0, 0, 28, 28] = 5.0          # a one-pixel speck elsewhere
    out = fill_holes(logits, 16)
    assert out[0, 0, 10, 10] == 0.1, "a hole becomes foreground, just"
    assert out[0, 0, 28, 28] == -0.1, "a speck becomes background, just"
    assert out[0, 0, 5, 5] == 5.0


def test_a_tiny_object_is_not_a_speck_of_itself():
    """Specks are limited to half the object's own area, so a small object survives."""
    logits = torch.full((1, 1, 16, 16), -5.0)
    logits[0, 0, 3:5, 3:5] = 5.0
    assert torch.equal(fill_holes(logits, 16), logits)


def test_nms_keeps_the_best_of_overlapping_detections():
    masks = torch.full((3, 8, 8), -1.0)
    masks[0, :4] = 1.0
    masks[1, :4] = 1.0
    masks[2, 4:] = 1.0
    keep = nms(torch.tensor([0.6, 0.9, 0.7]), masks, 0.5, 0.1)
    assert keep.tolist() == [False, True, True]


def test_nms_considers_only_detections_above_the_threshold():
    masks = torch.ones(2, 4, 4)
    assert nms(torch.tensor([0.4, 0.3]), masks, 0.5, 0.1).tolist() == [False, False]


def test_mask_iou_of_empty_masks_is_zero():
    empty = torch.zeros(1, 4, 4, dtype=torch.bool)
    assert mask_iou(empty, empty).item() == 0.0


def test_frames_pass_through_float16_as_meta_stores_them():
    frame = load_frame(np.full((10, 20, 3), 77, dtype=np.uint8))
    assert frame.shape == (1, 3, 1008, 1008) and frame.dtype == torch.float32
    assert torch.equal(frame, frame.half().float()), "every value is a float16 value"


def test_overlaps_are_resolved_to_the_strongest_logit():
    masks = torch.tensor([[[[2.0, -1.0]]], [[[1.0, 3.0]]]])
    out = non_overlapping(masks)
    assert out[0, 0, 0].tolist() == [2.0, -10.0]
    assert out[1, 0, 0].tolist() == [-10.0, 3.0]


# --- bounded memory ----------------------------------------------------------------------------

def test_a_batch_forgets_what_no_later_frame_can_read():
    group = ObjectGroup.__new__(ObjectGroup)
    group.net = _net()
    old = {"eff_iou_score": torch.tensor(1.0)}
    useless = {"eff_iou_score": torch.tensor(0.0)}
    group.store = {"cond_frame_outputs": {t: {} for t in range(0, 160, 16)},
                   "non_cond_frame_outputs": {t: (old if t < 50 else useless) for t in range(1, 200)}}
    group.tracked = set(range(200))
    group.consolidated = {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}
    group.prune(200)
    assert len(group.store["cond_frame_outputs"]) == COND_KEPT
    kept = sorted(group.store["non_cond_frame_outputs"])
    assert kept == sorted(set(range(50 - TRACKER.max_obj_ptrs, 50)) | set(range(200 - RECENT, 200)))


def test_the_recent_window_outlasts_probation():
    """Removing an object re-scores its batch; that only happens within probation."""
    assert group_module.RECENT > VIDEO.hotstart_delay


# --- prompts and refusals ----------------------------------------------------------------------

def test_each_prompt_is_one_object():
    prompts = split_prompts(points=[[[1, 2]], [[3, 4]]], labels=[[1], [1]], boxes=[[0, 0, 5, 5],
                                                                                   [1, 1, 6, 6]])
    assert len(prompts) == 2 and prompts[1]["box"].tolist() == [1, 1, 6, 6]
    with pytest.raises(ValueError, match="points and labels go together"):
        split_prompts(points=[[1, 2]])


class _Seg:
    click = object()
    tracker_net = object()
    device = "cpu"


@pytest.mark.parametrize("call,message", [
    ({}, "the first frame needs a prompt"),
    ({"texts": ["cat"], "prompts": [{"box": [0, 0, 1, 1]}]}, "phrases or clicks, not both"),
])
def test_a_first_frame_without_one_kind_of_prompt_is_refused(call, message):
    with pytest.raises(ValueError, match=message):
        VideoTracker(_Seg()).update(np.zeros((8, 8, 3), np.uint8), **call)


def _started(mode, texts=()):
    tracker = VideoTracker(_Seg())
    tracker.shape, tracker.mode, tracker.texts = (8, 8), mode, list(texts)
    return tracker


@pytest.mark.parametrize("tracker,call,message", [
    (lambda: _started("text", ["cat"]), {"texts": ["dog"]}, "phrases are fixed"),
    (lambda: _started("text", ["cat"]), {"prompts": [{"box": [0, 0, 1, 1]}]}, "first frame only"),
    (lambda: _started("clicks"), {"prompts": [{"box": [0, 0, 1, 1]}]}, "first frame only"),
    (lambda: _started("clicks"), {"texts": ["cat"]}, "phrases or clicks, not both"),
])
def test_later_frames_refuse_what_upstream_cannot_do(tracker, call, message):
    with pytest.raises(ValueError, match=message):
        tracker().update(np.zeros((8, 8, 3), np.uint8), **call)


def test_every_frame_must_be_the_size_of_the_first():
    with pytest.raises(ValueError, match="size of the first"):
        _started("text", ["cat"]).update(np.zeros((9, 8, 3), np.uint8))


def test_without_the_torch_encoder_there_is_no_tracker():
    class Graph:
        click = None
        tracker_net = None
    with pytest.raises(RuntimeError, match="torch image encoder"):
        VideoTracker(Graph())


# --- the adapter -------------------------------------------------------------------------------

class _FakeSession:
    """Returns a fixed frame and records what it was given."""

    def __init__(self, frame):
        self.frame, self.calls = frame, []

    def update(self, pixels, texts=None, prompts=None):
        self.calls.append((texts, prompts))
        return self.frame


def _frame(ids, prompts):
    masks = torch.zeros(len(ids), 6, 6, dtype=torch.bool)
    for row in range(len(ids)):
        masks[row, row, 1:4] = True
    return Frame(ids, prompts, masks, torch.linspace(0.6, 0.9, len(ids)))


def test_rows_carry_the_object_id_and_the_phrase():
    from mozo.adapters.sam3 import Sam3Tracker

    tracker = Sam3Tracker(_FakeSession(_frame([0, 2, 5], [0, 1, 0])))
    found = list(tracker.update(np.zeros((6, 6, 3), np.uint8), text=["person", "dog"]))
    assert [(r.tracker_id, r.class_id, r.class_name) for r in found] == [
        (0, 0, "person"), (2, 1, "dog"), (5, 0, "person")]
    assert found[1].bbox == [1.0, 1.0, 4.0, 2.0]


def test_repeating_the_phrases_is_the_same_as_passing_nothing():
    from mozo.adapters.sam3 import Sam3Tracker

    session = _FakeSession(_frame([], []))
    tracker = Sam3Tracker(session)
    tracker.update(np.zeros((6, 6, 3), np.uint8), text="cat")
    tracker.update(np.zeros((6, 6, 3), np.uint8), text="cat")
    tracker.update(np.zeros((6, 6, 3), np.uint8))
    assert [c[0] for c in session.calls] == [["cat"], None, None]


@pytest.mark.parametrize("call,message", [
    ({"text": ""}, "text was empty"),
    ({"text": []}, "no text was given"),
    ({"boxes": [[0, 0, 2, 2]] * 2, "name": ["a", "b", "c"]}, "3 names for 2 prompt"),
])
def test_the_adapter_refuses_before_touching_the_session(call, message):
    from mozo.adapters.sam3 import Sam3Tracker

    session = _FakeSession(_frame([], []))
    with pytest.raises(ValueError, match=message):
        Sam3Tracker(session).update(np.zeros((6, 6, 3), np.uint8), **call)
    assert session.calls == []


def test_clicked_objects_are_named_as_given():
    from mozo.adapters.sam3 import Sam3Tracker

    tracker = Sam3Tracker(_FakeSession(_frame([0, 1], [0, 1])))
    found = list(tracker.update(np.zeros((6, 6, 3), np.uint8), boxes=[[0, 0, 2, 2], [2, 2, 4, 4]],
                                name=["cup", "plate"]))
    assert [(r.tracker_id, r.class_id, r.class_name) for r in found] == [
        (0, 0, "cup"), (1, 1, "plate")]


def test_a_graph_runtime_has_no_tracker():
    from mozo.adapters.sam3 import Sam3Predictor

    model = Sam3Predictor.__new__(Sam3Predictor)
    model.runtime = "coreml-fp32"

    class Segmenter:
        tracker_net = None
    model._segmenter = Segmenter()
    with pytest.raises(ValueError, match="torch-fp32 only"):
        model.tracker()
