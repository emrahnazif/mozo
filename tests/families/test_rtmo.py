"""RTMO: the geometry, the decode around the network, and the shape of the answer.

Bit-exactness against MMPose is ``tools/verify/rtmo.py``'s job -- it needs MMPose installed, which
mozo never does -- and its results are recorded in ``rtmo_deploy/PROVENANCE.md``. What this file
pins is what would still be green if the model had quietly changed:

* the geometry written out in ``config.py``, held against the network it builds and against a
  strict load of each published checkpoint;
* the decode facts that make ``threshold`` safe to apply before NMS, and the letterbox centring;
* and the contract this family shares with ``rfdetr/keypoint-preview``: an image in, people with
  boxes, a class, a score and 17 named joints out.

The tests that load a checkpoint are skipped when the weights are absent. Point them at a local
tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_rtmo.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import COORD_STEP, published, require_weights
from mozo.vendors.rtmo_deploy.config import KEYPOINTS, SPECS, get_spec
from mozo.vendors.rtmo_deploy.network import RTMO, TRAINING_ONLY
from mozo.vendors.rtmo_deploy.postprocess import grid_priors, nms, select, to_frame
from mozo.vendors.rtmo_deploy.predictor import load_state
from mozo.vendors.rtmo_deploy.preprocess import letterbox
from mozo.weights import WeightsError, resolve

FAMILY = "rtmo"
ALL = ["tiny", "small", "medium", "large"]

JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
          "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
          "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


@pytest.fixture(scope="module")
def predictor_for():
    """One predictor per variant, kept while the module runs: loading is the slow part."""
    from mozo.adapters.rtmo import RTMOPredictor

    cache: dict[tuple, object] = {}

    def build(variant: str, runtime: str = "torch-fp32"):
        require_weights(FAMILY, variant, runtime)
        if (variant, runtime) not in cache:
            try:
                cache[variant, runtime] = RTMOPredictor(variant, device="cpu", runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"rtmo/{variant} weights unavailable: {error}")
        return cache[variant, runtime]

    return build


# --- the geometry ------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL)
def test_the_spec_builds_the_network_it_describes(variant):
    """The head splits the projector's output in half, one half per branch, so the projector has to
    be twice the head's input; the pose vector the head emits is what the DCC reads."""
    spec = get_spec(variant)
    with torch.device("meta"):
        model = RTMO(variant)
    projector = model.neck.projector.convs[0].conv.out_channels
    head_in = model.head.head_module.conv_cls[0][0].conv.in_channels
    assert projector == spec.pose_channels == 2 * head_in
    assert model.head.head_module.out_pose[0].out_channels == spec.pose_channels
    assert model.head.dcc.pose_to_kpts[0].in_features == spec.pose_channels
    assert model.head.head_module.out_kpt_vis[0].out_channels == KEYPOINTS
    assert spec.input_size % 32 == 0, "the stride-32 grid has to tile the input"


@pytest.mark.parametrize("variant", ALL)
def test_the_published_checkpoint_loads_strictly(variant):
    """``config.py`` is checked here against the real tensors: a wrong width is a wrong shape."""
    require_weights(FAMILY, variant)
    try:
        path = resolve(FAMILY, variant, "torch-fp32")
    except WeightsError as error:
        pytest.skip(f"rtmo/{variant} weights unavailable: {error}")
    with torch.device("meta"):
        model = RTMO(variant)
    model.load_state_dict(load_state(path), strict=True, assign=True)


def test_a_checkpoint_without_the_training_tensors_is_refused(tmp_path):
    """The dropped prefixes are named, and each must be there: a differently laid out checkpoint
    fails at the load instead of loading whatever happens to match."""
    path = tmp_path / "state.pth"
    torch.save({f"{TRAINING_ONLY[0]}x": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="no 'head.dcc.sigma_fc.' tensors"):
        load_state(path)


def test_an_unknown_variant_names_the_known_ones():
    with pytest.raises(ValueError, match="tiny"):
        get_spec("huge")


# --- the decode --------------------------------------------------------------------------------

def test_raising_the_threshold_only_removes_rows_from_the_bottom():
    """Why ``threshold`` can be applied before NMS: greedy NMS visits boxes best first, so a box
    below the threshold can never suppress one above it. Thresholding first and NMS-ing the rest
    keeps exactly the rows NMS-at-a-lower-threshold would have kept above it."""
    generator = torch.Generator().manual_seed(0)
    corners = torch.rand(300, 2, generator=generator) * 500
    boxes = torch.cat([corners, corners + 40 + torch.rand(300, 2, generator=generator) * 80], 1)
    scores = torch.rand(300, generator=generator)

    def kept(threshold):
        top, cells = select(scores, threshold)
        return cells[nms(boxes[cells], top, 0.65)].tolist()

    low, high = kept(0.1), kept(0.5)
    assert high == [cell for cell in low if scores[cell] > 0.5]
    assert len(low) > len(high) > 0, "the case does not exercise suppression across the threshold"


def test_the_grid_cells_sit_at_pixel_centres():
    """``centralize_points``: offset 0 plus ``(stride - 1) / 2``, so 7.5 at stride 16, not 8."""
    priors, strides = grid_priors([(2, 3), (1, 2)], (16, 32), torch.float32, torch.device("cpu"))
    assert priors[:3].tolist() == [[7.5, 7.5], [23.5, 7.5], [39.5, 7.5]]
    assert priors[6].tolist() == [15.5, 15.5]
    assert strides.tolist() == [16] * 6 + [32] * 2


def test_the_letterbox_is_centred_with_grey_bars():
    """MMPose's test-time letterbox centres the picture -- YOLOX's top-left letterbox is a
    different input, and nothing would fail if it crept in."""
    frame = np.zeros((100, 400, 3), np.uint8)
    canvas, center, scale = letterbox(frame, 640)
    rows = np.where((canvas != 114).any(axis=(1, 2)))[0]
    top, bottom = rows[0], 639 - rows[-1]
    assert canvas.shape == (640, 640, 3)
    assert abs(int(top) - int(bottom)) <= 1 and top > 200
    assert center.tolist() == [200.0, 50.0] and scale.tolist() == [400.0, 400.0]


def test_the_map_back_to_the_frame_is_float64():
    """Upstream divides a float32 array by a tuple of ints, which promotes; the answer is float64
    because of it, and float32 would move coordinates in the last place."""
    points = np.array([[320.0, 320.0]], np.float32)
    mapped = to_frame(points, (640, 640), np.array([200, 50], np.float32),
                      np.array([400, 400], np.float32))
    assert mapped.dtype == np.float64 and mapped.tolist() == [[200.0, 50.0]]


# --- what the family publishes -----------------------------------------------------------------

class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_ships_weights_names_and_terms(self, variant):
        keys = published(FAMILY, variant)
        if not keys:
            pytest.skip(f"{FAMILY}/{variant} is not in the manifest")
        assert {"torch-fp32", "labels"} <= set(keys)
        assert not [key for key in keys if key.startswith("coreml")]

    def test_registry_agrees_with_the_adapter(self):
        from mozo.adapters.rtmo import RTMOPredictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY[FAMILY]
        assert entry["adapter_class"] == RTMOPredictor.__name__
        assert entry["module"] == "mozo.adapters.rtmo"
        assert entry["task_type"] == "object_detection"
        assert entry["variants"] == list(RTMOPredictor.VARIANTS) == ALL
        assert set(SPECS) == set(RTMOPredictor.VARIANTS)

    def test_it_is_not_a_boxed_task(self):
        """RTMO finds the people itself. Registered as a boxed task, ``/predict`` would refuse it
        without ``?box=`` -- which is why it is ``object_detection``, as RF-DETR's keypoint
        variant is."""
        from mozo.registry import BOXED, MODEL_REGISTRY

        assert MODEL_REGISTRY[FAMILY]["task_type"] not in BOXED


# --- the answer --------------------------------------------------------------------------------

class TestTheAnswer:
    @pytest.mark.parametrize("variant", ALL)
    def test_people_with_named_joints_best_first(self, predictor_for, image, variant):
        found = predictor_for(variant).predict(image)
        rows = found.to_dict()
        assert rows, "nobody found in a photograph of several people"
        assert {row["class_id"] for row in rows} == {0}
        assert {row["class_name"] for row in rows} == {"person"}
        scores = [row["confidence"] for row in rows]
        assert scores == sorted(scores, reverse=True) and min(scores) > 0.5
        joints = rows[0]["keypoints"]
        assert [joint["name"] for joint in joints] == JOINTS
        assert [joint["id"] for joint in joints] == list(range(KEYPOINTS))
        height, width = image.shape[:2]
        for row in rows:
            x1, y1, x2, y2 = row["bbox"]
            assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height

    def test_a_lower_threshold_adds_rows_below_and_moves_none_above(self, predictor_for, image):
        """Boxes, scores and visibilities above the higher threshold are identical. The joints are
        not quite: the DCC runs once over all survivors, and a matrix product over 12 rows does
        not round like one over 4. Upstream does the same (the gate matches it at each threshold);
        the drift is float noise, at most one step of PixelFlow's 0.01 px rounding."""
        model = predictor_for("small")
        high = model.predict(image, threshold=0.5).to_dict()
        low = model.predict(image, threshold=0.1).to_dict()
        assert len(low) > len(high)
        for a, b in zip(high, low):
            assert a["bbox"] == b["bbox"] and a["confidence"] == b["confidence"]
            assert [j["confidence"] for j in a["keypoints"]] == [j["confidence"] for j in b["keypoints"]]
            moved = max(max(abs(j["x"] - k["x"]), abs(j["y"] - k["y"]))
                        for j, k in zip(a["keypoints"], b["keypoints"]))
            assert moved <= COORD_STEP + 1e-9

    def test_nobody_in_the_frame_is_an_answer_rather_than_an_error(self, predictor_for):
        found = predictor_for("small").predict(np.full((480, 640, 3), 127, np.uint8))
        assert len(found) == 0

    def test_caller_labels_replace_the_published_names(self, predictor_for, image):
        """A plain list names the class, as for any detector; the joints then go unnamed."""
        row = predictor_for("small").predict(image, labels=["human"]).to_dict()[0]
        assert row["class_name"] == "human"


# --- the graphs --------------------------------------------------------------------------------

def _rows(found):
    return [(row["bbox"], row["confidence"],
             [(j["x"], j["y"], j["confidence"]) for j in row["keypoints"]]) for row in found.to_dict()]


class TestGraphs:
    """The published graph replaces the dense network only; the decode and the DCC stay the
    vendor's, so a disagreement here is the graph."""

    @pytest.mark.parametrize("variant", ALL)
    def test_the_onnx_graph_finds_the_same_people(self, predictor_for, image, variant):
        require_weights(FAMILY, variant, "onnx-fp32")
        pytest.importorskip("onnxruntime")
        want = _rows(predictor_for(variant).predict(image))
        got = _rows(predictor_for(variant, "onnx-fp32").predict(image))
        assert len(want) == len(got) > 0
        for (box, score, joints), (box2, score2, joints2) in zip(want, got):
            # One step of PixelFlow's 0.01 px rounding at most: measured at 0.0003 px before it.
            assert max(abs(a - b) for a, b in zip(box, box2)) <= COORD_STEP + 1e-9
            assert abs(score - score2) < 1e-4
            seen = [(a, b) for a, b in zip(joints, joints2) if a[2] > 0.5]
            assert max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in seen) <= COORD_STEP + 1e-9

    def test_auto_stays_on_torch_on_the_cpu(self):
        from mozo.runtimes import select_runtime
        from mozo.weights import artifacts

        require_weights(FAMILY, "small", "onnx-fp32")
        assert select_runtime("cpu", artifacts(FAMILY, "small")) == "torch-fp32"


class TestTensorRT:
    @staticmethod
    def _choose(device, variant, runtime="auto"):
        from mozo.registry import DERIVES
        from mozo.runtimes import select_runtime
        from mozo.weights import artifacts

        return select_runtime(device, artifacts(FAMILY, variant), runtime,
                              derives=DERIVES[FAMILY])

    def test_auto_picks_the_exact_engine_on_cuda_when_tensorrt_is_usable(self, monkeypatch):
        import mozo.runtimes as runtimes

        monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: True)
        for variant in ALL:
            require_weights(FAMILY, variant, "onnx-fp32")
            assert self._choose("cuda", variant) == "tensorrt-fp32", variant

    def test_without_tensorrt_cuda_stays_on_torch(self, monkeypatch):
        import mozo.runtimes as runtimes

        monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: False)
        require_weights(FAMILY, "small", "onnx-fp32")
        assert self._choose("cuda", "small") == "torch-fp32"

    def test_the_fp32_engine_matches_ieee_torch(self, image):
        """Against torch with TF32 off: the engine is built with TF32 cleared."""
        from mozo.runtimes import _tensorrt_unusable

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        if (problem := _tensorrt_unusable()) is not None:
            pytest.skip(problem)
        require_weights(FAMILY, "small", "onnx-fp32")
        from mozo.adapters.rtmo import RTMOPredictor

        previous = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            want = _rows(RTMOPredictor("small", device="cuda", runtime="torch-fp32").predict(image))
        finally:
            torch.backends.cudnn.allow_tf32 = previous
        got = _rows(RTMOPredictor("small", device="cuda", runtime="tensorrt-fp32").predict(image))
        assert len(want) == len(got) > 0
        for (box, _, _), (box2, _, _) in zip(want, got):
            assert max(abs(a - b) for a, b in zip(box, box2)) <= COORD_STEP + 1e-9
