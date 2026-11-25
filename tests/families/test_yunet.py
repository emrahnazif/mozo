"""YuNet: the geometry, the decode around the network, and the shape of the answer.

Bit-exactness against the authors' PyTorch is ``tools/verify/yunet.py``'s job -- it needs their
repository checked out -- and its results are recorded in ``yunet_deploy/PROVENANCE.md``. What
this file pins is what would still be green if the model had quietly changed:

* the geometry written out in ``config.py``, held against a strict load of each checkpoint;
* the input the network sees, and the decode facts that make ``threshold`` safe before NMS and
  the IoU-matrix NMS equal to upstream's loop;
* and the contract: an image in, faces with a box, a class, a score and five named, unscored
  landmarks out.

The tests that load a checkpoint are skipped when the weights are absent. Point them at a local
tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_yunet.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import COORD_STEP, published, require_weights
from mozo.vendors.yunet_deploy import postprocess
from mozo.vendors.yunet_deploy.config import KEYPOINTS, SPECS, get_spec
from mozo.vendors.yunet_deploy.network import YuNet
from mozo.vendors.yunet_deploy.predictor import load_state
from mozo.vendors.yunet_deploy.preprocess import to_batch
from mozo.weights import WeightsError, resolve

FAMILY = "yunet"
ALL = ["small", "base"]
LANDMARKS = ["right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner"]


@pytest.fixture(scope="module")
def predictor_for():
    """One predictor per variant and runtime, kept while the module runs."""
    from mozo.adapters.yunet import YuNetPredictor

    cache: dict[tuple, object] = {}

    def build(variant: str, runtime: str = "torch-fp32", device: str = "cpu"):
        require_weights(FAMILY, variant, "torch-fp32" if runtime == "torch-fp32" else "onnx-fp32")
        if (variant, runtime, device) not in cache:
            try:
                cache[variant, runtime, device] = YuNetPredictor(variant, device=device,
                                                                 runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"yunet/{variant} weights unavailable: {error}")
        return cache[variant, runtime, device]

    return build


# --- the geometry ------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ALL)
def test_the_published_checkpoint_loads_strictly(variant):
    """``config.py`` is checked here against the real tensors: a wrong width is a wrong shape."""
    require_weights(FAMILY, variant)
    try:
        path = resolve(FAMILY, variant, "torch-fp32")
    except WeightsError as error:
        pytest.skip(f"yunet/{variant} weights unavailable: {error}")
    with torch.device("meta"):
        model = YuNet(get_spec(variant))
    model.load_state_dict(load_state(path), strict=True, assign=True)


def test_the_two_geometries_differ_where_upstream_says():
    """``base`` (upstream's ``yunet_n``, "normal") is the larger: a 64-wide stage 1 and one
    shared convolution per head level. Swapping the names would load neither checkpoint."""
    with torch.device("meta"):
        small, base = YuNet(get_spec("small")), YuNet(get_spec("base"))
    count = lambda model: sum(p.numel() for p in model.parameters())  # noqa: E731
    assert count(base) > count(small)
    assert not hasattr(small.bbox_head, "multi_level_share_convs")
    assert len(base.bbox_head.multi_level_share_convs) == 3
    assert base.bbox_head.multi_level_kps[0].conv1.out_channels == 2 * KEYPOINTS


def test_an_unknown_variant_names_the_known_ones():
    with pytest.raises(ValueError, match="small"):
        get_spec("nano")


# --- the input ---------------------------------------------------------------------------------

def test_the_network_sees_the_image_at_its_own_size_bgr_and_zero_padded():
    image = np.zeros((37, 70, 3), np.uint8)
    image[..., 0] = 10  # red
    image[..., 2] = 30  # blue
    batch = to_batch(image, 32, torch.device("cpu"))
    assert batch.shape == (1, 3, 64, 96) and batch.dtype == torch.float32
    assert batch[0, 0, :37, :70].unique().tolist() == [30.0]  # blue first: BGR
    assert batch[0, 2, :37, :70].unique().tolist() == [10.0]
    assert batch[0, :, 37:].abs().sum() == 0 and batch[0, :, :, 70:].abs().sum() == 0


# --- the decode --------------------------------------------------------------------------------

def test_a_prior_is_its_cell_s_corner_not_its_centre():
    priors = postprocess.grid_priors([(2, 3), (1, 1)], (8, 16), torch.float32, torch.device("cpu"))
    assert priors[:4].tolist() == [[0, 0, 8, 8], [8, 0, 8, 8], [16, 0, 8, 8], [0, 8, 8, 8]]
    assert priors[6].tolist() == [0, 0, 16, 16]


def _boxes(count: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    corners = torch.rand(count, 2, generator=generator) * 300
    boxes = torch.cat([corners, corners + 20 + torch.rand(count, 2, generator=generator) * 60], 1)
    return boxes, torch.rand(count, generator=generator)


@pytest.mark.parametrize("count", [1, 7, 300, 1500])
def test_the_iou_matrix_keeps_exactly_what_upstream_s_loop_keeps(monkeypatch, count):
    """Two paths, one answer: the matrix below ``MATRIX_LIMIT``, upstream's loop above it. Each
    is forced here on the same boxes."""
    boxes, scores = _boxes(count)
    matrix = postprocess.nms(boxes, scores, 0.45) if count <= postprocess.MATRIX_LIMIT else None
    monkeypatch.setattr(postprocess, "MATRIX_LIMIT", 0)
    loop = postprocess.nms(boxes, scores, 0.45)
    monkeypatch.setattr(postprocess, "MATRIX_LIMIT", 10 ** 9)
    assert torch.equal(postprocess.nms(boxes, scores, 0.45), loop)
    if matrix is not None:
        assert torch.equal(matrix, loop)
    assert 0 < len(loop) <= count


def test_raising_the_threshold_only_removes_rows_from_the_bottom():
    """Why ``threshold`` can be applied before NMS: greedy NMS visits boxes best first, so a box
    below the threshold can never suppress one above it."""
    boxes, scores = _boxes(300, seed=1)

    def kept(threshold):
        cells = torch.nonzero(scores >= threshold)[:, 0]
        return cells[postprocess.nms(boxes[cells], scores[cells], 0.45)].tolist()

    low, high = kept(0.1), kept(0.5)
    assert high == [cell for cell in low if scores[cell] >= 0.5]
    assert len(low) > len(high) > 0


# --- what the family publishes -----------------------------------------------------------------

class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_ships_weights_names_and_terms(self, variant):
        keys = published(FAMILY, variant)
        if not keys:
            pytest.skip(f"{FAMILY}/{variant} is not in the manifest")
        assert {"torch-fp32", "onnx-fp32", "labels"} <= set(keys)

        from mozo.weights import license_of
        assert license_of(FAMILY, variant) == "BSD-3-Clause"

    def test_registry_agrees_with_the_adapter(self):
        from mozo.adapters.yunet import YuNetPredictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY[FAMILY]
        assert entry["adapter_class"] == YuNetPredictor.__name__
        assert entry["module"] == "mozo.adapters.yunet"
        assert entry["task_type"] == "object_detection"
        assert entry["variants"] == list(YuNetPredictor.VARIANTS) == ALL
        assert set(SPECS) == set(YuNetPredictor.VARIANTS)

    def test_the_graph_outputs_are_named_for_what_the_network_returns(self):
        """The adapter maps a graph's outputs by these names; the exporter writes them."""
        from mozo.adapters.yunet import GRAPH_OUTPUTS

        assert GRAPH_OUTPUTS[:3] == ("cls_8", "cls_16", "cls_32")
        assert GRAPH_OUTPUTS[-1] == "kps_32" and len(GRAPH_OUTPUTS) == 12


# --- the answer --------------------------------------------------------------------------------

class TestTheAnswer:
    @pytest.mark.parametrize("variant", ALL)
    def test_faces_with_named_unscored_landmarks_best_first(self, predictor_for, image, variant):
        rows = predictor_for(variant).predict(image).to_dict()
        assert len(rows) >= 4, "the fixture photograph shows five faces"
        assert {row["class_id"] for row in rows} == {0}
        assert {row["class_name"] for row in rows} == {"face"}
        scores = [row["confidence"] for row in rows]
        assert scores == sorted(scores, reverse=True) and min(scores) >= 0.5
        for row in rows:
            points = row["keypoints"]
            assert [p["name"] for p in points] == LANDMARKS
            assert [p["id"] for p in points] == list(range(KEYPOINTS))
            assert {p["confidence"] for p in points} == {None}
            x1, y1, x2, y2 = row["bbox"]
            # The subject's right eye is on the picture's left, and both eyes sit in the box.
            assert x1 <= points[0]["x"] < points[1]["x"] <= x2
            assert y1 <= points[0]["y"] <= y2

    def test_a_lower_threshold_adds_rows_below_and_moves_none_above(self, predictor_for, image):
        model = predictor_for("small")
        high = model.predict(image, threshold=0.5).to_dict()
        low = model.predict(image, threshold=0.02).to_dict()
        assert len(low) > len(high)
        assert low[:len(high)] == high

    def test_nobody_in_the_picture_is_an_answer_rather_than_an_error(self, predictor_for):
        assert len(predictor_for("small").predict(np.full((480, 640, 3), 127, np.uint8))) == 0

    def test_a_one_pixel_image_is_padded_and_answered(self, predictor_for):
        assert len(predictor_for("small").predict(np.zeros((1, 1, 3), np.uint8))) == 0

    def test_caller_labels_replace_the_published_names(self, predictor_for, image):
        """A plain list names the class, as for any detector; the landmarks then go unnamed."""
        row = predictor_for("small").predict(image, labels=["head"]).to_dict()[0]
        assert row["class_name"] == "head"
        assert {p["name"] for p in row["keypoints"]} == {None}


# --- the graphs --------------------------------------------------------------------------------

def _rows(found):
    return [(row["bbox"], row["confidence"], [(p["x"], p["y"]) for p in row["keypoints"]])
            for row in found.to_dict()]


def _close(want, got):
    """Same faces, boxes and landmarks within one step of PixelFlow's 0.01 px rounding."""
    assert len(want) == len(got) > 0
    for (box, score, points), (box2, score2, points2) in zip(want, got):
        assert max(abs(a - b) for a, b in zip(box, box2)) <= COORD_STEP + 1e-9
        assert abs(score - score2) < 1e-4
        assert max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                   for a, b in zip(points, points2)) <= COORD_STEP + 1e-9


class TestGraphs:
    """The published graph replaces the network only; the decode stays the vendor's."""

    @pytest.mark.parametrize("variant", ALL)
    def test_the_onnx_graph_finds_the_same_faces(self, predictor_for, image, variant):
        require_weights(FAMILY, variant, "onnx-fp32")
        pytest.importorskip("onnxruntime")
        _close(_rows(predictor_for(variant).predict(image)),
               _rows(predictor_for(variant, "onnx-fp32").predict(image)))

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

    def test_auto_picks_the_engine_on_cuda_when_tensorrt_is_usable(self, monkeypatch):
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

    def test_a_tensorrt_load_fetches_the_checkpoint_for_large_images(self):
        from mozo.weights import _planned

        require_weights(FAMILY, "small", "onnx-fp32")
        _, files = _planned(FAMILY, "small", "tensorrt-fp32", "cuda", None)
        assert {"onnx-fp32", "torch-fp32", "labels"} <= set(files)

    @staticmethod
    def _engine(predictor_for):
        from mozo.runtimes import _tensorrt_unusable

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        if (problem := _tensorrt_unusable()) is not None:
            pytest.skip(problem)
        return predictor_for("small", "tensorrt-fp32", "cuda")

    def test_the_fp32_engine_matches_torch_on_the_cpu(self, predictor_for, image):
        """Against the CPU, not torch on CUDA: torch leaves TF32 on for convolutions there, which
        moves boxes by up to 3.5 px on WIDER Face; the engine is built with TF32 cleared."""
        _close(_rows(predictor_for("small").predict(image)),
               _rows(self._engine(predictor_for).predict(image)))

    def test_an_image_larger_than_the_engine_takes_runs_on_torch(self, predictor_for, image,
                                                                 monkeypatch):
        import mozo.adapters.yunet as adapter

        engine = self._engine(predictor_for)
        monkeypatch.setattr(adapter, "_TENSORRT_SHAPES", ((1, 3, 32, 32), (1, 3, 64, 64),
                                                          (1, 3, 512, 512)))
        # Torch on the same GPU, with the same arithmetic as torch-fp32 there: identical rows.
        torch_rows = _rows(predictor_for("small", "torch-fp32", "cuda").predict(image))
        assert _rows(engine.predict(image)) == torch_rows
        assert engine._large is not None and engine._large.model is not None
