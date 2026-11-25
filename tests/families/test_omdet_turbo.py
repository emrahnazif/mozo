"""Does OmDet-Turbo actually work.

These load a real checkpoint and run real inference, so they are slow and need the published
artifacts; they are skipped rather than failed when those are absent. Point them at a local
weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_omdet_turbo.py -q

What is protected: mozo must not change the vendor's answer; the name on a detection is the
caller's phrase and its id the phrase's index; the prompt contract is refused before anything
expensive; the text is encoded once per vocabulary without changing a bit; and the runtimes are
offered as measured. ``tools/verify/omdet_turbo.py`` is the bit-exact comparison against
``transformers``; it needs that package and does not run here.
"""

from __future__ import annotations

import pytest

from conftest import as_pixelflow_reports, published, require_weights
from mozo.registry import DERIVES, EXECUTES, MODEL_REGISTRY, NEEDS
from mozo.runtimes import select_runtime
from mozo.vendors.omdet_turbo_deploy import SPECS, VARIANTS
from mozo.vendors.omdet_turbo_deploy.predictor import task_sentence
from mozo.vendors.omdet_turbo_deploy.text.tokenizer import Tokenizer
from mozo.weights import artifacts

FAMILY = "omdet_turbo"

#: What the fixture photograph -- a desk scene -- reliably contains.
PROMPTS = ["person", "laptop", "cup"]


@pytest.fixture(scope="module")
def predictor():
    require_weights(FAMILY, "tiny")
    from mozo.adapters.omdet_turbo import OmDetTurboPredictor
    from mozo.weights import WeightsError

    try:
        return OmDetTurboPredictor("tiny", device="cpu", runtime="torch-fp32")
    except WeightsError as error:
        pytest.skip(str(error))


# --- the registry and the adapter must agree ---


def test_registry_agrees_with_the_adapter():
    from mozo.adapters.omdet_turbo import OmDetTurboPredictor

    assert set(MODEL_REGISTRY[FAMILY]["variants"]) == set(OmDetTurboPredictor.VARIANTS)
    assert list(VARIANTS) == ["tiny"]


def test_it_shares_the_open_vocabulary_task():
    assert MODEL_REGISTRY[FAMILY]["task_type"] == MODEL_REGISTRY["owlv2"]["task_type"]


def test_the_predict_signature_is_the_open_vocabulary_one():
    """Substitutable for OWLv2 and Grounding DINO: image, text, threshold, nothing else."""
    import inspect

    from mozo.adapters.omdet_turbo import OmDetTurboPredictor

    assert list(inspect.signature(OmDetTurboPredictor.predict).parameters) == [
        "self", "image", "text", "threshold"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_published_variant_ships_its_terms(variant):
    from mozo.weights import companions

    if not published(FAMILY, variant):
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert "LICENSE" in companions(FAMILY, variant)
    assert "NOTICE" in companions(FAMILY, variant)


def test_what_is_published_is_what_the_registry_derives_from():
    keys = published(FAMILY, "tiny")
    if not keys:
        pytest.skip(f"{FAMILY}/tiny publishes nothing")
    assert set(keys) == {"torch-fp32", "onnx-fp32", "onnx-fp16"}
    assert set(DERIVES[FAMILY].values()) <= set(keys)
    # The graph is the image path; the text tower comes from the checkpoint on every runtime.
    assert NEEDS[FAMILY] == {"onnx": ("torch-fp32",), "tensorrt": ("torch-fp32",)}


# --- runtimes, as measured ---


def _choose(device, runtime="auto"):
    return select_runtime(device, artifacts(FAMILY, "tiny"), runtime,
                          executes=EXECUTES[FAMILY], derives=DERIVES[FAMILY])


def test_auto_takes_the_exact_engine_on_cuda_when_tensorrt_is_usable(monkeypatch):
    import mozo.runtimes as runtimes

    require_weights(FAMILY, "tiny", "onnx-fp32")
    monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: True)
    assert _choose("cuda") == "tensorrt-fp32"
    assert _choose("cuda", "tensorrt-fp16") == "tensorrt-fp16"
    monkeypatch.setattr(runtimes, "_tensorrt_usable", lambda: False)
    assert _choose("cuda") == "torch-fp32"


def test_auto_stays_on_torch_on_the_cpu():
    require_weights(FAMILY, "tiny", "onnx-fp32")
    assert _choose("cpu") == "torch-fp32"


# --- the prompt contract ---


def test_it_refuses_no_prompt(predictor, image):
    with pytest.raises(ValueError, match="no text was given"):
        predictor.predict(image, [])


def test_it_refuses_a_blank_prompt(predictor, image):
    with pytest.raises(ValueError, match="text was empty"):
        predictor.predict(image, ["person", "  "])


def test_it_refuses_a_prompt_longer_than_the_text_tower(predictor, image):
    """Upstream truncates a class prompt silently; mozo refuses and names the length."""
    with pytest.raises(ValueError, match="76 tokens; at most 75 fit"):
        predictor.predict(image, [" ".join(["word"] * 76)])


def test_a_vocabulary_longer_than_the_task_sentence_still_runs(predictor, image):
    """The task sentence is truncated as upstream truncates it; every class is still scored."""
    prompts = [f"object number {n}" for n in range(40)] + ["person"]
    found = predictor.predict(image, prompts)
    assert {row["class_name"] for row in found.to_dict()} <= set(prompts)


# --- what comes back ---


def test_it_finds_the_scene(predictor, image):
    found = predictor.predict(image, PROMPTS)
    names = {row["class_name"] for row in found.to_dict()}
    assert names <= set(PROMPTS)
    assert "person" in names and "laptop" in names


def test_class_id_indexes_the_prompt_list(predictor, image):
    for row in predictor.predict(image, PROMPTS).to_dict():
        assert PROMPTS[row["class_id"]] == row["class_name"]


def test_detections_come_back_best_first(predictor, image):
    scores = [row["confidence"] for row in predictor.predict(image, PROMPTS).to_dict()]
    assert scores == sorted(scores, reverse=True)


def test_boxes_are_inside_the_source_image(predictor, image):
    height, width = image.shape[:2]
    for row in predictor.predict(image, PROMPTS).to_dict():
        x1, y1, x2, y2 = row["bbox"]
        assert 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height, row["bbox"]


def test_a_higher_threshold_keeps_fewer(predictor, image):
    loose = predictor.predict(image, PROMPTS, threshold=0.1)
    tight = predictor.predict(image, PROMPTS, threshold=0.6)
    assert len(tight) <= len(loose)


def test_the_text_cache_changes_nothing(predictor, image):
    """Encoded once per vocabulary, as one batch, so a repeat call is bit-identical -- and so is a
    call after another vocabulary has been encoded in between."""
    first = predictor.predict(image, PROMPTS).to_dict()
    predictor.predict(image, ["dog", "person"])
    assert predictor.predict(image, PROMPTS).to_dict() == first


# --- mozo must not change the vendor's answer ---


def test_mozo_agrees_with_the_vendor(predictor, image):
    from mozo.vendors.omdet_turbo_deploy import Detector
    from mozo.weights import resolve

    vendor = Detector(resolve(FAMILY, "tiny", "torch-fp32"), SPECS["tiny"])
    boxes, scores, labels = vendor(image, PROMPTS)
    got = predictor.predict(image, PROMPTS).to_dict()
    want_boxes, want_scores = as_pixelflow_reports(boxes, scores, labels)
    assert [list(b) for b in want_boxes] == [row["bbox"] for row in got]
    assert list(want_scores) == [row["confidence"] for row in got]
    assert [PROMPTS[i] for i in labels.tolist()] == [row["class_name"] for row in got]


# --- the pieces that can be checked without weights ---


def test_the_task_sentence_is_upstreams_default():
    assert task_sentence(["cat", "remote"]) == "Detect cat, remote."


def test_the_tokenizer_pads_with_the_end_marker_and_keeps_it_when_truncating():
    tokenizer = Tokenizer()
    ids, mask = tokenizer(["cat"], truncate=False)
    assert ids[0, :4].tolist() == [49406, 2368, 49407, 49407]
    assert mask[0, :4].tolist() == [1, 1, 1, 0]
    ids, mask = tokenizer([" ".join(["word"] * 100)], truncate=True)
    assert ids[0, -1].item() == 49407 and int(mask.sum()) == 77


def test_the_tokenizer_lowercases_as_the_reference_does_not_as_python_does():
    """Python's ``str.lower`` applies the Greek final-sigma rule; the reference's Rust does not."""
    tokenizer = Tokenizer()
    assert tokenizer.clean("ΑΣ") == "ασ" != "ΑΣ".lower()


def test_the_geometry_is_swin_tiny_at_640():
    spec = SPECS["tiny"]
    assert spec.swin.widths == (96, 192, 384, 768)
    assert spec.encoder_in_channels == spec.swin.widths[1:]
    assert spec.swin.image_size == 640 and spec.queries == 900


def test_the_shift_mask_is_timms_on_the_padded_grid():
    """160 is not a multiple of 7: timm pads to 161 and masks there."""
    from mozo.vendors.omdet_turbo_deploy.vision.swin import shift_mask

    mask = shift_mask(160, 7, 3)
    assert mask.shape == (23 * 23, 49, 49)
    assert set(mask.unique().tolist()) == {-100.0, 0.0}


# --- on CUDA ---


def _on_cuda(runtime):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if runtime.startswith("tensorrt"):
        from mozo.runtimes import _tensorrt_unusable

        if (problem := _tensorrt_unusable()) is not None:
            pytest.skip(problem)
    require_weights(FAMILY, "tiny", DERIVES[FAMILY].get(runtime, runtime))
    from mozo.adapters.omdet_turbo import OmDetTurboPredictor

    return OmDetTurboPredictor("tiny", device="cuda", runtime=runtime)


def test_the_fp32_engine_finds_what_torch_finds(image):
    want = _on_cuda("torch-fp32").predict(image, PROMPTS)
    got = _on_cuda("tensorrt-fp32").predict(image, PROMPTS)
    assert len(want) == len(got) > 0
    assert [d.class_name for d in want] == [d.class_name for d in got]
    worst = max(max(abs(a - b) for a, b in zip(x.bbox, y.bbox)) for x, y in zip(want, got))
    assert worst <= 0.5


def test_the_fp16_engine_keeps_the_confident_detections(image):
    import numpy as np

    want = _on_cuda("torch-fp32").predict(image, PROMPTS, threshold=0.5)
    got = _on_cuda("tensorrt-fp16").predict(image, PROMPTS, threshold=0.3)
    assert len(want) > 0
    for found in want:
        nearest = min(got, key=lambda g: float(np.abs(np.subtract(g.bbox, found.bbox)).max()))
        assert nearest.class_name == found.class_name
        assert float(np.abs(np.subtract(nearest.bbox, found.bbox)).max()) < 5.0
