"""Does MM-Grounding-DINO (LLMDet) actually work.

Unlike the module tests, these load a real checkpoint and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_mm_grounding_dino.py -q

The promises protected here are ``grounding_dino``'s, because the two families answer one task
through one endpoint: mozo must not change the vendor's answer, the name on a detection is the
caller's phrase, and the prompt contract is refused before anything expensive happens. Plus the
one this family adds: each decoder layer runs its *own* box head, which ``transformers`` 5 gets
wrong.

``tools/verify/mm_grounding_dino.py`` is the bit-exact comparison against ``transformers``; it
needs a 4.x install and does not run here.
"""

from __future__ import annotations

import pytest

from conftest import as_pixelflow_reports, published, require_weights
from mozo.registry import MODEL_REGISTRY
from mozo.vendors.mm_grounding_dino_deploy import SEPARATORS, SPECS, VARIANTS, caption_for
from mozo.vendors.mm_grounding_dino_deploy.text.tokenizer import Tokenizer

FAMILY = "mm_grounding_dino"
#: Read off the vendor, not retyped, so a fourth variant gets its licence and artifact coverage.
ALL = VARIANTS
SMALLEST = "llmdet-tiny"

#: What the fixture photograph -- a desk scene -- reliably contains.
PROMPTS = ["person", "laptop", "cup"]


@pytest.fixture(scope="module")
def predictor():
    """The smallest variant, built once for the file."""
    require_weights(FAMILY, SMALLEST)
    from mozo.adapters.mm_grounding_dino import MMGroundingDinoPredictor
    from mozo.weights import WeightsError

    try:
        return MMGroundingDinoPredictor(SMALLEST, device="cpu")
    except WeightsError as error:
        pytest.skip(str(error))


# --- the registry and the adapter must agree ---


def test_registry_agrees_with_the_adapter():
    from mozo.adapters.mm_grounding_dino import MMGroundingDinoPredictor

    assert set(MODEL_REGISTRY[FAMILY]["variants"]) == set(MMGroundingDinoPredictor.VARIANTS)
    assert MMGroundingDinoPredictor.VARIANTS[0] == SMALLEST


def test_the_registry_calls_this_open_vocabulary_detection():
    """It shares a task, an endpoint and a response shape with OWLv2 and Grounding DINO."""
    assert MODEL_REGISTRY[FAMILY]["task_type"] == "open_vocabulary_detection"
    assert MODEL_REGISTRY[FAMILY]["task_type"] == MODEL_REGISTRY["grounding_dino"]["task_type"]


def test_predict_has_grounding_dinos_signature():
    """Substitutable means the same call: a caller swapping families changes one name."""
    import inspect

    from mozo.adapters.grounding_dino import GroundingDinoPredictor
    from mozo.adapters.mm_grounding_dino import MMGroundingDinoPredictor

    ours = inspect.signature(MMGroundingDinoPredictor.predict)
    theirs = inspect.signature(GroundingDinoPredictor.predict)
    assert list(ours.parameters) == list(theirs.parameters)
    assert ours.parameters["threshold"].default is None
    init = inspect.signature(MMGroundingDinoPredictor.__init__).parameters
    assert set(init) == set(inspect.signature(GroundingDinoPredictor.__init__).parameters)


@pytest.mark.parametrize("variant", ALL)
def test_every_published_variant_ships_its_terms(variant):
    from mozo.weights import companions

    if not published(FAMILY, variant):
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert "LICENSE" in companions(FAMILY, variant)
    assert "NOTICE" in companions(FAMILY, variant)


@pytest.mark.parametrize("variant", ALL)
def test_only_torch_is_published_and_run(variant):
    """No graph is published and no TensorRT runtime offered: measured, TensorRT engines built from
    this model's graph -- fp32 as much as fp16 -- moved 3-6% of COCO detections where ONNX Runtime
    on the same graph moved none. If that changes, this is the reminder to measure it again."""
    from mozo.adapters.mm_grounding_dino import MMGroundingDinoPredictor

    keys = published(FAMILY, variant)
    if not keys:
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert keys == ["torch-fp32"]
    assert MMGroundingDinoPredictor.EXECUTES == ("torch",)
    assert MMGroundingDinoPredictor.DERIVES == {"torch-fp16": "torch-fp32"}


# --- the checkpoint ---


@pytest.mark.parametrize("variant", ALL)
def test_each_decoder_layer_loads_its_own_box_head(variant):
    """The trap ``transformers`` 5 falls into: the file stores layer 0's head as ``bbox_embed.0``
    and layers 1-5 under ``model.decoder.bbox_embed``, and reading that as six copies of layer 0
    loads, runs and moves every box. Each layer's head and the last score bias must be the
    file's own."""
    import torch

    from mozo.vendors.mm_grounding_dino_deploy.checkpoint import build, load_state_dict
    from mozo.weights import resolve

    require_weights(FAMILY, variant)
    path = resolve(FAMILY, variant, "torch-fp32")
    state = load_state_dict(path)
    model = build(SPECS[variant], path)
    for layer in range(6):
        key = ("bbox_embed.0" if layer == 0 else f"model.decoder.bbox_embed.{layer}")
        assert torch.equal(model.bbox_embed[layer].layers[0].weight, state[f"{key}.layers.0.weight"])
    assert not torch.equal(model.bbox_embed[1].layers[0].weight, model.bbox_embed[0].layers[0].weight)
    assert torch.equal(model.class_embed.bias, state["model.decoder.class_embed.5.bias"])


@pytest.mark.parametrize("variant", ALL)
def test_the_window_index_in_the_file_is_the_geometry(variant):
    """Swin's relative-position index is a persistent buffer in the checkpoint, and the strict
    load takes it from there. It must be the index the window size implies."""
    import torch

    from mozo.vendors.mm_grounding_dino_deploy.checkpoint import load_state_dict
    from mozo.weights import resolve

    require_weights(FAMILY, variant)
    window = SPECS[variant].swin.window_size
    coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window),
                                        indexing="ij")).flatten(1)
    relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (window - 1)
    expected = relative[:, :, 0] * (2 * window - 1) + relative[:, :, 1]
    state = load_state_dict(resolve(FAMILY, variant, "torch-fp32"))
    indices = [v for k, v in state.items() if k.endswith("relative_position_index")]
    assert indices and all(torch.equal(index, expected) for index in indices)


# --- the prompt contract ---


def test_it_refuses_no_prompt(predictor, image):
    with pytest.raises(ValueError, match="no text was given"):
        predictor.predict(image, [])


def test_it_refuses_a_blank_prompt(predictor, image):
    with pytest.raises(ValueError, match="text was empty"):
        predictor.predict(image, ["person", "  "])


def test_it_refuses_a_prompt_carrying_a_separator(predictor, image):
    with pytest.raises(ValueError, match="may not contain"):
        predictor.predict(image, ["a person. holding a mug"])


def test_it_refuses_more_prompts_than_the_token_budget(predictor, image):
    """Upstream truncates past 256 tokens without saying so, which drops prompts silently."""
    with pytest.raises(ValueError, match="token budget"):
        predictor.predict(image, [f"a photograph of object number {n}" for n in range(80)])


# --- what comes back ---


def test_it_finds_the_scene(predictor, image):
    found = predictor.predict(image, PROMPTS)
    names = {row["class_name"] for row in found.to_dict()}
    assert names <= set(PROMPTS), f"{names} contains something nobody asked for"
    assert {"person", "laptop", "cup"} <= names


def test_the_name_is_the_callers_phrase_not_a_decoded_span(predictor, image):
    prompts = ["a yellow school bus", "a person holding a mug"]
    for row in predictor.predict(image, prompts).to_dict():
        assert row["class_name"] in prompts


def test_class_id_indexes_the_prompt_list(predictor, image):
    for row in predictor.predict(image, PROMPTS).to_dict():
        assert PROMPTS[row["class_id"]] == row["class_name"]


def test_a_prompt_matching_nothing_returns_nothing(predictor, image):
    assert len(predictor.predict(image, ["dinosaur"])) == 0


def test_detections_come_back_best_first(predictor, image):
    scores = [row["confidence"] for row in predictor.predict(image, PROMPTS).to_dict()]
    assert scores == sorted(scores, reverse=True)


def test_boxes_are_inside_the_source_image(predictor, image):
    height, width = image.shape[:2]
    for row in predictor.predict(image, PROMPTS).to_dict():
        x1, y1, x2, y2 = row["bbox"]
        assert -1 <= x1 < x2 <= width + 1, row["bbox"]
        assert -1 <= y1 < y2 <= height + 1, row["bbox"]


def test_the_default_threshold_is_the_authors(predictor, image):
    """0.4, what the LLMDet README and model cards pass; omitting it must mean exactly that."""
    assert SPECS[SMALLEST].box_threshold == 0.4
    default = predictor.predict(image, PROMPTS).to_dict()
    explicit = predictor.predict(image, PROMPTS, threshold=0.4).to_dict()
    assert default == explicit
    assert all(row["confidence"] > 0.4 for row in default)


def test_a_higher_threshold_keeps_fewer(predictor, image):
    loose = predictor.predict(image, PROMPTS, threshold=0.2)
    tight = predictor.predict(image, PROMPTS, threshold=0.6)
    assert len(tight) <= len(loose)


# --- mozo must not change the vendor's answer ---


def test_mozo_agrees_with_the_vendor(predictor, image):
    """The adapter sorts, converts and rounds. It must do nothing else."""
    from mozo.vendors.mm_grounding_dino_deploy import Predictor
    from mozo.weights import resolve

    vendor = Predictor(resolve(FAMILY, SMALLEST, "torch-fp32"), SPECS[SMALLEST], device="cpu")
    raw = sorted(vendor(image, PROMPTS), key=lambda d: -d.score)
    got = predictor.predict(image, PROMPTS).to_dict()

    assert len(raw) == len(got)
    boxes, scores = as_pixelflow_reports(
        [d.box for d in raw], [d.score for d in raw], [d.prompt_index for d in raw])
    assert [list(b) for b in boxes] == [row["bbox"] for row in got]
    assert list(scores) == [row["confidence"] for row in got]
    assert [PROMPTS[d.prompt_index] for d in raw] == [row["class_name"] for row in got]


# --- the pieces that can be checked without weights ---


def test_the_caption_is_built_the_way_upstreams_processor_builds_it():
    """``_merge_candidate_labels_text``: stripped, lowercased, joined by '. ', trailing '.'."""
    assert caption_for(["Person", " laptop "]) == "person. laptop."


def test_the_tokenizer_matches_bert_base_uncased():
    tokenizer = Tokenizer()
    ids, types, mask = tokenizer.encode("person. laptop. cup.")
    assert ids == [101, 2711, 1012, 12191, 1012, 2452, 1012, 102]
    assert types == [0] * len(ids)
    assert mask == [1] * len(ids)
    assert tokenizer.convert_tokens_to_ids(list(SEPARATORS)) == [101, 102, 1012, 1029]


def test_phrases_are_isolated_from_each_other():
    import torch

    from mozo.vendors.mm_grounding_dino_deploy.network import phrase_masks

    tokenizer = Tokenizer()
    ids, _, _ = tokenizer.encode("person. laptop. cup.")
    separators = torch.tensor(tokenizer.convert_tokens_to_ids(list(SEPARATORS)))
    attention, positions, phrases = phrase_masks(torch.tensor([ids]), separators)

    assert phrases[0].shape[0] == 3
    assert phrases[0].sum().item() == 3
    assert (phrases[0].sum(dim=0) <= 1).all()
    assert not bool(attention[0, 1, 3])
    assert positions[0].max().item() < len(ids)


def test_the_resize_is_the_processors_not_grounding_dinos():
    """Every case read off ``transformers``' ``GroundingDinoImageProcessor``. The one to look at
    is the panorama: this processor scales the long side from the *unrounded* short side when the
    1333 cap binds, so 500x4000 becomes 167x1333 -- where Grounding DINO's own code, and mozo's
    ``grounding_dino``, return 167x1336. Same idea, different pixels."""
    from mozo.vendors.mm_grounding_dino_deploy.image import resized_size

    assert resized_size(1281, 1920, 800, 1333) == (800, 1199)
    assert resized_size(1920, 1281, 800, 1333) == (1199, 800)
    assert resized_size(500, 4000, 800, 1333) == (167, 1333)
    assert resized_size(4000, 500, 800, 1333) == (1333, 167)
    assert resized_size(800, 800, 800, 1333) == (800, 800)
    assert resized_size(100, 100, 800, 1333) == (800, 800)


def test_the_integer_bounds_test_is_upstreams_float_one_at_every_size():
    """``_inside`` replaces upstream's ``0.01 < (x + 0.5) / w < 0.99`` on float32 with integers,
    so a compiled graph cannot round its way across a bound. Every width to 2,048 and every cell,
    padding included, must get upstream's answer -- the ones exactly on 0.01 and 0.99 above all."""
    import torch

    from mozo.vendors.mm_grounding_dino_deploy.network import _inside

    index = torch.arange(2048)
    valid = torch.arange(0, 2049)
    quotient = (torch.linspace(0, 2047, 2048, dtype=torch.float32)[None, :] + 0.5) / valid[:, None]
    upstream = (quotient > 0.01) & (quotient < 0.99)
    assert torch.equal(_inside(index, valid), upstream)
    assert not bool(upstream[50, 49]) and not bool(upstream[50, 0])   # 0.99 and 0.01 exactly


def test_large_reads_four_backbone_stages_into_five_levels():
    """The one geometry difference between the sizes, and a different path through the neck."""
    assert SPECS["llmdet-large"].return_stages == (0, 1, 2, 3)
    assert SPECS["llmdet-large"].num_feature_levels == 5
    for variant in ("llmdet-tiny", "llmdet-base"):
        assert SPECS[variant].return_stages == (1, 2, 3)
        assert SPECS[variant].num_feature_levels == 4


# --- reduced precision and a fixed square ---


def test_tensorrt_is_not_offered():
    from mozo.registry import DERIVES, EXECUTES
    from mozo.runtimes import RuntimeError_, select_runtime
    from mozo.weights import artifacts

    with pytest.raises(RuntimeError_):
        select_runtime("cuda", artifacts(FAMILY, SMALLEST), "tensorrt-fp16",
                       executes=EXECUTES[FAMILY], derives=DERIVES[FAMILY])


def test_cuda_runs_fp16_by_default():
    from mozo.registry import DERIVES, EXECUTES
    from mozo.runtimes import select_runtime
    from mozo.weights import artifacts

    def choose(device):
        return select_runtime(device, artifacts(FAMILY, SMALLEST), "auto",
                              executes=EXECUTES[FAMILY], derives=DERIVES[FAMILY])

    assert choose("cuda") == "torch-fp16"
    assert choose("cpu") == choose("mps") == "torch-fp32"


@pytest.mark.parametrize("height, width", [(480, 640), (640, 480), (500, 4000), (800, 800), (3, 900)])
def test_fit_square_fills_the_long_side_and_masks_the_rest(height, width):
    import numpy as np

    from mozo.vendors.mm_grounding_dino_deploy.image import fit_square, fitted_size

    h, w = fitted_size(height, width, 800)
    assert max(h, w) == 800 and min(h, w) >= 1
    canvas, padding = fit_square(np.full((height, width, 3), 200, np.uint8), 800)
    assert canvas.shape == (3, 800, 800) and padding.shape == (800, 800)
    assert not padding[:h, :w].any() and padding.sum() == 800 * 800 - h * w
    assert (canvas[:, h:, :] == 0).all() and (canvas[:, :, w:] == 0).all()


def test_input_size_is_checked():
    from mozo.adapters.mm_grounding_dino import MMGroundingDinoPredictor

    with pytest.raises(ValueError, match="between 64 and 2048"):
        MMGroundingDinoPredictor(SMALLEST, "cpu", input_size=10)


def _on_cuda(runtime, **options):
    import torch

    from mozo import weights

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if not weights.is_cached(FAMILY, SMALLEST, "torch-fp32"):
        pytest.skip(f"{FAMILY}/{SMALLEST} is not downloaded here")
    from mozo.adapters.mm_grounding_dino import MMGroundingDinoPredictor

    return MMGroundingDinoPredictor(SMALLEST, "cuda", runtime=runtime, **options)


def test_fp16_finds_what_fp32_finds_on_the_fixture(image):
    """Not a parity gate -- fp16 moves borderline detections -- but the confident ones survive."""
    import numpy as np

    want = _on_cuda("torch-fp32").predict(image, list(PROMPTS))
    got = _on_cuda("torch-fp16").predict(image, list(PROMPTS), threshold=0.3)
    assert len(want) > 0
    for found in want:
        nearest = min(got, key=lambda g: float(np.abs(np.subtract(g.bbox, found.bbox)).max()))
        assert nearest.class_name == found.class_name
        assert float(np.abs(np.subtract(nearest.bbox, found.bbox)).max()) < 5.0
