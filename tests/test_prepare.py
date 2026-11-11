"""``prepare(image)``: a prompted model's image half, run before the first prompt.

In Python it warms the model's own cache, and an encode under way is waited for rather than
repeated. Over HTTP the server can also keep the image, so a click names it by ``?sha256=``
instead of uploading it again.
"""

from __future__ import annotations

import hashlib
import importlib
import threading

import numpy as np
import pixelflow as pf
import pytest

from conftest import FIXTURE
from mozo.adapters._prepare import Preparing
from mozo.image import load_image
from mozo.prepared import PreparedImages, limit_from_environment
from mozo.registry import MODEL_REGISTRY, PREPARES


class Fake(Preparing):
    """A model whose encode can be held open, so the interleavings are deterministic."""

    def __init__(self):
        self.encodes = 0
        self.started, self.release = threading.Event(), threading.Event()
        self.release.set()

    def _encode(self, pixels):
        self.encodes += 1
        self.started.set()
        assert self.release.wait(5)

    def predict(self, image):
        pixels = load_image(image)
        self._after_prepare(pixels)
        return self.encodes


@pytest.fixture(scope="module")
def pixels():
    return load_image(str(FIXTURE))


class TestPrepare:
    def test_it_encodes_the_image(self, pixels):
        model = Fake()
        model.prepare(pixels)
        assert model.encodes == 1

    def test_asked_twice_at_once_it_encodes_once(self, pixels):
        model = Fake()
        model.release.clear()
        first = threading.Thread(target=model.prepare, args=(pixels,))
        first.start()
        assert model.started.wait(5)
        second = threading.Thread(target=model.prepare, args=(pixels,))
        second.start()
        second.join(0.2)
        assert second.is_alive(), "the second prepare did not wait for the first"
        model.release.set()
        first.join(5)
        second.join(5)
        assert model.encodes == 1

    def test_a_click_during_prepare_waits_for_it_rather_than_encoding_again(self, pixels):
        """The case prepare exists for: the image is opened, and the user clicks at once."""
        model = Fake()
        model.release.clear()
        preparing = threading.Thread(target=model.prepare, args=(pixels,))
        preparing.start()
        assert model.started.wait(5)

        clicked = []
        click = threading.Thread(target=lambda: clicked.append(model.predict(pixels)))
        click.start()
        click.join(0.2)
        assert click.is_alive(), "the click did not wait for the encode under way"
        model.release.set()
        preparing.join(5)
        click.join(5)
        assert clicked == [1]

    def test_a_click_with_nothing_being_prepared_costs_nothing_extra(self, pixels, monkeypatch):
        """No encode under way, so nothing is hashed to look for one."""
        model = Fake()
        monkeypatch.setattr("mozo.adapters._prepare._key",
                            lambda pixels: pytest.fail("hashed with nothing being prepared"))
        model.predict(pixels)

    def test_a_click_on_another_image_does_not_wait(self, pixels):
        model = Fake()
        model.release.clear()
        threading.Thread(target=model.prepare, args=(pixels,)).start()
        assert model.started.wait(5)
        done = threading.Event()
        threading.Thread(target=lambda: (model.predict(pixels[:10]), done.set())).start()
        try:
            assert done.wait(2), "a click on a different image waited for this one"
        finally:
            model.release.set()


def test_the_registry_names_exactly_the_families_that_can_be_prepared():
    """The flag /models serves and the adapters that actually have prepare() agree."""
    found = set()
    for family, entry in MODEL_REGISTRY.items():
        adapter = getattr(importlib.import_module(entry["module"]), entry["adapter_class"])
        if hasattr(adapter, "prepare"):
            found.add(family)
    assert found == set(PREPARES)


def test_a_real_model_encodes_once_for_prepare_and_the_clicks_after(pixels):
    """SAM 2 tiny, offline from the local cache: prepare encodes, the clicks only decode."""
    from mozo import weights
    from mozo.manager import ModelManager

    weights.configure(offline=True)
    if not weights.is_cached("sam2", "tiny", "torch-fp32"):
        pytest.skip("sam2/tiny is not downloaded here")
    model = ModelManager().get_model("sam2", "tiny", device="cpu", runtime="torch-fp32")
    network = model._segmenter.network
    calls = []
    real = network.encode
    network.encode = lambda batch: calls.append(1) or real(batch)

    model.prepare(pixels)
    assert calls == [1]
    height, width = pixels.shape[:2]
    for x, y in [(width / 2, height / 2), (width / 3, height / 3)]:
        model.predict(pixels, points=[[x, y]], labels=[1])
    assert calls == [1]


class TestTheStore:
    def test_it_is_bounded_by_bytes_and_drops_the_least_recently_used(self):
        store = PreparedImages(limit=250)
        image = np.zeros(100, dtype=np.uint8)
        store.put("a", image)
        store.put("b", image)
        assert store.get("a") is image  # a is now the more recently used
        store.put("c", image)
        assert store.get("b") is None and store.get("a") is image and store.get("c") is image
        assert len(store) == 2

    def test_the_newest_is_kept_even_when_larger_than_the_bound(self):
        store = PreparedImages(limit=10)
        store.put("big", np.zeros(100, dtype=np.uint8))
        assert store.get("big") is not None

    def test_the_bound_comes_from_the_environment(self, monkeypatch):
        monkeypatch.delenv("MOZO_PREPARED_MB", raising=False)
        assert limit_from_environment() == 512_000_000
        monkeypatch.setenv("MOZO_PREPARED_MB", "64")
        assert limit_from_environment() == 64_000_000


class FakeModel:
    """What /prepare and /predict call, recording what they were handed."""

    def __init__(self):
        self.prepared, self.predicted = [], []

    def prepare(self, image):
        self.prepared.append(image)

    def predict(self, image, *args, **kwargs):
        # Prompted families get their text positionally: model.predict(image, text).
        self.predicted.append(image)
        return pf.Detections()


@pytest.fixture
def served(client, monkeypatch):
    """The server with a fake model and an empty store of its own."""
    model = FakeModel()
    monkeypatch.setattr(client.app.state.model_manager, "get_model",
                        lambda family, variant, **kw: model)
    monkeypatch.setattr(client.app.state, "prepared", PreparedImages(10**9))
    return client, model


class TestOverHTTP:
    CLICK = {"point": "10,10", "label": "1"}

    @pytest.fixture(scope="class")
    def upload(self):
        data = FIXTURE.read_bytes()
        return data, hashlib.sha256(data).hexdigest()

    def test_prepare_answers_with_the_files_own_sha256_and_encodes(self, served, upload):
        client, model = served
        data, key = upload
        response = client.post("/prepare/sam2/tiny", files={"file": ("a.jpg", data)})
        assert response.status_code == 200
        assert response.json() == {"sha256": key}
        assert len(model.prepared) == 1

    def test_a_click_by_sha256_needs_no_file(self, served, upload):
        client, model = served
        data, key = upload
        client.post("/prepare/sam2/tiny", files={"file": ("a.jpg", data)})
        response = client.post("/predict/sam2/tiny", params={**self.CLICK, "sha256": key})
        assert response.status_code == 200
        assert np.array_equal(model.predicted[0], load_image(data))

    def test_an_image_not_kept_is_a_404_saying_what_to_do(self, served, upload):
        client, _ = served
        _, key = upload
        response = client.post("/predict/sam2/tiny", params={**self.CLICK, "sha256": key})
        assert response.status_code == 404
        assert "Send the file again" in response.json()["detail"]

    def test_the_file_sent_with_its_sha256_is_kept_for_the_clicks_after(self, served, upload):
        """How a client recovers from a miss -- and how its first click can prepare too."""
        client, model = served
        data, key = upload
        first = client.post("/predict/sam2/tiny", params={**self.CLICK, "sha256": key},
                            files={"file": ("a.jpg", data)})
        assert first.status_code == 200
        later = client.post("/predict/sam2/tiny", params={**self.CLICK, "sha256": key})
        assert later.status_code == 200
        assert len(model.predicted) == 2

    def test_a_file_that_is_not_what_the_sha256_says_is_refused(self, served, upload):
        client, _ = served
        data, _ = upload
        response = client.post("/predict/sam2/tiny", params={**self.CLICK, "sha256": "0" * 64},
                               files={"file": ("a.jpg", data)})
        assert response.status_code == 400
        assert "the file's sha256 is" in response.json()["detail"]

    def test_a_file_already_kept_is_not_decoded_again(self, served, upload, monkeypatch):
        """Sent again with its sha256, or prepared twice: a hash, not a second decode."""
        import mozo.server

        client, _ = served
        data, key = upload
        decodes = []
        real = mozo.server._decode
        monkeypatch.setattr(mozo.server, "_decode", lambda data: decodes.append(1) or real(data))
        client.post("/prepare/sam2/tiny", files={"file": ("a.jpg", data)})
        client.post("/prepare/sam2/tiny", files={"file": ("a.jpg", data)})
        client.post("/predict/sam2/tiny", params={**self.CLICK, "sha256": key},
                    files={"file": ("a.jpg", data)})
        assert decodes == [1]


    def test_a_family_that_cannot_be_prepared_refuses_both(self, served, upload):
        client, _ = served
        data, key = upload
        assert client.post("/prepare/rfdetr/nano",
                           files={"file": ("a.jpg", data)}).status_code == 400
        response = client.post("/predict/rfdetr/nano", params={"sha256": key})
        assert response.status_code == 400 and "send the file" in response.json()["detail"]

    def test_models_says_which_families_can_be_prepared(self, client):
        body = client.get("/models").json()
        assert {family for family, info in body.items() if info["prepares"]} == set(PREPARES)


class TestNothingChangesForExistingClients:
    """``?sha256=`` made /predict's file optional. A client that never sends one must not notice.

    The missing-file answer below is what the release before ``?sha256=`` returned, recorded from
    it: FastAPI's own validation error, ahead of every other check.
    """

    MISSING_FILE = {"detail": [{"type": "missing", "loc": ["body", "file"],
                                "msg": "Field required", "input": None}]}

    #: A request each family that can be prepared accepted before, minus the file.
    REQUESTS = {
        "sam2/tiny": {"point": "10,10", "label": "1"},
        "edgetam/edgetam": {"point": "10,10", "label": "1"},
        "sam3/sam3": {"text": "person"},
        "owlv2/base": {"text": "a person"},
    }

    @pytest.mark.parametrize("model", list(REQUESTS))
    def test_a_request_with_its_file_is_answered_as_before(self, served, model):
        client, fake = served
        data = FIXTURE.read_bytes()
        response = client.post(f"/predict/{model}", params=self.REQUESTS[model],
                               files={"file": ("a.jpg", data)})
        assert response.status_code == 200
        assert np.array_equal(fake.predicted[0], load_image(data))
        assert len(client.app.state.prepared) == 0, "an ordinary upload was kept"

    @pytest.mark.parametrize("url", [
        *[f"/predict/{model}" for model in REQUESTS],
        "/predict/rfdetr/nano",            # a family that cannot be prepared
        "/predict/nosuchfamily/x",         # an unknown model: the missing file was still first
        "/predict/sam2/tiny?point=1,1",    # a bad prompt: the missing file was still first
    ])
    def test_a_request_without_a_file_gets_the_same_422_as_before(self, served, url):
        client, _ = served
        response = client.post(url)
        assert response.status_code == 422
        assert response.json() == self.MISSING_FILE

    def test_the_file_is_still_documented_as_the_image(self, client):
        """Optional in the schema now, and the schema says why, rather than vanishing from it."""
        schema = client.get("/openapi.json").json()
        body = next(c for path, item in schema["paths"].items() if path.startswith("/predict/")
                    for c in [item["post"]["requestBody"]["content"]["multipart/form-data"]])
        ref = body["schema"]["$ref"].rsplit("/", 1)[-1]
        assert "file" in schema["components"]["schemas"][ref]["properties"]
