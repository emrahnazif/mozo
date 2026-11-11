"""``prepare(image)``: a prompted model's image half, run before the first prompt.

In Python it warms the model's own cache, and an encode under way is waited for rather than
repeated. Over HTTP the server can also keep the image, so a click names it by ``?sha256=``
instead of uploading it again.
"""

from __future__ import annotations

import importlib
import threading

import numpy as np
import pytest

from conftest import FIXTURE
from mozo.adapters._prepare import Preparing
from mozo.image import load_image
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
