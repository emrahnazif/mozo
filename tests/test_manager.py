"""What the model cache promises: build once, hand back the same object, never twice at a time.

The manager it replaced had 408 lines, nine public methods, four of which nothing called, and no
tests at all -- which is how it came to hold a model under ``rfdetr/nano@cpu`` while offering an
unload that only ever looked for ``rfdetr/nano``. So the properties are pinned here rather than
assumed, on a toy adapter: none of this is about any particular model, and a real checkpoint
would only make it slow.

Nothing here tests eviction, because nothing is evicted -- see :mod:`mozo.manager` for why.

The toy is registered through :data:`mozo.registry.MODEL_REGISTRY` rather than by patching
:func:`mozo.manager._build`, so the registry lookup and the dynamic import are exercised too.
"""

from __future__ import annotations

from threading import Barrier, Event, Thread

import pytest

from mozo.manager import ModelManager
from mozo.registry import MODEL_REGISTRY


class ToyPredictor:
    """Stands in for an adapter: records every construction, and can be made slow."""

    #: Variant names, in construction order. A second entry for one variant means a cache miss
    #: that should have been a hit.
    built: list[str] = []
    #: When set, construction blocks on it -- so a load can be held open on another thread.
    gate: Event | None = None
    #: Set the moment construction begins, so a test can wait for a load to be genuinely under
    #: way rather than merely scheduled.
    entered = Event()

    def __init__(self, variant: str, device: str | None = None, **kwargs) -> None:
        self.variant, self.device, self.kwargs = variant, device, kwargs
        ToyPredictor.built.append(variant)
        ToyPredictor.entered.set()
        if ToyPredictor.gate is not None:
            assert ToyPredictor.gate.wait(timeout=10), "load was never released"


@pytest.fixture
def toy(monkeypatch):
    """Register the toy family, and hand back a clean construction log."""
    monkeypatch.setitem(MODEL_REGISTRY, "toy", {
        "adapter_class": "ToyPredictor",
        # pytest imports this file as a top-level module, so import_module finds it in sys.modules.
        "module": "test_manager",
        "task_type": "toy",
        "description": "not a real model",
        "variants": [],
    })
    ToyPredictor.built = []
    ToyPredictor.gate = None
    ToyPredictor.entered = Event()
    return ToyPredictor


class TestLoadingOnce:
    def test_a_second_request_reuses_the_first_model(self, toy):
        models = ModelManager()
        first = models.get_model("toy", "a")
        assert models.get_model("toy", "a") is first
        assert toy.built == ["a"]

    def test_the_device_is_part_of_the_identity(self, toy):
        """Two devices are two models, and the cache must not confuse them.

        The previous manager keyed on ``family/variant@device`` when loading and on
        ``family/variant`` when unloading, so a model loaded onto an explicit device could
        never be released and never appeared in the loaded list.
        """
        models = ModelManager()
        auto = models.get_model("toy", "a")
        pinned = models.get_model("toy", "a", device="cpu")

        assert auto is not pinned
        assert pinned.device == "cpu"
        assert models.loaded() == ["toy/a", "toy/a@cpu"]

    def test_adapter_arguments_are_passed_through(self, toy):
        model = ModelManager().get_model("toy", "a", revision="2026-01-01")
        assert model.kwargs == {"revision": "2026-01-01"}

    @pytest.mark.parametrize("first,second", [
        ({"checkpoint_path": "a.pth"}, {"checkpoint_path": "b.pth"}),   # two fine-tunes
        ({"runtime": "torch-fp32"}, {"runtime": "onnx-fp32"}),          # different numbers
        ({"revision": "2026-01-01"}, {"revision": "2026-02-01"}),       # different weights
        ({}, {"labels": ["hardhat"]}),
    ])
    def test_arguments_that_change_the_model_change_its_identity(self, toy, first, second):
        """Otherwise the second request silently gets the first request's model.

        RF-DETR accepts unpublished variant names so that a checkpoint of your own can be
        loaded under one, which means two people's ``my-training`` are routinely different
        models. Keyed on family and variant alone, the second would never be built.
        """
        models = ModelManager()
        assert models.get_model("toy", "a", **first) is not models.get_model("toy", "a", **second)
        assert len(models.loaded()) == 2

    def test_the_same_arguments_still_hit(self, toy):
        models = ModelManager()
        first = models.get_model("toy", "a", revision="2026-01-01")
        assert models.get_model("toy", "a", revision="2026-01-01") is first
        assert toy.built == ["a"]

    def test_an_unknown_family_names_the_ones_that_exist(self, toy):
        with pytest.raises(ValueError, match="Unknown model family: 'nope'"):
            ModelManager().get_model("nope", "a")


class TestConcurrency:
    def test_racing_threads_build_one_model_between_them(self, toy):
        """The expensive thing must happen once however many callers ask at the same moment."""
        models = ModelManager()
        start = Barrier(8)
        got = []

        def ask():
            start.wait(timeout=10)
            got.append(models.get_model("toy", "a"))

        threads = [Thread(target=ask) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        assert toy.built == ["a"]
        assert len(got) == 8 and all(model is got[0] for model in got)

    def test_a_resident_model_is_served_while_another_is_still_loading(self, toy):
        """Why the lock is held across the build and nowhere else.

        A load takes seconds. If reads were taken under the same lock, every request would queue
        behind the slowest cold start in flight -- including requests for models already in
        memory, which is the case that has to stay fast.
        """
        models = ModelManager()
        warm = models.get_model("toy", "warm")

        toy.gate = Event()
        toy.entered.clear()
        loading = Thread(target=models.get_model, args=("toy", "slow"))
        loading.start()
        try:
            assert toy.entered.wait(timeout=5), "the slow load never started"
            served = Event()

            def ask_for_the_warm_one():
                assert models.get_model("toy", "warm") is warm
                served.set()

            Thread(target=ask_for_the_warm_one).start()
            assert served.wait(timeout=5), "a cache hit blocked behind an unrelated load"
        finally:
            toy.gate.set()
            loading.join(timeout=10)

        assert models.loaded() == ["toy/warm", "toy/slow"]


class TestUnloading:
    """Released on request, and only the model asked for. Nothing is released unasked."""

    def test_load_unload_load_gives_a_new_object(self, toy):
        models = ModelManager()
        first = models.get_model("toy", "a")
        assert models.loaded() == ["toy/a"]

        assert models.unload("toy", "a") is True
        assert models.loaded() == []

        second = models.get_model("toy", "a")
        assert second is not first
        assert models.loaded() == ["toy/a"]
        assert toy.built == ["a", "a"]

    def test_unload_takes_what_get_model_took(self, toy):
        """The device and the adapter arguments are the identity, for unloading too."""
        models = ModelManager()
        models.get_model("toy", "a")
        models.get_model("toy", "a", device="cpu", labels=["x"])

        assert models.unload("toy", "a", device="cpu") is False  # not the one with labels
        assert models.unload("toy", "a", device="cpu", labels=["x"]) is True
        assert models.loaded() == ["toy/a"]

    def test_unloading_what_is_not_loaded_says_so(self, toy):
        assert ModelManager().unload("toy", "a") is False

    def test_clear_releases_everything_and_says_what(self, toy):
        models = ModelManager()
        models.get_model("toy", "a")
        models.get_model("toy", "b", device="cpu")
        assert models.clear() == ["toy/a", "toy/b@cpu"]
        assert models.loaded() == []
        assert models.clear() == []

    def test_it_gives_the_accelerator_memory_back(self, toy, monkeypatch):
        freed = []
        monkeypatch.setattr("mozo.manager._free_accelerator_memory", lambda: freed.append(1))
        models = ModelManager()
        models.get_model("toy", "a")
        models.unload("toy", "a")
        assert freed == [1]
        models.unload("toy", "a")  # nothing to release, nothing to free
        assert freed == [1]

    def test_a_model_still_held_elsewhere_is_named(self, toy, caplog):
        models = ModelManager()
        kept = models.get_model("toy", "a")
        with caplog.at_level("WARNING", logger="mozo.manager"):
            models.unload("toy", "a")
        assert "toy/a" in caplog.text and "still holds it" in caplog.text
        del kept

    def test_a_model_held_by_no_one_else_is_released_quietly(self, toy, caplog):
        models = ModelManager()
        models.get_model("toy", "a")
        with caplog.at_level("WARNING", logger="mozo.manager"):
            models.unload("toy", "a")
        assert caplog.text == ""

    def test_unloading_a_model_still_loading_waits_for_it(self, toy):
        models = ModelManager()
        toy.gate = Event()
        loading = Thread(target=models.get_model, args=("toy", "slow"))
        loading.start()
        assert toy.entered.wait(timeout=5)

        result = []
        unloading = Thread(target=lambda: result.append(models.unload("toy", "slow")))
        unloading.start()
        unloading.join(timeout=0.2)
        assert unloading.is_alive(), "unload did not wait for the build"

        toy.gate.set()
        loading.join(timeout=10)
        unloading.join(timeout=10)
        assert result == [True]
        assert models.loaded() == []

    def test_nothing_is_kept_for_a_build_once_it_ends(self, toy):
        models = ModelManager()
        models.get_model("toy", "a")
        assert models._builds == {}


class TestResident:
    def test_it_reports_device_last_use_and_order(self, toy):
        models = ModelManager()
        models.get_model("toy", "a", device="cpu")
        models.get_model("toy", "b")
        first = {row["id"]: row for row in models.resident()}
        assert list(first) == ["toy/a@cpu", "toy/b"]
        assert first["toy/a@cpu"]["device"] == "cpu"
        assert (first["toy/a@cpu"]["family"], first["toy/a@cpu"]["variant"]) == ("toy", "a")

        models.get_model("toy", "a", device="cpu")
        again = {row["id"]: row for row in models.resident()}
        assert again["toy/a@cpu"]["last_used"] >= first["toy/a@cpu"]["last_used"]
        assert again["toy/b"]["last_used"] == first["toy/b"]["last_used"]

    def test_a_size_it_cannot_estimate_is_none(self, toy):
        models = ModelManager()
        models.get_model("toy", "a")  # no runtime, no manifest entry
        assert models.resident()[0]["size"] is None

    def test_a_checkpoint_of_your_own_is_its_file_size(self, toy, tmp_path):
        checkpoint = tmp_path / "mine.pth"
        checkpoint.write_bytes(b"x" * 123)
        models = ModelManager()
        models.get_model("toy", "mine", checkpoint_path=str(checkpoint))
        assert models.resident()[0]["size"] == 123


def test_the_shared_cache_can_be_released_too(toy, monkeypatch):
    import mozo

    monkeypatch.setattr(mozo, "_shared", ModelManager())
    first = mozo.get_model("toy/a")
    assert mozo.unload("toy/a") is True
    assert mozo.get_model("toy", "a") is not first
    assert mozo.clear() == ["toy/a"]
