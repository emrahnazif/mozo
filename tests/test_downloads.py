"""Downloads a host can watch and stop, and loads that do not queue behind each other.

A desktop app runs a download as a job: it shows a progress bar, offers a cancel button, and
fetches on one lane while another model loads on the GPU. These run against the synthetic zoo
served from ``file://``, with a chunk size small enough that every file takes several chunks.
"""

from __future__ import annotations

import threading
import time

import pytest

from mozo.manager import ModelManager


@pytest.fixture
def chunked(weights, monkeypatch):
    """The zoo, read four bytes at a time, so a 14-byte artifact is a four-chunk download."""
    monkeypatch.setattr(weights, "_DOWNLOAD_CHUNK_BYTES", 4)
    return weights


def _leftovers(weights) -> list:
    return [row["path"] for row in weights.cached() if row["state"] == "partial"]


class TestProgress:
    def test_it_rises_across_every_file_and_ends_at_the_total(self, chunked):
        expected = chunked.download_size("toy", "split", "onnx-fp32")
        seen = []
        chunked.fetch("toy", "split", "onnx-fp32", progress=lambda done, total: seen.append((done, total)))

        dones = [done for done, _ in seen]
        assert len(seen) >= 4, "at least once per file"
        assert dones == sorted(set(dones)), "strictly rising, never restarting at zero per file"
        assert seen[-1] == (expected, expected)
        assert {total for _, total in seen} == {expected}

    def test_fetch_leaves_the_load_nothing_to_do(self, chunked):
        paths = chunked.fetch("toy", "split", "onnx-fp32")
        assert set(paths) == {"onnx-fp32-encoder", "onnx-fp32-decoder", "LICENSE", "NOTICE"}
        assert all(path.is_file() for path in paths.values())
        assert chunked.is_cached("toy", "split", "onnx-fp32")

        seen = []
        assert chunked.fetch("toy", "split", "onnx-fp32", progress=lambda *a: seen.append(a)) == paths
        assert seen == []

    def test_without_a_watcher_nothing_is_called(self, chunked):
        """The block's watch is gone once it closes; a later download reports to no one."""
        seen = []
        with chunked.watching(progress=lambda *a: seen.append(a)):
            pass
        chunked.resolve("toy", "beta")
        assert seen == []


class TestCancel:
    def test_halfway_raises_and_leaves_no_part_and_no_file(self, chunked):
        asked = []  # cancelled() is asked once per chunk; say stop on the third

        def cancelled():
            asked.append(1)
            return len(asked) >= 3

        with pytest.raises(chunked.DownloadCancelled):
            chunked.fetch("toy", "beta", cancelled=cancelled)
        target = chunked.cache_dir() / "toy" / "beta" / "2026-01-01" / "torch-fp32.pth"
        assert not target.exists()
        assert _leftovers(chunked) == []
        assert not chunked.is_cached("toy", "beta", device="cpu")

    def test_a_callback_that_raises_leaves_nothing_half_written(self, chunked):
        def explode(done, total):
            raise KeyError("the host's own bug")

        with pytest.raises(KeyError):
            chunked.fetch("toy", "beta", progress=explode)
        assert _leftovers(chunked) == []

    def test_remove_refuses_while_the_model_is_downloading(self, chunked):
        refused = []

        def try_to_remove(done, total):
            if not refused:
                try:
                    chunked.remove("toy", "beta")
                except chunked.WeightsError as error:
                    refused.append(str(error))

        chunked.fetch("toy", "beta", progress=try_to_remove)
        assert refused and "is downloading" in refused[0]
        assert chunked.is_cached("toy", "beta", device="cpu")


def test_two_writers_of_one_file_do_not_collide(chunked):
    """Two models can share a checkpoint, and two threads can fetch it at the same moment."""
    both_started = threading.Barrier(2, timeout=5)
    errors, paths = [], []

    def fetch():
        first = []

        def meet(done, total):
            if not first:
                first.append(done)
                both_started.wait()  # both are mid-transfer, each into its own partial file
        try:
            paths.append(chunked.fetch("toy", "beta", progress=meet)["torch-fp32"])
        except Exception as error:  # pragma: no cover - the failure being tested for
            errors.append(error)

    threads = [threading.Thread(target=fetch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    assert len(paths) == 2 and paths[0] == paths[1] and paths[0].is_file()
    assert _leftovers(chunked) == []


class TestLoadsDoNotQueue:
    @pytest.fixture
    def builds(self, monkeypatch):
        """A fake build: ``slow`` blocks until released, anything else returns at once."""
        state = {"started": threading.Event(), "release": threading.Event(), "count": 0}

        def build(family, variant, **kwargs):
            state["count"] += 1
            if family == "slow":
                state["started"].set()
                assert state["release"].wait(10)
            return object()

        monkeypatch.setattr("mozo.manager._build", build)
        return state

    def test_a_second_model_loads_while_the_first_is_still_loading(self, builds):
        manager = ModelManager()
        slow = threading.Thread(target=manager.get_model, args=("slow", "a"))
        slow.start()
        assert builds["started"].wait(5)

        done = threading.Event()
        threading.Thread(target=lambda: (manager.get_model("fast", "b"), done.set())).start()
        try:
            assert done.wait(2), "the second model queued behind the first"
            assert manager.loaded() == ["fast/b"]
        finally:
            builds["release"].set()
            slow.join(5)
        assert manager.loaded() == ["fast/b", "slow/a"]

    def test_one_model_asked_for_twice_at_once_is_built_once(self, builds):
        manager = ModelManager()
        got = []
        threads = [threading.Thread(target=lambda: got.append(manager.get_model("slow", "x")))
                   for _ in range(2)]
        for thread in threads:
            thread.start()
        assert builds["started"].wait(5)
        builds["release"].set()
        for thread in threads:
            thread.join(5)
        assert builds["count"] == 1
        assert len(got) == 2 and got[0] is got[1]


class TestThroughGetModel:
    @pytest.fixture
    def manager(self, chunked, monkeypatch):
        """A manager whose build fetches what a load of a toy model would."""
        monkeypatch.setattr("mozo.manager._build",
                            lambda family, variant, **kw: chunked.resolve(family, variant))
        return ModelManager()

    def test_progress_reaches_the_total_the_manifest_gives(self, manager, chunked):
        expected = chunked.download_size("toy", "alpha", device="cpu")
        seen = []
        manager.get_model("toy", "alpha", device="cpu", progress=lambda *a: seen.append(a))
        assert seen[-1] == (expected, expected)

    def test_cancelling_the_load_caches_nothing_and_loads_nothing(self, manager, chunked):
        with pytest.raises(chunked.DownloadCancelled):
            manager.get_model("toy", "alpha", device="cpu", cancelled=lambda: True)
        assert manager.loaded() == []
        assert _leftovers(chunked) == []

    def test_watching_is_not_part_of_which_model_it_is(self, manager):
        first = manager.get_model("toy", "alpha", device="cpu", progress=lambda *a: None)
        assert manager.get_model("toy", "alpha", device="cpu") is first
        assert manager.loaded() == ["toy/alpha@cpu"]


class TestWaitingOnAnotherCallersBuild:
    """A second caller for a model already loading joins that load rather than starting one."""

    @pytest.fixture
    def gated(self, chunked, monkeypatch):
        """A build that waits at a gate, then downloads toy/alpha -- or is cancelled by its caller."""
        state = {"entered": threading.Event(), "gate": threading.Event(), "builds": 0}

        def build(family, variant, **kwargs):
            state["builds"] += 1
            state["entered"].set()
            assert state["gate"].wait(10)
            return chunked.resolve(family, variant)

        monkeypatch.setattr("mozo.manager._build", build)
        return state

    def _start(self, manager, **watch):
        errors = []

        def run():
            try:
                manager.get_model("toy", "alpha", device="cpu", **watch)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        return thread, errors

    def test_a_waiter_hears_the_download_the_first_caller_started(self, gated, chunked):
        manager = ModelManager()
        expected = chunked.download_size("toy", "alpha", device="cpu")
        first_heard, second_heard = [], []
        first, _ = self._start(manager, progress=lambda *a: first_heard.append(a))
        assert gated["entered"].wait(5)
        second, _ = self._start(manager, progress=lambda *a: second_heard.append(a))
        deadline = time.monotonic() + 5
        while not manager._builds["toy/alpha@cpu"].followers[1:]:  # until the second has joined
            assert time.monotonic() < deadline, "the second caller never joined the build"
            time.sleep(0.001)
        gated["gate"].set()
        first.join(5)
        second.join(5)

        assert gated["builds"] == 1
        assert first_heard[-1] == second_heard[-1] == (expected, expected)

    def test_a_waiter_that_cancels_stops_waiting_and_leaves_the_build_alone(self, gated):
        manager = ModelManager()
        first, first_errors = self._start(manager)
        assert gated["entered"].wait(5)
        second, second_errors = self._start(manager, cancelled=lambda: True)
        second.join(5)
        assert not second.is_alive()
        assert [type(e).__name__ for e in second_errors] == ["DownloadCancelled"]

        gated["gate"].set()
        first.join(5)
        assert first_errors == []
        assert manager.loaded() == ["toy/alpha@cpu"]

    def test_when_the_first_caller_cancels_a_waiter_builds_it_instead(self, gated):
        manager = ModelManager()
        stop_first = threading.Event()
        first, first_errors = self._start(manager, cancelled=stop_first.is_set)
        assert gated["entered"].wait(5)
        second, second_errors = self._start(manager)
        stop_first.set()
        gated["gate"].set()
        first.join(5)
        second.join(5)

        assert [type(e).__name__ for e in first_errors] == ["DownloadCancelled"]
        assert second_errors == []
        assert gated["builds"] == 2
        assert manager.loaded() == ["toy/alpha@cpu"]


def test_resident_estimates_size_from_the_manifest(chunked, monkeypatch):
    """The bytes of the weights a model runs from: its runtime's files, not the licence."""
    from types import SimpleNamespace

    monkeypatch.setattr("mozo.manager._build", lambda family, variant, **kw: SimpleNamespace(
        device="cpu", runtime="onnx-fp32", path=chunked.resolve(family, variant, "onnx-fp32")))
    manager = ModelManager()
    manager.get_model("toy", "alpha", device="cpu", runtime="onnx-fp32")
    [row] = manager.resident()
    assert row["runtime"] == "onnx-fp32"
    onnx = chunked.cache_dir() / "toy" / "alpha" / "2026-02-01" / "onnx-fp32.onnx"
    assert row["size"] == onnx.stat().st_size
