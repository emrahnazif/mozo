"""The weights cache as an inventory: what is downloaded, what is left to fetch, and giving it back.

An application shows "852 MB to download" before it starts, keeps weights inside its own folder,
and offers to delete a model. None of that may cost a network call or a torch import, and the
first two are only worth anything if they agree with what a load will actually fetch.
"""

from __future__ import annotations

import json

import pytest

from conftest import executable_runtimes, fresh_interpreter


def _sizes(manifest_file, model_id: str, *keys: str) -> int:
    revisions = json.loads(manifest_file.read_text())["models"][model_id]["revisions"]
    latest = revisions[max(revisions)]["artifacts"]
    return sum(latest[key]["size"] for key in keys)


class TestConfigure:
    def test_code_wins_over_the_environment_and_none_gives_it_back(self, weights, tmp_path):
        from_env = weights.cache_dir()
        weights.configure(cache=tmp_path / "app")
        assert weights.cache_dir() == tmp_path / "app"
        assert weights.resolve("toy", "beta").is_relative_to(tmp_path / "app")
        weights.configure(cache=None)
        assert weights.cache_dir() == from_env

    def test_an_argument_left_out_is_left_alone(self, weights, tmp_path):
        weights.configure(cache=tmp_path / "app")
        weights.configure(offline=True)
        assert weights.cache_dir() == tmp_path / "app"

    def test_offline_in_code_refuses_and_says_how_it_was_set(self, weights, monkeypatch):
        monkeypatch.delenv("MOZO_OFFLINE", raising=False)
        weights.configure(offline=True)
        with pytest.raises(weights.WeightsError, match=r"configure\(offline=True\)"):
            weights.resolve("toy", "beta")

    def test_offline_false_in_code_overrides_the_environment(self, weights, monkeypatch):
        monkeypatch.setenv("MOZO_OFFLINE", "1")
        weights.configure(offline=False)
        assert weights.resolve("toy", "beta").is_file()

    def test_a_mirror_set_in_code_is_fetched_from(self, weights, monkeypatch, zoo):
        monkeypatch.setenv("MOZO_BASE_URL", "file:///nowhere")
        weights.configure(base_url=zoo.as_uri())
        assert weights.resolve("toy", "beta").is_file()

    def test_it_is_mozo_configure_too(self):
        import mozo
        from mozo import weights

        assert mozo.configure is weights.configure


class TestWhatALoadWouldFetch:
    def test_nothing_cached_means_everything_to_fetch(self, weights, manifest_file):
        assert not weights.is_cached("toy", "alpha", device="cpu")
        assert weights.download_size("toy", "alpha", device="cpu") == \
            _sizes(manifest_file, "toy/alpha", "torch-fp32", "LICENSE")

    def test_after_the_download_nothing_is_left(self, weights):
        weights.resolve("toy", "alpha")
        assert weights.is_cached("toy", "alpha", device="cpu")
        assert weights.download_size("toy", "alpha", device="cpu") == 0

    def test_a_runtime_by_name_counts_only_its_own_files(self, weights, manifest_file):
        weights.resolve("toy", "alpha")  # torch and the licence, but not the graph
        assert not weights.is_cached("toy", "alpha", "onnx-fp32")
        assert weights.download_size("toy", "alpha", "onnx-fp32") == \
            _sizes(manifest_file, "toy/alpha", "onnx-fp32")

    def test_a_split_runtime_is_every_part_and_the_notice(self, weights, manifest_file):
        assert weights.download_size("toy", "split", "onnx-fp32") == _sizes(
            manifest_file, "toy/split",
            "onnx-fp32-encoder", "onnx-fp32-decoder", "LICENSE", "NOTICE")
        weights.parts("toy", "split", "onnx-fp32")
        assert weights.is_cached("toy", "split", "onnx-fp32")

    def test_what_a_runtime_brings_with_it_is_counted(self, weights, manifest_file, monkeypatch):
        """A detector run from a graph reads its class names from the labels file."""
        from mozo import registry

        monkeypatch.setitem(registry.NEEDS, "toy", {"onnx": ("labels",)})
        assert weights.download_size("toy", "alpha", "onnx-fp32") == \
            _sizes(manifest_file, "toy/alpha", "onnx-fp32", "labels", "LICENSE")
        # Not on torch, and not a key the revision does not publish.
        assert weights.download_size("toy", "beta", device="cpu") == \
            _sizes(manifest_file, "toy/beta", "torch-fp32", "LICENSE")

    def test_auto_honours_what_the_family_can_execute(self, weights, monkeypatch, manifest_file):
        from mozo import registry

        weights.parts("toy", "split", "onnx-fp32")
        monkeypatch.setitem(registry.EXECUTES, "toy", ("onnx",))
        assert weights.is_cached("toy", "split", device="cpu")
        monkeypatch.setitem(registry.EXECUTES, "toy", ("torch",))
        assert not weights.is_cached("toy", "split", device="cpu")

    def test_a_runtime_it_cannot_run_says_so(self, weights):
        from mozo.runtimes import RuntimeError_

        with pytest.raises(RuntimeError_, match="not published"):
            weights.is_cached("toy", "beta", "onnx-fp32")

    def test_an_unpublished_model_says_so(self, weights):
        with pytest.raises(weights.WeightsError, match="publishes no weights"):
            weights.download_size("toy", "gamma", device="cpu")


def test_model_size_is_what_the_model_runs_from_cached_or_not(weights, manifest_file):
    """Every part of the runtime, but not the licence or NOTICE; the same before and after."""
    expected = _sizes(manifest_file, "toy/split", "onnx-fp32-encoder", "onnx-fp32-decoder")
    assert weights.model_size("toy", "split", "onnx-fp32") == expected
    weights.fetch("toy", "split", "onnx-fp32")
    assert weights.model_size("toy", "split", "onnx-fp32") == expected


class TestInventory:
    def test_an_empty_cache_holds_nothing(self, weights):
        assert weights.cached() == []

    def test_every_file_is_reported_with_its_key_runtime_and_size(self, weights):
        weights.resolve("toy", "alpha", "onnx-fp32")
        rows = {row["key"]: row for row in weights.cached()}
        assert set(rows) == {"onnx-fp32", "LICENSE"}
        onnx = rows["onnx-fp32"]
        assert (onnx["family"], onnx["variant"], onnx["revision"]) == ("toy", "alpha", "2026-02-01")
        assert onnx["runtime"] == "onnx-fp32" and onnx["state"] == "complete"
        assert onnx["size"] == onnx["path"].stat().st_size
        assert rows["LICENSE"]["runtime"] is None

    def test_a_part_of_a_split_runtime_reports_the_runtime(self, weights):
        weights.parts("toy", "split", "onnx-fp32")
        runtimes = {row["key"]: row["runtime"] for row in weights.cached()}
        assert runtimes["onnx-fp32-encoder"] == runtimes["onnx-fp32-decoder"] == "onnx-fp32"

    def test_leftovers_are_named_for_what_they_are(self, weights):
        path = weights.resolve("toy", "alpha", "onnx-fp32")
        revision = path.parent
        (revision / "torch-fp32.pth.part").write_bytes(b"half")
        package = revision / "onnx-fp32.mlpackage"  # what a CoreML archive unpacks to
        package.mkdir()
        (package / "weights.bin").write_bytes(b"x" * 10)
        stale = revision.parent / "1999-01-01"
        stale.mkdir()
        (stale / "torch-fp32.pth").write_bytes(b"old")

        rows = {(row["revision"], row["path"].name): row for row in weights.cached()}
        assert rows["2026-02-01", "torch-fp32.pth.part"]["state"] == "partial"
        assert rows["2026-02-01", "torch-fp32.pth.part"]["key"] == "torch-fp32"
        assert rows["2026-02-01", "onnx-fp32.mlpackage"]["state"] == "unpacked"
        assert rows["2026-02-01", "onnx-fp32.mlpackage"]["size"] == 10
        assert rows["1999-01-01", "torch-fp32.pth"]["state"] == "unlisted"
        assert rows["1999-01-01", "torch-fp32.pth"]["key"] is None


class TestRemove:
    def test_one_revision_goes_and_its_neighbours_stay(self, weights):
        old = weights.resolve("toy", "alpha", revision="2026-01-01")
        new = weights.resolve("toy", "alpha")
        other = weights.resolve("toy", "beta")
        expected = sum(p.stat().st_size for p in old.parent.iterdir())

        assert weights.remove("toy", "alpha", revision="2026-01-01") == expected
        assert not old.parent.exists()
        assert new.is_file() and other.is_file()

    def test_without_a_revision_every_one_goes_with_its_directories(self, weights):
        weights.resolve("toy", "alpha", revision="2026-01-01")
        weights.resolve("toy", "alpha")
        stale = weights.cache_dir() / "toy" / "alpha" / "1999-01-01"
        stale.mkdir()
        (stale / "leftover.part").write_bytes(b"x")

        assert weights.remove("toy", "alpha") > 0
        assert not (weights.cache_dir() / "toy" / "alpha").exists()
        assert weights.cache_dir().joinpath("toy").is_dir() is False  # nothing else was in it

    def test_the_family_stays_while_another_variant_is_in_it(self, weights):
        weights.resolve("toy", "alpha")
        beta = weights.resolve("toy", "beta")
        weights.remove("toy", "alpha")
        assert beta.is_file()

    def test_removing_what_is_not_there_frees_nothing(self, weights):
        assert weights.remove("toy", "alpha") == 0

    @pytest.mark.parametrize("names", [("..", "x"), ("toy", "../beta"), ("toy", "alpha", ".."),
                                       ("", "alpha")])
    def test_a_name_cannot_reach_outside_the_cache(self, weights, names):
        with pytest.raises(ValueError, match="not a model in the cache"):
            weights.remove(*names)


def test_none_of_it_downloads_or_imports_torch(tmp_path):
    """Against the real manifest, in a clean interpreter, with downloads forbidden."""
    finished = fresh_interpreter(
        "import sys\n"
        "from mozo import weights\n"
        f"weights.configure(cache={str(tmp_path)!r}, offline=True)\n"
        "print(weights.is_cached('sam2', 'large', device='mps'),"
        " weights.download_size('sam2', 'large', device='cpu') > 0,"
        " weights.download_size('sam3', 'sam3', 'coreml-fp32')"
        " > weights.download_size('sam3', 'sam3', 'torch-fp32'),"
        " weights.cached(), weights.remove('sam2', 'large'), 'torch' in sys.modules)\n"
    )
    assert finished.stdout.split() == ["False", "True", "True", "[]", "0", "False"]


def _loads():
    """One published variant per family, with every runtime it can execute on this machine."""
    from mozo.registry import MODEL_REGISTRY

    firsts = [(family, entry["variants"][0]) for family, entry in MODEL_REGISTRY.items()]
    for family, variant, runtime in executable_runtimes(firsts):
        yield pytest.param(family, variant, runtime, id=f"{family}/{variant}/{runtime}")


@pytest.mark.parametrize("family, variant, runtime", list(_loads()))
def test_the_plan_is_what_a_load_actually_fetches(family, variant, runtime):
    """Hold :data:`~mozo.registry.EXECUTES` and :data:`~mozo.registry.NEEDS` to real loads.

    They restate what each adapter does -- SAM 3 fetching its checkpoint beside its graph, a
    detector reading labels on a graph runtime -- so a load is run and every file it reaches for
    is recorded. Offline, from the real cache: a model not downloaded here is skipped, never
    fetched.
    """
    from mozo import get_model_info, weights

    weights.configure(offline=True)
    try:
        planned_dir, planned = weights._planned(family, variant, runtime, "cpu", None)
    except Exception as error:  # pragma: no cover - a runtime this machine cannot plan
        pytest.skip(str(error))
    if not weights.is_cached(family, variant, runtime, device="cpu"):
        pytest.skip(f"{family}/{variant} {runtime} is not downloaded here")

    import importlib
    from unittest import mock

    info = get_model_info(family)
    adapter = getattr(importlib.import_module(info["module"]), info["adapter_class"])
    # Moebius always runs torch and takes no runtime argument.
    choice = {} if family == "moebius" else {"runtime": runtime}
    with mock.patch.object(weights, "_obtain", wraps=weights._obtain) as obtain:
        adapter(variant=variant, device="cpu", **choice)

    reached = {artifact["path"] for (artifact, _), _ in obtain.call_args_list}
    assert sorted(reached) == sorted(a["path"] for a in planned.values())
