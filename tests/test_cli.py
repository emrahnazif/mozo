"""The command line, which had no test at all until a change broke it silently.

``mozo run wf.json --image photo.jpg`` wrote the uploaded path to a parameter named ``image``,
which was one node's vocabulary copied into a second place. Renaming that parameter fixed the HTTP
endpoint, left this one raising ``KeyError``, and the suite stayed green because nothing here ran
the command. So what this file holds is the entry point itself, not the engine under it.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from conftest import FIXTURE
from mozo.cli import cli



def test_a_file_given_on_the_command_line_is_what_it_runs_on(workflow):
    result = CliRunner().invoke(cli, ["run", str(workflow), "--file", str(FIXTURE)])
    assert result.exit_code == 0, result.output
    assert "1 item in" in result.output


def test_run_is_the_whole_source_not_one_item_of_it(tmp_path):
    """``mozo run`` says why this is the whole source rather than one item of it."""
    photos, out = tmp_path / "photos", tmp_path / "out"
    photos.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (photos / name).write_bytes(FIXTURE.read_bytes())
    document = tmp_path / "wf.json"
    document.write_text(json.dumps({
        "nodes": [{"id": "load", "type": "read_media",
                   "data": {"parameters": {"source": str(photos)}}},
                  {"id": "save", "type": "save_image",
                   "data": {"parameters": {"path": str(out)}}}],
        "edges": [{"source": "load", "sourceHandle": "image",
                   "target": "save", "targetHandle": "image"}]}))

    result = CliRunner().invoke(cli, ["run", str(document)])
    assert result.exit_code == 0, result.output
    assert "3 items in" in result.output
    # .png, because that is ``save_image``'s stated default; the point here is the count.
    assert sorted(p.name for p in out.iterdir()) == ["a.png", "b.png", "c.png"]


def test_test_describes_every_node_for_one_item(workflow):
    """The older behaviour, kept under its own name: what you want before pointing it at 10,000."""
    result = CliRunner().invoke(cli, ["run", str(workflow), "--file", str(FIXTURE), "--test"])
    assert result.exit_code == 0, result.output
    assert "1920x1281 image" in result.output


def test_the_file_is_bound_by_kind_not_by_a_name_written_here(workflow):
    """The parameter is found from its ``Source`` annotation, so this keeps working when it is
    renamed. The literal that used to be here did not."""
    from mozo.workflow import Workflow

    assert Workflow.load(str(workflow)).file_parameter == "source"
    assert CliRunner().invoke(
        cli, ["run", str(workflow), "--file", str(FIXTURE)]).exit_code == 0


def test_a_workflow_naming_files_that_are_not_here_says_so_cleanly(tmp_path):
    """One message listing every missing file, and no traceback."""
    document_path = tmp_path / "wf.json"
    document_path.write_text(json.dumps({
        "nodes": [{"id": "load", "type": "read_media",
                   "data": {"parameters": {"source": "/nowhere/frames"}}}],
        "edges": []}))
    result = CliRunner().invoke(cli, ["run", str(document_path)])
    assert result.exit_code == 1
    assert "1 file that is not here" in result.output and "/nowhere/frames" in result.output
    assert "Traceback" not in result.output


def test_downloads_draw_a_bar_only_on_a_terminal():
    """Piped or logged, a run keeps the plain log lines; a bar redrawn in place is unreadable."""
    from mozo.cli import _models_with_progress

    with CliRunner().isolation():
        assert _models_with_progress() is None


def test_the_bar_ends_its_line_before_anything_else_prints(capsys):
    import logging

    from mozo.cli import _bar, _showing_logs

    with _showing_logs():
        _bar("rfdetr/nano", 50, 100)
        logging.getLogger("mozo.weights").info("[mozo] downloading LICENSE (0.0 MB)")
        _bar("rfdetr/nano", 100, 100)
    lines = capsys.readouterr().out.split("\n")
    assert "rfdetr/nano" in lines[0] and f"[{'#' * 15}{'.' * 15}]" in lines[0]  # half way
    assert lines[1] == "[mozo] downloading LICENSE (0.0 MB)"
    assert lines[2].strip().startswith("rfdetr/nano") and lines[2].endswith("MB")
