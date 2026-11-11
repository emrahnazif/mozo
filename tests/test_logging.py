"""mozo as a library prints nothing; mozo as a command prints what it always has.

An application that embeds mozo needs the "downloading", "loading" and "ready on" lines as log
records it can route into its own job logs, not as writes to a stdout it may not have. So the
package logs under ``mozo``, and only the command line turns those records back into lines.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import pickle

import pytest
from click.testing import CliRunner

from conftest import FIXTURE, ROOT, fresh_interpreter
from mozo import cli as cli_module
from mozo.cli import _showing_logs, cli


@pytest.fixture(scope="module")
def library_calls() -> list[tuple[str, ast.Call]]:
    """Every call in mozo's own modules, with where it is.

    Not the command line, which listens on ``mozo`` and prints by design, and not the vendors,
    which are kept as their own code and hold no prints.
    """
    calls = []
    for path in sorted((ROOT / "mozo").rglob("*.py")):
        if path.name == "cli.py" or "vendors" in path.parts:
            continue
        where = path.relative_to(ROOT)
        calls += [(f"{where}:{node.lineno}", node)
                  for node in ast.walk(ast.parse(path.read_text()))
                  if isinstance(node, ast.Call)]
    return calls


def test_no_library_module_calls_print(library_calls):
    offenders = [where for where, call in library_calls
                 if isinstance(call.func, ast.Name) and call.func.id == "print"]
    assert not offenders, "use logging.getLogger(__name__) instead: " + ", ".join(offenders)


def test_every_logger_is_under_mozo(library_calls):
    """One handler on ``logging.getLogger("mozo")`` hears everything."""
    offenders = [where for where, call in library_calls
                 if isinstance(call.func, ast.Attribute) and call.func.attr == "getLogger"
                 and not (len(call.args) == 1 and isinstance(call.args[0], ast.Name)
                          and call.args[0].id == "__name__")]
    assert not offenders, "log through getLogger(__name__): " + ", ".join(offenders)


def test_a_library_caller_hears_nothing_until_they_configure_logging(tmp_path):
    """A real download, a status line and a warning print nothing; one basicConfig later, they do.

    In a fresh interpreter, because pytest's handler on the root logger stops Python's
    last-resort printer from ever running here -- a missing ``NullHandler`` would pass.
    """
    payload = tmp_path / "weights.bin"
    payload.write_bytes(b"x" * 4096)
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    finished = fresh_interpreter(
        "import logging, sys\n"
        "from pathlib import Path\n"
        "from mozo import weights\n"
        f"weights._fetch({payload.as_uri()!r}, Path({str(tmp_path / 'out.bin')!r}),"
        f" size=4096, sha256={digest!r})\n"
        "logging.getLogger('mozo.adapters._yolo').warning('no class names')\n"
        "print('configured', flush=True)\n"
        "logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(name)s %(message)s')\n"
        "logging.getLogger('mozo.manager').info('[mozo] loading rfdetr/nano')\n"
    )
    assert (tmp_path / "out.bin").read_bytes() == payload.read_bytes()
    assert finished.stdout == "configured\nmozo.manager [mozo] loading rfdetr/nano\n"
    assert finished.stderr == ""


def test_the_command_line_prints_the_bare_message(monkeypatch, workflow):
    """``mozo run`` prints mozo's records exactly as the old ``print`` calls did."""
    def logs_instead(built, overrides):
        logging.getLogger("mozo.manager").info("[mozo] loading %s", "rfdetr/nano")
        logging.getLogger("mozo.adapters.rfdetr").info("RF-DETR nano ready on cpu via torch-fp32.")

    monkeypatch.setattr(cli_module, "_process", logs_instead)
    result = CliRunner().invoke(cli, ["run", str(workflow), "--file", str(FIXTURE)])
    assert result.exit_code == 0, result.output
    assert result.output == "[mozo] loading rfdetr/nano\nRF-DETR nano ready on cpu via torch-fp32.\n"


def test_warnings_go_to_stderr_and_the_handler_leaves_with_the_command():
    logger = logging.getLogger("mozo")
    before = list(logger.handlers), logger.level
    with CliRunner(mix_stderr=False).isolation() as (out, err):
        with _showing_logs():
            logging.getLogger("mozo.runtimes").warning("asked for cuda, running on cpu")
            logging.getLogger("mozo.weights").debug("not shown")
        stdout, stderr = out.getvalue(), err.getvalue()
    assert stdout == b""
    assert stderr == b"asked for cuda, running on cpu\n"
    assert (list(logger.handlers), logger.level) == before


def test_the_server_logging_reaches_child_processes():
    """``mozo start --reload`` or ``--workers`` runs the app in children, which get this config.

    It has to survive pickling to get there, and once applied it has to print mozo's records. In
    a fresh interpreter because applying it replaces uvicorn's and mozo's handlers process-wide.
    """
    pickle.dumps(cli_module._server_logging())
    finished = fresh_interpreter(
        "import logging, logging.config\n"
        "from mozo.cli import _server_logging\n"
        "logging.config.dictConfig(_server_logging())\n"
        "logging.getLogger('mozo.server').info('[mozo] loading rfdetr/nano')\n"
        "logging.getLogger('mozo.server').warning('MOZO_ENABLE: nosuchfamily.')\n"
    )
    assert finished.stdout == "[mozo] loading rfdetr/nano\n"
    assert finished.stderr == "MOZO_ENABLE: nosuchfamily.\n"
