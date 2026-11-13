"""The editor: that it is served, and that it and the server still agree.

The editor is a built artifact -- ``ui/`` at the repository root produces it, and only the product
is committed. So the thing worth testing here is not its behaviour, which needs a browser, but the
seam: that the page and its files are reachable, and that what the bundle asks the server for is
what the server offers. Those two drift silently, and a stale bundle looks exactly like a working
one until someone clicks Run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urljoin

import pixelflow
import pytest

import mozo
from mozo.workflow import get
from workflow_nodes import shipped

EDITOR = Path(mozo.__file__).parent / "workflow" / "static"


@pytest.fixture(scope="module")
def bundle() -> str:
    """The built editor's JavaScript."""
    built = EDITOR / "assets" / "editor.js"
    if not built.is_file():
        pytest.skip("the editor is not built; run `npm install && npm run build` in ui/")
    return built.read_text()


def _asked_for(page: str, at: str) -> list:
    """Every file *page*, served at *at*, makes the browser fetch: its script, and the stylesheet
    the script attaches -- each resolved as a browser would, the stylesheet against the script."""
    scripts = [urljoin(at, src) for src in re.findall(r'from "([^"]+\.js)"', page)]
    return scripts + [urljoin(script, "editor.css") for script in scripts]


class TestItIsServed:

    def test_the_editor_answers(self, client):
        response = client.get("/workflow")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_every_file_the_page_asks_for_is_there(self, client):
        """A page that loads with a 404 script is a blank screen and no explanation.

        Each URL is resolved against the page's own address the way a browser would, rather than
        being pasted onto a prefix by the test. That distinction is the whole point here: the page
        is served at ``/workflow`` with no trailing slash, so a relative ``./assets/x.js`` resolves
        to ``/assets/x.js`` -- which nothing serves. An earlier version of this test joined the
        prefix itself and passed against exactly that broken page.
        """
        wanted = _asked_for(client.get("/workflow").text, "/workflow")
        assert wanted, "the page asks for nothing, so it cannot be the built editor"
        for resolved in wanted:
            assert client.get(resolved).status_code == 200, f"nothing at {resolved}"

    def test_its_files_are_asked_for_again_rather_than_kept(self, client):
        """Stable names -- a host imports ``editor.js`` by name -- so a cached copy would outlive
        a mozo upgrade unless the browser is told to check."""
        assert client.get("/workflow/assets/editor.js").headers["cache-control"] == "no-cache"

    def test_the_icons_licence_ships_beside_them(self, client):
        assert "MIT" in client.get("/workflow/assets/heroicons-LICENSE").text

    def test_a_name_that_climbs_out_of_the_directory_reaches_nothing(self, client):
        assert client.get("/workflow/assets/../../server.py").status_code == 404
        assert client.get("/workflow/assets/..%2F..%2Fserver.py").status_code == 404


class TestTheEditorAndTheServerAgree:
    """What the bundle was built against, checked against what is served now."""

    def test_it_calls_endpoints_that_exist(self):
        """Every path the editor's API client asks for, relative to the router, is a route.

        Read from the client's source: the bundler renames the functions that carry the paths."""
        from mozo.workflow.api import router

        client = (Path(__file__).resolve().parents[2] / "ui/src/lib/api.js").read_text()
        called = set(re.findall(r"""(?:url|events)\(\s*['"](\w+)['"]""", client))
        assert called >= {"nodes", "stream", "process"}, called
        offered = {route.path for route in router.routes}
        assert {f"/workflow/{path}" for path in called} <= offered

    def test_it_carries_nothing_from_the_project_it_came_from(self, bundle):
        """agentui's API and its cloud are both gone; a bundle still calling them is a stale one."""
        for gone in ("datamarkin", "agentui", "/api/tools", "/api/workflow", "toolType",
                     "MediaInput"):
            assert gone not in bundle, f"the built editor still references {gone!r}"

    def test_the_server_offers_every_widget_the_panel_can_draw(self):
        """The panel switches on `kind`, and anything it has not heard of falls to a text box.

        Read out of the panel rather than listed here, for the reason the catalogue is read out of
        the node functions: a list kept alongside is a second place to keep in step, and this one
        would drift silently -- a new kind would render as a text box and pass.

        `str` is the text box, so the fall-through is right for it and for nothing else.
        """
        panel = (Path(__file__).resolve().parents[2] / "ui/src/lib/PropertiesPanel.svelte").read_text()
        drawn = set(re.findall(r"field\.kind === '(\w+)'", panel)) | {"str"}
        served = {parameter.kind for name in shipped() for parameter in get(name).parameters}
        assert served <= drawn, f"the server offers widgets the editor cannot draw: {served - drawn}"

    def test_a_connection_is_refused_before_it_is_made_rather_than_after(self):
        """Svelte Flow's `isValidConnection`, not its `connect` event.

        The event is a notification: by the time it fires the edge is on the canvas, and returning
        early from it leaves it there. Checked that way, the editor accepted a detections output
        into an image input and only the server noticed -- 400 at the end of a drag that looked
        like it had worked. `isValidConnection` is asked while the wire is still in the air.
        """
        app = (Path(__file__).resolve().parents[2] / "ui/src/App.svelte").read_text()
        assert "{isValidConnection}" in app, "the canvas does not validate connections at all"
        assert "on:connect=" not in app, (
            "the canvas checks connections on the `connect` event, which fires after the edge "
            "has already been added")

    def test_every_category_has_an_icon(self, client):
        """A category the icons do not know renders as a bare circle in the palette.

        Eight of the ten did, at one point: the icons came from the project the editor came from
        and named its categories, not mozo's.
        """
        icons = (Path(__file__).resolve().parents[2] / "ui/src/lib/CategoryIcon.svelte").read_text()
        drawn = set(re.findall(r"^    (\w+): '", icons, re.M))
        declared = {get(name).category for name in shipped()}
        assert declared <= drawn, f"these categories have no icon: {sorted(declared - drawn)}"

    def test_every_port_type_has_a_colour(self, client):
        """Handles are coloured by port type, and two of a colour is what "wireable" looks like."""
        css = EDITOR / "assets" / "editor.css"
        if not css.is_file():
            pytest.skip("the editor is not built")
        styled = set(re.findall(r"\.mz-port-([a-z]+)", css.read_text()))
        served = {port.type.value for name in shipped()
                  for port in get(name).inputs + get(name).outputs}
        assert served <= styled, f"these port types have no colour: {served - styled}"


class TestInsideAHostApplication:
    """A host mounts the router under its own prefix, with its own model loading."""

    @pytest.fixture
    def host(self):
        """A host app with the router at ``/mozo``, loading models through a recording fake."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from mozo.workflow.api import workflow_router

        asked = []

        class Model:
            def predict(self, image, **kwargs):
                return pixelflow.Detections()

        def models(family, variant, **kwargs):
            asked.append((family, variant))
            return Model()

        app = FastAPI()
        app.include_router(workflow_router(models=models, process=False), prefix="/mozo")
        return TestClient(app), asked

    def test_the_page_its_files_and_its_api_are_found_under_the_prefix(self, host):
        """Nothing is configured: the page's script resolves against the page, and the editor's
        API against the script."""
        client, _ = host
        wanted = _asked_for(client.get("/mozo/workflow").text, "/mozo/workflow")
        assert wanted == ["/mozo/workflow/assets/editor.js", "/mozo/workflow/assets/editor.css"]
        for resolved in wanted:
            assert client.get(resolved).status_code == 200, f"nothing at {resolved}"
        assert client.get("/mozo/workflow/nodes").status_code == 200
        valid = client.post("/mozo/workflow/validate", data={"workflow": json.dumps(
            {"nodes": [{"id": "a", "type": "read_media", "data": {"parameters": {}}}], "edges": []})})
        assert valid.json()["valid"] is True

    def test_the_page_works_with_and_without_a_trailing_slash_and_no_redirect(self):
        """A host may turn redirects off, and a proxy may add the slash a redirect takes away, so
        each address is served with the script path that resolves from it."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from mozo.workflow.api import workflow_router

        app = FastAPI(redirect_slashes=False)
        app.include_router(workflow_router(), prefix="/mozo")
        client = TestClient(app)
        for at in ("/mozo/workflow", "/mozo/workflow/"):
            page = client.get(at, follow_redirects=False)
            assert page.status_code == 200, at
            assert _asked_for(page.text, at)[0] == "/mozo/workflow/assets/editor.js", at

    def test_a_run_loads_its_models_the_host_s_way(self, host):
        from conftest import FIXTURE, document

        client, asked = host
        workflow = json.dumps(document(
            {"load": ("read_media", {"source": str(FIXTURE)}), "find": ("sam3", {})},
            [("load", "image", "find", "image")]))
        response = client.post("/mozo/workflow/run", data={"workflow": workflow})
        assert response.status_code == 200, response.text
        assert asked == [("sam3", "sam3")]

        streamed = client.post("/mozo/workflow/stream", data={"workflow": workflow})
        assert '"done": true' in streamed.text
        assert asked == [("sam3", "sam3")] * 2

    def test_a_host_that_leaves_out_whole_source_runs_does_not_offer_them(self, host, client):
        hosted, _ = host
        assert hosted.get("/mozo/workflow/nodes").json()["process"] is False
        assert hosted.post("/mozo/workflow/process", data={"workflow": "{}"}).status_code in (404, 405)
        assert client.get("/workflow/nodes").json()["process"] is True
