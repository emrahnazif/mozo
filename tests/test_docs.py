"""The docs in ``docs/`` held to the site's rules, and to the code they describe.

``docs/`` is published at https://emrah.fr/mozo/ as it is: the file names, folders and each page's
header decide the URLs, the menu and the titles, so a page that breaks one of ``DOCS-GUIDE.md``'s
rules breaks the site, not only the page. Those rules are mechanical, so they are checked here
rather than remembered.

The second half is what a reader cannot see is missing. A family with no page, a public function
with no reference entry, a route the reference never mentions: none of these fail anywhere else,
and each is the kind of gap that stays open because nothing reports it. So every name the code
exports, every route the server documents and every command the CLI has must have an entry, and a
family added without a page fails here rather than shipping undocumented.
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

import mozo
import mozo.weights
import mozo.workflow
from conftest import ROOT
from mozo.registry import MODEL_REGISTRY

DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md"))
SITE = "https://emrah.fr/mozo/"

#: A fenced code block, opened and closed by the same run of backticks at the same indent; group 3
#: is the language named after the opening fence.
FENCE = re.compile(r"^(\s*)(`{3,})(\S*).*?^\1\2\s*$", re.S | re.M)

each_page = pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(DOCS)))


@cache
def text(page: Path) -> str:
    return page.read_text(encoding="utf-8")


def prose(page: Path) -> str:
    """*page* without its code blocks and inline code, where links and headings are not links
    and headings."""
    return re.sub(r"`[^`\n]*`", "", FENCE.sub("", text(page)))


@cache
def anchors(page: Path) -> frozenset[str]:
    """The anchors the site gives *page*'s headings: lower case, spaces to ``-``, everything but
    letters, digits, ``-`` and ``_`` dropped, and ``-1``, ``-2`` on repeats."""
    seen: dict[str, int] = {}
    found = set()
    for heading in re.findall(r"^#{1,6} (.+)$", FENCE.sub("", text(page)), re.M):
        slug = re.sub(r"[^a-z0-9_-]", "", heading.strip().lower().replace(" ", "-"))
        count = seen.get(slug, 0)
        found.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return frozenset(found)


def header(page: Path) -> dict[str, str]:
    """The ``key: value`` fields of the comment after a page's title, or ``{}`` without one."""
    found = re.match(r"# .*\n\n<!--\n(.*?)\n-->\n", text(page), re.S)
    if not found:
        return {}
    return dict(line.split(": ", 1) for line in found.group(1).split("\n") if ": " in line)


@cache
def entries() -> frozenset[str]:
    """What ``docs/reference/`` has an entry for: every heading, markdown escapes removed, and the
    name in the first column of every table row that starts with one in backticks."""
    found = set()
    for page in sorted((DOCS / "reference").glob("*.md")):
        body = FENCE.sub("", text(page))
        found |= {line.lstrip("#").strip().replace("\\", "")
                  for line in body.split("\n") if line.startswith("##")}
        found |= set(re.findall(r"^\| `([^`]+)` \|", body, re.M))
    return frozenset(found)


def test_docs_holds_only_pages_and_assets():
    paths = [path.relative_to(DOCS) for path in DOCS.rglob("*")]
    outside = [path for path in paths if path.parts[0] != "assets"]
    strays = [path for path in outside if (DOCS / path).is_file()
              and path.suffix != ".md" and path != Path("llms.txt")]
    nested = [path for path in outside if (DOCS / path).is_dir() and len(path.parts) > 1]
    assert not strays, f"not a page and not in docs/assets/: {strays}"
    assert not nested, f"one level of folders only: {nested}"
    assert (DOCS / "index.md").is_file()


@each_page
def test_page_follows_the_site_rules(page: Path):
    title = text(page).split("\n", 1)[0]
    fields = header(page)
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*\.md", page.name), "lower case, words split by -"
    assert title.startswith("# ") and len(title) - 2 <= 55, f"title: {title!r}"
    assert fields.get("summary") and len(fields["summary"]) < 160, "summary under 160 characters"
    assert fields.get("label") and len(fields["label"].split()) <= 3, "label of 1-3 words"
    assert fields.get("keywords", "").startswith("mozo"), "keywords start with the library name"
    assert (fields.get("template") == "docs") == (page == DOCS / "index.md"), \
        "template: docs belongs on index.md and nowhere else"
    assert not re.search(r"^# ", FENCE.sub("", text(page)).split("\n", 1)[1], re.M), \
        "one # heading, the first line"
    assert all(block.group(3) for block in FENCE.finditer(text(page))), \
        "every code block names its language"


@each_page
def test_links_and_images_resolve(page: Path):
    for target in re.findall(r"\]\(([^)\s]+)\)", prose(page)):
        if target.startswith(("http://", "https://", "mailto:")):
            assert not target.startswith(SITE), \
                f"a page of this library is linked by its /docs/ path, not its URL: {target}"
            continue
        file, _, anchor = target.partition("#")
        assert not file or file.startswith("/docs/"), f"links start with /docs/: {target}"
        linked = ROOT / file.lstrip("/") if file else page
        assert linked.exists(), f"links to a missing file: {target}"
        assert not anchor or anchor in anchors(linked), f"links to a missing section: {target}"
    assert "![]" not in prose(page), "every image has a description"


def test_every_asset_is_used():
    used = "".join(text(page) for page in PAGES)
    unused = [asset.name for asset in (DOCS / "assets").glob("*") if asset.name not in used]
    assert not unused, f"docs/assets/ files no page shows: {unused}"


def test_llms_txt_is_generated_from_the_pages():
    """``docs/llms.txt`` is the index an agent reads first, built from every page's title and
    summary by ``tools/docs_llms.py``. Comparing what it would write with what is committed catches
    a new page, a renamed one, a deleted one and an edited summary alike."""
    check = subprocess.run([sys.executable, str(ROOT / "tools" / "docs_llms.py"), "--check"],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stdout


@pytest.mark.parametrize("family", sorted(MODEL_REGISTRY))
def test_every_family_has_a_page(family: str):
    page = f"{family.replace('_', '-')}.md"
    assert (DOCS / "models" / page).is_file(), f"docs/models/{page} is missing"
    assert f"(/docs/models/{page})" in text(DOCS / "models" / "all-models.md"), \
        f"docs/models/all-models.md does not list {family}"


def routes() -> list[str]:
    """``METHOD /path`` for every route in the server's OpenAPI document: what it declares
    public, and none of what it marks as its own plumbing."""
    from mozo.server import app

    return sorted(f"{method.upper()} {path}"
                  for path, operations in app.openapi()["paths"].items() for method in operations)


def commands() -> list[str]:
    from mozo.cli import cli

    return [f"mozo {command}" for command in cli.commands]


@pytest.mark.parametrize("name", [
    *mozo.__all__, "load_image", "encode_image", *mozo.weights.__all__, *mozo.workflow.__all__,
    *routes(), *commands()])
def test_every_public_name_route_and_command_has_an_entry(name: str):
    assert name in entries(), f"docs/reference/ has no entry for {name}"
