#!/usr/bin/env python3
"""Write ``docs/llms.txt``, the index an agent reads before the docs, from the pages themselves.

    python tools/docs_llms.py            # rewrite docs/llms.txt
    python tools/docs_llms.py --check    # exit 1 if it is out of date

Generated rather than written, because every line of it already exists somewhere: each entry is
a page's title and the ``summary`` from its header, and the sections are the site's menu -- the
top-level pages first, then each folder, in the order DOCS-GUIDE.md says the site sorts them. A
hand-kept copy goes stale the moment a summary is edited or a page renamed, and the agent reading
it is the one reader who cannot tell. ``tests/test_docs.py`` runs ``--check``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SITE = "https://emrah.fr/mozo/"

PREAMBLE = """\
Mozo is a Python package (`pip install mozo`, Python 3.10+). A model is named `family/variant`, \
the same in Python and over HTTP:

```python
import mozo
detections = mozo.get_model("rfdetr/medium").predict("photo.jpg")   # a PixelFlow Detections
```

```bash
mozo start
curl -X POST http://localhost:8000/predict/rfdetr/medium -F file=@photo.jpg
```

Weights download on first use, are verified by sha256 and cached in ~/.cache/mozo. Detection \
results are PixelFlow objects (https://emrah.fr/pixelflow/). Each page below is also available as \
markdown in the repository under docs/."""


def header(page: Path) -> tuple[str, dict[str, str]]:
    """A page's title and the ``key: value`` fields of the comment under it."""
    text = page.read_text(encoding="utf-8")
    fields = re.match(r"# .*\n\n<!--\n(.*?)\n-->\n", text, re.S)
    return text.split("\n", 1)[0][2:], dict(
        line.split(": ", 1) for line in (fields.group(1).split("\n") if fields else ()) if ": " in line)


def url(page: Path) -> str:
    """The page's address on the site: its path without ``.md``, ``index`` being the root."""
    path = page.relative_to(DOCS).with_suffix("")
    return SITE if path == Path("index") else f"{SITE}{path.as_posix()}/"


def render() -> str:
    """The whole file, from the pages as they are now."""
    top = [DOCS / "index.md", *sorted(p for p in DOCS.glob("*.md") if p.name != "index.md")]
    sections = [("Start", top)] + [
        (folder.name.replace("-", " ").capitalize(), sorted(folder.glob("*.md")))
        for folder in sorted(DOCS.iterdir()) if folder.is_dir() and folder.name != "assets"]
    lines = ["# Mozo", "", f"> {header(DOCS / 'index.md')[1]['summary']}", "", PREAMBLE, ""]
    for name, pages in sections:
        lines += [f"## {name}", ""]
        lines += [f"- [{title}]({url(page)}): {fields['summary']}"
                  for page in pages for title, fields in [header(page)]]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    target = DOCS / "llms.txt"
    wanted = render()
    if "--check" in sys.argv[1:]:
        if target.read_text(encoding="utf-8") != wanted:
            print(f"{target.relative_to(ROOT)} is out of date: run python tools/docs_llms.py")
            return 1
        return 0
    target.write_text(wanted, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
