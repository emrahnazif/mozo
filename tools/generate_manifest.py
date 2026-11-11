#!/usr/bin/env python3
"""Generate ``mozo/manifest.json`` from the local ``weights/`` directory.

The directory is the source of truth. Paths, sizes, hashes, artifact keys and the ``latest``
pointer are all read off the disk, so the manifest cannot disagree with what is actually there,
and nothing in it is typed by hand.

Layout::

    weights/<family>/<variant>/<revision>/<key>.<ext>

``<revision>`` is the publish date in ISO form, so revisions sort lexically and the newest is
``latest``. Each file's *stem* is its artifact key: ``torch-fp32.pth`` publishes as ``torch-fp32``.
``LICENSE`` has no extension, so it is simply an artifact named ``LICENSE`` -- required in every
revision, and fetched alongside whatever else a caller asks for. Shipping the licence text with
the weights is the whole of the compliance story.

Each revision also states which licence that is, by name, so a caller can show or
filter on it before downloading anything. It is read off the LICENSE file like everything else
(see :data:`LICENCES`), never typed, so it cannot disagree with the text it names. A file that
matches no known licence stops the build rather than publishing a guess.

Run from the repository root::

    python tools/generate_manifest.py

Output is sorted and stably formatted, so re-running it on unchanged inputs produces a
byte-identical file. That makes "did someone forget to regenerate?" a one-line check in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE_URL = "https://dtmfiles.com/mozo/v1"

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
MANIFEST_FILE = ROOT / "mozo" / "manifest.json"

#: Present in every revision. Downloading weights without their terms is not a thing we do.
LICENCE_KEY = "LICENSE"

_HASH_CHUNK_BYTES = 1 << 20

#: How a LICENSE file's opening lines say which licence it is: its identifier, and phrases
#: that must all appear in the first :data:`_HEAD_CHARS` characters, compared case- and
#: whitespace-insensitively. The heading rather than a hash, because upstream copies of one
#: licence differ in a copyright line or a trailing newline -- mozo already publishes three
#: Apache-2.0 texts and two MIT ones -- and those are the same terms.
#:
#: Each identifier is what the text itself says, and nothing it does not: the AGPL text is
#: version 3 of the GNU Affero GPL, so it is ``AGPL-3.0``, and SAM 3's calls itself the
#: ``SAM License``. Where an SPDX identifier exists it is the one used.
LICENCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Apache-2.0", ("apache license", "version 2.0")),
    ("MIT", ("mit license", "permission is hereby granted, free of charge")),
    ("AGPL-3.0", ("gnu affero general public license", "version 3")),
    ("CC-BY-NC-4.0", ("attribution-noncommercial 4.0 international",)),
    ("SAM License", ("sam license",)),
)

_HEAD_CHARS = 400


class GenerateError(RuntimeError):
    """Raised when a revision directory is malformed."""


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of *path*, read in chunks so large checkpoints do not land in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify_licence(path: Path) -> str:
    """Return the identifier of the licence text at *path*, from its opening lines.

    Raises:
        GenerateError: If the text matches no entry in :data:`LICENCES`, or more than one.
    """
    head = " ".join(path.read_text(encoding="utf-8").split()).lower()[:_HEAD_CHARS]
    matches = [name for name, phrases in LICENCES if all(p in head for p in phrases)]
    if len(matches) != 1:
        found = f"matches {', '.join(matches)}" if matches else "matches no known licence"
        raise GenerateError(
            f"{path}: {found}. Add its heading to LICENCES in {Path(__file__).name} "
            "rather than guessing."
        )
    return matches[0]


def _artifact_files(revision_dir: Path) -> list[Path]:
    """Return the publishable files in *revision_dir*: regular files, no dotfiles, no subdirectories.

    Dotfiles are skipped rather than rejected because ``.DS_Store`` appears unbidden on macOS and
    failing the build over it would teach people to ignore the build.
    """
    return sorted(p for p in revision_dir.iterdir() if p.is_file() and not p.name.startswith("."))


def _scan_revision(revision_dir: Path, root: Path) -> dict[str, dict[str, object]]:
    """Hash every artifact in one revision directory and return them keyed by file stem.

    Raises:
        GenerateError: If the directory holds no artifacts, two files share a stem, or ``LICENSE``
            is absent.
    """
    artifacts: dict[str, dict[str, object]] = {}
    for path in _artifact_files(revision_dir):
        key = path.stem
        if key in artifacts:
            raise GenerateError(
                f"{revision_dir}: two files share the artifact key {key!r}. "
                "Each artifact must have a unique stem."
            )
        artifacts[key] = {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }

    if not artifacts:
        raise GenerateError(f"{revision_dir}: no artifacts found.")
    if LICENCE_KEY not in artifacts:
        raise GenerateError(
            f"{revision_dir}: no {LICENCE_KEY} file. Every revision ships upstream's licence "
            "alongside the weights."
        )
    _check_runtime_parts(revision_dir, artifacts)
    return artifacts


def _check_runtime_parts(revision_dir: Path, artifacts: dict[str, dict[str, object]]) -> None:
    """Refuse a runtime split across parts unless every part of it is here.

    A key is ``<framework>-<precision>`` with an optional ``-<part>``, and the third segment is
    whatever someone named the file. That makes composition a naming convention, which is fine
    until a half arrives on its own: ``mozo.runtimes.runnable`` would offer ``onnx-fp32``,
    ``auto`` could select it, and it would fail at load time on somebody else's machine. The
    manifest is generated from the tree and is where that has to be caught, for the same reason
    a missing licence is caught here rather than trusted.

    A lone third segment is the ambiguous case -- ``onnx-int8-dynamic`` is one file describing a
    flavour, not half of anything -- so it is refused too, rather than silently welded into a
    runtime with whatever else shares its prefix.

    Raises:
        GenerateError: If a runtime is composed of exactly one part.
    """
    from collections import defaultdict

    # ``LICENSE``, ``NOTICE`` and ``labels`` carry no hyphen, so they never reach the count.
    composed: dict[str, list[str]] = defaultdict(list)
    for key in artifacts:
        segments = key.split("-")
        if len(segments) > 2:
            composed["-".join(segments[:2])].append(key)

    for runtime, keys in sorted(composed.items()):
        if len(keys) == 1:
            raise GenerateError(
                f"{revision_dir}: {keys[0]!r} names a part of {runtime!r} but is the only one. "
                f"A runtime split across files needs every part present, and a single suffixed "
                f"file is indistinguishable from one. Publish the rest, or drop the suffix."
            )


def _scan_variant(variant_dir: Path, root: Path) -> dict[str, object]:
    """Scan every revision of one model variant and return its manifest entry.

    Raises:
        GenerateError: If the variant has no revision directories.
    """
    revisions = {
        d.name: {"artifacts": _scan_revision(d, root),
                 "license": identify_licence(d / LICENCE_KEY)}
        for d in sorted(p for p in variant_dir.iterdir() if p.is_dir())
    }
    if not revisions:
        raise GenerateError(f"{variant_dir}: no revision directories.")

    # ISO dates sort lexically, so the newest name is the newest revision.
    return {"latest": max(revisions), "revisions": revisions}


def build_manifest(weights_dir: Path = WEIGHTS_DIR) -> dict[str, object]:
    """Walk *weights_dir* and return the manifest as a dict."""
    if not weights_dir.is_dir():
        raise GenerateError(f"{weights_dir} does not exist.")

    models: dict[str, object] = {}
    for family_dir in sorted(p for p in weights_dir.iterdir() if p.is_dir()):
        for variant_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            models[f"{family_dir.name}/{variant_dir.name}"] = _scan_variant(variant_dir, weights_dir)

    # 2: every revision carries a ``license`` identifier beside its artifacts.
    return {"schema": 2, "base_url": BASE_URL, "models": models}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR, help="tree to scan")
    parser.add_argument("--out", type=Path, default=MANIFEST_FILE, help="manifest to write")
    args = parser.parse_args()

    try:
        manifest = build_manifest(args.weights_dir)
    except GenerateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    artifacts = sum(
        len(revision["artifacts"])
        for model in manifest["models"].values()
        for revision in model["revisions"].values()
    )
    print(f"wrote {args.out}: {len(manifest['models'])} models, {artifacts} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
