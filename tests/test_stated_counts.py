"""Every number mozo states about itself, held against the manifest that decides it.

The package docstring said "Seventeen published variants across two families" for eight families
longer than it was true, and the README said 52 while the manifest said 54. Both are the first
thing a reader sees -- ``help(mozo)`` opens with one and PyPI opens with the other -- and both
were wrong because nothing checked them.

Prose cannot be generated: a module docstring has to be a literal for ``help()`` and doctest, and
a README is written for people. So the numbers stay written by hand and this file is what stops
them drifting, exactly as ``tests/families/test_*.py::test_registry_agrees_with_the_adapter``
stops the catalogue drifting from the adapters.

The manifest is the authority because it is generated from the weights tree
(``tools/generate_manifest.py``) rather than typed, so it cannot itself be the thing that is
stale.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

import mozo
from conftest import GENERATOR
from mozo.registry import MODEL_REGISTRY
from mozo.weights import license_of, manifest

ROOT = Path(__file__).resolve().parent.parent

#: The irregular part of writing a number out. Everything from twenty up is composed from these.
_UNITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
          "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def spell(number: int) -> str:
    """Write *number* out in words, the way prose does.

    Derived rather than tabulated because the alternative -- a dict of the counts mozo happens to
    have today -- is the same hand-maintained number this file exists to stop drifting, one level
    up. It covers 0-99, which outlasts any catalogue anyone will read in one sitting.
    """
    if not 0 <= number < 100:
        raise ValueError(f"{number} is past where a docstring should be spelling numbers out")
    if number < 20:
        return _UNITS[number]
    tens, unit = divmod(number, 10)
    return _TENS[tens] + (f"-{_UNITS[unit]}" if unit else "")


def published_variants() -> int:
    """How many variants the manifest publishes."""
    return len(manifest()["models"])


def published_families() -> int:
    """How many distinct families those variants belong to."""
    return len({model_id.split("/")[0] for model_id in manifest()["models"]})


def test_the_package_docstring_states_the_published_counts():
    """``help(mozo)`` opens with these two numbers, so they are the ones read most and checked
    least."""
    text = mozo.__doc__
    variants, families = published_variants(), published_families()

    assert f"{spell(variants).capitalize()} published variants" in text, (
        f"the package docstring does not open with {spell(variants)} published variants:"
        f"\n{text[:200]}"
    )
    assert f"{spell(families)} families" in text, (
        f"the package docstring does not say {spell(families)} families"
    )


def test_the_readme_headline_states_the_published_count():
    """The first line of the README is the first thing anyone reads about mozo."""
    readme = (ROOT / "README.md").read_text()
    variants = published_variants()

    headline = re.search(r"^### (\d+) computer vision models", readme, re.M)
    assert headline, "the README no longer opens with a model count"
    assert int(headline.group(1)) == variants, (
        f"README headline says {headline.group(1)}, manifest publishes {variants}"
    )


#: Where the README states the model count, as the words either side of the number. Tight on
#: purpose: matching a bare "the NN" would fire on "the 16 tokens" in the OWLv2 row and fail the
#: suite for an unrelated edit, which is worse than not checking -- a test nobody trusts gets
#: deleted. Reword the README and this list is what you update.
COUNT_CONTEXTS = (
    r"### (\d+) computer vision models",
    r"runs all (\d+)",
    r"# all (\d+), no torch import",
    r"a curated (\d+)",
    r"Of the (\d+) published variants",
)


def test_the_readme_repeats_that_count_consistently():
    """It appears four more times after the headline. One stale copy contradicts the page."""
    readme = (ROOT / "README.md").read_text()
    variants = published_variants()

    for pattern in COUNT_CONTEXTS:
        found = re.findall(pattern, readme)
        assert found, f"the README no longer contains {pattern!r} -- update COUNT_CONTEXTS"
        for number in found:
            assert int(number) == variants, (
                f"README says {number} in {pattern!r}, manifest publishes {variants}"
            )


def test_the_registry_and_the_manifest_agree_on_the_families():
    """A family in the catalogue that publishes nothing is fine -- it takes your own checkpoint.
    A family in the manifest that the catalogue does not know is not reachable at all."""
    catalogued = set(MODEL_REGISTRY)
    published = {model_id.split("/")[0] for model_id in manifest()["models"]}
    assert published <= catalogued, f"published but not in the registry: {published - catalogued}"


def test_every_published_variant_is_one_the_registry_names():
    """The manifest is generated from directory names; a typo there publishes bytes under a
    variant no adapter will ever ask for."""
    for model_id in manifest()["models"]:
        family, variant = model_id.split("/", 1)
        known = MODEL_REGISTRY[family]["variants"]
        assert not known or variant in known, f"{model_id} is published but {family} lists {known}"


def test_every_registered_variant_is_one_the_manifest_publishes():
    """The other direction, and the one that reaches a user.

    A variant the manifest does not carry is a name the catalogue offers and no request can
    satisfy: ``/models`` lists it, the test page puts it in the dropdown, and picking it fails on
    a download that was never going to find anything. SigLIP 2 shipped that way -- fifteen
    registered against three published -- and the symptom was a 500 rather than a red test.

    An empty ``variants`` list means the family accepts any name and publishes nothing of its own,
    which ``registry.py`` documents; those are skipped rather than failed.
    """
    models = manifest()["models"]
    for family, entry in MODEL_REGISTRY.items():
        missing = [v for v in entry["variants"] if f"{family}/{v}" not in models]
        assert not missing, f"{family} registers {missing} but the manifest publishes no weights"


def test_each_family_counts_its_own_variants_correctly():
    """Eight descriptions open by counting their variants, and ``/models`` serves that text.

    The count is written by hand next to the list it counts, so it is the one number in the
    catalogue that can contradict the entry carrying it.
    """
    for family, entry in MODEL_REGISTRY.items():
        stated = re.search(r"(\d+) variants?\b", entry["description"])
        if not stated or not entry["variants"]:
            continue
        assert int(stated.group(1)) == len(entry["variants"]), (
            f"{family} says {stated.group(1)} variants and lists {len(entry['variants'])}"
        )


#: The licences under which the README's MOZO_ENABLE example may serve weights. Only the README's
#: claim needs this word; mozo itself reports identifiers and leaves the judgement to its caller.
PERMISSIVE = frozenset({"Apache-2.0", "MIT"})


def licences() -> dict[tuple[str, str], str]:
    """Every published variant's licence, as its latest revision states it."""
    return {(family, variant): license_of(family, variant)
            for family, variant in (model_id.split("/", 1) for model_id in manifest()["models"])}


def test_the_licence_breakdown_states_the_manifest_s_counts():
    """The README splits the published variants by licence, and every number in it is checked.

    Summing was all that could be checked while the manifest recorded only a LICENSE file's hash,
    and a sum misses two numbers moving in opposite directions. Now each count is the manifest's.
    """
    readme = (ROOT / "README.md").read_text()
    sentence = re.search(r"Of the \d+ published variants.*?SAM License", readme, re.S)
    assert sentence, "the README no longer states a licence breakdown -- update this test"
    plain = " ".join(sentence.group(0).replace("*", "").split())
    stated = {name: int(n) for n, name in
              re.findall(r"(\d+) (?:are|carries) (?:Meta's )?(SAM License|[\w.-]+)", plain)}
    assert stated == dict(Counter(licences().values()))


def _licence_tables(readme: str):
    """Yield ``(family, variant tokens, stated licence)`` for each row of a "Weights licence" table.

    A blank family cell continues the row above, as Depth Anything's three rows do, and a token
    ending in a hyphen (``indoor-``) stands for every variant it begins.
    """
    column = family = None
    for line in readme.splitlines():
        if not line.startswith("|"):
            column = None
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if "Weights licence" in cells:
            column = cells.index("Weights licence")
            continue
        if column is None or set(line) <= set("|-"):
            continue
        family = cells[0].strip("`") or family
        yield family, re.findall(r"`([^`]+)`", cells[1]), cells[column]


def test_every_readme_table_states_each_variant_s_manifest_licence():
    """The tables are where a reader picks a model, so a wrong licence there is the costly one."""
    published = licences()
    seen, wrong = set(), []
    for family, tokens, stated in _licence_tables((ROOT / "README.md").read_text()):
        for token in tokens:
            matched = [variant for fam, variant in published
                       if fam == family and (variant == token
                                             or token.endswith("-") and variant.startswith(token))]
            assert matched, f"the README lists {family}/{token}, which mozo does not publish"
            for variant in matched:
                seen.add((family, variant))
                expected = published[family, variant]
                if stated != expected:
                    wrong.append(f"{family}/{variant}: README says {stated}, manifest {expected}")
    assert not wrong, "\n".join(wrong)
    assert seen == set(published), f"no licence stated for {sorted(set(published) - seen)}"


def test_the_readme_states_how_many_workflow_nodes_there_are():
    """The workflow section opens with the node count, and a new node is easy to forget there."""
    from mozo.workflow.registry import get, names

    # Only mozo's own: other tests register nodes of their own into the same registry.
    shipped = [n for n in names() if get(n).run.__module__.startswith("mozo.workflow.nodes")]
    readme = (ROOT / "README.md").read_text()
    stated = re.search(r"^(\d+) nodes: every model family", readme, re.M)
    assert stated, "the README no longer states the node count -- update this test"
    assert int(stated.group(1)) == len(shipped)


def test_the_readme_states_how_many_families_ship_a_verification_gate():
    """`tools/verify/` is the claim the whole README rests on, so its arithmetic is checked."""
    gates = {
        path.stem for path in (ROOT / "tools" / "verify").glob("*.py")
        if not path.stem.startswith("_") and not path.stem.endswith("_reference")
    }
    families = published_families()
    readme = (ROOT / "README.md").read_text()

    assert gates <= set(MODEL_REGISTRY), f"gates for unknown families: {gates - set(MODEL_REGISTRY)}"
    assert f"{spell(len(gates)).capitalize()} of the {spell(families)} families ship one" in readme, (
        f"{len(gates)} of {families} families have a gate; the README says otherwise"
    )


def non_permissive() -> dict[tuple[str, str], str]:
    """Every published variant whose weights are not permissively licensed, from the manifest.

    Per variant, not per family: Depth Anything is seven Apache-2.0 variants and two that are not,
    so a list of family names would only notice a non-permissive *family* being added.
    """
    return {model: name for model, name in licences().items() if name not in PERMISSIVE}


def test_one_licence_file_is_one_licence():
    """Revisions that ship the same LICENSE bytes must name the same licence.

    The identifier is derived from the file when the manifest is generated; this holds the
    published manifest to that without needing the files on disk.
    """
    named: dict[str, set[str]] = {}
    for entry in manifest()["models"].values():
        for revision in entry["revisions"].values():
            named.setdefault(revision["artifacts"]["LICENSE"]["sha256"], set()).add(revision["license"])
    assert all(len(ids) == 1 for ids in named.values()), named


def test_the_licence_files_on_this_machine_say_what_the_manifest_says():
    """Where the weights tree is present, re-read every LICENSE file and compare."""
    import importlib.util

    weights = ROOT / "weights"
    if not weights.is_dir():
        pytest.skip("no weights/ tree here; test_one_licence_file_is_one_licence still holds")
    spec = importlib.util.spec_from_file_location("generate_manifest", GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    for model_id, entry in manifest()["models"].items():
        for name, revision in entry["revisions"].items():
            path = weights / revision["artifacts"]["LICENSE"]["path"]
            if path.is_file():
                assert generator.identify_licence(path) == revision["license"], f"{model_id}@{name}"


def test_the_readme_s_permissive_deployment_offers_what_it_claims(deploy):
    """The README shows a ``MOZO_ENABLE`` line for serving only the permissively licensed weights,
    and says how many that is. It is the one example whose being wrong is not a documentation bug.

    An operator pastes it to avoid taking on AGPL section 13 obligations, so a variant that slips
    through is served under a licence they were reading this section specifically to decline. And
    it is exactly the kind of list that rots: adding a sixth YOLO size would leave the line still
    looking right while quietly offering it.

    Three independent things are checked, because each can be wrong on its own -- the count the
    prose states, that nothing non-permissive is offered, and that nothing permissive was dropped
    by accident along the way.
    """
    readme = (ROOT / "README.md").read_text()
    example = re.search(r"```bash\nMOZO_ENABLE=(.*?)\n```", readme, re.S)
    assert example, "the README no longer shows a MOZO_ENABLE example -- update this test"
    stated = re.search(r"That is the (\d+) Apache-2\.0 and MIT variants", readme)
    assert stated, "the README no longer states what that line offers -- update this test"

    deployment = deploy(example.group(1).replace("\\\n", ""))
    offered = {(family, v) for family, variants in deployment.items() for v in variants}

    assert len(offered) == int(stated.group(1)), (
        f"the README's MOZO_ENABLE line offers {len(offered)} variants, and says {stated.group(1)}"
    )

    restricted = non_permissive()
    leaked = {model: restricted[model] for model in offered & set(restricted)}
    assert not leaked, f"the README's MOZO_ENABLE line offers non-permissive weights: {leaked}"

    everything = {(family, v) for family, entry in MODEL_REGISTRY.items()
                  for v in entry["variants"]}
    assert everything - offered == set(restricted), (
        f"it also leaves out {sorted(everything - offered - set(restricted))}, which are permissive"
    )


#: What mozo installs. The workflow runtime was absorbed from a separate project that carried
#: flask, flask-cors, requests and tqdm; none of them came, and the whole merge added nothing. That
#: was the constraint the work was done under, so it is stated here rather than left as intent --
#: it is the invariant most easily broken by an ordinary-looking commit.
REQUIRED = {
    "fastapi", "uvicorn", "opencv-python-headless", "torch", "Pillow", "numpy", "pixelflow",
    "click", "python-multipart", "regex", "ftfy", "torchvision",
}


def test_mozo_installs_exactly_what_it_says_it_does():
    # Read rather than parsed with tomllib, which is 3.11+ where mozo supports 3.9 -- and adding
    # tomli to read it would break the very thing this test exists to hold.
    block = re.search(r"^dependencies = \[(.*?)^\]", (ROOT / "pyproject.toml").read_text(),
                      re.S | re.M).group(1)
    named = {re.match(r'"([A-Za-z0-9_.-]+)', line.strip()).group(1)
             for line in block.splitlines()
             if line.strip().startswith('"')}
    assert named == REQUIRED, (
        f"dependencies changed: added {named - REQUIRED}, removed {REQUIRED - named}. "
        f"If that is deliberate, say so here.")
