"""The registry's per-family tables agree with each other and name only real families.

A duplicate key in a dict literal is legal Python: the last one silently wins. Merging two
families' branches left exactly that in ``DERIVES`` once -- a stale entry offering a runtime the
family had stopped running -- and only that family's own tests noticed. These checks catch the
class of mistake for every family, without loading a model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mozo import registry
from mozo.weights import framework_of

SOURCE = Path(registry.__file__)
TABLES = ("EXECUTES", "DERIVES", "PREFERS", "NEEDS", "ENCODES")


def _written_keys(node: ast.Dict) -> list[str]:
    """The keys a dict literal spells out, including those of a ``**{f: ... for f in (...)}``."""
    keys = []
    for key, value in zip(node.keys, node.values):
        if key is not None:
            keys.append(ast.literal_eval(key))
        elif isinstance(value, ast.DictComp):
            [generator] = value.generators
            keys.extend(ast.literal_eval(generator.iter))
    return keys


@pytest.mark.parametrize("table", TABLES)
def test_no_family_is_written_twice(table):
    tree = ast.parse(SOURCE.read_text())
    [node] = [n.value for n in tree.body
              if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == table]
    keys = _written_keys(node)
    assert len(keys) == len(set(keys)), sorted(k for k in set(keys) if keys.count(k) > 1)


@pytest.mark.parametrize("table", TABLES)
def test_every_key_is_a_registered_family(table):
    assert set(getattr(registry, table)) <= set(registry.MODEL_REGISTRY)


@pytest.mark.parametrize("family", sorted(set(registry.DERIVES) | set(registry.PREFERS)))
def test_what_a_family_derives_or_prefers_is_something_it_executes(family):
    """A family restricted by ``EXECUTES`` cannot derive or prefer a runtime outside it."""
    executes = registry.EXECUTES.get(family)
    if executes is None:
        return
    wanted = set(registry.DERIVES.get(family, {}))
    wanted |= {key for keys in registry.PREFERS.get(family, {}).values() for key in keys}
    assert {framework_of(key) for key in wanted} <= set(executes), family
