"""ARCHITECTURE.md's module table must name symbols that actually exist.

The "IntentGraph layer" table is the entry point for library users, and
AGENTS.md points agents at the docs before the code. A table row that names
a symbol that does not exist (or a renamed entry point the table still
shows under its old name) sends every reader to code that raises.

These tests parse the table itself rather than restating it, so adding a row
extends the coverage automatically and renaming a function in the code fails
here until the doc is updated too.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

_ARCHITECTURE = Path(__file__).parent.parent / "ARCHITECTURE.md"

# A table row: | `<module>.py` | <entry points> | <input → output> |
_ROW = re.compile(r"^\|\s*`(\w+)\.py`\s*\|(.*?)\|", re.MULTILINE)
# A function reference inside a cell: `name(...)` or a bare `name`.
_FUNC = re.compile(r"`([a-z_][a-z0-9_]*)(?:\(|`)")


def _documented_symbols() -> list[tuple[str, str]]:
    """(module, function) for every lowercase symbol named in the table."""
    text = _ARCHITECTURE.read_text(encoding="utf-8")
    start = text.index("## IntentGraph layer")
    # a section can be the last one in the file: no following "##" exists
    try:
        end = text.index("##", start + 3)
    except ValueError:
        end = len(text)
    pairs: list[tuple[str, str]] = []
    for module, cell in _ROW.findall(text[start:end]):
        for func in _FUNC.findall(cell):
            pairs.append((f"intentgraph.{module}", func))
    return pairs


_SYMBOLS = _documented_symbols()


def test_the_table_was_actually_parsed():
    """Guard the parser itself: a regex that silently matches nothing would make
    every parametrized test below vacuous."""
    assert len(_SYMBOLS) >= 30, (
        f"parsed too few symbols, regex likely broken: {_SYMBOLS}")
    modules = {m for m, _ in _SYMBOLS}
    assert {"intentgraph.ingest", "intentgraph.retrieve",
            "intentgraph.serve"} <= modules, modules
    # the table must cover the whole package, not a stale subset
    assert len(modules) >= 18, modules


@pytest.mark.parametrize("module,func", _SYMBOLS)
def test_architecture_table_symbols_exist(module, func):
    mod = importlib.import_module(module)
    assert hasattr(mod, func), (
        f"ARCHITECTURE.md documents {module}.{func}, which does not exist. "
        f"Update the table (or re-export the symbol)."
    )
