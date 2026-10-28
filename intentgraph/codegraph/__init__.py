"""IntentGraph's native codebase-graph engine (independently written).

A deterministic, evidence-backed code graph for a repository, built from
first principles with the standard library (Python's ``ast`` parser) and
networkx — no external code-graph engine is installed, imported, or used.

Pipeline: ``scan → parse → index symbols → resolve references → build graph
→ analyze → export``.

Supported languages: **Python only** (stdlib ``ast``). The parser interface
(``python_parser.LanguageParser`` / ``PARSERS``) is the extension point for
more; nothing else in the engine is language-specific.

The export is the node-link ``graph.json`` the code-graph bridge and the
renderer already consume, identified by a ``graph`` attribute block
(``intentgraph-codegraph`` schema v1) so consumers can tell native files
from legacy ones. See ``docs/intentgraph.md`` and ``ARCHITECTURE.md``.
"""
from __future__ import annotations

from intentgraph.codegraph.schema import SCHEMA_NAME, SCHEMA_VERSION

# Public surface (lazily resolved via __getattr__): build_graph, BuildOptions,
# CodeBuild, scan_repository, ScanOptions, load_graph — plus the schema
# constants exported eagerly above.


def __getattr__(name: str):
    # Lazy imports: `import intentgraph.codegraph` stays cheap and does not
    # pull networkx-heavy modules until they are actually used.
    _map = {
        "build_graph": ("intentgraph.codegraph.build", "build_graph"),
        "BuildOptions": ("intentgraph.codegraph.build", "BuildOptions"),
        "CodeBuild": ("intentgraph.codegraph.build", "CodeBuild"),
        "scan_repository": ("intentgraph.codegraph.scan", "scan_repository"),
        "ScanOptions": ("intentgraph.codegraph.scan", "ScanOptions"),
        "load_graph": ("intentgraph.codegraph.analyze", "load_graph"),
    }
    if name in _map:
        import importlib

        mod_name, attr = _map[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module 'intentgraph.codegraph' has no "
                         f"attribute {name!r}")
