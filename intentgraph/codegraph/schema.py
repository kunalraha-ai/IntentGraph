"""Codegraph schema v1: the native, versioned data model.

Node and edge vocabulary, evidence labels, and the stable-id scheme.

Id scheme (documented contract — the same scheme the code-graph bridge
documents for anchors, so memory ``graph_anchors`` written against one
generated graph keep matching later rebuilds of the same tree):

* file/module id — repository-relative path, lowercased, filename
  extension stripped, ``/`` and ``.`` flattened to ``_``:
  ``intentgraph/models.py`` → ``intentgraph_models``.
* symbol id — file id + ``_`` + the lowercased dotted qualname with dots
  flattened: class ``Feedback`` in ``intentgraph/models.py`` →
  ``intentgraph_models_feedback``; method ``MemoryStore.add_edge`` in
  ``intentgraph/store.py`` → ``intentgraph_store_memorystore_add_edge``.
* directory id — ``dir_`` + flattened path: ``intentgraph`` →
  ``dir_intentgraph``.
* repository id — the constant ``repository`` (one graph covers one root).

IDs are derived from repository-relative identity only — never absolute
machine paths, never scan order. Literal underscores in file names are not
distinguished from path separators (a ``my_file.py`` and ``my/file.py``
flatten identically); the builder detects such collisions and appends a
deterministic ``__2``-style suffix, so IDs of colliding entities can change
when the colliding set changes. Renames/moves change the flattened id by
construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SCHEMA_NAME", "SCHEMA_VERSION", "NODE_TYPES", "RELATIONS",
    "EXTRACTED", "INFERRED", "IdRegistry", "file_id", "directory_id",
    "symbol_id_part", "repository_id",
]

#: Identifies graphs produced by IntentGraph's native engine (stored in the
#: node-link ``graph`` attribute block; the bridge reports it, consumers can
#: check it). Legacy graph.json files carry no marker and are treated as
#: legacy-compatible.
SCHEMA_NAME = "intentgraph-codegraph"
SCHEMA_VERSION = 1

#: Node types. ``file`` is reserved for future parsers whose files are not
#: importable modules; Python files are their modules, so they are typed
#: ``module`` (or ``test`` for test files).
NODE_TYPES = ("repository", "directory", "file", "module", "test",
              "class", "function", "method")

#: Edge (relation) types. Structural: contains/defines/imports/tests;
#: symbol-level evidence: calls/inherits/references.
RELATIONS = ("contains", "defines", "imports", "calls", "inherits",
             "references", "tests")

#: Evidence labels — the same discipline as the memory graph: EXTRACTED means
#: statically proven from source; INFERRED means derived by a documented
#: heuristic. Unresolvable relationships never become edges.
EXTRACTED = "EXTRACTED"
INFERRED = "INFERRED"

REPOSITORY_ID = "repository"


def _flatten(relpath: str) -> str:
    """Repository-relative path → flattened id stem (lowercase, no dots)."""
    posix = relpath.replace("\\", "/")
    name = posix.rsplit("/", 1)[-1]
    if "." in name:
        posix = posix[: len(posix) - len(name)] + name.rsplit(".", 1)[0]
    return posix.lower().replace("/", "_").replace(".", "_")


def file_id(relpath: str) -> str:
    """Stable id for a source file (see module docstring)."""
    return _flatten(relpath)


def directory_id(relpath: str) -> str:
    """Stable id for a directory (``""`` never passed — repository is its own)."""
    return "dir_" + _flatten(relpath)


def symbol_id_part(qualname: str) -> str:
    """Id suffix for a symbol qualname (lowercased, dots flattened)."""
    return qualname.lower().replace(".", "_")


repository_id = REPOSITORY_ID


class IdRegistry:
    """Assigns unique node ids, suffixing rare collisions deterministically.

    The builder registers ids in a sorted, stable order (sorted relpaths),
    so which entity keeps the bare id and which gets ``__2`` is a function
    of the tree, not of scan order.
    """

    def __init__(self) -> None:
        self._taken: set[str] = set()
        self.collisions: list[dict[str, str]] = []

    def register(self, raw_id: str, owner: str) -> str:
        """Return a unique id for ``owner`` (provenance of the collision)."""
        if raw_id not in self._taken:
            self._taken.add(raw_id)
            return raw_id
        suffix = 2
        candidate = f"{raw_id}__{suffix}"
        while candidate in self._taken:
            suffix += 1
            candidate = f"{raw_id}__{suffix}"
        self._taken.add(candidate)
        self.collisions.append({"raw": raw_id, "id": candidate, "owner": owner})
        return candidate


@dataclass
class CodeEdge:
    """One evidence-backed relationship between two graph nodes."""

    source: str
    target: str
    relation: str
    confidence: str
    source_file: str = ""
    source_location: str = ""
    extraction: str = "ast"
    basis: str = ""
    locations: list[str] = field(default_factory=list)

    def merge_location(self, location: str, confidence: str) -> None:
        """Same (source, target, relation) seen again: keep every location,
        upgrade to EXTRACTED if any occurrence was proven."""
        if location and location not in self.locations:
            self.locations.append(location)
        if confidence == EXTRACTED:
            self.confidence = EXTRACTED
