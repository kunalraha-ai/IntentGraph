"""Deterministic exports: the bridge/renderer-compatible graph.json.

The node-link shape matches what the existing code-graph bridge and the
renderer already consume (``nodes``/``links`` with ``source``/``target``/
``relation``/``confidence``/``source_file``/``source_location``, an
undirected envelope, and per-link ``_src``/``_tgt`` logical-direction
markers), so a freshly built graph is immediately usable by both — plus a
``graph`` attribute block identifying the native schema and version.
Legacy graph.json files (no schema marker) remain equally valid inputs to
the bridge; nothing is silently relabeled.

Exports are byte-deterministic for a byte-identical tree: node and link
order is sorted, keys are sorted, and no timestamps or machine paths are
written. The repository node's label carries the checkout directory's
name, so two checkouts of the same tree under different directory names
differ in that one label only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from intentgraph.codegraph.build import CodeBuild
from intentgraph.codegraph.schema import SCHEMA_NAME, SCHEMA_VERSION

__all__ = ["to_node_link", "write_json", "write_graphml",
           "communities_by_directory"]

#: Node attribute keys exported, in this order for every node (stable).
_NODE_KEYS = ("label", "type", "source_file", "source_location", "language",
              "file_type", "extraction", "is_test", "entry_point",
              "parse_error", "imports_unresolved", "imports_external")

_EDGE_KEYS = ("relation", "confidence", "source_file", "source_location",
              "extraction", "basis", "locations")


def to_node_link(build: CodeBuild) -> dict[str, Any]:
    """The build's graph as a deterministic node-link dict (schema v1)."""
    G = build.graph
    languages = sorted({lang for _, d in G.nodes(data=True)
                        if isinstance((lang := d.get("language")), str)})
    nodes: list[dict[str, Any]] = []
    for nid in sorted(G.nodes):
        d = G.nodes[nid]
        entry: dict[str, Any] = {"id": nid}
        for key in _NODE_KEYS:
            if key in d:
                entry[key] = d[key]
        nodes.append(entry)

    links: list[dict[str, Any]] = []
    for u, v, d in G.edges(data=True):
        link: dict[str, Any] = {
            "source": d.get("_src", u), "target": d.get("_tgt", v),
            "_src": d.get("_src", u), "_tgt": d.get("_tgt", v),
        }
        for key in _EDGE_KEYS:
            if key in d:
                link[key] = d[key]
        links.append(link)
    links.sort(key=lambda l: (l["source"], l["target"], l["relation"],
                              l.get("source_location", "")))

    return {
        "directed": False,
        "multigraph": True,
        "graph": {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "generator": "intentgraph-codegraph",
            "root": build.stats.get("root_label", ""),
            "languages": languages,
            "nodes": len(nodes),
            "links": len(links),
        },
        "nodes": nodes,
        "links": links,
    }


def write_json(build: CodeBuild, path: str | Path) -> Path:
    """Write the deterministic node-link JSON; returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = to_node_link(build)
    out.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")), encoding="utf-8")
    return out


def write_graphml(build: CodeBuild, path: str | Path) -> Path:
    """Write GraphML (scalar attributes only; list attributes are dropped and
    parallel edges between the same pair collapse to one, keeping the first
    in sorted link order)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    H = nx.Graph()
    data = to_node_link(build)
    for node in data["nodes"]:
        attrs = {k: v for k, v in node.items()
                 if isinstance(v, (str, int, float, bool)) and k != "id"}
        H.add_node(node["id"], **attrs)
    for link in data["links"]:
        attrs = {k: v for k, v in link.items()
                 if isinstance(v, (str, int, float, bool))
                 and k not in ("source", "target", "_src", "_tgt")}
        H.add_edge(link["source"], link["target"], **attrs)
    nx.write_graphml(H, out)
    return out


def communities_by_directory(G: nx.Graph
                             ) -> tuple[dict[int, list[str]], dict[int, str]]:
    """Deterministic renderer communities: group nodes by the top-level
    directory of the file that owns them (root files group together)."""
    groups: dict[str, list[str]] = {}
    for nid in sorted(G.nodes):
        d = G.nodes[nid]
        ntype = d.get("type", "")
        owner = d.get("source_file") or ""
        if ntype in ("module", "test", "class", "function", "method") and owner:
            top = owner.split("/")[0] if "/" in owner else "(root files)"
        else:
            top = "(structure)"
        groups.setdefault(top, []).append(nid)
    names = sorted(groups)
    communities = {i: groups[name] for i, name in enumerate(names)}
    labels = {i: name for i, name in enumerate(names)}
    return communities, labels
