"""Analysis over a built (or saved) code graph: stats, cycles, hubs, orphan
candidates, paths, bounded neighborhoods, and components.

All operations are deterministic and bounded: neighbor sets are sorted,
cycle enumeration is capped, and traversal depth/fan-out/caps are explicit
(so a pathological repository degrades, it does not hang).

``load_graph`` accepts both native schema files (``graph`` block carries
``intentgraph-codegraph``) and legacy graph.json files with no marker. A
native file whose schema *version* is newer than this build understands is
refused with a clear error — never silently reinterpreted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph as _json_graph

from intentgraph.codegraph.schema import SCHEMA_NAME, SCHEMA_VERSION

__all__ = ["GraphFileError", "load_graph", "graph_stats", "import_cycles",
           "hubs", "orphan_candidates", "find_paths", "neighborhood",
           "components", "node_report"]

_MAX_CYCLES_COLLECTED = 200
_MAX_REPORTED = 50
_MAX_PATHS_COLLECTED = 200


class GraphFileError(Exception):
    """A graph file exists but cannot be used as-is."""


def load_graph(path: str | Path
               ) -> tuple[nx.Graph, nx.DiGraph | None, dict[str, Any]]:
    """Load a graph.json → (undirected view, directed view or None, meta).

    The undirected view matches what the code-graph bridge traverses; the
    directed view (rebuilt from ``_src``/``_tgt`` markers) drives direction-
    aware analyses like import cycles.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"graph file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "nodes" not in data:
        raise GraphFileError(f"not a node-link graph file: {p}")
    meta = data.get("graph") or {}
    schema = meta.get("schema")
    version = meta.get("schema_version")
    if schema == SCHEMA_NAME and isinstance(version, int) \
            and version > SCHEMA_VERSION:
        raise GraphFileError(
            f"graph schema {schema} v{version} is newer than this build "
            f"supports (v{SCHEMA_VERSION}): rebuild or upgrade IntentGraph")
    links = data.get("links") or data.get("edges") or []
    normalized = dict(data, directed=False, links=[
        {**link, "_src": link.get("_src", link.get("source")),
         "_tgt": link.get("_tgt", link.get("target"))}
        for link in links])
    try:
        G = _json_graph.node_link_graph(normalized, edges="links")
    except TypeError:  # pragma: no cover - older networkx signatures
        G = _json_graph.node_link_graph(normalized)
    if G.is_directed():
        H = nx.MultiGraph() if G.is_multigraph() else nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            H.add_edge(u, v, **d)
        G = H
    D: nx.MultiDiGraph | None = None
    if all(link.get("_src") and link.get("_tgt") for link in links):
        D = nx.MultiDiGraph()
        D.add_nodes_from(G.nodes(data=True))
        for link in links:
            if link["_src"] != link["_tgt"]:
                D.add_edge(link["_src"], link["_tgt"],
                           key=link.get("relation", ""), **{
                               k: v for k, v in link.items()
                               if k not in ("source", "target", "_src",
                                            "_tgt")})
    return G, D, dict(meta)


def _relations_between(G: nx.Graph, u: str, v: str) -> str:
    """First relation (sorted) among the edges between u and v."""
    data = G.get_edge_data(u, v)
    if not data:
        return "?"
    if G.is_multigraph():
        rels = sorted({d.get("relation", "?") for d in data.values()})
    else:
        rels = sorted({data.get("relation", "?")})
    return rels[0]


# -- analyses ---------------------------------------------------------------------


def graph_stats(G: nx.Graph, D: nx.DiGraph | None = None,
                meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deterministic summary of a code graph."""
    by_type: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        by_type[d.get("type", "?")] = by_type.get(d.get("type", "?"), 0) + 1
    by_relation: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        by_relation[d.get("relation", "?")] = \
            by_relation.get(d.get("relation", "?"), 0) + 1
    comp = components(G)
    cycles = import_cycles(D if D is not None else G)
    orphans = orphan_candidates(G, limit=None)   # exact count, not a cap
    out: dict[str, Any] = {
        "schema": (meta or {}).get("schema"),
        "schema_version": (meta or {}).get("schema_version"),
        "root": (meta or {}).get("root"),
        "languages": (meta or {}).get("languages", []),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "nodes_by_type": dict(sorted(by_type.items())),
        "edges_by_relation": dict(sorted(by_relation.items())),
        "components": comp,
        "import_cycles": len(cycles["cycles"]),
        "import_cycles_truncated": cycles["truncated"],
        "hubs": [dict(h) for h in hubs(G, top=5)],
        "orphan_candidates": len(orphans),
        "orphan_candidate_examples": [o["id"] for o in orphans[:10]],
    }
    return out


def _module_subgraph(D: nx.DiGraph) -> nx.DiGraph:
    M = nx.DiGraph()
    for nid in sorted(D.nodes):
        if D.nodes[nid].get("type") in ("module", "test"):
            M.add_node(nid)
    for u, v, d in D.edges(data=True):
        if d.get("relation") == "imports" and u in M and v in M:
            M.add_edge(u, v)
    return M


def import_cycles(D_or_G: nx.DiGraph | nx.Graph, max_cycles: int = _MAX_REPORTED
                  ) -> dict[str, Any]:
    """Import dependency cycles among modules (bounded, deterministic)."""
    D = D_or_G if isinstance(D_or_G, nx.DiGraph) else nx.DiGraph()
    if not isinstance(D_or_G, nx.DiGraph):
        for u, v, d in D_or_G.edges(data=True):
            if d.get("relation") == "imports" and u != v:
                D.add_edge(u, v)
    M = _module_subgraph(D) if M_check(D) else D
    if M.number_of_edges() == 0:
        return {"cycles": [], "truncated": False}
    collected: list[list[str]] = []
    truncated = False
    try:
        for cycle in nx.simple_cycles(M):
            collected.append(sorted(cycle))
            if len(collected) >= _MAX_CYCLES_COLLECTED:
                truncated = True
                break
    except Exception:  # noqa: BLE001 — analysis must never crash a command
        return {"cycles": sorted(collected)[:max_cycles],
                "truncated": True}
    collected = sorted(collected)
    truncated = truncated or len(collected) > max_cycles
    return {"cycles": collected[:max_cycles], "truncated": truncated}


def M_check(D: nx.DiGraph) -> bool:  # noqa: N802 — helper predicate
    """Cycle over the module-only imports subgraph when node types exist."""
    return any(D.nodes[n].get("type") in ("module", "test") for n in D.nodes)


def hubs(G: nx.Graph, top: int = 10) -> list[dict[str, Any]]:
    """Highest-degree code nodes (degree over the undirected view)."""
    entries = []
    for nid, d in G.nodes(data=True):
        if d.get("type") in ("repository", "directory"):
            continue
        entries.append({"id": nid, "label": d.get("label", nid),
                        "type": d.get("type", "?"),
                        "degree": G.degree(nid)})
    entries.sort(key=lambda e: (-e["degree"], e["id"]))
    return entries[:top]


def orphan_candidates(G: nx.Graph, limit: int | None = _MAX_REPORTED
                      ) -> list[dict[str, Any]]:
    """Symbols nothing points at — candidates for review, not proven dead
    code (dynamic dispatch and framework discovery are invisible to static
    analysis, so these are labelled candidates).

    Direction comes from the edge ``_src``/``_tgt`` markers (present on
    both freshly built and loaded graphs); without them — a legacy
    undirected file with no direction info — any touch disqualifies.

    Dunder methods (protocol-invoked) and test symbols (framework-discovered
    by pytest) are excluded by policy.
    """
    incoming: set[str] = set()
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "defines":
            continue
        tgt = d.get("_tgt")
        if tgt:
            incoming.add(tgt)
        else:  # no direction info available: any touch counts
            incoming.add(u)
            incoming.add(v)
    out: list[dict[str, Any]] = []
    for nid in sorted(G.nodes):
        d = G.nodes[nid]
        if d.get("type") not in ("class", "function", "method"):
            continue
        if nid in incoming:
            continue
        name = d.get("label", nid)
        if name.startswith(".") and name.endswith("()"):
            bare = name[1:-2]
            if bare.startswith("__") and bare.endswith("__"):
                continue  # dunder: implicitly protocol-invoked
        if d.get("is_test"):
            continue  # discovered by the test framework, not "uncalled"
        out.append({"id": nid, "label": d.get("label", nid),
                    "source_file": d.get("source_file", "")})
        if limit is not None and len(out) >= limit:
            break
    return out


def find_paths(G: nx.Graph, a: str, b: str, max_paths: int = 5
               ) -> dict[str, Any]:
    """Deterministic shortest paths between two nodes, with edge relations."""
    if a not in G:
        raise KeyError(f"no such node: {a}")
    if b not in G:
        raise KeyError(f"no such node: {b}")
    if a == b:
        return {"found": True, "length": 0, "paths": [{"nodes": [a],
                                                       "relations": []}]}
    if not nx.has_path(G, a, b):
        return {"found": False, "length": None, "paths": [],
                "reason": "no path (different components)"}
    gen = nx.all_shortest_paths(G, a, b)
    collected: list[list[str]] = []
    for nodes in gen:
        collected.append(nodes)
        if len(collected) >= _MAX_PATHS_COLLECTED:
            break
    paths = []
    for nodes in sorted(collected)[:max_paths]:
        relations = [_relations_between(G, u, v)
                     for u, v in zip(nodes, nodes[1:])]
        paths.append({"nodes": nodes, "relations": relations})
    return {"found": True, "length": len(paths[0]["nodes"]) - 1,
            "paths": paths}


def neighborhood(G: nx.Graph, node: str, depth: int = 2, fanout: int = 12,
                 cap: int = 200) -> list[str]:
    """Bounded BFS neighborhood — the same traversal semantics as the
    memory bridge (sorted neighbors, per-hop fan-out, total cap)."""
    if node not in G:
        raise KeyError(f"no such node: {node}")
    out: set[str] = {node}
    frontier = [node]
    for _ in range(max(0, depth)):
        nxt: list[str] = []
        for n in frontier:
            if len(out) >= cap:
                return sorted(out)
            for nb in sorted(G.neighbors(n))[:fanout]:
                if nb not in out:
                    out.add(nb)
                    nxt.append(nb)
                    if len(out) >= cap:
                        return sorted(out)
        frontier = nxt
    return sorted(out)


def components(G: nx.Graph) -> dict[str, Any]:
    """Connected components: count, largest, top sizes (undirected view)."""
    sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    return {"count": len(sizes),
            "largest": sizes[0] if sizes else 0,
            "largest_sizes": sizes[:20]}


def node_report(G: nx.Graph, node: str, D: nx.DiGraph | None = None,
                edge_limit: int = 50) -> dict[str, Any]:
    """One node's attributes and its typed, bounded relationships."""
    if node not in G:
        raise KeyError(f"no such node: {node}")
    attrs = {k: v for k, v in G.nodes[node].items()}
    out: dict[str, Any] = {"node": node, "attributes": attrs}

    def edge_entry(other: str, d: dict[str, Any], direction: str) -> dict[str, Any]:
        return {"direction": direction, "relation": d.get("relation", "?"),
                "confidence": d.get("confidence", "?"),
                "other": other, "source_file": d.get("source_file", ""),
                "source_location": d.get("source_location", "")}

    outgoing: list[dict[str, Any]] = []
    incoming: list[dict[str, Any]] = []
    if D is not None and node in D:
        for _, v, d in sorted(D.out_edges(node, data=True),
                              key=lambda t: (t[1], t[2].get("relation", ""))):
            outgoing.append(edge_entry(v, d, "outgoing"))
        for u, _, d in sorted(D.in_edges(node, data=True),
                              key=lambda t: (t[0], t[2].get("relation", ""))):
            incoming.append(edge_entry(u, d, "incoming"))
    else:
        for u, v, d in sorted(G.edges(node, data=True),
                              key=lambda t: (t[1], t[2].get("relation", ""))):
            other = v if u == node else u
            if d.get("_src") == node:
                outgoing.append(edge_entry(other, d, "outgoing"))
            else:
                incoming.append(edge_entry(other, d, "incoming"))
    out["outgoing"] = outgoing[:edge_limit]
    out["incoming"] = incoming[:edge_limit]
    out["outgoing_truncated"] = len(outgoing) > edge_limit
    out["incoming_truncated"] = len(incoming) > edge_limit
    return out
