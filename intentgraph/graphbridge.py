"""Code-graph bridge: link memories into a graph.json code graph.

A memory record may carry ``graph_anchors``: node ids from a ``graph.json``
code graph (file nodes, symbol nodes) that lives at ``codegraph-out/graph.json``
(written by IntentGraph's native ``intentgraph codegraph build``) or
``graphify-out/graph.json`` (legacy-compatible) — a data-format integration:
the file is loaded here, no code-graph engine is imported. **Anchor ids
follow the graph.json id scheme** — path-flattened lowercase, e.g.
``intentgraph_models`` for ``intentgraph/models.py``,
``intentgraph_models_memoryrecord`` for a class, as produced by
``intentgraph codegraph build`` (schema documented in
``intentgraph/codegraph/schema.py``). This bridge lets retrieval follow
*code structure*, not just memory edges: "the decision about the auth
middleware" can surface "the test run for auth middleware" because both
records anchor inside the same neighborhood of the code graph, even with no
memory edge between them.

The node-link graph is loaded with networkx's ``json_graph.node_link_graph``
(undirected traversal view, per-edge ``_src``/``_tgt`` direction markers
preserved) in ~20 lines — so anchor ids and hop semantics match what any
consumer of the same file sees, without a dependency beyond networkx.

All expansion is bounded (hop depth, fan-out, total nodes) and deterministic.
If the graph file is missing, malformed, or empty, the bridge degrades to
"no code-adjacency" and says so — never to an exception in the query path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from intentgraph.models import MemoryRecord

__all__ = ["GraphifyBridge"]

_MAX_GRAPH_BYTES = 200 * 1024 * 1024  # sanity cap on a loaded graph file


class GraphifyBridge:
    """Bounded neighbor queries over a Graphify graph.json."""

    def __init__(self, graph_path: str | Path | None):
        self.graph_path = str(graph_path) if graph_path else ""
        self._G = None
        self._anchor_index: dict[str, set[str]] = {}  # node id → record ids
        self._load_error: str = ""
        self._schema: dict[str, Any] = {}
        if graph_path:
            self._load(graph_path)

    # -- loading ---------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._G is not None and self._G.number_of_nodes() > 0

    @property
    def load_error(self) -> str:
        return self._load_error

    @property
    def schema(self) -> dict[str, Any]:
        """The loaded file's ``graph`` attribute block, when present — native
        engine files identify themselves (schema name + version); legacy
        files carry nothing (empty dict)."""
        return dict(self._schema)

    @property
    def node_count(self) -> int:
        return self._G.number_of_nodes() if self._G is not None else 0

    @property
    def graph(self) -> Any:
        """The loaded undirected multigraph (typed edges with ``_src``/
        ``_tgt`` direction markers), or None when unavailable — the same
        graph the cross-graph retriever (:mod:`intentgraph.codequery`)
        builds its symbol index over."""
        return self._G

    def _load(self, graph_path: str | Path) -> None:
        try:
            p = Path(graph_path)
            if not p.exists():
                self._load_error = f"graph file not found: {p}"
                return
            if p.stat().st_size > _MAX_GRAPH_BYTES:
                self._load_error = (f"graph file exceeds safety cap "
                                    f"({_MAX_GRAPH_BYTES} bytes): {p}")
                return
            import networkx as nx
            from networkx.readwrite import json_graph

            data = json.loads(p.read_text(encoding="utf-8"))
            meta = data.get("graph")
            if isinstance(meta, dict):
                self._schema = dict(meta)
            if "links" not in data and "edges" in data:
                data = dict(data, links=data["edges"])
            # Mirror the CLI `query` loader: traverse undirected so a leaf
            # symbol's callers are reachable, keep true direction per edge.
            data = dict(data, directed=False)
            data = dict(
                data,
                links=[
                    {**link,
                     "_src": link.get("_src", link.get("source")),
                     "_tgt": link.get("_tgt", link.get("target"))}
                    for link in data.get("links", [])
                ],
            )
            try:
                G = json_graph.node_link_graph(data, edges="links")
            except TypeError:
                G = json_graph.node_link_graph(data)
            if G.is_directed():
                H = nx.MultiGraph() if G.is_multigraph() else nx.Graph()
                H.add_nodes_from(G.nodes(data=True))
                for u, v, d in G.edges(data=True):
                    H.add_edge(u, v, **d)
                G = H
            self._G = G
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self._load_error = f"could not load graph: {exc}"

    # -- anchor indexing ---------------------------------------------------------

    def index_records(self, records: Iterable[MemoryRecord]) -> int:
        """(Re)build node-id → record-ids index from the archive's anchors."""
        self._anchor_index.clear()
        n = 0
        for rec in records:
            for anchor in rec.graph_anchors:
                self._anchor_index.setdefault(anchor, set()).add(rec.id)
                n += 1
        return n

    def index_code_links(self, links: Iterable[dict[str, Any]]) -> int:
        """Union VERIFIED code links (explicit/extracted only) into the
        anchor index — records linked to a code node become retrievable
        through code adjacency exactly like anchored records. Candidate
        links are deliberately excluded: similarity is not evidence."""
        n = 0
        for l in links:
            if l.get("link_type") not in ("explicit", "extracted"):
                continue
            self._anchor_index.setdefault(l["node_id"], set()).add(
                l["record_id"])
            n += 1
        return n

    def add_anchor(self, record_id: str, node_id: str) -> None:
        """Incrementally register one anchor/link so records ingested after
        :meth:`index_records` still participate in code adjacency."""
        self._anchor_index.setdefault(node_id, set()).add(record_id)

    def records_at(self, node_id: str) -> set[str]:
        return set(self._anchor_index.get(node_id, ()))

    def _neighbors_bounded(self, start: str, depth: int, fanout: int,
                           cap: int) -> dict[str, int]:
        """Bounded BFS neighborhood of a node with per-hop and total caps.

        Returns node → BFS distance (the true shortest distance within the
        walked bound: computing it during the walk replaces a per-pair
        shortest_path_length call that made big graphs quadratic — same
        distances, one traversal per anchor).
        """
        G = self._G
        if G is None:  # pragma: no cover — callers gate on `available`
            return {start: 0}
        out: dict[str, int] = {start: 0}
        frontier = [start]
        for level in range(1, max(0, depth) + 1):
            nxt: list[str] = []
            for node in frontier:
                if len(out) >= cap:
                    return out
                neigh = sorted(G.neighbors(node))
                for nb in neigh[:fanout]:
                    if nb not in out:
                        out[nb] = level
                        nxt.append(nb)
                        if len(out) >= cap:
                            return out
            frontier = nxt
        return out

    # -- queries ------------------------------------------------------------------

    def related_records(
        self,
        seed_record_ids: Sequence[str],
        records_by_id: dict[str, MemoryRecord],
        *,
        depth: int = 2,
        fanout: int = 12,
        cap: int = 200,
    ) -> list[dict[str, Any]]:
        """Record ids code-adjacent to the seeds, with the hop path.

        Returns candidates shaped like retriever hits:
        ``{"id", "score", "via", "anchor"}`` where ``via`` is the chain
        of node hops from a seed's anchor to the related record's anchor.
        Deterministic: neighbors sorted, dedup by best score.
        """
        if not self.available:
            return []

        G = self._G
        assert G is not None  # available implies _G loaded
        seeds = [rid for rid in seed_record_ids if rid in records_by_id]
        if not seeds:
            return []
        best: dict[str, dict[str, Any]] = {}
        for rid in seeds:
            rec = records_by_id[rid]
            for anchor in rec.graph_anchors:
                if anchor not in G:
                    continue
                hood = self._neighbors_bounded(anchor, depth, fanout, cap)
                for node, dist in hood.items():
                    for other_rid in self.records_at(node):
                        if other_rid == rid:
                            continue
                        # dist is the BFS distance from the anchor, computed
                        # during the bounded walk itself
                        score = 0.5 / (1.0 + dist)
                        prev = best.get(other_rid)
                        if prev is None or score > prev["score"]:
                            best[other_rid] = {
                                "id": other_rid,
                                "score": round(score, 6),
                                "via": f"{anchor} ~{dist} hop(s)~ {node}",
                                "anchor": node,
                                "matched_terms": [anchor],
                                "term_scores": {},
                            }
        out = sorted(best.values(), key=lambda c: (-c["score"], c["id"]))
        return out
