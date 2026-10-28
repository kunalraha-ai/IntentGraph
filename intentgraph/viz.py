"""IntentGraph visualization adapter: real memory data on the existing
Graphify renderer.

This is a THIN adapter, not a renderer. It converts the memory archive's
actual records and edges into the ``nx.Graph`` shape Graphify's existing
``to_html`` consumes (plus the generic ``viz_*`` overlay attributes the
renderer copies conditionally), then calls that renderer unchanged:

    build_memory_graph(ig)      → nx.Graph of real records + typed edges
    retrieval_overlay(ig, q)    → highlight dict from a REAL retrieve() call
    write_memory_html(...)      → to_html(...) with communities/labels wired

Design rules, in order of authority:

* **The database is the source of truth.** The adapter reads real records
  and edges; it never invents a node, edge, score, or status. The one
  derived relationship it draws — chronological ordering between
  consecutive conversation turns of the same session — is built from real
  timestamps and is marked INFERRED with an explicit basis, and it exists
  only in the visualization graph, never in the store.
* **Honesty in the render**: superseded records stay visible (gray ring,
  dashed) with their successor named; declared contradictions stay visible
  on BOTH endpoints (amber) and the edge is amber — nothing resolves a
  conflict in the picture. A retrieval score is displayed as a retrieval
  fact ("rank #2, via lexical+graph"), never as a truth value. Missing
  metadata renders as "not recorded", not as an invented value.
* **Community colors keep their meaning**: the community a node belongs to
  is its memory-type group (Decisions, Facts, …), exactly as Graphify uses
  communities to group related things. Memory *status* rides the border
  ring/dashes — a second channel — so status can never masquerade as
  community membership.
* **Colors follow the spec palette**: cyan #22d3ee = directly retrieved,
  violet #8b5cf6 = graph-expanded supporting context, amber #f59e0b =
  contradiction, gray #9ca3af = superseded/historical. The renderer's own
  community palette is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from intentgraph.models import MemoryRecord

__all__ = [
    "build_memory_graph",
    "build_combined_graph",
    "retrieval_overlay",
    "write_memory_html",
    "RETRIEVAL_DIRECT",
    "RETRIEVAL_EXPANDED",
    "CONFLICT_RING",
    "SUPERSEDED_RING",
    "CODE_LINK",
]

# status ring colors — the second visual channel (border), never background
RETRIEVAL_DIRECT = "#22d3ee"   # cyan: directly retrieved evidence
RETRIEVAL_EXPANDED = "#8b5cf6"  # violet: graph-expanded supporting context
CONFLICT_RING = "#f59e0b"      # amber: contradictory evidence
SUPERSEDED_RING = "#9ca3af"    # gray: superseded / historical
CODE_LINK = "#10b981"          # green: verified memory ↔ code link

#: node shape per code-entity type (memory types stay dots/diamonds)
_CODE_SHAPES = {"module": "box", "test": "box", "class": "triangle",
                "directory": "box", "repository": "box"}

#: combined-view community order: memory types first, then code groups
_CODE_TYPE_LABELS = {"module": "Code modules", "class": "Classes",
                     "function": "Functions", "method": "Methods",
                     "test": "Test modules", "directory": "Directories",
                     "repository": "Repository"}

#: hard cap on code nodes in a combined render — the memory graph must stay
#: readable; use `intentgraph codegraph render` for the full code view
_COMBINED_CODE_CAP = 2000

#: record status → (ring, dashed). Rings stack by precedence below:
#: conflict > superseded. Rings are additive via the border only.
_STATUS_RING: dict[str, tuple[str, bool]] = {
    "superseded": (SUPERSEDED_RING, True),
    "deleted": (SUPERSEDED_RING, True),
}

#: display order for the communities (= memory-type groups)
_TYPE_COMMUNITY_ORDER = (
    "conversation", "decision", "fact", "experience", "action",
    "document", "preference", "task", "project_state", "code_reference",
)

_MAX_LABEL = 60


def _node_label(rec: MemoryRecord) -> str:
    """One-line display label: first words of the content, truncated."""
    flat = " ".join(rec.content.split())
    if len(flat) > _MAX_LABEL:
        return flat[:_MAX_LABEL - 1] + "…"
    return flat or rec.id


def _time_or_not(recorded: str | None) -> str:
    """Honest absence: 'not recorded' for missing metadata."""
    return str(recorded) if recorded else "not recorded"


def _provenance_summary(rec: MemoryRecord, conflicts: list[str]) -> list[tuple[str, str]]:
    """Evidence-panel rows for one record — values are REAL fields only."""
    if rec.source_id:
        source = f"{rec.source_type}: {rec.source_id}"
    elif rec.source_type and rec.source_type != "manual":
        # a distinctive source type without an id (e.g. tool, url)
        source = rec.source_type
    else:
        # "manual" is the ingest DEFAULT: with no source_id it carries no
        # provenance information, and pretending it does would be inventing
        source = "not recorded"
    fields: list[tuple[str, str]] = [
        ("Memory type", rec.type),
        ("Record ID", rec.id),
        ("Event time", _time_or_not(rec.event_time)),
        ("Ingested", _time_or_not(rec.ingested_at)),
        ("Source", source),
    ]
    if rec.status != "active":
        fields.append(("Status", rec.status
                       + (f" (superseded by {rec.superseded_by})"
                          if rec.superseded_by else "")))
    else:
        fields.append(("Status", "active (current)"))
    if rec.tags:
        fields.append(("Tags", ", ".join(rec.tags)))
    if rec.provenance:
        keys = sorted(str(k) for k in rec.provenance)[:6]
        fields.append(("Provenance", ", ".join(keys)))
    if conflicts:
        fields.append(("Conflicts with", ", ".join(conflicts)))
    return fields


def _conflict_map(records: list[MemoryRecord], edges: list[Any]) -> dict[str, list[str]]:
    """record id → ids of records it has a declared contradicts edge with."""
    by_id = {r.id: r for r in records}
    out: dict[str, list[str]] = {}
    for e in edges:
        if e.relation != "contradicts":
            continue
        if e.source in by_id and e.target in by_id:
            out.setdefault(e.source, []).append(e.target)
            out.setdefault(e.target, []).append(e.source)
    return out


def build_memory_graph(
    ig: Any,
    *,
    include_conversation: bool = True,
    include_superseded: bool = True,
    include_deleted: bool = False,
) -> nx.Graph:
    """Convert the real memory archive into a renderable nx.Graph.

    Nodes are the archive's records (stable ``mem_*`` ids); edges are the
    archive's typed relationships, directed source→target via the
    ``_src``/``_tgt`` convention the renderer already honors. The graph is a
    *view*: nothing is written back to the store.
    """
    records = [r for r in ig.store.iter_all_records(include_deleted=True)
               if (include_deleted if r.status == "deleted"
                   else include_superseded if r.status == "superseded"
                   else True)]
    if not include_conversation:
        records = [r for r in records if r.type != "conversation"]
    edges = [e for e in ig.store.iter_edges()
             if e.source in {r.id for r in records}
             and e.target in {r.id for r in records}]
    conflicts = _conflict_map(records, edges)

    G = nx.Graph()
    for rec in records:
        viz_ring, dashed = None, False
        if rec.id in conflicts:
            viz_ring, dashed = CONFLICT_RING, False
        elif rec.status in _STATUS_RING:
            viz_ring, dashed = _STATUS_RING[rec.status]
        node_attrs: dict[str, Any] = {
            "label": _node_label(rec),
            "file_type": "memory",
            "source_file": rec.source_id or rec.source_type,
            # real memory type: drives the type-group communities and the
            # panel; the renderer treats it as an opaque string
            "viz_record_type": rec.type,
            "viz_fields": _provenance_summary(rec, conflicts.get(rec.id, [])),
        }
        if viz_ring:
            node_attrs["viz_ring"] = viz_ring
            node_attrs["viz_dashes"] = dashed
        if rec.type == "conversation":
            # captured turn ≠ verified fact: a distinct shape + an explicit
            # panel row say so; the node body keeps its community color
            node_attrs["viz_shape"] = "diamond"
            node_attrs["viz_fields"] = (
                [("Layer", "conversation event (captured turn — not a verified fact)")]
                + node_attrs["viz_fields"])
        G.add_node(rec.id, **node_attrs)

    for e in edges:
        G.add_edge(e.source, e.target, relation=e.relation,
                   # the renderer dashes non-EXTRACTED edges and shows the
                   # confidence in the tooltip — exactly the honest
                   # explicit/inferred distinction we want
                   confidence="EXTRACTED" if e.explicit else "INFERRED",
                   _src=e.source, _tgt=e.target)
        if e.relation == "contradicts":
            G[e.source][e.target]["viz_color"] = CONFLICT_RING

    # Chronological ordering between consecutive conversation turns of the
    # same session — DERIVED (viz-only), marked INFERRED with the basis, so
    # it can never be mistaken for a user-declared edge.
    if include_conversation:
        conv = sorted((r for r in records if r.type == "conversation"),
                      key=lambda r: (r.source_id, r.effective_time()))
        prev_by_session: dict[str, str] = {}
        for rec in conv:
            prev = prev_by_session.get(rec.source_id or rec.id)
            if prev is not None and prev != rec.id and not G.has_edge(prev, rec.id):
                G.add_edge(prev, rec.id, relation="occurred_before",
                           confidence="INFERRED", _src=prev, _tgt=rec.id)
            prev_by_session[rec.source_id or rec.id] = rec.id
    return G


def build_combined_graph(
    ig: Any,
    *,
    include_conversation: bool = True,
    include_superseded: bool = True,
    code_cap: int = _COMBINED_CODE_CAP,
) -> nx.Graph:
    """Memory graph + the project's code graph + their verified links.

    Code nodes come from the loaded code graph (bounded, deterministic
    selection: modules/tests first, then classes, functions, methods —
    sorted by id). Cross edges are REAL wiring only: record anchors and
    code links, green when verified, dashed-INFERRED when candidate. No
    similarity-only association is ever drawn as solid evidence.
    """
    G = build_memory_graph(ig, include_conversation=include_conversation,
                            include_superseded=include_superseded)
    index = getattr(ig, "code_index", None)
    if index is None:
        return G  # no code graph: the memory graph is the combined graph
    code_nodes: list[str] = []
    for group in (("module", "test"), ("class",), ("function",),
                  ("method",)):
        code_nodes.extend(
            nid for nid in sorted(index.G.nodes)
            if index.node(nid) and index.node(nid).get("type") in group)
        if len(code_nodes) >= code_cap:
            break
    code_nodes = sorted(code_nodes[:code_cap])

    for nid in code_nodes:
        d = index.node(nid) or {}
        attrs: dict[str, Any] = {
            "label": d.get("label", nid),
            "file_type": "code",
            "source_file": d.get("source_file", ""),
            "viz_record_type": d.get("type", "module"),
            "viz_fields": [
                ("Code type", d.get("type", "?")),
                ("File", d.get("source_file") or "not recorded"),
                ("Location", d.get("source_location") or "not recorded"),
            ] + ([("Symbol", d["qualname"])]
                 if d.get("qualname") else []),
        }
        shape = _CODE_SHAPES.get(d.get("type", ""))
        if shape:
            attrs["viz_shape"] = shape
        G.add_node(nid, **attrs)

    # real code-graph edges between included nodes (typed, evidence-labelled)
    for u in code_nodes:
        for v in sorted(index.G.neighbors(u)):
            if v not in G or G.nodes[v].get("file_type") != "code":
                continue
            data = index.G.get_edge_data(u, v)
            if not data:
                continue
            if not G.has_edge(u, v):
                attrs_by_key = (data.values() if index.G.is_multigraph()
                                else [data])
                best = sorted(attrs_by_key,
                              key=lambda d_: d_.get("relation", ""))[0]
                G.add_edge(u, v, relation=best.get("relation", "?"),
                           confidence=best.get("confidence", "EXTRACTED"),
                           _src=best.get("_src", u), _tgt=best.get("_tgt", v))

    # cross edges: record anchors (explicit) + code links (tiered). These
    # are the ONLY memory↔code connections ever drawn.
    for rec in ig.store.iter_all_records(include_deleted=True):
        if rec.id not in G or rec.status not in ("active", "superseded"):
            continue
        for anchor in rec.graph_anchors:
            if anchor in G and G.nodes[anchor].get("file_type") == "code":
                G.add_edge(rec.id, anchor, relation="code_link",
                           confidence="EXTRACTED", _src=rec.id, _tgt=anchor,
                           viz_color=CODE_LINK)
        for l in ig.store.iter_code_links(record_id=rec.id):
            if l["node_id"] in G and \
                    G.nodes[l["node_id"]].get("file_type") == "code":
                verified = l["link_type"] in ("explicit", "extracted")
                G.add_edge(rec.id, l["node_id"], relation="code_link",
                           confidence="EXTRACTED" if verified else "INFERRED",
                           _src=rec.id, _tgt=l["node_id"],
                           viz_color=CODE_LINK if verified else "#6b7280")
                G[rec.id][l["node_id"]]["viz_link_type"] = l["link_type"]
    return G


def retrieval_overlay(ig: Any, query: str, *, top_k: int = 12,
                      include_code: bool = False) -> dict[str, Any]:
    """Run a REAL retrieval and convert it to the renderer highlight dict.

    Tiers come from the retrieval's own transparency: candidates carrying a
    ``graph_path`` (graph expansion) are 'expanded' supporting context; the
    rest are 'direct'. Rank/score/methods are appended to the highlighted
    nodes' panel fields — as retrieval facts, never truth claims. Empty
    results return an overlay the renderer renders honestly ("no evidence
    returned — nothing highlighted"), never a fabricated match.

    ``include_code`` appends the query's REAL code-graph search hits
    (deterministic symbol/path matches) to the highlighted set — only the
    nodes the query actually returned are highlighted, never decorative ones.
    """
    result = ig.retrieve(query, mode="hybrid", top_k=top_k)
    overlay: dict[str, Any] = {"query": query, "results": []}
    for rank, cand in enumerate(result["candidates"], 1):
        tier = "expanded" if cand.get("graph_path") else "direct"
        overlay["results"].append({"id": cand["memory_id"], "tier": tier,
                                   "rank": rank, "score": cand.get("score"),
                                   "methods": cand.get("methods")})
    if include_code:
        index = getattr(ig, "code_index", None)
        if index is None:
            overlay.setdefault("note_extra", []).append(
                "no code graph for this project — memory highlights only")
        else:
            for hit in index.search(query, top_k=8):
                if hit["id"] in {r["id"] for r in overlay["results"]}:
                    continue
                overlay["results"].append({
                    "id": hit["id"], "tier": "direct",
                    "rank": len(overlay["results"]) + 1,
                    "score": hit["score"] / 100.0,
                    "methods": ["code"]})
    notes = result.get("notes") or []
    if not overlay["results"]:
        overlay["note"] = ("retrieval returned no evidence for this query"
                           + (f" ({notes[0]})" if notes else ""))
    return overlay


def write_memory_html(
    ig: Any,
    output_path: str | Path,
    *,
    query: str | None = None,
    include_conversation: bool = True,
    include_superseded: bool = True,
    with_code: bool = False,
) -> bool:
    """Render the memory archive (and optional real retrieval highlight) to
    an interactive graph.html via IntentGraph's own renderer
    (:mod:`intentgraph.renderer` — same call contract the borrowed renderer
    exposed, written fresh for the memory-graph use case).

    ``with_code`` renders the COMBINED graph: memory records + the
    project's code entities + the verified (green) / candidate (dashed
    gray) links between them. Nothing is fabricated: every node and edge
    in the render exists in the archive or the code graph."""
    from intentgraph.renderer import to_html

    if with_code:
        G = build_combined_graph(ig, include_conversation=include_conversation,
                                 include_superseded=include_superseded)
        communities, labels = _combined_communities(G)
    else:
        G = build_memory_graph(
            ig, include_conversation=include_conversation,
            include_superseded=include_superseded)
        communities, labels = _type_communities(G)
    retrieval = retrieval_overlay(ig, query, include_code=with_code) \
        if query else None

    # legend entries describe exactly what this render actually contains
    legend_extra: list[dict[str, Any]] = []
    edge_data = [d for _, _, d in G.edges(data=True)]
    node_data = [d for _, d in G.nodes(data=True)]
    if retrieval is not None:
        legend_extra.append({"color": RETRIEVAL_DIRECT,
                            "label": "Retrieved evidence (direct)"})
        if any(r["tier"] == "expanded" for r in retrieval["results"]):
            legend_extra.append({"color": RETRIEVAL_EXPANDED,
                                "label": "Supporting context (graph-expanded)"})
    if any(d.get("viz_link_type") in ("explicit", "extracted")
           or (d.get("relation") == "code_link" and d.get("viz_color") == CODE_LINK)
           for d in edge_data):
        legend_extra.append({"color": CODE_LINK,
                             "label": "Verified memory-code link"})
    if any(d.get("relation") == "code_link" and d.get("viz_color") == "#6b7280"
           for d in edge_data):
        legend_extra.append({"color": "#6b7280", "label": "Candidate link (unverified)",
                             "dashes": True})
    if any(d.get("relation") == "contradicts" for d in edge_data):
        legend_extra.append({"color": CONFLICT_RING, "label": "Contradiction"})
    if any(d.get("viz_ring") == SUPERSEDED_RING for d in node_data):
        legend_extra.append({"color": SUPERSEDED_RING, "label": "Superseded / historical",
                             "dashes": True})
    if any(d.get("viz_shape") == "diamond" for d in node_data):
        legend_extra.append({"color": "#4E79A7", "label": "Conversation event (captured turn)"})

    # retrieval rank/score join the panel fields of the highlighted nodes —
    # real retrieval facts, added AFTER the honest base fields
    if retrieval:
        by_id = {r["id"]: r for r in retrieval["results"]}
        for nid, d in G.nodes(data=True):
            hit = by_id.get(nid)
            if hit is not None:
                methods = "+".join(hit.get("methods") or ["?"])
                d["viz_fields"] = list(d.get("viz_fields") or []) + [
                    ("Retrieval rank", f"#{hit['rank']} (score {hit['score']}, via {methods})"),
                ]
    return to_html(G, communities, str(output_path),
                   community_labels=labels,
                   retrieval=retrieval, legend_extra=legend_extra)


def _combined_communities(G: nx.Graph) -> tuple[dict[int, list[str]], dict[int, str]]:
    """Communities for the combined render: memory types (known order),
    then code-entity groups with human labels."""
    order: list[str] = [t for t in _TYPE_COMMUNITY_ORDER
                        if any(d.get("viz_record_type") == t
                               for _, d in G.nodes(data=True))]
    for t in ("module", "class", "function", "method", "test", "directory",
              "repository"):
        if any(d.get("viz_record_type") == t for _, d in G.nodes(data=True)) \
                and t not in order:
            order.append(t)
    communities: dict[int, list[str]] = {}
    for nid, d in G.nodes(data=True):
        t = d.get("viz_record_type") or "other"
        communities.setdefault(order.index(t) if t in order else len(order),
                               []).append(nid)
    labels: dict[int, str] = {}
    for cid, t in enumerate(order):
        if t in _CODE_TYPE_LABELS:
            labels[cid] = _CODE_TYPE_LABELS[t]
        elif t == "conversation":
            labels[cid] = "Conversation events"
        else:
            labels[cid] = f"{t}s"
    return communities, labels


def _type_communities(G: nx.Graph) -> tuple[dict[int, list[str]], dict[int, str]]:
    """Group nodes by their real memory type into renderer communities."""
    # node type is recorded on the node attrs at build time
    order = [t for t in _TYPE_COMMUNITY_ORDER
             if any(d.get("viz_record_type") == t for _, d in G.nodes(data=True))]
    for nid, d in G.nodes(data=True):
        t = d.get("viz_record_type")
        if t is not None and t not in order:
            order.append(t)
    communities: dict[int, list[str]] = {}
    for nid, d in G.nodes(data=True):
        t = d.get("viz_record_type") or "other"
        cid = order.index(t) if t in order else len(order)
        communities.setdefault(cid, []).append(nid)
    labels = {cid: (f"{t}s" if t != "conversation" else "Conversation events")
              for cid, t in enumerate(order)}
    return communities, labels
