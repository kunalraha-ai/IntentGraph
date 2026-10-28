"""Cross-graph retrieval: code entities and memories, retrieved together.

The milestone differentiator. A query can reach both knowledge sources at
once and follow the verified wiring between them:

1. memories — the existing deterministic hybrid (exact + lexical +
   semantic-when-configured + graph expansion) — untouched;
2. code entities — a deterministic name/path/symbol search over the
   project's code graph (``CodeIndex``); no fuzzy matching, ranked by
   match specificity with the match kind recorded;
3. verified links (``explicit``/``extracted`` code links) are followed in
   both directions, each result carrying the link's provenance
   (type, basis, matched text);
4. candidate links (similarity suggestions) are NEVER followed as
   evidence — they are listed, labelled, and stay that way until a human
   verifies them.

Honesty rules inherited from the rest of the system: no invented
associations, no similarity-as-proof, empty results say so, and a missing
code graph is an optional-capability note — memory retrieval continues to
work exactly as before.
"""
from __future__ import annotations

import json
import re
from typing import Any

import networkx as nx

from intentgraph.tokens import estimate_tokens

__all__ = ["CodeIndex", "combined_retrieve"]

_EXCERPT = 240
_RELATED_PER_HIT = 6
_LINKED_PER_HIT = 6

#: bare tokens shorter than this, or in this stopword set, never resolve on
#: their own (they would be noise: every repo has a `run` or `test`); they
#: still resolve inside dotted chains, backticks, and file paths.
_MIN_BARE_TOKEN = 4
_STOPWORDS = frozenset({
    "what", "when", "where", "which", "who", "whom", "why", "how", "this",
    "that", "these", "those", "with", "from", "into", "onto", "over",
    "code", "file", "files", "data", "info", "test", "tests", "case",
    "cases", "time", "work", "works", "make", "made", "take", "taken",
    "give", "given", "thing", "things", "name", "names", "here", "there",
    "then", "than", "them", "they", "were", "will", "would", "could",
    "should", "does", "done", "used", "uses", "using", "have", "hash",
    "some", "such", "same", "only", "also", "just", "like", "want",
    "about", "after", "before", "between", "because", "change", "changed",
    "query", "system", "memory", "check", "run", "runs", "list", "item",
    "each", "more", "most", "other", "another", "every", "both",
})

_FILE_TOKEN = re.compile(r"[A-Za-z0-9_][\w./-]*\.[A-Za-z]{1,4}\b")
_DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_BARE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
_BACKTICK = re.compile(r"`([^`\n]{2,200})`")


def _relations_between(G: nx.Graph, u: str, v: str) -> list[str]:
    data = G.get_edge_data(u, v)
    if not data:
        return []
    if G.is_multigraph():
        return sorted({d.get("relation", "?") for d in data.values()})
    return sorted({data.get("relation", "?")})


def _dotted_of(source_file: str) -> str:
    """intentgraph/store.py → intentgraph.store (__init__ → the package)."""
    posix = source_file.replace("\\", "/")
    parts = [p for p in posix.split("/") if p]
    if not parts:
        return ""
    name = parts[-1]
    if name == "__init__.py":
        parts = parts[:-1]
    elif name.endswith(".py"):
        parts[-1] = name[: -len(".py")]
    return ".".join(parts)


class CodeIndex:
    """Deterministic symbol/path index over a loaded code graph.

    Built once per SDK instance from the bridge's loaded graph (nodes carry
    id/label/type/qualname/source_file). All lookups are exact or structural
    — identical queries return identical results in identical order.
    """

    def __init__(self, G: nx.Graph) -> None:
        self.G = G
        self._by_id: dict[str, dict[str, Any]] = {}
        self._qualname: dict[str, list[str]] = {}   # lower(qualname) → ids
        self._simple: dict[str, list[str]] = {}     # lower(simple name) → ids
        self._full: dict[str, list[str]] = {}       # module(.class).symbol → ids
        self._module_files: dict[str, str] = {}     # module dotted → source_file
        self._paths: dict[str, list[str]] = {}      # source_file → module/test ids
        for nid in sorted(G.nodes):
            d = G.nodes[nid]
            self._by_id[nid] = dict(d)
            ntype = d.get("type")
            source = d.get("source_file") or ""
            qualname = d.get("qualname")
            if ntype in ("module", "test"):
                self._paths.setdefault(source, []).append(nid)
                if source:
                    self._module_files.setdefault(_dotted_of(source), source)
            elif ntype in ("class", "function", "method") and qualname:
                low = qualname.lower()
                self._qualname.setdefault(low, []).append(nid)
                self._simple.setdefault(low.rsplit(".", 1)[-1], []).append(nid)
                if source:
                    module = _dotted_of(source)
                    self._full.setdefault(f"{module}.{low}", []).append(nid)
                    # methods also answer the "module.method" phrasing
                    if low.count(".") >= 1:
                        self._full.setdefault(
                            f"{module}.{low.rsplit('.', 1)[-1]}",
                            []).append(nid)

    # -- resolution -----------------------------------------------------------

    def resolve(self, token: str) -> list[dict[str, Any]]:
        """Resolve one written token to code nodes with match provenance.

        ``[{"id", "score", "match"}]`` ordered by score then id. Scores:
        node id 100 · module name 95 · symbol qualname 90 · module-qualified
        symbol 80 · file path 75 · path suffix 60 · bare symbol name 55 ·
        path substring 35 · label substring 20.
        """
        token = token.strip().strip("`\"'")
        if not token:
            return []
        low = token.lower()
        hits: dict[str, dict[str, Any]] = {}

        def add(ids: list[str], score: int, match: str) -> None:
            for nid in ids:
                prev = hits.get(nid)
                if prev is None or score > prev["score"]:
                    hits[nid] = {"id": nid, "score": score, "match": match}

        if token in self._by_id:
            return [{"id": token, "score": 100, "match": "node id"}]
        if low in self._qualname:
            add(self._qualname[low], 90, "symbol qualname")
        if low in self._full:
            add(self._full[low], 80, "module-qualified symbol")
        # a bare token that exactly names a module/package resolves to its
        # module node at the HIGHEST symbol score: the dotted name is the
        # module's importable identity, while a case-insensitive qualname
        # hit (a class named `IntentGraph` vs the token `intentgraph`) is
        # only a naming collision — so `intentgraph` links to the package,
        # never to a same-named class
        if low in self._module_files:
            add(self._paths.get(self._module_files[low], []),
                95, "module name")
        # module-qualified split: longest known module prefix + remainder
        for i in range(low.count("."), 0, -1):
            parts = low.rsplit(".", i)
            prefix, rest = parts[0], ".".join(parts[1:])
            module_file = self._module_files.get(prefix)
            if not module_file or not rest:
                continue
            ids = [nid for nid in self._simple.get(rest, [])
                   if self._by_id.get(nid, {}).get("source_file") == module_file]
            add(sorted(set(ids)), 80, "module-qualified symbol")
            break
        if low in self._paths:
            add(self._paths[low], 75, "file path")
        if "/" in token or re.search(r"\.[A-Za-z]{1,4}$", token):
            add([nid for path, nids in self._paths.items()
                 if path and path != low and path.endswith("/" + low)
                 for nid in nids], 60, "file path (suffix)")
            add([nid for path, nids in self._paths.items()
                 if path and path != low and low in path.lower()
                 for nid in nids], 35, "file path (substring)")
        if len(low) >= _MIN_BARE_TOKEN and low not in _STOPWORDS:
            add(self._simple.get(low, []), 55, "symbol name")
        if not hits:
            for nid, d in self._by_id.items():
                if low in str(d.get("label", "")).lower():
                    hits[nid] = {"id": nid, "score": 20,
                                 "match": "label substring"}
        return [hits[k] for k in sorted(hits,
                                        key=lambda k: (-hits[k]["score"], k))]

    def node(self, nid: str) -> dict[str, Any] | None:
        return self._by_id.get(nid)

    # -- search ---------------------------------------------------------------

    def search(self, query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
        """Deterministic code search for a natural-language query.

        Extracts file-path, dotted-chain, backtick, and (guarded) bare
        tokens; overlapping spans don't double-count (longer matches win).
        """
        tokens: list[tuple[str, int, int]] = []
        covered: list[tuple[int, int]] = []

        def take(match: re.Match, group: int = 0) -> None:
            s, e = match.start(group), match.end(group)
            if any(s < ce and e > cs for cs, ce in covered):
                return
            tokens.append((match.group(group), s, e))
            covered.append((s, e))

        for m in _BACKTICK.finditer(query):
            take(m, 1)
        for m in _FILE_TOKEN.finditer(query):
            take(m)
        for m in _DOTTED.finditer(query):
            take(m)
        for m in _BARE.finditer(query):
            take(m)

        hits: dict[str, dict[str, Any]] = {}
        for token, _, _ in tokens:
            for hit in self.resolve(token):
                prev = hits.get(hit["id"])
                if prev is None or hit["score"] > prev["score"]:
                    hits[hit["id"]] = dict(hit, token=token)

        out: list[dict[str, Any]] = []
        for nid in sorted(hits, key=lambda k: (-hits[k]["score"], k)):
            d = self._by_id.get(nid, {})
            out.append({
                "id": nid,
                "label": d.get("label", nid),
                "type": d.get("type", "?"),
                "qualname": d.get("qualname"),
                "source_file": d.get("source_file", ""),
                "source_location": d.get("source_location", ""),
                "score": hits[nid]["score"],
                "match": hits[nid]["match"],
                "matched_token": hits[nid]["token"],
            })
            if len(out) >= max(1, top_k):
                break
        return out

    def expand(self, node_ids: list[str], *, depth: int = 1, fanout: int = 8,
               cap: int = 24) -> dict[str, dict[str, Any]]:
        """Bounded neighborhood: node → {"relations": [...], "hops": n}."""
        out: dict[str, dict[str, Any]] = {}
        frontier: dict[str, int] = {}
        for n in node_ids:
            if n in self._by_id:
                frontier[n] = 0
                out[n] = {"relations": [], "hops": 0}
        for _ in range(max(0, depth)):
            nxt: dict[str, int] = {}
            for n in sorted(frontier):
                if len(out) >= cap:
                    return out
                for other in sorted(self.G.neighbors(n))[:fanout]:
                    if other in out:
                        continue
                    if other not in nxt:
                        nxt[other] = out[n]["hops"] + 1
                    out[other] = {
                        "relations": _relations_between(self.G, n, other),
                        "hops": out[n]["hops"] + 1}
                    if len(out) >= cap:
                        return out
            frontier = nxt
        return out


# --- combined retrieval ---------------------------------------------------------


def combined_retrieve(
    ig: Any,
    query: str,
    *,
    top_k: int = 12,
    code_top_k: int = 8,
    depth: int = 1,
    fanout: int = 8,
    budget_tokens: int | None = None,
    include_candidates: bool = False,
) -> dict[str, Any]:
    """One query across both graphs: memories + code entities + verified links.

    Every returned entry carries provenance — source paths, source
    locations, memory ids, timestamps, link types and bases — so a consumer
    can always tell WHY something is in the result. A missing code graph is
    a note, never an error: memory retrieval is the same call it has always
    been.
    """
    result = ig.retrieve(query, mode="hybrid", top_k=top_k)
    notes: list[str] = list(result.get("notes") or [])
    found_via: set[str] = set()

    memories: list[dict[str, Any]] = []
    for c in result["candidates"]:
        rec = ig.store.get_record(c["memory_id"])
        if rec is None:
            continue
        memories.append(_memory_entry(rec, c, ig))
        found_via.add(rec.id)

    index: CodeIndex | None = getattr(ig, "code_index", None)
    code: list[dict[str, Any]] = []
    if index is None:
        notes.append("no code graph for this project (optional capability) "
                     "— memory retrieval only")
    else:
        for hit in index.search(query, top_k=code_top_k):
            entry = dict(hit)
            related = index.expand([hit["id"]], depth=depth, fanout=fanout)
            related.pop(hit["id"], None)
            entry["related"] = [
                {"id": nid,
                 "label": (index.node(nid) or {}).get("label", nid),
                 "type": (index.node(nid) or {}).get("type", "?"),
                 "source_file": (index.node(nid) or {}).get("source_file", ""),
                 "via": info["relations"], "hops": info["hops"]}
                for nid, info in sorted(related.items())[:_RELATED_PER_HIT]]
            linked = [l for l in ig.store.iter_code_links(node_id=hit["id"])
                      if l["link_type"] in ("explicit", "extracted")]
            entry["linked_memories"] = [
                {"record_id": l["record_id"], "link_type": l["link_type"],
                 "basis": l["basis"], "match": l["match"]}
                for l in linked[:_LINKED_PER_HIT]]
            code.append(entry)
            # follow verified links: a code hit may pull in the record that
            # explains it — appended after the direct hits, and the entry
            # itself says it arrived via a code link
            for l in linked[:_LINKED_PER_HIT]:
                if l["record_id"] in found_via:
                    continue
                rec = ig.store.get_record(l["record_id"])
                if rec is None or not rec.is_active:
                    continue
                memories.append(_memory_entry(
                    rec, {"memory_id": rec.id, "score": None,
                          "methods": ["code_link"]}, ig,
                    link={"type": l["link_type"], "basis": l["basis"],
                          "node": hit["id"]}))
                found_via.add(rec.id)

    candidates: list[dict[str, Any]] = []
    if include_candidates and index is not None:
        seen: set[tuple[str, str]] = set()
        for hit in code:
            for l in ig.store.iter_code_links(node_id=hit["id"],
                                              link_type="candidate"):
                key = (l["record_id"], l["node_id"])
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "record_id": l["record_id"], "node_id": l["node_id"],
                    "basis": l["basis"], "match": l["match"],
                    "note": "similarity suggestion — NOT verified evidence"})

    payload: dict[str, Any] = {
        "query": query,
        "memories": memories,
        "code": code,
        "candidates": candidates,
        "notes": notes,
    }
    if budget_tokens is not None:
        _fit_budget(payload, budget_tokens)
    return payload


def _memory_entry(rec: Any, cand: dict[str, Any], ig: Any,
                  link: dict[str, Any] | None = None) -> dict[str, Any]:
    flat = " ".join(rec.content.split())
    entry: dict[str, Any] = {
        "id": rec.id,
        "type": rec.type,
        "score": cand.get("score"),
        "methods": cand.get("methods") or [cand.get("retriever", "?")],
        "source_type": rec.source_type,
        "source_id": rec.source_id,
        "event_time": rec.event_time,
        "status": rec.status,
        "excerpt": flat[:_EXCERPT] + ("…" if len(flat) > _EXCERPT else ""),
    }
    if cand.get("graph_path"):
        entry["via"] = " --".join(
            f"{rel}--> {mid}" for rel, mid in cand["graph_path"])
    if link is not None:
        entry["via_code_link"] = link
    linked_code = [l for l in ig.store.iter_code_links(record_id=rec.id)
                   if l["link_type"] in ("explicit", "extracted")]
    if linked_code:
        entry["linked_code"] = [
            {"id": l["node_id"], "link_type": l["link_type"],
             "basis": l["basis"]} for l in linked_code[:_LINKED_PER_HIT]]
    return entry


def _fit_budget(payload: dict[str, Any], budget: int) -> None:
    """Drop tail entries (code first, then memories) until the payload fits
    the token budget — and say so. Deterministic, never cuts mid-entry."""
    budget = max(256, int(budget))
    trimmed: dict[str, int] = {}

    def size() -> int:
        return estimate_tokens(json.dumps(payload, ensure_ascii=False,
                                           default=str))

    for section in ("code", "candidates"):
        while payload[section] and size() > budget:
            payload[section].pop()
            trimmed[section] = trimmed.get(section, 0) + 1
    while payload["memories"] and size() > budget:
        payload["memories"].pop()
        trimmed["memories"] = trimmed.get("memories", 0) + 1
    if trimmed:
        payload["trimmed"] = trimmed
        payload["notes"].append(
            "result trimmed to the token budget: "
            + ", ".join(f"{k} -{v}" for k, v in sorted(trimmed.items())))
