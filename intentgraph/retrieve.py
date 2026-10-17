"""Hybrid retrieval: lexical + semantic + graph, fused with RRF.

Structure:

* Each retriever is an independent callable producing a ranked list of
  :class:`Candidate` dicts with full scoring transparency (per-term scores,
  matched terms, and — for graph expansion — the exact edge path taken).
* :func:`fuse_rrf` merges retriever outputs with weighted Reciprocal Rank
  Fusion. RRF is chosen deliberately: BM25 scores, cosine similarities, and
  decayed graph-walk scores live on different scales, and normalizing them
  into one scale invites tuning lies. RRF needs only *ranks*, which every
  retriever produces honestly.
* **Deduplication preserves provenance**: when two retrievers find the same
  memory, the fused candidate carries BOTH methods and BOTH scores. A memory
  found by one retriever only is never dropped for being single-source —
  unique evidence is often exactly the evidence the others missed.
* Execution is sequential on purpose: every retriever is in-memory except
  the query embedding (one HTTP call at most), so a thread pool would add
  nondeterminism without measurable latency benefit. The controller calls
  retrieval rounds in a bounded loop (see controller.py).

All ranking is deterministic: score desc, then memory id asc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from intentgraph.lexindex import BM25Index
from intentgraph.models import RELATIONS

__all__ = [
    "MODES",
    "Candidate",
    "LexicalRetriever",
    "ExactRetriever",
    "SemanticRetriever",
    "GraphRetriever",
    "fuse_rrf",
    "RetrievalEngine",
]

#: Retrieval pipeline modes (what the CLI/API call "strategy").
MODES = ("lexical", "semantic", "graph", "hybrid", "adaptive")

# --- candidates -----------------------------------------------------------------


@dataclass
class Candidate:
    """One retrieval result, with everything needed to explain it."""

    memory_id: str
    retriever: str
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    term_scores: dict[str, float] = field(default_factory=dict)
    #: graph expansion explanation: list of (relation, other_memory_id) hops
    graph_path: list[tuple[str, str]] = field(default_factory=list)
    #: depth for graph-expanded candidates (0 = direct hit)
    depth: int = 0
    #: which retrievers found it (fused candidates carry the union)
    methods: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "retriever": self.retriever,
            "methods": self.methods or [self.retriever],
            "score": self.score,
            "matched_terms": self.matched_terms,
            "term_scores": self.term_scores,
            "graph_path": [[rel, mid] for rel, mid in self.graph_path],
            "depth": self.depth,
        }


def _as_candidate(hit: dict[str, Any], retriever: str, **extra: Any) -> Candidate:
    return Candidate(
        memory_id=hit["id"],
        retriever=retriever,
        score=float(hit.get("score", 0.0)),
        matched_terms=list(hit.get("matched_terms", [])),
        term_scores=dict(hit.get("term_scores", {})),
        **extra,
    )


# --- retrievers -------------------------------------------------------------------


class LexicalRetriever:
    """BM25 keyword search over record text (graphify-free, deterministic)."""

    name = "lexical"

    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(self, query: str, top_k: int = 25,
                 doc_filter: Callable[[str], bool] | None = None) -> list[Candidate]:
        hits = self.index.search(query, top_k=top_k, doc_filter=doc_filter)
        return [_as_candidate(h, self.name) for h in hits]


class ExactRetriever:
    """Exact identifiers: record ids, source ids, and exact phrases.

    Guarantees a query naming a known thing ("mem_abc...", "ADR-007",
    "auth_middleware.py") finds it regardless of BM25's term statistics —
    a rare failure mode of bag-of-words ranking that memory lookup cannot
    afford (an exact name being missed is the most visible possible bug).
    """

    name = "exact"

    def __init__(self, index: BM25Index, lookup_source_id: Callable[[str], list[str]],
                 lookup_record_id: Callable[[str], list[str]]):
        self.index = index
        self._source_id = lookup_source_id
        self._record_id = lookup_record_id

    def retrieve(self, query: str, top_k: int = 25,
                 doc_filter: Callable[[str], bool] | None = None) -> list[Candidate]:
        out: dict[str, Candidate] = {}
        # 1. direct record id mention
        import re

        for rid in set(re.findall(r"\bmem_[0-9a-f]{16}\b", query)):
            for hit_id in self._record_id(rid):
                if doc_filter and not doc_filter(hit_id):
                    continue
                out[hit_id] = Candidate(hit_id, self.name, 10.0,
                                         matched_terms=[rid],
                                         term_scores={rid: 10.0})
        # 2. source-id / distinctive-token exact match
        for phrase in _distinctive_phrases(query):
            for hit_id in self._source_id(phrase):
                if doc_filter and not doc_filter(hit_id):
                    continue
                cand = Candidate(hit_id, self.name, 8.0,
                                 matched_terms=[phrase],
                                 term_scores={phrase: 8.0})
                out.setdefault(hit_id, cand)
            for hit_id in self.index.exact_lookup(phrase):
                if doc_filter and not doc_filter(hit_id):
                    continue
                cand = Candidate(hit_id, self.name, 6.0,
                                 matched_terms=[phrase],
                                 term_scores={phrase: 6.0})
                out.setdefault(hit_id, cand)
        return sorted(out.values(), key=lambda c: (-c.score, c.memory_id))[:top_k]


_DISTINCTIVE_RE = None


def _distinctive_phrases(query: str) -> list[str]:
    """Phrases likely to be exact names: dotted paths, quoted strings,
    ALL_CAPS tokens, dotted/dashed identifiers, bare symbols ≥ 4 chars."""
    import re

    global _DISTINCTIVE_RE
    if _DISTINCTIVE_RE is None:
        _DISTINCTIVE_RE = re.compile(
            r"[\w./\\-]+\.[\w]+"          # file-ish: auth.py, a/b.test.ts
            r"|ADR-\d+"                    # decision records
            r"|\"[^\"]{3,}\"|'[^']{3,}'"    # quoted phrases
            r"|\b[A-Z][A-Za-z0-9_]{3,}\b"  # CamelCase-ish symbol
        )
    return _DISTINCTIVE_RE.findall(query)


class SemanticRetriever:
    """Cosine search over embeddings; reports unavailable when no provider."""

    name = "semantic"

    def __init__(self, semantic_index: Any):
        self.index = semantic_index  # embeddings.SemanticIndex

    def retrieve(self, query: str, top_k: int = 25,
                 doc_filter: Callable[[str], bool] | None = None) -> list[Candidate]:
        hits = self.index.search(query, top_k=top_k, doc_filter=doc_filter)
        return [_as_candidate(h, self.name) for h in hits]


# --- graph expansion ---------------------------------------------------------------

#: How much each relation is trusted to carry retrieval weight when expanding.
#: These are HEURISTICS, not learned parameters: relations that name a
#: causal/evidential mechanism (caused_by, validated_by, failed_because_of)
#: transfer more relevance than grouping ones (belongs_to, related_to).
#: duplicate_of is kept low on purpose: near-dups rarely add new information
#: to a context packet — but the edge is still traversable, just ranked lower.
_RELATION_WEIGHTS: dict[str, float] = {
    "caused_by": 0.95,
    "failed_because_of": 0.95,
    "validated_by": 0.90,
    "contradicts": 0.90,      # contradictions must ride along evidence
    "supersedes": 0.85,
    "derived_from": 0.80,
    "depends_on": 0.70,
    "occurred_before": 0.60,
    "occurred_after": 0.60,
    "mentions": 0.50,
    "belongs_to": 0.45,
    "related_to": 0.40,
    "duplicate_of": 0.25,
}

#: Relations the verifier treats as "evidence-bearing" follow-up edges.
HIGH_SIGNAL_RELATIONS = ("caused_by", "failed_because_of", "validated_by",
                         "contradicts", "supersedes", "derived_from")


class GraphRetriever:
    """Expand seed memories through typed relationship edges, bounded.

    Expansion is per-seed BFS over the memory edge graph with:

    * **edge-type weights** (see :data:`_RELATION_WEIGHTS`),
    * **depth decay** (each hop multiplies by ``decay``, default 0.6),
    * **per-hop fan-out cap** and a **global candidate cap**,
    * **active-record filtering** via ``doc_filter``.

    Deterministic: frontier is processed in (score desc, id asc) order.
    """

    name = "graph"

    def __init__(self, edges_of: Callable[[str], list[Any]]):
        #: edges_of(memory_id) → list of MemoryEdge touching that memory
        self._edges_of = edges_of

    def retrieve(
        self,
        seeds: Sequence[Candidate | str],
        *,
        top_k: int = 25,
        max_depth: int = 2,
        max_fanout: int = 8,
        decay: float = 0.6,
        relations: Iterable[str] | None = None,
        doc_filter: Callable[[str], bool] | None = None,
        seed_scores: dict[str, float] | None = None,
    ) -> list[Candidate]:
        if not seeds:
            return []
        allowed = set(relations) if relations is not None else set(RELATIONS)
        # seed bookkeeping: id → (base score, path so far)
        frontier: dict[str, tuple[float, list[tuple[str, str]]]] = {}
        for s in seeds:
            sid = s.memory_id if isinstance(s, Candidate) else str(s)
            base = (s.score if isinstance(s, Candidate) else 1.0) * 0.9  # expansion beats nothing that directly matched
            frontier[sid] = (base, [])
        best: dict[str, Candidate] = {
            sid: Candidate(sid, self.name, score, graph_path=list(path), depth=0)
            for sid, (score, path) in frontier.items()
        }
        current_depth = 0
        while frontier and current_depth < max_depth:
            current_depth += 1
            next_frontier: dict[str, tuple[float, list[tuple[str, str]]]] = {}
            # deterministic expansion order
            for sid in sorted(frontier, key=lambda x: (-frontier[x][0], x)):
                base_score, path = frontier[sid]
                out_edges = self._edges_of(sid)
                # deterministic edge order: relation weight desc, then id
                out_edges = sorted(
                    out_edges,
                    key=lambda e: (-_RELATION_WEIGHTS.get(e.relation, 0.3), e.id),
                )
                taken = 0
                for edge in out_edges:
                    if edge.relation not in allowed or taken >= max_fanout:
                        continue
                    other = edge.target if edge.source == sid else edge.source
                    if doc_filter is not None and not doc_filter(other):
                        continue
                    taken += 1
                    rel_w = _RELATION_WEIGHTS.get(edge.relation, 0.3)
                    new_score = base_score * rel_w * decay
                    new_path = path + [(edge.relation, other)]
                    if other in best:
                        if new_score > best[other].score:
                            best[other] = Candidate(
                                other, self.name, new_score,
                                graph_path=new_path, depth=current_depth)
                    else:
                        best[other] = Candidate(
                            other, self.name, new_score,
                            graph_path=new_path, depth=current_depth)
                    if other not in next_frontier or new_score > next_frontier[other][0]:
                        # don't re-expand beyond depth via duplicate_of noise
                        if edge.relation != "duplicate_of":
                            next_frontier[other] = (new_score, new_path)
            frontier = {k: v for k, v in next_frontier.items()
                        if v[0] > 0.05}  # prune: nothing that weak is evidence
        expanded = [c for c in best.values()
                    if c.depth > 0 and (doc_filter is None or doc_filter(c.memory_id))]
        expanded.sort(key=lambda c: (-c.score, c.memory_id))
        return expanded[:top_k]


# --- fusion -----------------------------------------------------------------------


def fuse_rrf(
    retriever_results: dict[str, list[Candidate]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int = 50,
) -> list[Candidate]:
    """Weighted Reciprocal Rank Fusion over per-retriever rankings.

    ``score = Σ_retriever weight / (k + rank)`` with k=60 (the standard
    constant that dampens the difference between top ranks so a rank-1 vs
    rank-3 disagreement across retrievers does not dominate).

    Dedup keeps a union of methods and per-retriever scores: a memory found
    by lexical only survives next to one found by all three.
    """
    weights = weights or {}
    fused: dict[str, dict[str, Any]] = {}
    for retriever, results in retriever_results.items():
        w = float(weights.get(retriever, 1.0))
        for rank, cand in enumerate(results):
            contribution = w / (k + rank + 1)
            slot = fused.setdefault(cand.memory_id, {
                "memory_id": cand.memory_id,
                "methods": [],
                "retriever_scores": {},
                "matched_terms": [],
                "term_scores": {},
                "graph_path": [],
                "depth": cand.depth,
                "rrf": 0.0,
            })
            if cand.retriever not in slot["methods"]:
                slot["methods"].append(cand.retriever)
            slot["retriever_scores"][cand.retriever] = max(
                cand.score, slot["retriever_scores"].get(cand.retriever, 0.0))
            for t in cand.matched_terms:
                if t not in slot["matched_terms"]:
                    slot["matched_terms"].append(t)
            slot["term_scores"].update(cand.term_scores)
            if cand.graph_path and not slot["graph_path"]:
                slot["graph_path"] = [[r, m] for r, m in cand.graph_path]
            slot["depth"] = min(slot["depth"], cand.depth)
            slot["rrf"] += contribution
    out = [
        Candidate(
            memory_id=slot["memory_id"],
            retriever="+".join(slot["methods"]),
            score=round(slot["rrf"], 6),
            matched_terms=slot["matched_terms"],
            term_scores=slot["term_scores"],
            graph_path=[(r, m) for r, m in slot["graph_path"]],
            depth=slot["depth"],
            methods=slot["methods"],
        )
        for slot in fused.values()
    ]
    out.sort(key=lambda c: (-c.score, c.memory_id))
    return out[: max(0, top_k)]


def candidates_to_dicts(cands: Iterable[Candidate]) -> list[dict[str, Any]]:
    """Serialize candidates with the methods union as a first-class field."""
    return [c.to_dict() for c in cands]


class RetrievalEngine:
    """One-stop hybrid retrieval over a store's indexes.

    ``mode`` selects which retrievers run:

    * ``lexical``  — BM25 only
    * ``semantic`` — embeddings only (fails loudly if unconfigured → the
      engine falls back to lexical with a note, since returning zero
      results for every query would read as "archive empty")
    * ``graph``    — exact-seed + graph expansion only
    * ``hybrid``   — lexical + semantic (+ exact), then graph expansion of
      the fused top seeds
    * ``adaptive`` — the controller drives :meth:`retrieve_round` itself
    """

    def __init__(
        self,
        *,
        lexical: BM25Index,
        semantic_index: Any,
        edges_of: Callable[[str], list[Any]],
        lookup_source_id: Callable[[str], list[str]],
        lookup_record_id: Callable[[str], list[str]],
        weights: dict[str, float] | None = None,
        graph_depth: int = 2,
        max_candidates: int = 50,
    ):
        self.lexical = LexicalRetriever(lexical)
        self.exact = ExactRetriever(lexical, lookup_source_id, lookup_record_id)
        self.semantic = SemanticRetriever(semantic_index)
        self.graph = GraphRetriever(edges_of)
        self.weights = weights or {}
        self.graph_depth = graph_depth
        self.max_candidates = max_candidates

    def retrieve(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int | None = None,
        doc_filter: Callable[[str], bool] | None = None,
        notes: list[str] | None = None,
        seed_ids: Sequence[str] | None = None,
    ) -> list[Candidate]:
        """Run one retrieval round in the given mode.

        ``notes`` collects honest status lines (e.g. "semantic unavailable")
        for the caller to surface — the engine never silently degrades.
        """
        top_k = top_k or self.max_candidates
        notes = notes if notes is not None else []

        if mode == "lexical":
            return self.lexical.retrieve(query, top_k, doc_filter)[:top_k]
        if mode == "exact":
            return self.exact.retrieve(query, top_k, doc_filter)
        if mode == "semantic":
            if not self.semantic.index.ready:
                notes.append("semantic retrieval unavailable (no embedding provider configured); "
                             "fell back to lexical")
                return self.lexical.retrieve(query, top_k, doc_filter)[:top_k]
            return self.semantic.retrieve(query, top_k, doc_filter)[:top_k]
        if mode == "graph":
            seeds = list(seed_ids or [])
            if not seeds:
                # graph mode with no seeds: find seeds via exact identifiers
                seeds = [c.memory_id for c in self.exact.retrieve(query, top_k, doc_filter)]
            return self.graph.retrieve(
                seeds, top_k=top_k, max_depth=self.graph_depth,
                doc_filter=doc_filter)
        if mode in ("hybrid", "adaptive"):
            results: dict[str, list[Candidate]] = {}
            results["exact"] = self.exact.retrieve(query, top_k, doc_filter)
            results["lexical"] = self.lexical.retrieve(query, top_k, doc_filter)
            if self.semantic.index.ready:
                results["semantic"] = self.semantic.retrieve(query, top_k, doc_filter)
            else:
                notes.append("semantic retrieval unavailable (no embedding provider configured)")
            fused = fuse_rrf(results, weights=self.weights, top_k=top_k)
            if mode == "hybrid":
                # expand the fused top seeds through the memory graph
                expanded = self.graph.retrieve(
                    [c.memory_id for c in fused[:5]],
                    top_k=top_k, max_depth=self.graph_depth, doc_filter=doc_filter)
                combined: dict[str, list[Candidate]] = {"fused": fused, "graph": expanded}
                return fuse_rrf(combined, weights=self.weights, top_k=top_k)
            return fused
        raise ValueError(f"unknown retrieval mode: {mode!r}")

    def expand(self, seed_ids: Sequence[str], *, top_k: int = 25,
               relations: Iterable[str] | None = None,
               doc_filter: Callable[[str], bool] | None = None) -> list[Candidate]:
        """Targeted graph expansion (the controller's follow-up tool)."""
        return self.graph.retrieve(
            seed_ids, top_k=top_k, max_depth=self.graph_depth,
            relations=relations, doc_filter=doc_filter)
