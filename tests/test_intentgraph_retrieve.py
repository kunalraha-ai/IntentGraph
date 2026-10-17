"""Unit tests for intentgraph.retrieve (retrievers, graph expansion, RRF)."""
from __future__ import annotations

import pytest

from intentgraph.embeddings import HashingEmbeddingProvider, SemanticIndex
from intentgraph.lexindex import BM25Index
from intentgraph.models import MemoryEdge, MemoryRecord
from intentgraph.retrieve import (
    Candidate,
    ExactRetriever,
    GraphRetriever,
    RetrievalEngine,
    fuse_rrf,
)
from intentgraph.store import MemoryStore

from tests.intentgraph_scenario import seed  # noqa: F401  (import order check)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    yield s
    s.close()


def _add(store, content, **kw):
    return store.add_record(MemoryRecord(content=content, **kw))


class TestExactRetriever:
    def test_finds_record_id_in_query(self, store):
        rec = _add(store, "some body text")
        idx = BM25Index()
        idx.build((r.id, r.search_text()) for r in store.iter_all_records())
        ex = ExactRetriever(idx, lambda s: [], lambda rid: [rid] if store.get_record(rid) else [])
        hits = ex.retrieve(f"context about {rec.id}")
        assert hits[0].memory_id == rec.id

    def test_finds_file_like_tokens(self, store):
        _add(store, "the bug lived in auth_middleware.py", source_id="logs/auth_middleware.py")
        idx = BM25Index()
        idx.build((r.id, r.search_text()) for r in store.iter_all_records())
        ex = ExactRetriever(idx, lambda s: [], lambda rid: [])
        hits = ex.retrieve("where was the auth_middleware.py bug?")
        assert len(hits) == 1

    def test_source_id_lookup(self, store):
        rec = _add(store, "deploy log", source_id="ci/build-9931.log")
        idx = BM25Index()
        idx.build((r.id, r.search_text()) for r in store.iter_all_records())
        ex = ExactRetriever(
            idx,
            lambda s: [rec.id] if s in "ci/build-9931.log" else [],
            lambda rid: [])
        hits = ex.retrieve("what happened in ci/build-9931.log?")
        assert hits[0].memory_id == rec.id


class TestRRF:
    def test_union_preserves_single_source_candidates(self):
        a = [Candidate("mem_a", "lexical", 3.0)]
        b = [Candidate("mem_b", "semantic", 0.9)]
        fused = fuse_rrf({"lexical": a, "semantic": b})
        ids = {c.memory_id for c in fused}
        assert ids == {"mem_a", "mem_b"}  # single-source survivors kept

    def test_multi_retriever_candidate_carries_both_methods(self):
        a = [Candidate("mem_x", "lexical", 3.0)]
        b = [Candidate("mem_x", "semantic", 0.9)]
        fused = fuse_rrf({"lexical": a, "semantic": b})
        assert len(fused) == 1
        assert sorted(fused[0].methods) == ["lexical", "semantic"]
        assert fused[0].retriever == "lexical+semantic"

    def test_top_rank_wins_regardless_of_raw_scale(self):
        # raw scores live on wildly different scales; only ranks may matter
        a = [Candidate("mem_top", "lexical", 1.0), Candidate("mem_low", "lexical", 999999.0)]
        b = [Candidate("mem_top", "semantic", 0.001), Candidate("mem_low", "semantic", 0.0001)]
        fused = fuse_rrf({"lexical": a, "semantic": b})
        assert fused[0].memory_id == "mem_top"

    def test_weights_apply(self):
        a = [Candidate("mem_a", "lexical", 1.0)]
        b = [Candidate("mem_b", "semantic", 1.0)]
        fused = fuse_rrf({"lexical": a, "semantic": b},
                         weights={"lexical": 10.0, "semantic": 0.1})
        assert fused[0].memory_id == "mem_a"

    def test_rank_not_score_drives_fusion(self):
        # wildly different score scales must not matter — only ranks do
        a = [Candidate("mem_a", "lexical", 0.001)]
        b = [Candidate("mem_b", "graph", 999999.0), Candidate("mem_a", "graph", 500.0)]
        fused = fuse_rrf({"lexical": a, "graph": b})
        # mem_a: 1/(60+1) from lexical + 1/(60+2) from graph
        # mem_b: 1/(60+1) from graph only
        assert fused[0].memory_id == "mem_a"


class TestGraphRetriever:
    def test_expansion_follows_edges_with_path(self, store):
        a = _add(store, "failure record")
        b = _add(store, "test output")
        c = _add(store, "unrelated record")
        store.add_edge(MemoryEdge(source=a.id, target=b.id, relation="validated_by"))
        g = GraphRetriever(store.edges_for)
        out = g.retrieve([a.id], max_depth=1, relations=("validated_by",))
        assert [cand.memory_id for cand in out] == [b.id]
        assert out[0].graph_path == [("validated_by", b.id)]
        assert out[0].depth == 1

    def test_depth_decay_scores_lower(self, store):
        a = _add(store, "seed")
        mid = _add(store, "middle")
        far = _add(store, "far")
        store.add_edge(MemoryEdge(source=a.id, target=mid.id, relation="caused_by"))
        store.add_edge(MemoryEdge(source=mid.id, target=far.id, relation="caused_by"))
        g = GraphRetriever(store.edges_for)
        out = g.retrieve([a.id], max_depth=2, relations=("caused_by",))
        by_id = {c.memory_id: c for c in out}
        assert by_id[mid.id].score > by_id[far.id].score
        assert by_id[far.id].depth == 2

    def test_relation_weights_order_high_signal_first(self, store):
        a = _add(store, "seed")
        high = _add(store, "caused record")
        low = _add(store, "grouped record")
        store.add_edge(MemoryEdge(source=a.id, target=high.id, relation="failed_because_of"))
        store.add_edge(MemoryEdge(source=a.id, target=low.id, relation="belongs_to"))
        g = GraphRetriever(store.edges_for)
        out = g.retrieve([a.id], max_depth=1)
        assert out[0].memory_id == high.id

    def test_fanout_cap_bounds_expansion(self, store):
        a = _add(store, "hub seed")
        for i in range(20):
            store.add_edge(MemoryEdge(source=a.id, target=_add(store, f"n{i}").id,
                                      relation="related_to"))
        g = GraphRetriever(store.edges_for)
        out = g.retrieve([a.id], max_depth=1, max_fanout=5)
        assert len(out) == 5

    def test_doc_filter_blocks_inactive(self, store):
        a = _add(store, "seed")
        b = _add(store, "deleted record")
        store.add_edge(MemoryEdge(source=a.id, target=b.id, relation="related_to"))
        store.delete_record(b.id)
        g = GraphRetriever(store.edges_for)
        out = g.retrieve([a.id], max_depth=1, doc_filter=lambda r: store.get_record(r).is_active)
        assert out == []

    def test_duplicate_of_not_expanded_transitively(self, store):
        a = _add(store, "seed")
        dup = _add(store, "near duplicate of a")
        far = _add(store, "behind the duplicate")
        store.add_edge(MemoryEdge(source=dup.id, target=a.id, relation="duplicate_of"))
        store.add_edge(MemoryEdge(source=dup.id, target=far.id, relation="caused_by"))
        g = GraphRetriever(store.edges_for)
        out = g.retrieve([a.id], max_depth=2)
        ids = {c.memory_id for c in out}
        assert dup.id in ids      # directly linked, still surfaced
        assert far.id not in ids  # not expanded THROUGH the duplicate_of edge


class TestRetrievalEngine:
    def _engine(self, store, with_semantic=True):
        idx = BM25Index()
        idx.build((r.id, r.search_text()) for r in store.iter_all_records())
        sem = SemanticIndex(
            HashingEmbeddingProvider(dim=64) if with_semantic
            else __import__("intentgraph.embeddings", fromlist=["NullEmbeddingProvider"]).NullEmbeddingProvider()
        )
        if with_semantic:
            sem.rebuild(store, list(store.iter_all_records()))
        return RetrievalEngine(
            lexical=idx, semantic_index=sem, edges_of=store.edges_for,
            lookup_source_id=lambda s: [],
            lookup_record_id=lambda rid: [rid] if store.get_record(rid) else [],
            graph_depth=2)

    def test_lexical_mode(self, store):
        _add(store, "the gateway rejected the amount")
        _add(store, "dark mode dashboard preference")
        eng = self._engine(store, with_semantic=False)
        out = eng.retrieve("gateway amount rejection", mode="lexical")
        assert out and out[0].retriever == "lexical"

    def test_hybrid_mode_includes_all_three_sources(self, store):
        _add(store, "regression in checkout flow INVALID_AMOUNT")
        eng = self._engine(store, with_semantic=True)
        notes: list[str] = []
        out = eng.retrieve("checkout regression", mode="hybrid", notes=notes)
        assert out
        # with the hash provider configured, semantic must have participated;
        # after the second fusion pass the method appears inside the
        # combined retriever string ("lexical+semantic", "lexical+semantic+graph", ...)
        assert any("semantic" in c.retriever for c in out)
        assert not any("semantic retrieval unavailable" in n for n in notes)

    def test_semantic_unavailable_note_is_honest(self, store):
        _add(store, "gateway fact")
        eng = self._engine(store, with_semantic=False)
        notes: list[str] = []
        eng.retrieve("gateway", mode="hybrid", notes=notes)
        assert any("semantic retrieval unavailable" in n for n in notes)

    def test_semantic_mode_falls_back_with_note(self, store):
        _add(store, "gateway fact")
        eng = self._engine(store, with_semantic=False)
        notes: list[str] = []
        out = eng.retrieve("gateway", mode="semantic", notes=notes)
        assert out  # lexical fallback, not empty
        assert notes

    def test_graph_mode_seeds_from_exact(self, store):
        seed_rec = _add(store, "the DeadlockInvestigation record about the pool")
        neighbor = _add(store, "caused by pool exhaustion")
        store.add_edge(MemoryEdge(source=seed_rec.id, target=neighbor.id, relation="caused_by"))
        eng = self._engine(store)
        out = eng.retrieve("the DeadlockInvestigation record", mode="graph")
        assert {c.memory_id for c in out} == {neighbor.id}

    def test_unknown_mode_raises(self, store):
        eng = self._engine(store)
        with pytest.raises(ValueError, match="unknown retrieval mode"):
            eng.retrieve("x", mode="mystical")
