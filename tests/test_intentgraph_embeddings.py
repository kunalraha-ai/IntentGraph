"""Unit tests for intentgraph.embeddings (provider interface + vector index)."""
from __future__ import annotations

import json

import pytest

from intentgraph.embeddings import (
    HashingEmbeddingProvider,
    NullEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    SemanticIndex,
    provider_from_config,
)
from intentgraph.store import MemoryStore


class TestNullProvider:
    def test_not_configured_and_raises(self):
        p = NullEmbeddingProvider()
        assert p.is_configured() is False
        with pytest.raises(RuntimeError, match="no embedding provider"):
            p.embed_texts(["x"])

    def test_provider_from_config_default_is_none(self):
        assert isinstance(provider_from_config(None), NullEmbeddingProvider)
        assert isinstance(provider_from_config({}), NullEmbeddingProvider)
        assert isinstance(provider_from_config({"provider": "none"}), NullEmbeddingProvider)


class TestHashingProvider:
    def test_deterministic_across_instances(self):
        a = HashingEmbeddingProvider(dim=64)
        b = HashingEmbeddingProvider(dim=64)
        v1 = a.embed_texts(["the gateway rejected the amount"])[0]
        v2 = b.embed_texts(["the gateway rejected the amount"])[0]
        assert v1 == v2

    def test_shape_and_normalization(self):
        p = HashingEmbeddingProvider(dim=64)
        for v in p.embed_texts(["hello world", "totally different", ""]):
            assert len(v) == 64
            norm = sum(x * x for x in v) ** 0.5
            assert norm == pytest.approx(1.0, abs=1e-6) or norm == pytest.approx(0.0)

    def test_overlap_scores_higher_than_disjoint(self):
        p = HashingEmbeddingProvider(dim=256)

        def cos(a, b):
            return sum(x * y for x, y in zip(a, b))

        shared = p.embed_texts(["checkout regression invalid amount gateway"])[0]
        similar = p.embed_texts(["checkout regression amount gateway error"])[0]
        disjoint = p.embed_texts(["dark mode analytics dashboard theme"])[0]
        assert cos(shared, similar) > cos(shared, disjoint)

    def test_dim_validation(self):
        with pytest.raises(ValueError):
            HashingEmbeddingProvider(dim=8)


class TestOpenAICompatibleProvider:
    def test_requires_model(self):
        with pytest.raises(ValueError, match="model"):
            OpenAICompatibleEmbeddingProvider(base_url="https://x/v1", model="")

    def test_from_env_requires_signal(self, monkeypatch):
        monkeypatch.delenv("INTENTGRAPH_EMBEDDING_BASE_URL", raising=False)
        monkeypatch.delenv("INTENTGRAPH_EMBEDDING_MODEL", raising=False)
        assert OpenAICompatibleEmbeddingProvider.from_env() is None
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_MODEL", "text-embedding-3-small")
        prov = OpenAICompatibleEmbeddingProvider.from_env()
        assert prov is not None
        assert prov.model == "text-embedding-3-small"

    def test_embed_sends_correct_payload_and_parses(self, monkeypatch):
        captured = {}

        class _Resp:
            def __init__(self, body):
                self._body = body

            def read(self):
                return json.dumps(self._body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["auth"] = req.get_header("Authorization")
            captured["body"] = json.loads(req.data.decode())
            assert timeout == 60.0
            return _Resp({"data": [
                {"index": 1, "embedding": [0.2, 0.1]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]})

        monkeypatch.setattr("intentgraph.embeddings.urllib.request.urlopen", fake_urlopen)
        p = OpenAICompatibleEmbeddingProvider(
            base_url="https://api.example.com/v1", model="m1",
            api_key="sk-test")
        out = p.embed_texts(["a", "b"])
        # response order must be re-sorted by index
        assert out[0] == [0.1, 0.2] and out[1] == [0.2, 0.1]
        assert captured["url"] == "https://api.example.com/v1/embeddings"
        assert captured["auth"] == "Bearer sk-test"
        assert captured["body"] == {"model": "m1", "input": ["a", "b"]}

    def test_provider_from_config_openai_needs_model(self):
        with pytest.raises(ValueError, match="INTENTGRAPH_EMBEDDING_MODEL"):
            provider_from_config({"provider": "openai"})

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown embedding provider"):
            provider_from_config({"provider": "mystical"})


class TestSemanticIndex:
    def _store_with_records(self, tmp_path):
        s = MemoryStore(tmp_path / "m.db")
        from intentgraph.models import MemoryRecord
        for i in range(3):
            s.add_record(MemoryRecord(content=f"record number {i} about gateways"))
        return s

    def test_rebuild_and_search(self, tmp_path):
        store = self._store_with_records(tmp_path)
        provider = HashingEmbeddingProvider(dim=64)
        idx = SemanticIndex(provider)
        records = list(store.iter_all_records())
        n = idx.rebuild(store, records)
        assert n == 3
        assert idx.ready
        hits = idx.search("gateway record", top_k=3)
        assert len(hits) >= 1
        assert all("score" in h for h in hits)

    def test_unconfigured_index_not_ready(self, tmp_path):
        store = self._store_with_records(tmp_path)
        idx = SemanticIndex(NullEmbeddingProvider())
        assert idx.rebuild(store, list(store.iter_all_records())) == 0
        assert not idx.ready
        assert idx.search("anything") == []

    def test_load_from_store_persisted_vectors(self, tmp_path):
        store = self._store_with_records(tmp_path)
        provider = HashingEmbeddingProvider(dim=64)
        idx = SemanticIndex(provider)
        idx.rebuild(store, list(store.iter_all_records()))
        # fresh index, same provider name → loads from the store, no recompute
        idx2 = SemanticIndex(provider)
        loaded = idx2.load(store)
        assert loaded == 3
        assert idx2.search("record")[0]["id"] == idx.search("record")[0]["id"]

    def test_doc_filter(self, tmp_path):
        store = self._store_with_records(tmp_path)
        provider = HashingEmbeddingProvider(dim=64)
        idx = SemanticIndex(provider)
        idx.rebuild(store, list(store.iter_all_records()))
        all_ids = {r.id for r in store.iter_all_records()}
        one = sorted(all_ids)[0]
        hits = idx.search("record", top_k=5, doc_filter=lambda d: d == one)
        assert {h["id"] for h in hits} <= {one}
        store.close()
