"""Tests for the Nebius Token Factory provider integrations (all mocked).

Covers: the nebius embedding alias + credential precedence, the OpenAI-
compatible embeddings client hardening (batch ordering, dimension validation,
empty input, retries, honest failures, model listing, model-change detection),
and the /v1/rerank reranker (request shape, coverage validation, classified
errors, ordering-only semantics in the engine).
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from intentgraph.embeddings import (
    NEBIUS_DEFAULT_BASE_URL,
    NullEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    nebius_api_key_from_env,
    provider_from_config,
    SemanticIndex,
)
from intentgraph.rerank import (
    NebiusReranker,
    NullReranker,
    RerankError,
    reranker_from_config,
)


class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_urlopen(monkeypatch, responses, capture=None):
    calls = capture if capture is not None else []
    state = {"i": 0}

    def fake(req, timeout=None):
        calls.append({"url": req.full_url,
                      "auth": req.get_header("Authorization"),
                      "method": req.method,
                      "body": (json.loads(req.data.decode())
                               if getattr(req, "data", None) else None)})
        item = responses[state["i"]]
        state["i"] += 1
        if isinstance(item, urllib.error.HTTPError):
            raise item
        return _FakeResponse(item)

    monkeypatch.setattr("intentgraph.embeddings.urllib.request.urlopen", fake)
    return calls


def _http_error(code, detail="err"):
    try:
        raise urllib.error.HTTPError("https://x/v1", code, detail, {},
                                      io.BytesIO(b"{}"))
    except urllib.error.HTTPError as e:
        return e


# --- construction + credentials ---------------------------------------------------


class TestNebiusConstruction:
    def test_nebius_alias_uses_documented_base_and_requires_model(self, monkeypatch):
        monkeypatch.setenv("NEBIUS_API_KEY", "nb-key-1")
        p = provider_from_config({"provider": "nebius", "model": "some/model"})
        assert isinstance(p, OpenAICompatibleEmbeddingProvider)
        assert p.base_url == NEBIUS_DEFAULT_BASE_URL.rstrip("/")
        assert "nebius" in NEBIUS_DEFAULT_BASE_URL
        with pytest.raises(ValueError, match="requires an explicitly configured model"):
            provider_from_config({"provider": "nebius", "model": ""})
        with pytest.raises(ValueError, match="requires an explicitly configured model"):
            provider_from_config({"provider": "nebius"})

    def test_credential_precedence_nebius_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("NEBIUS_API_KEY", "primary")
        monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "alias")
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_API_KEY", "generic")
        assert nebius_api_key_from_env() == "primary"
        monkeypatch.delenv("NEBIUS_API_KEY")
        assert nebius_api_key_from_env() == "alias"
        monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_KEY")
        assert nebius_api_key_from_env() == ""
        # provider construction falls back to the generic var, never to a key
        p = provider_from_config({"provider": "nebius", "model": "m"})
        assert p.api_key == "generic"

    def test_no_credentials_anywhere_is_honest_null(self, monkeypatch):
        for var in ("NEBIUS_API_KEY", "NEBIUS_TOKEN_FACTORY_KEY",
                    "INTENTGRAPH_EMBEDDING_API_KEY", "OPENAI_API_KEY",
                    "INTENTGRAPH_EMBEDDING_BASE_URL", "INTENTGRAPH_EMBEDDING_MODEL"):
            monkeypatch.delenv(var, raising=False)
        assert isinstance(provider_from_config({"provider": "none"}),
                          NullEmbeddingProvider)

    def test_local_only_still_refuses_nebius(self, tmp_path):
        from intentgraph.config import load_config
        with pytest.raises(PermissionError, match="local_only"):
            load_config(data_dir=str(tmp_path / "d"),
                        overrides={"privacy.local_only": True,
                                   "embedding.provider": "nebius",
                                   "embedding.model": "some-model"})


# --- embeddings client hardening ---------------------------------------------------


class TestEmbedClient:
    def _provider(self, model="BAAI/bge-test", batch=2):
        return OpenAICompatibleEmbeddingProvider(
            base_url=NEBIUS_DEFAULT_BASE_URL, model=model,
            api_key="k", batch=batch, timeout_s=5)

    def test_empty_input_no_request(self, monkeypatch):
        calls = _mock_urlopen(monkeypatch, [])
        assert self._provider().embed_texts([]) == []
        assert calls == []

    def test_batching_and_response_reorder(self, monkeypatch):
        # 5 texts, batch=2 → 3 requests; responses return indices reversed
        p = self._provider(batch=2)
        responses = []
        for start, n in ((0, 2), (2, 2), (4, 1)):
            responses.append({"data": [
                {"index": start + j, "embedding": [float(start + j)] * 4}
                for j in reversed(range(n))],
                "usage": {"prompt_tokens": 10}})
        calls = _mock_urlopen(monkeypatch, responses)
        out = p.embed_texts(["a", "b", "c", "d", "e"])
        assert out[0][0] == 0.0 and out[4][0] == 4.0  # order restored by index
        assert len(calls) == 3
        assert all(c["body"]["model"] == "BAAI/bge-test" for c in calls)
        assert calls[0]["auth"] == "Bearer k"
        assert calls[0]["url"].endswith("/embeddings")
        assert p.last_usage == {"prompt_tokens": 10}

    def test_dimension_consistency_enforced(self, monkeypatch):
        p = self._provider()
        _mock_urlopen(monkeypatch, [{"data": [
            {"index": 0, "embedding": [1.0, 2.0]},
            {"index": 1, "embedding": [1.0, 2.0, 3.0]},  # ragged
        ]}])
        with pytest.raises(RuntimeError, match="malformed"):
            p.embed_texts(["a", "b"])

    def test_dimension_change_vs_provider_binding_is_an_error(self, monkeypatch):
        p = self._provider()
        _mock_urlopen(monkeypatch, [{"data": [
            {"index": 0, "embedding": [1.0, 2.0, 3.0]}]}])
        p.embed_texts(["first"])
        _mock_urlopen(monkeypatch, [{"data": [
            {"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}]}])
        with pytest.raises(RuntimeError, match="dimensions"):
            p.embed_texts(["second"])

    def test_incomplete_response_is_an_error(self, monkeypatch):
        p = self._provider()
        _mock_urlopen(monkeypatch, [{"data": [
            {"index": 0, "embedding": [1.0]}]}])
        with pytest.raises(RuntimeError, match="1 of 2"):
            p.embed_texts(["a", "b"])

    def test_rate_limit_retries_then_fails_honestly(self, monkeypatch):
        p = self._provider()
        calls = _mock_urlopen(monkeypatch,
                              [_http_error(429)] * 3)
        with pytest.raises(RuntimeError, match="after retries"):
            p.embed_texts(["a"])
        assert len(calls) == 3

    def test_auth_failure_fails_fast_no_retry(self, monkeypatch):
        p = self._provider()
        calls = _mock_urlopen(monkeypatch, [_http_error(401)])
        with pytest.raises(RuntimeError, match="401"):
            p.embed_texts(["a"])
        assert len(calls) == 1

    def test_never_fakes_embeddings_on_failure(self, monkeypatch):
        p = self._provider()
        _mock_urlopen(monkeypatch, [_http_error(500)] * 3)
        with pytest.raises(RuntimeError):
            p.embed_texts(["a"])   # raises — never returns zeros/hashes


class TestModelListing:
    def test_list_models_returns_sorted_ids(self, monkeypatch):
        p = OpenAICompatibleEmbeddingProvider(
            base_url=NEBIUS_DEFAULT_BASE_URL, model="m", api_key="k")
        calls = _mock_urlopen(monkeypatch, [{"data": [
            {"id": "BAAI/bge-en-icl"}, {"id": "Qwen/Qwen3-Reranker-8B"},
            {"id": "AA/bbb"}]}])
        models = p.list_models()
        assert models == ["AA/bbb", "BAAI/bge-en-icl", "Qwen/Qwen3-Reranker-8B"]
        assert calls[0]["method"] == "GET"
        assert calls[0]["url"].endswith("/models")
        assert calls[0]["auth"] == "Bearer k"

    def test_local_provider_cannot_list(self):
        with pytest.raises(AttributeError):
            NullEmbeddingProvider().list_models()


class TestModelChangeDetection:
    def _store(self, tmp_path):
        from intentgraph.store import MemoryStore
        from intentgraph.models import MemoryRecord
        s = MemoryStore(tmp_path / "m.db")
        s.add_record(MemoryRecord(content="some text"))
        return s

    @staticmethod
    def _mock_embeddings(monkeypatch, dim=4):
        _mock_urlopen(monkeypatch, [{"data": [
            {"index": 0, "embedding": [0.5] * dim}]}])

    def test_stored_vectors_from_other_model_refuse_to_load(self, tmp_path,
                                                            monkeypatch):
        store = self._store(tmp_path)
        old = OpenAICompatibleEmbeddingProvider(
            base_url=NEBIUS_DEFAULT_BASE_URL, model="old-model", api_key="k")
        idx = SemanticIndex(old)
        self._mock_embeddings(monkeypatch)
        idx.rebuild(store, list(store.iter_all_records()))
        # a NEW provider with a different model: the stale vectors must be
        # rejected with instructions, never mixed into one cosine space
        new = OpenAICompatibleEmbeddingProvider(
            base_url=NEBIUS_DEFAULT_BASE_URL, model="new-model", api_key="k")
        idx2 = SemanticIndex(new)
        loaded = idx2.load(store)
        assert loaded == 0
        assert not idx2.ready
        assert "reindex" in idx2.stale_model_note
        assert "old-model" in idx2.stale_model_note
        store.close()

    def test_same_model_loads(self, tmp_path, monkeypatch):
        store = self._store(tmp_path)
        old = OpenAICompatibleEmbeddingProvider(
            base_url=NEBIUS_DEFAULT_BASE_URL, model="same-model", api_key="k")
        idx = SemanticIndex(old)
        self._mock_embeddings(monkeypatch)
        idx.rebuild(store, list(store.iter_all_records()))
        idx2 = SemanticIndex(old)
        assert idx2.load(store) == 1
        assert idx2.ready
        store.close()


# --- reranker ------------------------------------------------------------------------


class TestReranker:
    def _reranker(self, model="Qwen/Qwen3-Reranker-8B"):
        return NebiusReranker(model=model, api_key="k",
                               base_url=NEBIUS_DEFAULT_BASE_URL)

    def test_request_shape_matches_vllm_protocol(self, monkeypatch):
        calls = _mock_urlopen(monkeypatch, [
            {"results": [{"index": 0, "document": {"text": "a"},
                          "relevance_score": 0.9}],
             "usage": {"prompt_tokens": 5, "total_tokens": 5}}])
        r = self._reranker()
        out = r.rerank("q", ["a"])
        req = calls[0]
        assert req["url"].endswith("/rerank")
        assert req["body"] == {"model": "Qwen/Qwen3-Reranker-8B",
                               "query": "q", "documents": ["a"]}
        assert out == [{"index": 0, "score": 0.9}]
        assert r.last_usage["prompt_tokens"] == 5

    def test_results_ordered_and_covering(self, monkeypatch):
        _mock_urlopen(monkeypatch, [
            {"results": [
                {"index": 2, "document": {"text": "c"}, "relevance_score": 0.1},
                {"index": 0, "document": {"text": "a"}, "relevance_score": 0.9},
                {"index": 1, "document": {"text": "b"}, "relevance_score": 0.5},
            ]}])
        out = self._reranker().rerank("q", ["a", "b", "c"])
        assert [item["index"] for item in out] == [0, 1, 2]

    def test_missing_index_is_rejected_not_faked(self, monkeypatch):
        _mock_urlopen(monkeypatch, [
            {"results": [
                {"index": 0, "document": {"text": "a"}, "relevance_score": 0.9},
            ]}])
        with pytest.raises(RerankError, match="1 of 2"):
            self._reranker().rerank("q", ["a", "b"])

    def test_out_of_range_index_rejected(self, monkeypatch):
        _mock_urlopen(monkeypatch, [
            {"results": [{"index": 7, "document": {"text": "?"},
                          "relevance_score": 0.9}]}])
        with pytest.raises(RerankError, match="malformed"):
            self._reranker().rerank("q", ["a"])

    def test_no_key_is_rejected_at_use_time(self):
        with pytest.raises(RerankError, match="no API key"):
            NebiusReranker(model="m", api_key="",
                           base_url=NEBIUS_DEFAULT_BASE_URL).rerank("q", ["a"])

    def test_model_required(self):
        with pytest.raises(ValueError, match="explicitly configured model"):
            NebiusReranker(model="", api_key="k",
                           base_url=NEBIUS_DEFAULT_BASE_URL)

    def test_404_reported_as_availability_not_auth(self, monkeypatch):
        calls = _mock_urlopen(monkeypatch, [_http_error(404)])
        with pytest.raises(RerankError, match="not be served"):
            self._reranker().rerank("q", ["a"])
        assert len(calls) == 1

    def test_reranker_from_config(self):
        assert isinstance(reranker_from_config(None), NullReranker)
        r = reranker_from_config({"provider": "nebius",
                                  "model": "Qwen/Qwen3-Reranker-8B"})
        assert isinstance(r, NebiusReranker)
        with pytest.raises(PermissionError, match="local_only"):
            reranker_from_config({"provider": "nebius", "model": "m"},
                                 local_only=True)

    def test_rerank_in_sdk_retrieve_is_ordering_only(self, tmp_path, monkeypatch):
        # reranking reorders but adds/drops nothing; failure keeps order
        from intentgraph.sdk import IntentGraph
        ig = IntentGraph(data_dir=str(tmp_path / "ig"),
                         overrides={"reranker.provider": "nebius",
                                    "reranker.model": "rm",
                                    "embedding.provider": "hash"})
        ig.ingest("decision: store amounts as integer minor units",
                  type="decision")
        ig.ingest("preference: use dark mode in dashboards", type="preference")
        # no credentials → reranker unavailable → honest note, fused order kept
        out = ig.retrieve("integer minor units decision", mode="hybrid")
        assert any("no credentials" in n for n in out["notes"])
        assert out["candidates"]
        ig.close()


class TestNvidiaConstruction:
    """NVIDIA NIM (`nvidia` provider alias): same client, documented base,
    its own credential convention. Model availability differs per account —
    this account serves nvidia/nemotron-3-embed-1b on NIM but NOT on Token
    Factory (the verified reason the alias exists)."""

    def test_alias_uses_nim_base_and_requires_model(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        p = provider_from_config({"provider": "nvidia",
                                  "model": "nvidia/nemotron-3-embed-1b"})
        assert isinstance(p, OpenAICompatibleEmbeddingProvider)
        assert p.base_url == "https://integrate.api.nvidia.com/v1"
        with pytest.raises(ValueError, match="requires an explicitly configured model"):
            provider_from_config({"provider": "nvidia"})

    def test_credential_resolution_nvidia_api_key_wins(self, monkeypatch):
        from intentgraph.embeddings import nvidia_api_key_from_env
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-a")
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_API_KEY", "generic")
        assert nvidia_api_key_from_env() == "nvapi-a"
        p = provider_from_config({"provider": "nvidia", "model": "m"})
        assert p.api_key == "nvapi-a"
        # falls back to the generic var only when NVIDIA_API_KEY is unset
        monkeypatch.delenv("NVIDIA_API_KEY")
        assert nvidia_api_key_from_env() == ""
        p = provider_from_config({"provider": "nvidia", "model": "m"})
        assert p.api_key == "generic"

    def test_local_only_refuses_nvidia(self, tmp_path):
        from intentgraph.config import load_config
        with pytest.raises(PermissionError, match="local_only"):
            load_config(data_dir=str(tmp_path / "d"),
                        overrides={"privacy.local_only": True,
                                   "embedding.provider": "nvidia",
                                   "embedding.model": "m"})


class TestDualProviderSupport:
    """Nebius and NVIDIA coexist as first-class providers. Nebius is the
    documented default; switching is a two-line config change; neither
    provider's credentials contaminate the other; and embeddings from
    different models are never mixed in one index."""

    def test_both_providers_construct_side_by_side(self, monkeypatch):
        monkeypatch.setenv("NEBIUS_API_KEY", "neb-key")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvid-key")
        neb = provider_from_config({"provider": "nebius",
                                    "model": "Qwen/Qwen3-Embedding-8B"})
        nvid = provider_from_config({"provider": "nvidia",
                                     "model": "nvidia/nemotron-3-embed-1b"})
        assert neb.base_url == NEBIUS_DEFAULT_BASE_URL.rstrip("/")
        assert nvid.base_url == "https://integrate.api.nvidia.com/v1"
        # each provider uses ITS OWN credential, never the other's
        assert neb.api_key == "neb-key" and neb.alias == "nebius"
        assert nvid.api_key == "nvid-key" and nvid.alias == "nvidia"

    def test_nebius_is_the_documented_default(self, monkeypatch, tmp_path):
        # a config file with no embedding section resolves to provider none;
        # the DEFAULT AMBIENT CONFIGURATION (as shipped in .env.example and
        # docs) selects nebius — the alias continues to work unchanged
        from intentgraph.config import load_config
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_PROVIDER", "nebius")
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
        monkeypatch.setenv("NEBIUS_API_KEY", "neb-key")
        cfg = load_config(data_dir=str(tmp_path / "d"))
        p = provider_from_config(cfg.embedding)
        assert p.base_url == NEBIUS_DEFAULT_BASE_URL.rstrip("/")
        assert p.model == "Qwen/Qwen3-Embedding-8B"

    def test_configuration_backwards_compatibility(self, tmp_path, monkeypatch):
        """A Phase-3.1-era configuration (no llm/reranker/web_search
        sections; nebius embedding) loads and resolves identically."""
        from intentgraph.config import load_config
        import json as _json
        home = tmp_path / "legacy-home"
        home.mkdir()
        (home / "intentgraph.json").write_text(_json.dumps({
            "embedding": {"provider": "nebius",
                          "model": "Qwen/Qwen3-Embedding-8B"},
        }), encoding="utf-8")
        monkeypatch.setenv("NEBIUS_API_KEY", "neb-key")
        cfg = load_config(data_dir=str(home))
        assert cfg.embedding["model"] == "Qwen/Qwen3-Embedding-8B"
        # new sections default to OFF — old configs never gain providers
        # they did not opt into
        assert cfg.llm["provider"] == "none"
        assert cfg.reranker["provider"] == "none"
        assert cfg.web_search["provider"] == "none"
        assert cfg.web_search["enabled"] is False
        p = provider_from_config(cfg.embedding)
        assert p.base_url == NEBIUS_DEFAULT_BASE_URL.rstrip("/")

    def test_embeddings_from_different_providers_never_mix(self, tmp_path,
                                                            monkeypatch):
        """The real-world switch: Nebius vectors stored, NVIDIA provider
        configured — the index must refuse the old vectors (different model
        AND endpoint), report the stale-model note, and a reindex embed
        cleanly under the new key."""
        from intentgraph.store import MemoryStore
        from intentgraph.models import MemoryRecord
        store = MemoryStore(tmp_path / "dual.db")
        store.add_record(MemoryRecord(content="shared evidence text"))
        neb = provider_from_config({"provider": "nebius",
                                    "model": "Qwen/Qwen3-Embedding-8B",
                                    "api_key": "k"})
        monkeypatch.setattr("intentgraph.embeddings.urllib.request.urlopen",
                            lambda req, timeout=None: _FakeResponse(
                                {"data": [{"index": 0,
                                           "embedding": [0.5] * 4096}]}))
        idx = SemanticIndex(neb)
        idx.rebuild(store, list(store.iter_all_records()))
        assert idx.ready

        nvid = provider_from_config({"provider": "nvidia",
                                     "model": "nvidia/nemotron-3-embed-1b",
                                     "api_key": "k"})
        idx2 = SemanticIndex(nvid)
        loaded = idx2.load(store)
        assert loaded == 0 and not idx2.ready
        assert "Qwen/Qwen3-Embedding-8B" in idx2.stale_model_note
        assert "reindex" in idx2.stale_model_note
        # after a clean reindex under the new provider, search works
        monkeypatch.setattr("intentgraph.embeddings.urllib.request.urlopen",
                            lambda req, timeout=None: _FakeResponse(
                                {"data": [{"index": 0,
                                           "embedding": [0.25] * 2048}]}))
        idx3 = SemanticIndex(nvid)
        idx3.rebuild(store, list(store.iter_all_records()))
        assert idx3.ready
        assert idx3.search("shared evidence")[0]["id"] == \
            list(store.iter_all_records())[0].id
        store.close()

    def test_same_dimensions_different_models_still_refuse_to_mix(self, tmp_path,
                                                                   monkeypatch):
        """Isolation is by provider NAME (model+endpoint), not dimensions —
        two same-width vector spaces are not comparable just because the
        widths match."""
        from intentgraph.store import MemoryStore
        from intentgraph.models import MemoryRecord
        store = MemoryStore(tmp_path / "samedim.db")
        store.add_record(MemoryRecord(content="x"))
        a = OpenAICompatibleEmbeddingProvider(
            base_url="https://host-a/v1", model="model-a", api_key="k")
        b = OpenAICompatibleEmbeddingProvider(
            base_url="https://host-b/v1", model="model-b", api_key="k")
        # mocked embeddings (offline suite): both providers would produce
        # 4096-wide vectors below — the refusal is about IDENTITY, not width
        monkeypatch.setattr("intentgraph.embeddings.urllib.request.urlopen",
                            lambda req, timeout=None: _FakeResponse(
                                {"data": [{"index": 0,
                                           "embedding": [0.5] * 4096}]}))
        SemanticIndex(a).rebuild(store, list(store.iter_all_records()))
        idx_b = SemanticIndex(b)
        assert idx_b.load(store) == 0
        assert idx_b.stale_model_note and "reindex" in idx_b.stale_model_note
        store.close()


class TestKeylessProviderBehavior:
    """A selected-but-keyless remote provider is unusable — and the system
    says so BEFORE any request, in the right place, with the right words."""

    def _keyless(self, monkeypatch, provider, model):
        for var in ("NEBIUS_API_KEY", "NEBIUS_TOKEN_FACTORY_KEY",
                    "NVIDIA_API_KEY", "INTENTGRAPH_EMBEDDING_API_KEY",
                    "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("intentgraph.embeddings.urllib.request.urlopen",
                            lambda *a, **k: pytest.fail(
                                "no request may be made without a key"))
        return provider_from_config({"provider": provider, "model": model})

    @pytest.mark.parametrize("alias,env_hint", [
        ("nebius", "NEBIUS_API_KEY"),
        ("nvidia", "NVIDIA_API_KEY"),
    ])
    def test_embed_and_list_refuse_before_any_request(self, monkeypatch,
                                                      alias, env_hint):
        p = self._keyless(monkeypatch, alias, "some/model")
        assert p.is_configured() is False
        assert p.available is False
        with pytest.raises(RuntimeError, match=env_hint):
            p.embed_texts(["text"])
        with pytest.raises(RuntimeError, match=env_hint):
            p.list_models()
        with pytest.raises(RuntimeError, match=alias):
            p.embed_texts(["text"])  # the message names the configured alias

    def test_reindex_reports_missing_credentials_distinctly(self, tmp_path,
                                                            monkeypatch):
        from intentgraph.sdk import IntentGraph
        self._keyless(monkeypatch, "nebius", "Qwen/Qwen3-Embedding-8B")
        ig = IntentGraph(data_dir=str(tmp_path / "keyless"),
                         overrides={"embedding.provider": "nebius",
                                    "embedding.model": "Qwen/Qwen3-Embedding-8B"})
        try:
            ig.ingest("some record", type="fact")
            report = ig.reindex_semantic()
            assert report["embedded"] == 0
            assert "no API key resolved" in report["note"]
            assert report["provider"]["api_key_present"] is False
        finally:
            ig.close()
