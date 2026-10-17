"""Tests for the Tavily web-search provider (all mocked; no network, no key).

Every request-formatting and response-parsing behavior asserted against the
documented API shape (POST /search, Bearer auth, include_published_date,
include_usage, results[].{title,url,content,score,published_date}).
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from intentgraph.websearch import (
    ExternalEvidence,
    NullWebSearchProvider,
    TavilySearchProvider,
    WebSearchError,
    provider_from_config,
    store_external_result,
)


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _mock_urlopen(monkeypatch, responses, capture=None):
    """responses: list of (status_class, body_dict | HTTPError)."""
    calls = capture if capture is not None else []
    state = {"i": 0}

    def fake_urlopen(req, timeout=None):
        calls.append({
            "url": req.full_url,
            "auth": req.get_header("Authorization"),
            "body": json.loads(req.data.decode()),
            "timeout": timeout,
        })
        item = responses[state["i"]]
        state["i"] += 1
        if isinstance(item, urllib.error.HTTPError):
            raise item
        return _FakeResponse(item)

    monkeypatch.setattr("intentgraph.websearch.urllib.request.urlopen", fake_urlopen)
    return calls


def _http_error(code: int, detail: str = "boom") -> urllib.error.HTTPError:
    try:
        raise urllib.error.HTTPError("https://api.tavily.com/search", code,
                                     detail, {},
                                     io.BytesIO(json.dumps(
                                         {"detail": {"error": detail}}).encode()))
    except urllib.error.HTTPError as e:
        return e


@pytest.fixture
def tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key-000")
    return TavilySearchProvider(max_results=3)


class TestConstruction:
    def test_no_key_reports_unavailable_and_search_fails_honestly(
            self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        p = TavilySearchProvider()
        assert p.available is False
        with pytest.raises(WebSearchError) as ei:
            p.search("anything")
        assert ei.value.kind == "auth"
        assert "TAVILY_API_KEY" in str(ei.value)
        assert "tvly" not in str(ei.value)  # no key material in errors

    def test_param_validation(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        with pytest.raises(ValueError, match="search_depth"):
            TavilySearchProvider(search_depth="deepest")
        with pytest.raises(ValueError, match="topic"):
            TavilySearchProvider(topic="gossip")
        with pytest.raises(ValueError, match="time_range"):
            TavilySearchProvider(time_range="epoch")

    def test_provider_from_config(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        p = provider_from_config({"provider": "tavily", "max_results": 7})
        assert isinstance(p, TavilySearchProvider)
        assert p.max_results == 7
        assert isinstance(provider_from_config(None), NullWebSearchProvider)
        assert isinstance(provider_from_config({"provider": "none"}),
                          NullWebSearchProvider)
        with pytest.raises(ValueError, match="unknown web search provider"):
            provider_from_config({"provider": "metacrawler"})

    def test_local_only_refuses_tavily(self):
        with pytest.raises(PermissionError, match="local_only"):
            provider_from_config({"provider": "tavily"}, local_only=True)

    def test_max_results_clamped_to_documented_cap(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        assert TavilySearchProvider(max_results=500).max_results == 20
        assert TavilySearchProvider(max_results=0).max_results == 1


class TestRequestFormatting:
    def test_exact_documented_request_shape(self, tavily, monkeypatch):
        calls = _mock_urlopen(monkeypatch, [
            {"query": "q", "results": [], "images": [], "response_time": 0.1,
             "answer": None},
        ])
        tavily.search("sqlite WAL mode")
        req = calls[0]
        assert req["url"] == "https://api.tavily.com/search"
        assert req["auth"] == "Bearer tvly-test-key-000"
        body = req["body"]
        assert body["query"] == "sqlite WAL mode"
        assert body["max_results"] == 3
        assert body["search_depth"] == "basic"
        assert body["topic"] == "general"
        # the ONLY way dates come back; and usage metadata is requested
        assert body["include_published_date"] is True
        assert body["include_usage"] is True
        # fields the user never configured must not be sent
        assert "include_answer" not in body
        assert "include_raw_content" not in body

    def test_domain_filters_and_time_range_sent_when_configured(
            self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        p = TavilySearchProvider(include_domains=["docs.x.dev"],
                                  exclude_domains=["spam.example"],
                                  time_range="week", topic="news")
        calls = _mock_urlopen(monkeypatch, [
            {"query": "q", "results": [], "images": [], "response_time": 0.1,
             "answer": None},
        ])
        p.search("release notes")
        body = calls[0]["body"]
        assert body["include_domains"] == ["docs.x.dev"]
        assert body["exclude_domains"] == ["spam.example"]
        assert body["time_range"] == "week"
        assert body["topic"] == "news"


class TestNormalization:
    def _body(self, results, **extra):
        return {"query": "q", "results": results, "images": [],
                "response_time": 1.25, "answer": None,
                "request_id": "req-1", **extra}

    def test_full_provenance_preserved(self, tavily, monkeypatch):
        _mock_urlopen(monkeypatch, [self._body([
            {"title": "SQLite WAL", "url": "https://sqlite.org/wal.html",
             "content": "WAL allows concurrent readers",
             "score": 0.81, "published_date": "Tue, 11 Mar 2025 17:00:00 GMT"},
        ], usage={"credits": 1})])
        out = tavily.search("sqlite wal")
        assert len(out) == 1
        ev = out[0]
        assert ev.url == "https://sqlite.org/wal.html"
        assert ev.title == "SQLite WAL"
        assert "concurrent readers" in ev.excerpt
        assert ev.score == 0.81
        assert ev.published_date == "Tue, 11 Mar 2025 17:00:00 GMT"
        assert ev.retrieved_at  # always stamped
        assert ev.provider == "tavily"
        assert ev.query == "sqlite wal"
        assert ev.usage_credits == 1
        assert ev.response_time_s == 1.25
        assert ev.request_id == "req-1"
        assert ev.trust["origin"] == "external"
        assert ev.trust["verified"] is False

    def test_missing_published_date_stays_none_never_fabricated(
            self, tavily, monkeypatch):
        _mock_urlopen(monkeypatch, [self._body([
            {"title": "No date", "url": "https://x.dev/a", "content": "c",
             "score": 0.5, "published_date": None},
            {"title": "No date field at all", "url": "https://x.dev/b",
             "content": "c", "score": 0.5},
        ])])
        out = tavily.search("something")
        assert [e.published_date for e in out] == [None, None]

    def test_result_without_url_is_dropped_and_counted(
            self, tavily, monkeypatch):
        _mock_urlopen(monkeypatch, [self._body([
            {"title": "ghost", "url": "", "content": "no url", "score": 0.9},
            {"title": "ok", "url": "https://x.dev/a", "content": "c",
             "score": 0.5},
        ])])
        out = tavily.search("x")
        assert len(out) == 1 and out[0].url == "https://x.dev/a"
        assert tavily.last_usage["skipped_missing_url"] == 1

    def test_excerpt_capped_with_honest_truncation_counter(
            self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        p = TavilySearchProvider(content_max_chars=300)
        _mock_urlopen(monkeypatch, [self._body([
            {"title": "long", "url": "https://x.dev/a",
             "content": "x" * 900, "score": 0.5},
        ])])
        out = p.search("x")
        assert len(out[0].excerpt) == 300
        assert out[0].excerpt_truncated_chars == 600

    def test_empty_query_rejected(self, tavily):
        with pytest.raises(WebSearchError, match="empty"):
            tavily.search("   ")


class TestErrorsAndRetries:
    def test_auth_failure_is_classified_not_retried(self, tavily, monkeypatch):
        calls = _mock_urlopen(monkeypatch, [_http_error(401, "bad key")])
        with pytest.raises(WebSearchError) as ei:
            tavily.search("q")
        assert ei.value.kind == "auth"
        assert len(calls) == 1

    def test_rate_limit_retries_then_fails_with_kind(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        p = TavilySearchProvider(retries=2)
        calls = _mock_urlopen(monkeypatch, [_http_error(429)] * 3)
        with pytest.raises(WebSearchError) as ei:
            p.search("q")
        assert ei.value.kind == "rate_limit"
        assert len(calls) == 3  # initial + 2 bounded retries

    def test_plan_limit_never_retried(self, tavily, monkeypatch):
        calls = _mock_urlopen(monkeypatch, [_http_error(432, "plan limit")])
        with pytest.raises(WebSearchError) as ei:
            tavily.search("q")
        assert ei.value.kind == "plan"
        assert len(calls) == 1

    def test_server_error_retries_then_fails(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        p = TavilySearchProvider(retries=1)
        calls = _mock_urlopen(monkeypatch, [_http_error(500), _http_error(502)])
        with pytest.raises(WebSearchError) as ei:
            p.search("q")
        assert ei.value.kind == "server"
        assert len(calls) == 2

    def test_network_timeout_classified(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        p = TavilySearchProvider(retries=0)

        def boom(req, timeout=None):
            raise TimeoutError("slow")

        monkeypatch.setattr("intentgraph.websearch.urllib.request.urlopen", boom)
        with pytest.raises(WebSearchError) as ei:
            p.search("q")
        assert ei.value.kind == "network"

    def test_null_provider_never_constructs_requests(self):
        p = NullWebSearchProvider()
        with pytest.raises(WebSearchError) as ei:
            p.search("q")
        assert ei.value.kind == "disabled"


class TestExplicitStorage:
    def test_store_external_result_writes_full_provenance(self):
        class _FakeIG:
            def __init__(self):
                self.calls = []

            def ingest(self, content, **kw):
                self.calls.append((content, kw))
                from intentgraph.ingest import IngestResult
                from intentgraph.models import MemoryRecord
                rec = MemoryRecord(content=content)
                return IngestResult(rec)

        ig = _FakeIG()
        ev = ExternalEvidence(url="https://x.dev/a", title="T", excerpt="E",
                              query="q", provider="tavily", score=0.5,
                              published_date="2026-01-01",
                              retrieved_at="2026-10-16T00:00:00+00:00")
        rid = store_external_result(ig, ev, query="q")
        content, kw = ig.calls[0]
        assert kw["source_type"] == "url"
        assert kw["source_id"] == "https://x.dev/a"
        assert "external" in kw["tags"]
        prov = kw["provenance"]
        assert prov["origin"] == "external_search"
        assert prov["url"] == "https://x.dev/a"
        assert prov["published_date_is_provider_estimate"] is True
        assert prov["search_query"] == "q"
        assert rid and rid.startswith("mem_")


# --- the .env loader (Phase 3 credential wiring) ---------------------------------------


class TestLoadEnvFile:
    def test_loads_unset_vars_and_never_overrides(self, tmp_path, monkeypatch):
        from intentgraph.config import load_env_file
        monkeypatch.delenv("INTENTGRAPH_DISABLE_DOTENV", raising=False)
        env = tmp_path / ".env"
        env.write_text(
            "# comment\n"
            "INTENTGRAPH_TEST_KEY_A=alpha\n"
            'INTENTGRAPH_TEST_KEY_B="quoted value"\n'
            "INTENTGRAPH_TEST_KEY_C=1\n"
            "not-a-valid-line\n",
            encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        # a real env var always wins over the file
        monkeypatch.setenv("INTENTGRAPH_TEST_KEY_A", "from-real-env")
        applied = load_env_file()
        import os
        assert os.environ["INTENTGRAPH_TEST_KEY_A"] == "from-real-env"
        assert os.environ["INTENTGRAPH_TEST_KEY_B"] == "quoted value"
        assert os.environ["INTENTGRAPH_TEST_KEY_C"] == "1"
        assert applied == 2  # only the two unset variables applied

    def test_parent_walk_finds_workspace_env(self, tmp_path, monkeypatch):
        from intentgraph.config import load_env_file
        monkeypatch.delenv("INTENTGRAPH_DISABLE_DOTENV", raising=False)
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (tmp_path / ".env").write_text(
            "INTENTGRAPH_TEST_WALK=found-it\n", encoding="utf-8")
        monkeypatch.chdir(nested)  # three levels below the .env
        load_env_file()
        import os
        assert os.environ["INTENTGRAPH_TEST_WALK"] == "found-it"

    def test_disable_flag_is_a_complete_noop(self, tmp_path, monkeypatch):
        from intentgraph.config import load_env_file
        env = tmp_path / ".env"
        env.write_text("INTENTGRAPH_TEST_DISABLED=x\n", encoding="utf-8")
        monkeypatch.setenv("INTENTGRAPH_DISABLE_DOTENV", "1")
        monkeypatch.chdir(tmp_path)
        assert load_env_file() == 0
        import os
        assert "INTENTGRAPH_TEST_DISABLED" not in os.environ

    def test_missing_file_is_silently_fine(self, tmp_path, monkeypatch):
        from intentgraph.config import load_env_file
        monkeypatch.delenv("INTENTGRAPH_DISABLE_DOTENV", raising=False)
        monkeypatch.chdir(tmp_path)
        assert load_env_file() == 0

    def test_secret_values_never_echoed_by_loader(self, tmp_path, monkeypatch,
                                                   capsys):
        from intentgraph.config import load_env_file
        monkeypatch.delenv("INTENTGRAPH_DISABLE_DOTENV", raising=False)
        (tmp_path / ".env").write_text(
            "NEBIUS_API_KEY=v1.supersecret-not-a-real-key\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        load_env_file()
        out = capsys.readouterr()
        assert "supersecret" not in out.out + out.err
