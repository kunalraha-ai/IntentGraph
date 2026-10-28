"""External web search: the Tavily provider behind a provider interface.

External evidence is a *different class of thing* from user memories, and
this module enforces that structurally:

* Search results are :class:`ExternalEvidence` objects — never MemoryRecords
  and never mixed into the store implicitly. They carry their own provenance
  (canonical URL, title, provider-supplied excerpt, provider's *estimated*
  publish date or an honest None, retrieval timestamp, the query that found
  them, provider name, relevance score, and request/usage metadata).
* Persisting an external finding into the memory archive is an explicit,
  opt-in action (:meth:`TavilySearchProvider.store_result`, wired to the
  ``web_search.store_results`` config flag, default **false**). By default a
  search result lives only inside the context packet for the query that
  produced it.
* An external result never overwrites or "corrects" a user memory. The
  controller keeps both sides visible with their own provenance and lets the
  consuming model compare them.
* Web content is untrusted input: excerpts are length-capped and rendered
  inside the packet's untrusted-evidence fences with their source URL.

Everything is off by default: ``web_search.provider`` defaults to ``none``,
and a local-only configuration refuses to construct the Tavily provider at
all. ``TAVILY_API_KEY`` is read from the environment only — never from a
config file and never echoed into logs or errors.

API shape (verified against the published OpenAPI, 2026-10):
``POST https://api.tavily.com/search``, Bearer auth, request fields
``query`` (required), ``search_depth`` (basic|advanced|fast|ultra-fast),
``max_results`` (≤20), ``topic`` (general|news|finance), ``time_range``,
``include_published_date`` (beta — the only way ``published_date`` is
returned; results without a detectable date return ``null``, which is
preserved as None, never invented), ``include_domains`` /
``exclude_domains``, ``include_usage``. Errors: 400/401/422 (no retry),
429 (retry honoring ``Retry-After``, bounded), 432/433 (plan limits, no
retry — reported distinctly), 5xx (bounded retry).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "ExternalEvidence",
    "WebSearchError",
    "WebSearchProvider",
    "NullWebSearchProvider",
    "TavilySearchProvider",
    "parse_provider_date",
    "provider_from_config",
    "store_external_result",
]

TAVILY_DEFAULT_BASE_URL = "https://api.tavily.com"

_MAX_RESULTS_CAP = 20          # Tavily's documented maximum
_EXCERPT_DEFAULT_CHARS = 2000  # per-result content cap (config: content_max_chars)
_REQUEST_BODY_CAP = 8_000      # query + parameters payload cap


class WebSearchError(Exception):
    """Search failure with an honest, non-retryable classification.

    ``kind`` distinguishes what a consumer should do: ``auth`` (bad/missing
    key), ``plan`` (quota/plan limit), ``rate_limit`` (retries exhausted),
    ``bad_request``, ``server``, ``network`` — so "provider unavailable" is
    never confused with "genuinely insufficient evidence".
    """

    def __init__(self, message: str, *, kind: str = "server"):
        super().__init__(message)
        self.kind = kind


@dataclass
class ExternalEvidence:
    """One web search result, kept structurally separate from memories."""

    url: str
    title: str
    excerpt: str                       # length-capped provider content
    query: str                         # the query that retrieved this
    provider: str                       # e.g. "tavily"
    score: float | None = None         # provider-supplied relevance, 0..1
    published_date: str | None = None  # provider's estimate; None = undated
    retrieved_at: str = ""             # ISO retrieval timestamp (always set)
    request_id: str = ""
    response_time_s: float | None = None
    usage_credits: int | None = None
    #: explicit trust notes — what this evidence IS and IS NOT
    trust: dict[str, Any] = field(default_factory=lambda: {
        "origin": "external",
        "verified": False,            # never auto-verified, regardless of score
        "content_is": "provider-supplied excerpt, not the full page",
    })
    excerpt_truncated_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url, "title": self.title, "excerpt": self.excerpt,
            "query": self.query, "provider": self.provider, "score": self.score,
            "published_date": self.published_date,
            "retrieved_at": self.retrieved_at,
            "request_id": self.request_id,
            "response_time_s": self.response_time_s,
            "usage_credits": self.usage_credits,
            "trust": self.trust,
            "excerpt_truncated_chars": self.excerpt_truncated_chars,
        }


class WebSearchProvider:
    """Interface: query → external evidence (or a classified error)."""

    name = "null"
    available = False

    def search(self, query: str) -> list[ExternalEvidence]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "available": self.available}


class NullWebSearchProvider(WebSearchProvider):
    """Web search disabled (the default). Search reports that honestly."""

    # "none" (not "null") so status surfaces read uniformly: the configured
    # provider IS none, which is a different statement from unconfigured
    name = "none"

    def search(self, query: str) -> list[ExternalEvidence]:
        raise WebSearchError("web search is not enabled (web_search.provider is "
                             "'none')", kind="disabled")


def _now() -> str:
    from intentgraph.models import utcnow
    return utcnow()


class TavilySearchProvider(WebSearchProvider):
    """Tavily search over the documented POST /search API (stdlib urllib).

    Configuration (all optional beyond the key):
    ``max_results`` (1..20, default 5), ``search_depth``
    (basic|advanced|fast|ultra-fast, default basic — advanced costs 2 credits),
    ``topic`` (general|news|finance), ``time_range``
    (day|week|month|year), ``include_domains``/``exclude_domains`` lists,
    ``timeout_s`` (default 15), ``retries`` (default 2, on 429/5xx only),
    ``content_max_chars`` (default 2000, excerpt cap).
    """

    name = "tavily"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = TAVILY_DEFAULT_BASE_URL,
        max_results: int = 5,
        search_depth: str = "basic",
        topic: str = "general",
        time_range: str = "",
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        timeout_s: float = 15.0,
        retries: int = 2,
        content_max_chars: int = _EXCERPT_DEFAULT_CHARS,
    ):
        self.api_key = api_key if api_key is not None else \
            os.environ.get("TAVILY_API_KEY", "")
        # no key = provider constructed but not available: the failure is
        # reported at first use (and by `available`), never a silent no-op
        self.base_url = base_url.rstrip("/")
        self.max_results = max(1, min(_MAX_RESULTS_CAP, int(max_results)))
        if search_depth not in ("basic", "advanced", "fast", "ultra-fast"):
            raise ValueError(f"unknown search_depth: {search_depth!r}")
        if topic not in ("general", "news", "finance"):
            raise ValueError(f"unknown topic: {topic!r}")
        if time_range and time_range not in ("day", "week", "month", "year",
                                             "d", "w", "m", "y"):
            raise ValueError(f"unknown time_range: {time_range!r}")
        self.search_depth = search_depth
        self.topic = topic
        self.time_range = time_range
        self.include_domains = [str(d) for d in (include_domains or [])][:300]
        self.exclude_domains = [str(d) for d in (exclude_domains or [])][:150]
        self.timeout_s = float(timeout_s)
        self.retries = max(0, int(retries))
        self.content_max_chars = max(200, int(content_max_chars))
        self.last_usage: dict[str, Any] = {}  # usage/request metadata, no content
        #: number of search() invocations (not retried HTTP attempts) —
        #: the observable a demo/test can assert "zero external requests"
        #: with against the REAL provider, not just the mock
        self.calls_made = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> dict[str, Any]:
        # never the key itself — only whether one is present
        return {"provider": self.name, "available": self.available,
                "base_url": self.base_url,
                "max_results": self.max_results,
                "search_depth": self.search_depth,
                "api_key_present": bool(self.api_key)}

    def search(self, query: str) -> list[ExternalEvidence]:
        if not self.api_key:
            raise WebSearchError(
                "Tavily search is configured but TAVILY_API_KEY is not set "
                "(export it or unset web_search)", kind="auth")
        query = str(query).strip()
        if not query:
            raise WebSearchError("empty search query", kind="bad_request")
        if len(query) > _REQUEST_BODY_CAP:
            query = query[:_REQUEST_BODY_CAP]
        self.calls_made += 1

        payload: dict[str, Any] = {
            "query": query,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "topic": self.topic,
            # the only way published_date comes back at all; results without
            # a detectable date return null — preserved as None, never faked
            "include_published_date": True,
            "include_usage": True,
        }
        if self.time_range:
            payload["time_range"] = self.time_range
        if self.include_domains:
            payload["include_domains"] = self.include_domains
        if self.exclude_domains:
            payload["exclude_domains"] = self.exclude_domains

        body = self._post(payload)
        results = body.get("results") or []
        out: list[ExternalEvidence] = []
        skipped_missing_url = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url.strip():
                # a result with no URL has no citable source: skip it and
                # report the count rather than inventing a reference
                skipped_missing_url += 1
                continue
            content = item.get("content") or ""
            truncated = 0
            if len(content) > self.content_max_chars:
                truncated = len(content) - self.content_max_chars
                content = content[: self.content_max_chars]
            pub = item.get("published_date")
            out.append(ExternalEvidence(
                url=url.strip(),
                title=str(item.get("title") or url),
                excerpt=content,
                query=query,
                provider=self.name,
                score=float(item["score"]) if isinstance(item.get("score"),
                                                          (int, float)) else None,
                published_date=str(pub) if pub else None,
                retrieved_at=_now(),
                request_id=str(body.get("request_id") or ""),
                response_time_s=(float(body["response_time"])
                                 if isinstance(body.get("response_time"),
                                               (int, float)) else None),
                usage_credits=(int(body.get("usage", {}).get("credits"))
                               if isinstance(body.get("usage"), dict) else None),
                excerpt_truncated_chars=truncated,
            ))
        self.last_usage = {
            "request_id": body.get("request_id"),
            "response_time_s": body.get("response_time"),
            "usage": body.get("usage"),
            "results": len(out),
            "skipped_missing_url": skipped_missing_url,
        }
        return out

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/search", data=data, method="POST",
            headers={"Content-Type": "application/json",
                     # key goes in the header only; never logged, never in
                     # exception text (errors quote status, not the request)
                     "Authorization": f"Bearer {self.api_key}"})
        last_exc: WebSearchError | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                code = exc.code
                detail = self._error_detail(exc)
                if code in (400, 401, 422):
                    raise WebSearchError(
                        f"tavily rejected the request (HTTP {code}): {detail}",
                        kind="auth" if code == 401 else "bad_request") from exc
                if code in (432, 433):
                    raise WebSearchError(
                        f"tavily plan/usage limit reached (HTTP {code}): {detail}",
                        kind="plan") from exc
                if code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    wait = (float(retry_after) if isinstance(retry_after, str)
                            and _is_number(retry_after)
                            else 0.5 * (attempt + 1))
                    last_exc = WebSearchError(
                        f"tavily rate limited (HTTP 429): {detail}", kind="rate_limit")
                    if attempt < self.retries:
                        time.sleep(min(wait, 30.0))
                        continue
                    raise last_exc from exc
                if code >= 500:
                    last_exc = WebSearchError(
                        f"tavily server error (HTTP {code})", kind="server")
                    if attempt < self.retries:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    raise last_exc from exc
                raise WebSearchError(f"tavily HTTP {code}: {detail}",
                                     kind="server") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = WebSearchError(f"tavily unreachable: {type(exc).__name__}",
                                          kind="network")
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise last_exc from exc
            except json.JSONDecodeError as exc:
                raise WebSearchError(f"tavily returned invalid JSON: {exc}",
                                     kind="server") from exc
        raise last_exc or WebSearchError("tavily search failed", kind="server")

    @staticmethod
    def _error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            detail = parsed.get("detail")
            if isinstance(detail, dict):
                return str(detail.get("error", body))[:300]
            return str(detail or body)[:300]
        except Exception:
            return f"HTTP {exc.code}"


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def parse_provider_date(raw: str | None):
    """Parse a provider-supplied publication date into an aware datetime.

    Tavily's documented example format is RFC 822 ("Tue, 11 Mar 2025
    17:00:00 GMT"); ISO-8601 is also accepted. Returns None when the
    provider sent nothing parseable — never a fabricated date.
    """
    if not raw:
        return None
    from datetime import timezone

    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(str(raw))
    except (TypeError, ValueError, IndexError):
        from intentgraph.models import parse_timestamp
        return parse_timestamp(str(raw))
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def provider_from_config(cfg: dict[str, Any] | None,
                         *, local_only: bool = False) -> WebSearchProvider:
    """Build the configured search provider; ``none`` is the default.

    ``local_only`` refuses the Tavily provider outright — a privacy
    hard-stop raised before any request could be made, mirroring the
    embedding provider's rule.
    """
    cfg = dict(cfg or {})
    kind = str(cfg.get("provider", "none")).strip().lower()
    if kind in ("", "none", "null"):
        return NullWebSearchProvider()
    if kind in ("tavily", "tavily-search"):
        if local_only:
            raise PermissionError(
                "privacy.local_only is set but web search is configured "
                f"({kind}); external search would send the query off-machine. "
                "Remove the provider or unset local_only.")
        key = cfg.get("api_key") or os.environ.get("TAVILY_API_KEY", "")
        return TavilySearchProvider(
            api_key=key,
            base_url=str(cfg.get("base_url") or TAVILY_DEFAULT_BASE_URL),
            max_results=int(cfg.get("max_results", 5)),
            search_depth=str(cfg.get("search_depth", "basic")),
            topic=str(cfg.get("topic", "general")),
            time_range=str(cfg.get("time_range", "")),
            include_domains=cfg.get("include_domains") or None,
            exclude_domains=cfg.get("exclude_domains") or None,
            timeout_s=float(cfg.get("timeout_s", 15)),
            retries=int(cfg.get("retries", 2)),
            content_max_chars=int(cfg.get("content_max_chars",
                                           _EXCERPT_DEFAULT_CHARS)),
        )
    raise ValueError(f"unknown web search provider: {kind!r} (none|tavily)")


def store_external_result(ig: Any, ev: ExternalEvidence, *, query: str) -> str | None:
    """Explicitly persist one external finding as a memory record.

    Only called when the user opted in (``web_search.store_results: true``).
    The record's provenance carries the FULL external metadata (url,
    provider, request id, retrieval time, score, the fact the date is the
    provider's estimate) so a future reader can audit exactly where the
    content came from — and delete or supersede it like any memory.
    """
    content = ev.excerpt if ev.excerpt.strip() else ev.title
    res = ig.ingest(
        content,
        type="document",
        tags=["external", "web", ev.provider],
        source_type="url",
        source_id=ev.url,
        provenance={
            "origin": "external_search",
            "provider": ev.provider,
            "url": ev.url,
            "title": ev.title,
            "search_query": query,
            "retrieved_at": ev.retrieved_at,
            "published_date_is_provider_estimate": True,
            "published_date": ev.published_date,
            "provider_score": ev.score,
            "request_id": ev.request_id,
            "note": "ingested from web search; content is the provider's "
                    "excerpt, not the full page",
        },
    )
    return res.record.id if res.duplicate_of is None else res.duplicate_of.id
