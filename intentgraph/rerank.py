"""Optional remote reranking — Nebius Token Factory's /v1/rerank endpoint.

Verified against the published OpenAPI (2026-10): Token Factory exposes a
vLLM-style rerank protocol —

    POST /v1/rerank  {"model": "...", "query": "...", "documents": [str]}
    → {"results": [{"index": int, "document": {"text": str},
                    "relevance_score": float}], "usage": {...}}

**Availability caveat, stated honestly:** the endpoint is documented, but
whether a specific reranker model is served to a given account is an
account-level fact we cannot verify without credentials. The model id must
be configured explicitly (the API's own example uses
``Qwen/Qwen3-Reranker-8B`` — that is a documented example, NOT a default we
assume is universally available). Run the live smoke
(``INTENTGRAPH_LIVE_TESTS=1``) or ``intentgraph embed --list-models`` against
your account to see what yours serves.

Reranking improves ORDER only: it reorders the already-fused candidate list.
It never adds or drops evidence and never decides factual truth — scores are
relevance estimates from the provider, carried as metadata.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from intentgraph.embeddings import NEBIUS_DEFAULT_BASE_URL, nebius_api_key_from_env

__all__ = [
    "RerankError",
    "Reranker",
    "NullReranker",
    "NebiusReranker",
    "reranker_from_config",
]

_MAX_DOCS = 64          # bounded: rerank candidates, never whole archives
_MAX_DOC_CHARS = 4000   # per-document cap sent to the provider


class RerankError(Exception):
    """Reranking failure — recoverable (caller keeps the fused order)."""


class Reranker:
    """Interface: (query, documents) → relevance order."""

    name = "null"
    available = False

    def rerank(self, query: str, documents: Sequence[str]) -> list[dict[str, Any]]:
        """→ [{"index": i, "score": s}, ...] ordered best-first, covering
        every input index exactly once."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "available": self.available}


class NullReranker(Reranker):
    """Reranking disabled (the default) — fused order stands."""

    # "none" for uniform status reporting (see NullWebSearchProvider)
    name = "none"

    def rerank(self, query: str, documents: Sequence[str]) -> list[dict[str, Any]]:
        return [{"index": i, "score": 1.0} for i in range(len(documents))]


class NebiusReranker(Reranker):
    """Token Factory /v1/rerank (vLLM scoring protocol, Bearer auth).

    Bounded retries on 429/5xx only; per-document char caps (the provider
    sees excerpts, not whole records); validates that the response covers
    every input index exactly once; captures usage without content.
    """

    name = "nebius"

    def __init__(self, *, model: str, api_key: str, base_url: str,
                 timeout_s: float = 20.0, retries: int = 2):
        if not model:
            raise ValueError("reranker requires an explicitly configured model — "
                             "check what your account serves with "
                             "`intentgraph embed --list-models`")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.retries = max(0, int(retries))
        self.last_usage: dict[str, Any] = {}
        self.calls_made = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model,
                "available": self.available,
                "api_key_present": bool(self.api_key)}

    def rerank(self, query: str, documents: Sequence[str]) -> list[dict[str, Any]]:
        if not documents:
            return []
        if not self.api_key:
            raise RerankError("reranker configured but no API key resolved")
        docs = [d[:_MAX_DOC_CHARS] for d in documents[:_MAX_DOCS]]
        payload = json.dumps({"model": self.model, "query": query,
                              "documents": docs}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/rerank", data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        self.calls_made += 1
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                if isinstance(body.get("usage"), dict):
                    self.last_usage = dict(body["usage"])
                results = body.get("results") or []
                out = []
                seen: set[int] = set()
                for item in results:
                    idx = item.get("index")
                    score = item.get("relevance_score")
                    if not isinstance(idx, int) or idx in seen \
                            or idx < 0 or idx >= len(docs) \
                            or not isinstance(score, (int, float)):
                        raise RerankError("reranker returned a malformed result")
                    seen.add(idx)
                    out.append({"index": idx, "score": float(score)})
                if seen != set(range(len(docs))):
                    raise RerankError(
                        f"reranker covered {len(seen)} of {len(docs)} documents")
                out.sort(key=lambda r: (-r["score"], r["index"]))
                return out
            except RerankError:
                raise
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise RerankError(
                        f"reranker rejected credentials (HTTP {exc.code})") from exc
                if exc.code == 404:
                    raise RerankError(
                        "rerank endpoint or model not found (HTTP 404) — the "
                        "model may not be served to this account") from exc
                last = exc
                if exc.code in (429, 500, 502, 503) and attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RerankError(f"rerank request failed (HTTP {exc.code})") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RerankError(
                    f"reranker unreachable: {type(exc).__name__}") from exc
            except json.JSONDecodeError as exc:
                raise RerankError(f"reranker returned invalid JSON: {exc}") from exc
        raise RerankError(f"rerank failed after retries: {last}")


def reranker_from_config(cfg: dict[str, Any] | None,
                         *, local_only: bool = False) -> Reranker:
    """Build the configured reranker; ``none`` is the default.

    ``local_only`` refuses a remote reranker — sending excerpts off-machine
    is a network provider like any other.
    """
    cfg = dict(cfg or {})
    kind = str(cfg.get("provider", "none")).strip().lower()
    if kind in ("", "none", "null"):
        return NullReranker()
    if kind in ("nebius", "openai_compatible", "openai-compatible"):
        if local_only:
            raise PermissionError(
                "privacy.local_only is set but a reranker is configured "
                f"({kind}) — reranking would send record excerpts off-machine")
        base = cfg.get("base_url") or NEBIUS_DEFAULT_BASE_URL
        model = cfg.get("model") or ""
        key = (cfg.get("api_key") or nebius_api_key_from_env()
               or "")
        return NebiusReranker(model=model, api_key=key, base_url=base,
                               timeout_s=float(cfg.get("timeout_s", 20)),
                               retries=int(cfg.get("retries", 2)))
    raise ValueError(f"unknown reranker provider: {kind!r} (none|nebius)")
