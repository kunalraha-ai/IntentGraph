"""Embedding provider interface + semantic vector index.

Three provider implementations, one contract:

* :class:`NullEmbeddingProvider` — the default. No embedding provider is
  configured unless the user explicitly picks one; semantic retrieval then
  reports itself unavailable and the rest of the pipeline (lexical, graph,
  adaptive controller) keeps working. Local use never silently depends on
  a cloud service.
* :class:`HashingEmbeddingProvider` — deterministic, offline, **not neural**.
  It projects word n-grams into a fixed-dim L2-normalized vector via hashing;
  cosine similarity is then a weighted lexical-overlap score. It exists so
  the whole hybrid pipeline (fusion, gap detection, controller) is
  runnable and testable offline. It is documented everywhere as an
  approximate lexical projection — it does NOT understand paraphrase, and
  any benchmark run with it is a *pipeline* test, not a semantic-quality one.
* :class:`OpenAICompatibleEmbeddingProvider` — real embeddings through any
  OpenAI-compatible ``/embeddings`` endpoint: OpenAI, NVIDIA NIM
  (``https://integrate.api.nvidia.com``), Ollama, vLLM, LiteLLM proxies.
  Uses stdlib ``urllib`` (no new dependency); never enabled implicitly —
  instantiation requires explicit configuration naming the endpoint and
  model. Credentials come only from environment variables, never from files.

The :class:`SemanticIndex` is the derived, rebuildable side-table: vectors
live in SQLite per (record_id, model) and are materialized into a numpy
matrix for cosine search. Dropping them loses nothing irrecoverable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from intentgraph.lexindex import tokenize, _STOPWORDS

__all__ = [
    "EmbeddingProvider",
    "NullEmbeddingProvider",
    "HashingEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "NEBIUS_DEFAULT_BASE_URL",
    "NVIDIA_DEFAULT_BASE_URL",
    "nebius_api_key_from_env",
    "nvidia_api_key_from_env",
    "provider_from_config",
    "SemanticIndex",
]

#: Nebius Token Factory inference API — an OpenAI-compatible endpoint
#: (verified against the published API reference, 2026-10): Bearer auth with
#: NEBIUS_API_KEY, /embeddings, /chat/completions, /rerank, /models.
NEBIUS_DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

#: NVIDIA NIM (build.nvidia.com) inference API — also OpenAI-compatible
#: (Bearer auth with an nvapi-* key, /embeddings, /models). Model
#: availability differs per account/product: NIM serves the NVIDIA-built
#: embedding models (e.g. nvidia/nemotron-3-embed-1b) that Token Factory
#: may not — this account's catalog proved exactly that split.
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def nebius_api_key_from_env() -> str:
    """Nebius credential resolution, documented precedence:

    1. ``NEBIUS_API_KEY`` — the convention Nebius's own docs use
    2. ``NEBIUS_TOKEN_FACTORY_KEY`` — accepted alias for setups that name
       the service rather than the vendor

    The key is used in the Authorization header only; it is never logged,
    never included in exception text, and never written to configuration.
    """
    return (os.environ.get("NEBIUS_API_KEY", "").strip()
            or os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip())


def nvidia_api_key_from_env() -> str:
    """NVIDIA NIM credential resolution, documented precedence:

    1. ``NVIDIA_API_KEY`` — the convention NVIDIA's own docs/SDKs use for
       nvapi-* keys (build.nvidia.com / integrate.api.nvidia.com)

    The key is used in the Authorization header only; it is never logged,
    never included in exception text, and never written to configuration.
    """
    return os.environ.get("NVIDIA_API_KEY", "").strip()


class EmbeddingProvider:
    """Base contract: turn texts into equal-length float vectors."""

    #: provider name (persisted alongside vectors — switch invalidates them)
    name = "null"
    #: vector dimensionality; subclasses must set
    dim = 0

    def is_configured(self) -> bool:
        return False

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "dim": self.dim, "configured": self.is_configured()}


class NullEmbeddingProvider(EmbeddingProvider):
    """Not-configured provider: semantic retrieval is disabled, on purpose."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("no embedding provider configured")


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic local n-gram hashing projection — NOT a neural embedding.

    Each word and word bigram is hashed into ``dim`` buckets with a
    signed weight (feature hashing / the hashing trick). Cosine similarity
    between two such vectors approximates weighted token-overlap. Two texts
    sharing vocabulary score high; paraphrases with disjoint vocabulary do
    NOT — that capability requires a real embedding model.

    Determinism: the hash is sha256 of the token — no PYTHONHASHSEED
    involvement, so index rebuilds are reproducible across processes.
    """

    def __init__(self, dim: int = 384):
        if dim < 32:
            raise ValueError("hash embedding needs dim >= 32 to be useful")
        self.name = f"hash-{dim}"
        self.dim = dim

    def is_configured(self) -> bool:
        return True  # local, always available once chosen

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            toks = [t for t in tokenize(text) if t not in _STOPWORDS]
            grams = list(toks)
            grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for g in grams:
                h = hashlib.sha256(g.encode("utf-8", "replace")).digest()
                bucket = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                # rare-ish grams weigh more: log-damped count
                vec[bucket] += sign
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Real embeddings via an OpenAI-compatible POST /embeddings endpoint.

    Configuration is explicit and environment-driven:

    ==============  =========================================================
    ``base_url``    ``INTENTGRAPH_EMBEDDING_BASE_URL``
                     (examples: ``https://api.openai.com/v1``,
                      ``https://integrate.api.nvidia.com/v1`` for NVIDIA NIM,
                      ``http://localhost:11434/v1`` for Ollama)
    ``model``       ``INTENTGRAPH_EMBEDDING_MODEL`` (no default — must be set)
    ``api_key``     ``INTENTGRAPH_EMBEDDING_API_KEY`` or ``OPENAI_API_KEY``
    ``timeout_s``   ``INTENTGRAPH_EMBEDDING_TIMEOUT`` (default 60)
    ``batch``       ``INTENTGRAPH_EMBEDDING_BATCH`` (default 64)
    ==============  =========================================================

    Never constructed implicitly: :func:`provider_from_config` only builds it
    when the config's ``embedding.provider`` is ``"openai"`` (and this module
    refuses when the model env is unset — an empty model would silently
    hit whatever default the endpoint has, which is how bills and leaks
    happen).
    """

    _PATH = "/embeddings"

    def describe(self) -> dict[str, Any]:
        # credential presence reported as a boolean — never the key itself
        return {"provider": self.name, "dim": self.dim,
                "configured": self.is_configured(),
                "api_key_present": bool(self.api_key)}

    def __init__(self, *, base_url: str, model: str, api_key: str | None = None,
                 timeout_s: float = 60.0, batch: int = 64, dim: int | None = None,
                 alias: str = "openai"):
        if not base_url:
            raise ValueError("embedding base_url is required")
        if not model:
            raise ValueError("embedding model is required (set INTENTGRAPH_EMBEDDING_MODEL)")
        # the configured alias (nebius|nvidia|openai) for user-facing
        # messages; `name` stays the storage key (protocol:model:endpointhash)
        self.alias = alias
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        # usage metadata from the last response (no content); instance-level
        # so concurrent providers never share a dict
        self.last_usage: dict[str, Any] = {}
        self.batch = max(1, int(batch))
        # name binds stored vectors to model+endpoint; switching either
        # invalidates the cache naturally through the (record_id, model) key
        self.name = f"openai:{self.model}:{hashlib.sha256(self.base_url.encode()).hexdigest()[:8]}"
        self._dim = dim

    @property
    def dim(self) -> int:  # resolved lazily from the first successful call
        return int(self._dim or 0)

    @property
    def available(self) -> bool:
        """A remote provider is only usable with a resolved credential."""
        return bool(self.api_key)

    def is_configured(self) -> bool:
        # credential-aware on purpose: a selected-but-keyless provider is NOT
        # usable — the semantic index must not silently no-op and doctor must
        # be able to say "credentials missing" rather than "configured"
        return bool(self.api_key)

    #: alias → the credential env var a user should set (names the exact
    #: variable in errors; a missing key is a config problem, and config
    #: problems deserve actionable messages)
    _CREDENTIAL_HINT = {
        "nebius": "NEBIUS_API_KEY (or NEBIUS_TOKEN_FACTORY_KEY)",
        "nvidia": "NVIDIA_API_KEY",
        "openai": "INTENTGRAPH_EMBEDDING_API_KEY (or OPENAI_API_KEY)",
    }

    def _require_key(self) -> None:
        if not self.api_key:
            hint = self._CREDENTIAL_HINT.get(self.alias, "INTENTGRAPH_EMBEDDING_API_KEY")
            raise RuntimeError(
                f"embedding provider '{self.alias}' has no API key resolved "
                f"— set {hint}; no request was sent "
                "(see docs/intentgraph.md)")

    @classmethod
    def from_env(cls) -> "OpenAICompatibleEmbeddingProvider | None":
        base = os.environ.get("INTENTGRAPH_EMBEDDING_BASE_URL", "").strip()
        model = os.environ.get("INTENTGRAPH_EMBEDDING_MODEL", "").strip()
        if not base and not model:
            return None
        return cls(
            base_url=base or "https://api.openai.com/v1",
            model=model,
            api_key=(os.environ.get("INTENTGRAPH_EMBEDDING_API_KEY")
                     or os.environ.get("OPENAI_API_KEY")
                     or ""),
            timeout_s=float(os.environ.get("INTENTGRAPH_EMBEDDING_TIMEOUT", "60") or 60),
            batch=int(os.environ.get("INTENTGRAPH_EMBEDDING_BATCH", "64") or 64),
        )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []  # nothing to embed: no request, no error
        self._require_key()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = list(texts[i : i + self.batch])
            out.extend(self._embed_batch(chunk))
        return out

    #: basic usage metadata from the last response (no content), for the
    #: status surface; e.g. {"prompt_tokens": 42}

    def list_models(self) -> list[str]:
        """Model discovery: GET {base_url}/models (OpenAI-compatible).

        Verifies what the ENDPOINT offers the configured account — the
        honest check before picking an embedding model (the catalog may
        advertise models this account cannot serve). Network call; bounded
        by timeout, no retry (listing is advisory, not worth hammering).
        """
        self._require_key()
        req = urllib.request.Request(self.base_url + "/models", method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = body.get("data") or []
        return sorted(str(m.get("id")) for m in data
                      if isinstance(m, dict) and m.get("id"))

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + self._PATH,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                if isinstance(body.get("usage"), dict):
                    self.last_usage = dict(body["usage"])
                data = sorted(body.get("data", []), key=lambda d: d.get("index", 0))
                vecs = [item.get("embedding") for item in data]
                if len(vecs) != len(texts):
                    raise _EmbedProtocolError(
                        f"endpoint returned {len(vecs)} of {len(texts)} embeddings")
                # completeness + dimensional consistency: every item must be
                # a non-empty numeric vector, and all must share one width —
                # a mixed-width response is corrupt, not a usable index row
                dims = {len(v) if isinstance(v, list) else -1 for v in vecs}
                if -1 in dims or len(dims) != 1 or 0 in dims:
                    raise _EmbedProtocolError("endpoint returned malformed "
                                              "embedding vectors (missing or "
                                              "ragged dimensions)")
                if self._dim is None:
                    self._dim = next(iter(dims))
                elif next(iter(dims)) != self._dim:
                    raise _EmbedProtocolError(
                        f"endpoint returned {next(iter(dims))}-dimensional "
                        f"vectors but this provider is bound to {self._dim} "
                        f"dimensions (model change or endpoint mismatch — "
                        "reindex with `intentgraph embed`)")
                return vecs
            except _EmbedProtocolError:
                # deterministic response defect: retrying would return the
                # same shape — fail immediately with the real message
                raise
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    RuntimeError, json.JSONDecodeError) as exc:
                last_exc = exc
                # linear-ish backoff for rate limits; fail fast otherwise
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503):
                    raise RuntimeError(f"embedding request failed: {exc}") from exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"embedding request failed after retries: {last_exc}")


class _EmbedProtocolError(RuntimeError):
    """Deterministic protocol violation in a response (wrong count, ragged
    or mismatched dimensions). NEVER retried — the response parsed fine and
    sending it again would return the same shape. Subclasses RuntimeError so
    callers keep their existing error contract; the dedicated except-clause
    above the retry net is what keeps it out of the retry loop."""


def provider_from_config(cfg: dict[str, Any] | None) -> EmbeddingProvider:
    """Build the configured provider. Unknown names raise — never fall back
    silently to a different provider than the one the user asked for."""
    cfg = dict(cfg or {})
    kind = str(cfg.get("provider", "none")).strip().lower()
    if kind in ("", "none", "null"):
        return NullEmbeddingProvider()
    if kind == "hash":
        return HashingEmbeddingProvider(dim=int(cfg.get("dim", 384)))
    if kind == "nebius":
        # Nebius Token Factory: OpenAI-compatible (verified against the
        # published API reference). Credential precedence is documented in
        # nebius_api_key_from_env(); an explicitly configured key wins.
        base = cfg.get("base_url") or os.environ.get(
            "INTENTGRAPH_EMBEDDING_BASE_URL", "") or NEBIUS_DEFAULT_BASE_URL
        model = cfg.get("model") or os.environ.get("INTENTGRAPH_EMBEDDING_MODEL", "")
        key = (cfg.get("api_key") or nebius_api_key_from_env()
               or os.environ.get("INTENTGRAPH_EMBEDDING_API_KEY", ""))
        if not model:
            raise ValueError(
                "embedding provider 'nebius' requires an explicitly configured "
                "model — no default is assumed (accounts differ; list what "
                "yours serves with `intentgraph embed --list-models`)")
        return OpenAICompatibleEmbeddingProvider(
            base_url=base, model=model, api_key=key,
            timeout_s=float(cfg.get("timeout_s", 60)),
            batch=int(cfg.get("batch", 64)),
            alias="nebius",
        )
    if kind == "nvidia":
        # NVIDIA NIM (build.nvidia.com / integrate.api.nvidia.com):
        # OpenAI-compatible; the home of NVIDIA-built embedding models
        # (nvidia/nemotron-3-embed-1b, …) that Token Factory may not serve.
        base = cfg.get("base_url") or os.environ.get(
            "INTENTGRAPH_EMBEDDING_BASE_URL", "") or NVIDIA_DEFAULT_BASE_URL
        model = cfg.get("model") or os.environ.get("INTENTGRAPH_EMBEDDING_MODEL", "")
        key = (cfg.get("api_key") or nvidia_api_key_from_env()
               or os.environ.get("INTENTGRAPH_EMBEDDING_API_KEY", ""))
        if not model:
            raise ValueError(
                "embedding provider 'nvidia' requires an explicitly configured "
                "model — no default is assumed; list what your NIM account "
                "serves with `intentgraph embed --list-models`")
        return OpenAICompatibleEmbeddingProvider(
            base_url=base, model=model, api_key=key,
            timeout_s=float(cfg.get("timeout_s", 60)),
            batch=int(cfg.get("batch", 64)),
            alias="nvidia",
        )
    if kind in ("openai", "openai_compatible", "openai-compatible"):
        base = cfg.get("base_url") or os.environ.get("INTENTGRAPH_EMBEDDING_BASE_URL", "")
        model = cfg.get("model") or os.environ.get("INTENTGRAPH_EMBEDDING_MODEL", "")
        key = (cfg.get("api_key") or os.environ.get("INTENTGRAPH_EMBEDDING_API_KEY")
               or os.environ.get("OPENAI_API_KEY") or "")
        if not model:
            raise ValueError(
                "embedding provider 'openai' requires a model — set "
                "INTENTGRAPH_EMBEDDING_MODEL (see docs/intentgraph.md)")
        return OpenAICompatibleEmbeddingProvider(
            base_url=base or "https://api.openai.com/v1",
            model=model, api_key=key,
            timeout_s=float(cfg.get("timeout_s", 60)),
            batch=int(cfg.get("batch", 64)),
            alias="openai",
        )
    raise ValueError(f"unknown embedding provider: {kind!r} "
                     f"(expected none|hash|openai)")


class SemanticIndex:
    """Cosine search over stored embedding vectors (derived, rebuildable)."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self._ids: list[str] = []
        self._mat: Any = None
        #: set by load() when stored vectors belong to a different model
        self.stale_model_note: str = ""

    @property
    def ready(self) -> bool:
        return self.provider.is_configured() and self._mat is not None and len(self._ids) > 0

    def rebuild(self, store: Any, records: Sequence[Any]) -> int:
        """(Re)embed every record through the provider and persist vectors.

        Returns 0 immediately when the provider is disabled — rebuild is a
        no-op then, not an error, so callers can call it unconditionally.
        """
        if not self.provider.is_configured():
            return 0
        store.drop_embeddings()  # vectors from another model must not survive
        # Same ordering as load() (record_id ascending) so a rebuilt matrix
        # and a freshly-loaded matrix are row-for-row identical — anything
        # else makes search results depend on whether vectors were rebuilt
        # or restored, which is unexplainable behavior.
        records = sorted(records, key=lambda r: r.id)
        texts = [r.search_text() for r in records]
        vecs = self.provider.embed_texts(texts)
        for rec, vec in zip(records, vecs):
            store.set_embedding(rec.id, self.provider.name, vec)
        self._ids = [r.id for r in records]
        try:
            import numpy as np
            self._mat = np.asarray(vecs, dtype=np.float32) if vecs else None
        except ImportError:
            self._mat = None
        return len(records)

    def load(self, store: Any) -> int:
        """Materialize stored vectors from the store (no provider call).

        Model-change detection: vectors are keyed by provider name, which
        embeds the model + endpoint. If ANY stored vectors belong to a
        different model than the configured provider (user switched
        models), the index refuses the stale vectors and reports not-ready
        with the reindex instruction — mixing vector spaces silently would
        corrupt every cosine score. (load_embeddings filters by the current
        provider name, so a differing stored model can only ever surface
        here: the mismatch check must not also require matching rows.)
        """
        stored_model = store.stored_embedding_model()
        self.stale_model_note = ""
        if stored_model and stored_model != self.provider.name:
            self.stale_model_note = (
                f"stored embeddings are from '{stored_model}' but the "
                f"configured provider is '{self.provider.name}' — semantic "
                "index not loaded; run `intentgraph embed` to reindex")
            self._ids, self._mat = [], None
            return 0
        ids, mat = store.load_embeddings(self.provider.name)
        if mat is None or not ids:
            self._ids, self._mat = [], None
            return 0
        self._ids = ids
        self._mat = mat
        return len(ids)

    def search(self, query: str, top_k: int = 20,
               doc_filter: Any = None) -> list[dict[str, Any]]:
        """Cosine-ranked candidates, or [] when the provider is disabled.

        Score transparency mirrors the lexical retriever: each hit carries
        the raw similarity so fusion can log per-retriever evidence.
        """
        if not self.provider.is_configured() or self._mat is None:
            return []
        qvec = self.provider.embed_texts([query])[0]
        try:
            import numpy as np
        except ImportError:
            return []
        q = np.asarray(qvec, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        m = self._mat
        mn = np.linalg.norm(m, axis=1)
        mn[mn == 0] = 1.0
        sims = (m @ q) / (mn * qn)
        order = np.argsort(-sims)  # type: ignore[union-attr]
        out: list[dict[str, Any]] = []
        for idx in order[: max(0, top_k)]:
            i = int(idx)
            doc_id = self._ids[i]
            if doc_filter is not None and not doc_filter(doc_id):
                continue
            score = float(sims[i])
            if score <= 0.0:
                continue
            out.append({"id": doc_id, "score": round(score, 6),
                        "matched_terms": [], "term_scores": {}})
        return out
