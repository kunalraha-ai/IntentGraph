"""IntentGraph SDK — the one class everything else (CLI, API, tests) uses.

:func:`IntentGraph` is a facade over the subsystems with one job: make the
vertical slice — ingest → connect → retrieve → verify → context — a few
lines of code, while keeping every subsystem usable on its own.

Lifecycle and index policy:

* The lexical index is rebuilt from the store on open (fast; it is O(archive)
  and runs in well under a second for realistic sizes) and updated
  incrementally on every ingest/update/delete through this facade.
* The semantic index materializes stored vectors on open (no provider call);
  embeddings are only computed when (a) a provider is configured AND
  (b) :meth:`reindex_semantic` is invoked (ingest or explicit). A remote
  provider therefore never causes a surprise per-query network call, and
  local-only operation is the default state, not a degraded mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from intentgraph.config import Config, load_config
from intentgraph.context import compile_context
from intentgraph.controller import (
    AdaptiveMemoryEngine,
    RetrievalTrace,
    RuleBasedPolicy,
)
from intentgraph.embeddings import (
    HashingEmbeddingProvider,
    NullEmbeddingProvider,
    provider_from_config,
    SemanticIndex,
)
from intentgraph.graphbridge import GraphifyBridge
from intentgraph.ingest import IngestResult, Ingestor
from intentgraph.lexindex import BM25Index
from intentgraph.models import MemoryEdge, MemoryRecord
from intentgraph.retrieve import Candidate, RetrievalEngine, candidates_to_dicts, fuse_rrf
from intentgraph.store import MemoryStore, StoreError
from intentgraph.verify import EvidenceReport

__all__ = ["IntentGraph"]


def _build_reranker(config: Config):
    """Reranker from config; honest ValueError when misconfigured."""
    from intentgraph.rerank import reranker_from_config
    return reranker_from_config(
        config.reranker,
        local_only=bool(config.privacy.get("local_only")))


def _build_llm_policy(config: Config, web_search: Any):
    """Optional LLM planning policy; None = deterministic default."""
    from intentgraph.llmassist import llm_policy_from_config
    # the policy may propose external search only when the provider is
    # configured AND enabled — enforced again at validation time
    external_enabled = (bool(config.web_search.get("enabled", False))
                        and getattr(web_search, "available", False))
    return llm_policy_from_config(
        config.llm,
        local_only=bool(config.privacy.get("local_only")),
        external_enabled=external_enabled)


class IntentGraph:
    """Programmatic entry point: a memory archive + retrieval intelligence."""

    def __init__(self, config: Config | None = None, **config_overrides: Any):
        # ``overrides={...}`` is accepted as an explicit dict of dotted config
        # paths (the API/tests pass it that way); any other kwarg must itself
        # be a valid dotted path or top-level field — an unknown key raises
        # instead of being silently dropped, because a silently-ignored
        # setting is exactly how a memory system gets misconfigured without
        # anyone noticing.
        overrides = dict(config_overrides.pop("overrides", {}) or {})
        for key, value in config_overrides.items():
            overrides[key] = value
        self.config = config or load_config(overrides=overrides or None)
        if not isinstance(self.config, Config):
            raise TypeError("config must be a intentgraph.config.Config")
        self.store = MemoryStore(self.config.db_path)
        self.ingestor = Ingestor(self.store)
        self.lexical = BM25Index()
        self.embedding_provider = provider_from_config(self.config.embedding)
        self.semantic_index = SemanticIndex(self.embedding_provider)
        self.bridge = GraphifyBridge(self.config.graphify.get("graph_path") or None)

        # Optional external providers — every one constructed ONLY when
        # explicitly configured (privacy.local_only is enforced inside
        # load_config before any of this can build a network client).
        from intentgraph.websearch import provider_from_config as _ws_from_config
        self.web_search = _ws_from_config(
            self.config.web_search,
            local_only=bool(self.config.privacy.get("local_only")))
        self.reranker = _build_reranker(self.config)
        self.llm_policy = _build_llm_policy(self.config, self.web_search)

        self._rebuild_indexes()
        # Full store mirror (active, superseded, deleted): visibility is a
        # query-time decision (_doc_filter), and bridge/source-id lookups
        # must see the whole archive to be able to honor include_inactive.
        self._records_cache: dict[str, MemoryRecord] = {
            r.id: r for r in self.store.iter_all_records(include_deleted=True)
        }
        self.engine = self._build_engine()
        self.controller = AdaptiveMemoryEngine(
            self.engine, self.store,
            policy=self.llm_policy or RuleBasedPolicy(),
            max_iterations=int(self.config.retrieval.get("max_iterations", 3)),
            max_working_set=int(self.config.retrieval.get("max_working_set", 24)),
            timeout_s=float(self.config.retrieval.get("timeout_s", 30.0)),
            web_search=(self.web_search
                        if self.config.web_search.get("enabled", False)
                        else None),
            external_max_calls=int(
                self.config.web_search.get("max_calls_per_query", 1)),
            external_store_results=bool(
                self.config.web_search.get("store_results", False)),
            ingest_hook=self,  # controller uses .ingest only when storing
        )

    # -- index plumbing --------------------------------------------------------

    def _rebuild_indexes(self) -> None:
        # Index the FULL archive — superseded and deleted records included —
        # and let doc_filter decide visibility per query. Indexing only the
        # active slice (the previous behavior) made the include_inactive
        # audit path structurally unable to retrieve superseded records
        # lexically: they simply were not in the postings. Filtering at the
        # candidate stage is where staleness policy belongs, because there
        # the reason for exclusion (active-only default vs audit view) is
        # known; at index build time it is not.
        records = list(self.store.iter_all_records(include_deleted=True))
        self.lexical.build((r.id, r.search_text()) for r in records)
        self.semantic_index.load(self.store)
        if self.bridge and self.config.graphify.get("graph_path"):
            self.bridge.index_records(records)

    def _doc_filter(self, include_inactive: bool = False):
        if include_inactive:
            return None
        return lambda rid: (rid in self._records_cache
                            and self._records_cache[rid].is_active)

    def _build_engine(self) -> RetrievalEngine:
        def lookup_source_id(needle: str) -> list[str]:
            out = []
            for rid, rec in self._records_cache.items():
                if needle and needle.lower() in (rec.source_id or "").lower():
                    out.append(rid)
            return sorted(out)

        def lookup_record_id(rid: str) -> list[str]:
            rec = self.store.get_record(rid)
            return [rid] if rec is not None else []

        return RetrievalEngine(
            lexical=self.lexical,
            semantic_index=self.semantic_index,
            edges_of=self.store.edges_for,
            lookup_source_id=lookup_source_id,
            lookup_record_id=lookup_record_id,
            weights=self.config.retrieval.get("weights"),
            graph_depth=int(self.config.retrieval.get("graph_depth", 2)),
            max_candidates=int(self.config.retrieval.get("max_candidates", 50)),
        )

    def reindex_semantic(self) -> dict[str, Any]:
        """(Re)compute embeddings for all active records through the provider.

        Only active records are embedded — vectors serve the default
        retrieval surface; superseded bodies stay reachable by id, via
        lexical audit, and through graph edges. No-op (reported, not
        silent) when no provider is configured — or when a remote provider
        is selected but its credential is missing, which is a distinct
        condition with its own message.
        """
        records = list(self.store.iter_all_records())
        n = self.semantic_index.rebuild(self.store, records)
        note = "" if n else "no provider configured or archive empty"
        if not n and self.embedding_provider.is_configured() is False \
                and not isinstance(self.embedding_provider,
                                   (NullEmbeddingProvider,
                                    HashingEmbeddingProvider)):
            note = (f"embedding provider '{self.embedding_provider.name}' is "
                    "selected but no API key resolved — set its credential "
                    "environment variable (see docs/intentgraph.md)")
        return {"embedded": n,
                "provider": self.embedding_provider.describe(),
                "note": note}

    # -- ingest / records ---------------------------------------------------------

    def ingest(self, content: str, **kwargs: Any) -> IngestResult:
        res = self.ingestor.ingest(content, **kwargs)
        if res.duplicate_of is None:
            self._post_add(res.record)
        return res

    def ingest_file(self, path: str | Path, **kwargs: Any) -> IngestResult:
        res = self.ingestor.ingest_file(path, **kwargs)
        if res.duplicate_of is None:
            self._post_add(res.record)
        return res

    def ingest_dict(self, data: dict[str, Any], **overrides: Any) -> IngestResult:
        res = self.ingestor.ingest_dict(data, **overrides)
        if res.duplicate_of is None:
            self._post_add(res.record)
        return res

    def _post_add(self, record: MemoryRecord) -> None:
        self.lexical.add(record.id, record.search_text())
        self._records_cache[record.id] = record
        if self.semantic_index.provider.is_configured() and self.semantic_index._mat is not None:
            # keep vectors fresh for incremental ingests when the provider is
            # local; a remote provider is batch-refreshed via reindex_semantic
            # (per-ingest HTTP calls would make ingest latency unbounded).
            pass

    def get(self, record_id: str, version: int | None = None) -> dict[str, Any] | None:
        rec = self.store.get_record(record_id, version=version)
        return rec.to_dict() if rec else None

    def get_versions(self, record_id: str) -> list[dict[str, Any]]:
        return self.store.record_versions(record_id)

    def list(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        recs, total = self.store.list_records(**kwargs)
        return [r.to_dict() for r in recs], total

    def link(
        self,
        source: str,
        target: str,
        relation: str,
        *,
        explicit: bool = True,
        basis: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        edge = self.store.add_edge(MemoryEdge(
            source=source, target=target, relation=relation,
            explicit=explicit, basis=basis or ("explicit link via API"
                                               if explicit else "inferred"),
            metadata=metadata))
        return edge.to_dict()

    def update(self, record_id: str, **kwargs: Any) -> dict[str, Any]:
        new = self.store.update_record(record_id, **kwargs)
        self.lexical.add(new.id, new.search_text())
        self._records_cache[new.id] = new
        return new.to_dict()

    def supersede(self, old_id: str, new_id: str, *, basis: str = "") -> dict[str, Any]:
        rec = self.store.supersede(old_id, new_id, basis=basis)
        if rec is not None:
            self._records_cache[rec.id] = rec
        return rec.to_dict() if rec else {}

    def delete(self, record_id: str) -> dict[str, Any]:
        rec = self.store.delete_record(record_id)
        if rec is not None:
            self._records_cache[rec.id] = rec
            self.lexical.remove(record_id)
        return rec.to_dict() if rec else {}

    def restore(self, record_id: str) -> dict[str, Any]:
        rec = self.store.restore_record(record_id)
        if rec is not None:
            self._records_cache[rec.id] = rec
            self.lexical.add(rec.id, rec.search_text())
        return rec.to_dict() if rec else {}

    def purge(self, record_id: str) -> bool:
        ok = self.store.purge_record(record_id)
        if ok:
            self._records_cache.pop(record_id, None)
            self.lexical.remove(record_id)
        return ok

    def edges(self, record_id: str) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.store.edges_for(record_id)]

    # -- retrieval ----------------------------------------------------------------

    def _apply_reranker(self, query: str, cands: list[Candidate]) -> tuple[list[Candidate], str]:
        """Reorder the fused top slice through the configured reranker.

        Returns (candidates, note). Every failure mode degrades to the
        fused order with a note — reranking is an ordering nicety and must
        never break retrieval.
        """
        from intentgraph.rerank import NullReranker, RerankError
        if self.reranker is None or isinstance(self.reranker, NullReranker):
            return cands, ""
        if not getattr(self.reranker, "available", False):
            return cands, "reranker configured but no credentials — fused order kept"
        top_n = min(len(cands), max(1, int(self.config.reranker.get("top_n", 10))))
        head, tail = cands[:top_n], cands[top_n:]
        docs = []
        for c in head:
            rec = self._records_cache.get(c.memory_id) \
                or self.store.get_record(c.memory_id)
            docs.append((rec.search_text() if rec else "")[:4000])
        try:
            order = self.reranker.rerank(query, docs)
        except RerankError as exc:
            return cands, f"reranker failed ({exc}) — fused order kept"
        new_head = [head[item["index"]] for item in order]
        for c, item in zip(new_head, order):
            c.term_scores["__rerank_score__"] = round(item["score"], 6)
        return new_head + tail, \
            f"reranked top {top_n} candidates via {self.reranker.name} " \
            "(scores in __rerank_score__; fused scores unchanged)"

    def retrieve(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int | None = None,
        include_inactive: bool = False,
        use_graphify_bridge: bool = True,
    ) -> dict[str, Any]:
        """One retrieval round; returns candidates + resolved records + notes.

        The Graphify bridge is applied in hybrid/adaptive modes after fusion:
        records whose code anchors sit near the fused seeds' anchors join as
        ``code-adjacent`` candidates (marked as such — provenance of method).
        """
        notes: list[str] = []
        top_k = top_k or int(self.config.retrieval.get("max_candidates", 50))
        cands: list[Candidate] = self.engine.retrieve(
            query, mode=mode, top_k=top_k,
            doc_filter=self._doc_filter(include_inactive), notes=notes)
        # Optional reranking: improves ORDER of the fused top slice only —
        # it never adds or drops evidence, and a provider failure keeps the
        # fused order with an honest note (never a crash, never silence).
        if mode in ("hybrid", "adaptive") and cands:
            cands, rerank_note = self._apply_reranker(query, cands)
            if rerank_note:
                notes.append(rerank_note)
        if use_graphify_bridge and mode in ("hybrid", "adaptive") \
                and self.bridge.available:
            bridged = self.bridge.related_records(
                [c.memory_id for c in cands[:5]], self._records_cache)
            doc_filter = self._doc_filter(include_inactive)
            # bridged candidates must pass the same visibility gate as every
            # other source — without this, a superseded or deleted record
            # could slip into a default-mode answer via a code-graph hop
            extra = [Candidate(b["id"], "graphify-bridge", b["score"],
                              matched_terms=b["matched_terms"],
                              graph_path=[("code_adjacent", b["via"])])
                     for b in bridged
                     if doc_filter is None or doc_filter(b["id"])]
            cands = fuse_rrf({"round": cands, "bridge": extra},
                             weights=self.config.retrieval.get("weights"),
                             top_k=top_k)
        records = self.store.get_records_by_ids([c.memory_id for c in cands])
        return {
            "query": query,
            "mode": mode,
            "candidates": candidates_to_dicts(cands),
            "records": {rid: records[rid].to_dict() for rid in records},
            "notes": notes,
            "graphify_bridge": "active" if self.bridge.available else "unavailable",
        }

    def remember(
        self,
        query: str,
        *,
        budget_tokens: int | None = None,
        include_inactive: bool = False,
    ) -> dict[str, Any]:
        """Full adaptive activation: iterative retrieval → verified context.

        This is the flagship call: it runs the controller loop, verifies the
        evidence, compiles the token-budgeted packet, and returns the packet
        (dict + ready-to-paste text), the trace, and the full report —
        nothing is summarized away behind a convenience field.
        """
        budget = int(budget_tokens or self.config.retrieval.get("context_budget_tokens", 4000))
        trace = self.controller.run(query, doc_filter=self._doc_filter(include_inactive))
        packet = compile_context(query, trace, self.store, budget_tokens=budget)
        return {
            "packet": packet.to_dict(),
            "text": packet.to_text(),
            "trace": trace.to_dict(),
            "report": trace.report,
        }

    # -- feedback / portability -------------------------------------------------------

    def feedback(self, query: str, outcome: str, *,
                 memory_ids: list[str] | None = None, note: str = "",
                 metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        from intentgraph.models import Feedback
        fb = Feedback(query=query, outcome=outcome, memory_ids=memory_ids,
                      note=note, metadata=metadata)
        return self.store.add_feedback(fb).to_dict()

    def export(self, path: str | Path) -> dict[str, Any]:
        return self.store.export_bundle(path)

    def import_(self, path: str | Path) -> dict[str, Any]:
        counts = self.store.import_bundle(path)
        self._records_cache = {
            r.id: r for r in self.store.iter_all_records(include_deleted=True)}
        self._rebuild_indexes()
        return counts

    def status(self) -> dict[str, Any]:
        stats = self.store.stats()
        stats["data_dir"] = str(self.config.data_dir)
        stats["embedding"] = self.embedding_provider.describe()
        if getattr(self.semantic_index, "stale_model_note", ""):
            stats["embedding"]["stale_model"] = self.semantic_index.stale_model_note
        stats["llm"] = {"provider": self.config.llm.get("provider", "none"),
                        "active": self.llm_policy is not None,
                        "llm_calls": getattr(self.llm_policy, "llm_calls", 0),
                        "fallbacks": getattr(self.llm_policy, "fallbacks", 0)}
        stats["reranker"] = self.reranker.describe() if self.reranker else {
            "provider": "none"}
        stats["web_search"] = self.web_search.describe()
        stats["web_search"]["enabled"] = bool(
            self.config.web_search.get("enabled", False))
        stats["graphify_bridge"] = {
            "path": self.bridge.graph_path,
            "available": self.bridge.available,
            "nodes": self.bridge.node_count,
            "error": self.bridge.load_error,
        }
        stats["lexical_index"] = {"docs": len(self.lexical),
                                  "vocabulary": self.lexical.vocabulary_size}
        return stats

    def list_embedding_models(self) -> list[str] | dict[str, str]:
        """Discover what the configured embedding endpoint serves THIS account.

        Network call (bounded, no retry). Returns sorted model ids, or an
        error dict when the provider cannot list (e.g. hash/null provider).
        """
        provider = self.embedding_provider
        list_models = getattr(provider, "list_models", None)
        if not callable(list_models):
            return {"error": f"provider '{provider.name}' does not support "
                             "model listing (configure nebius/openai first)"}
        result = list_models()
        return list(result) if isinstance(result, (list, tuple)) \
            else {"error": "unexpected model listing response"}

    def search_web(self, query: str) -> list[dict[str, Any]]:
        """Direct external search (explicit; provider must be enabled).

        For the CLI/API surfaces and tests. The adaptive controller calls
        the provider on its own only when the question warrants it; this
        method is the always-available manual escape hatch.
        """
        results = self.web_search.search(query)
        return [ev.to_dict() for ev in results]

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "IntentGraph":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
