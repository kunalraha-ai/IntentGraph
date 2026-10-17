"""Adaptive memory activation: iterative, gap-targeted retrieval.

The engine that makes IntentGraph more than "vector search with extra steps":

1. **Analyze** the question into evidence requirements (what kinds of
   evidence would a complete answer need?).
2. **Retrieve** a broad first round (exact + lexical + semantic when
   configured) and fuse it (RRF).
3. **Expand** promising seeds through the memory graph's high-signal
   relations (caused_by, failed_because_of, validated_by, supersedes,
   derived_from) — bounded by depth and fan-out.
4. **Verify** the working set (verify.verify_evidence): coverage per
   requirement, conflicts, supersession, temporal ambiguity.
5. **Target the gaps**: if a requirement is unmet, run *follow-up retrieval
   whose query is sharpened toward exactly that missing evidence shape*
   ("we tried", "reverted", "didn't work" for attempted-fixes) — never a
   blind expansion of every neighborhood.
6. **Stop** when: every requirement is covered, or the iteration cap is
   hit, or a round adds no new covered requirement and no candidate above
   the gain threshold, or the wall-clock timeout fires, or the candidate
   budget is exhausted.
7. **Return a trace**: the selected candidates, every round's queries and
   yields, the gaps found, the stop reason, and the verification report —
   so the consuming model can see *why* these memories and not others.

The controller interface is intentionally replaceable
(:class:`ActivationPolicy`): the default :class:`RuleBasedPolicy` is fully
deterministic; a learned or LLM-assisted policy can be dropped in later
without touching the retrieval/verification plumbing.

Never fabricates: if a requirement stays unmet, it is reported as a gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from intentgraph.retrieve import Candidate, RetrievalEngine, candidates_to_dicts, fuse_rrf
from intentgraph.verify import (
    EvidenceReport,
    Requirement,
    analyze_requirements,
    verify_evidence,
)

__all__ = ["RetrievalTrace", "RuleBasedPolicy", "ActivationPolicy", "AdaptiveMemoryEngine"]


# --- policy interface ---------------------------------------------------------------


class ActivationPolicy:
    """Replaceable decision layer of the controller.

    A policy decides (a) how to read the question into requirements and
    (b) how to phrase follow-up rounds for a specific missing requirement.
    Keeping it a plain class (not a Protocol) means a custom policy can be a
    simple subclass; the engine only calls the two methods below.
    """

    def requirements(self, query: str) -> list[Requirement]:
        return analyze_requirements(query)

    def follow_up_query(self, query: str, missing: Requirement) -> str:
        raise NotImplementedError

    def needs_external(self, query: str, report: Any = None) -> bool:
        """Does this question warrant external web search? Deterministic
        default: False (subclass RuleBasedPolicy provides the heuristic;
        LLMPolicy delegates to the model with this as fallback)."""
        return False

    def external_query(self, query: str) -> str:
        """The query sent to the external provider, bounded."""
        return query[:600]

    def plan_next_action(self, query: str, report_summary: dict[str, Any],
                         *, external_ok: bool, iteration: int) -> dict[str, Any] | None:
        """Optional retrieval-action selection hook (LLM policies override).

        Returning None (the base behavior) means "use the controller's own
        deterministic action logic" — so an ordinary policy never has to
        think about this, and a failed LLM call degrades to the default.
        """
        return None


class RuleBasedPolicy(ActivationPolicy):
    """Deterministic policy: requirement rules + signal-sharpened follow-ups."""

    def follow_up_query(self, query: str, missing: Requirement) -> str:
        # Sharpen the ORIGINAL question toward the missing evidence shape:
        # original terms stay (they name the subject), requirement signals
        # are appended so BM25/semantic rank the right records this round.
        # Take the first two content signals — a signal-dump query becomes
        # its own noise.
        extra = " ".join(missing.content_signals[:3]) or missing.description
        return f"{query} {extra}"

    # -- deterministic external-search heuristic --------------------------------
    #
    # HEURISTIC by design (documented as such, like every Phase-1 weight):
    # a question warrants web search when it names a *freshness demand*
    # (latest/current/up-to-date, changelogs, current documentation, "what
    # does <X> support these days") that a fixed local archive cannot
    # satisfy in principle. Local-history questions ("what did we decide")
    # never trigger it. Both-class questions ("compare our decision with
    # current docs") retrieve local first, then external as a complement.

    _EXTERNAL_NEED_SIGNALS = (
        "latest", "newest", "current version", "recent version",
        "up to date", "up-to-date", "changelog", "release notes",
        "current documentation", "latest documentation", "these days",
        "right now", "as of today", "latest release", "current release",
        "current best practices", "modern best practices", "recently released",
        "what does the latest", "what does the current", "current support",
        "still supported", "deprecated in", "has this been deprecated",
        "current status of",
    )
    # co-occurrence rule: a freshness adjective anywhere in the question plus
    # a documentation-flavored noun anywhere in it ("current gateway
    # documentation", "the newest SDK docs") — exact-substring signals alone
    # miss the natural word orders agents actually write.
    _FRESHNESS_ADJ = (
        "latest", "current", "newest", "recent", "up-to-date",
        "up to date", "modern", "today's",
    )
    _DOC_NOUN = (
        "documentation", "docs", "changelog", "release notes", "sdk",
        "api reference", "best practices",
    )
    _LOCAL_HISTORY_SIGNALS = (
        "what did we", "we decided", "our decision", "what we tried",
        "did we try", "have we tried", "our architecture", "last week we",
        "remember", "our team", "in this project",
    )

    def needs_external(self, query: str, report: Any = None) -> bool:
        """Freshness-demand heuristic: only current-world questions trigger.

        A *coverage gap alone never triggers external search* — a missing
        local record is a local problem, and searching the web for it
        would both leak the query and fabricate relevance. The trigger is
        the question ASKING for something that changes over time.
        """
        lowered = query.lower()
        if any(sig in lowered for sig in self._EXTERNAL_NEED_SIGNALS):
            return True
        if (any(adj in lowered for adj in self._FRESHNESS_ADJ)
                and any(noun in lowered for noun in self._DOC_NOUN)):
            return True
        # "compare our X with current docs" is BOTH — that still needs
        # external, so local signals do not veto; they only matter when the
        # freshness signal is weak/absent, which is handled above.
        return False

    def external_query(self, query: str) -> str:
        """The query sent to the search provider — bounded."""
        return query[:600]


# --- trace ---------------------------------------------------------------------------


@dataclass
class RoundRecord:
    """What one retrieval round did and found."""

    round_index: int
    query: str
    purpose: str
    found: int = 0
    new: int = 0
    covered_after: list[str] = field(default_factory=list)
    still_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index, "query": self.query, "purpose": self.purpose,
            "found": self.found, "new": self.new,
            "covered_after": self.covered_after,
            "still_missing": self.still_missing,
        }


@dataclass
class RetrievalTrace:
    """Everything the adaptive loop did, in auditable form."""

    query: str
    selected: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    stop_reason: str = ""
    iterations: int = 0
    report: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    #: external web search outcome — evidence, call count, and a status that
    #: distinguishes "not needed" from "disabled" from "provider failed"
    external_evidence: list[dict[str, Any]] = field(default_factory=list)
    external_calls: int = 0
    external_status: str = "not_needed"   # not_needed|ok|failed:<kind>|disabled

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "requirements": self.requirements,
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "rounds": [r.to_dict() for r in self.rounds],
            "selected": self.selected,
            "report": self.report,
            "notes": self.notes,
            "duration_ms": round(self.duration_ms, 1),
            "external_evidence": self.external_evidence,
            "external_calls": self.external_calls,
            "external_status": self.external_status,
        }


# --- engine ---------------------------------------------------------------------------


class AdaptiveMemoryEngine:
    """Iterative retrieve → expand → verify → target-the-gap loop.

    Bounded by every knob that could otherwise run away: iteration count,
    working-set size, graph depth/fan-out, per-round candidate limits, and
    a wall-clock timeout. The loop is deterministic end-to-end when the
    underlying retrievers are (they are, except a remote embedding
    provider).
    """

    def __init__(
        self,
        engine: RetrievalEngine,
        store: Any,
        *,
        policy: ActivationPolicy | None = None,
        max_iterations: int = 3,
        max_working_set: int = 24,
        max_candidates_per_round: int = 30,
        timeout_s: float = 30.0,
        min_new_candidates: int = 2,
        graph_relations: Sequence[str] | None = None,
        web_search: Any = None,
        external_max_calls: int = 1,
        external_store_results: bool = False,
        ingest_hook: Any = None,
    ):
        self.engine = engine
        self.store = store
        self.policy = policy or RuleBasedPolicy()
        self.max_iterations = max(1, int(max_iterations))
        self.max_working_set = max(1, int(max_working_set))
        self.max_candidates_per_round = max(1, int(max_candidates_per_round))
        self.timeout_s = float(timeout_s)
        self.min_new_candidates = max(0, int(min_new_candidates))
        self.graph_relations = tuple(graph_relations) if graph_relations else None
        # external search: provider (websearch.WebSearchProvider), budgets,
        # and the explicit opt-in for persisting external findings as
        # memories. ingest_hook (sdk.ingest, injected to avoid a cycle) is
        # only used when external_store_results is True.
        self.web_search = web_search
        self.external_max_calls = max(0, int(external_max_calls))
        self.external_store_results = bool(external_store_results)
        self.ingest_hook = ingest_hook

    # -- internals ---------------------------------------------------------

    def _active_filter(self) -> Callable[[str], bool]:
        """Only current, non-deleted records may enter the working set.

        Superseded/deleted records are still retrievable by id for audit —
        but they must not crowd the working set of an answer that should
        reflect the current state of memory.
        """
        cache: dict[str, bool] = {}

        def allowed(rid: str) -> bool:
            if rid not in cache:
                rec = self.store.get_record(rid)
                cache[rid] = bool(rec and rec.is_active)
            return cache[rid]
        return allowed

    def _select_working_set(self, cands: list[Candidate]) -> list[Candidate]:
        return cands[: self.max_working_set]

    def run(
        self,
        query: str,
        *,
        doc_filter: Callable[[str], bool] | None = None,
    ) -> RetrievalTrace:
        t0 = time.perf_counter()
        notes: list[str] = []
        requirements = self.policy.requirements(query)
        trace = RetrievalTrace(
            query=query,
            requirements=[r.name for r in requirements],
            notes=notes,
        )
        active = doc_filter or self._active_filter()

        # cumulative fused pool across rounds: memory_id → best candidate
        pool: dict[str, Candidate] = {}
        report: EvidenceReport | None = None
        stop_reason = ""

        for round_index in range(self.max_iterations):
            if time.perf_counter() - t0 > self.timeout_s:
                stop_reason = "timeout"
                break

            # --- decide this round's query --------------------------------
            if round_index == 0:
                round_query = query
                purpose = "broad first pass (exact + lexical + semantic + graph)"
            else:
                missing_reqs = [r for r in requirements
                                if report is not None and r.name in report.missing]
                if not missing_reqs:
                    # No requirement stayed uncovered (or none were derivable
                    # at all) — nothing left to target.
                    stop_reason = "no evidence gaps identified"
                    break
                gap = missing_reqs[0]  # deterministic: first missing requirement
                round_query = self.policy.follow_up_query(query, gap)
                purpose = f"targeted follow-up for missing evidence: {gap.name}"

            # --- retrieve + fuse + expand ----------------------------------
            fused = self.engine.retrieve(
                round_query, mode="adaptive",
                top_k=self.max_candidates_per_round,
                doc_filter=active, notes=notes)
            # deep expansion of the round's own top seeds through the
            # memory graph, restricted to evidence-bearing relations
            if fused:
                expanded = self.engine.expand(
                    [c.memory_id for c in fused[:4]],
                    top_k=self.max_candidates_per_round,
                    relations=self.graph_relations, doc_filter=active)
                fused = fuse_rrf({"round": fused, "expand": expanded},
                                 top_k=self.max_candidates_per_round)

            new_count = sum(1 for c in fused if c.memory_id not in pool)
            for c in fused:
                if c.memory_id not in pool or c.score > pool[c.memory_id].score:
                    if c.memory_id in pool:
                        # keep the union of methods across rounds
                        merged_methods = list(dict.fromkeys(
                            pool[c.memory_id].methods + c.methods))
                        c.methods = merged_methods
                    pool[c.memory_id] = c

            round_rec = RoundRecord(round_index, round_query, purpose,
                                     found=len(fused), new=new_count)

            # --- verify the working set --------------------------------------
            working = self._select_working_set(
                sorted(pool.values(), key=lambda c: (-c.score, c.memory_id)))
            records = [self.store.get_record(c.memory_id) for c in working]
            records = [r for r in records if r is not None]
            cand_dicts = {c.memory_id: d for c, d in zip(working, candidates_to_dicts(working))}
            report = verify_evidence(
                query, records, self.store,
                candidates=cand_dicts, requirements=requirements)
            round_rec.covered_after = [k for k, v in report.coverage.items() if v["covered"]]
            round_rec.still_missing = list(report.missing)
            trace.rounds.append(round_rec)

            # --- stopping conditions ------------------------------------------
            if not report.missing and requirements:
                stop_reason = "requirements covered"
                break
            if round_index > 0 and new_count < self.min_new_candidates:
                stop_reason = (f"no progress: round added {new_count} new candidates "
                               f"(threshold {self.min_new_candidates})")
                break
            if len(pool) >= self.max_candidates_per_round * self.max_iterations:
                stop_reason = "candidate budget exhausted"
                break
        else:
            stop_reason = "iteration limit reached"

        # final report over the final working set (the loop's break conditions
        # may have run before this round's verification)
        working = self._select_working_set(
            sorted(pool.values(), key=lambda c: (-c.score, c.memory_id)))
        records = [self.store.get_record(c.memory_id) for c in working]
        records = [r for r in records if r is not None]
        cand_dicts = {c.memory_id: d for c, d in zip(working, candidates_to_dicts(working))}
        report = verify_evidence(query, records, self.store,
                                 candidates=cand_dicts, requirements=requirements)

        # --- external research phase ---------------------------------------------
        # Runs AFTER local retrieval, only when the question asks for
        # something an archive cannot hold (freshness), the provider is
        # configured AND enabled, and call/budget limits allow. A local-only
        # question never triggers a request; "provider unavailable" is a
        # distinct status from "insufficient evidence".
        self._run_external_phase(query, requirements, report, trace, records, t0)

        # policies keep their own notes (LLM fallback events, usage);
        # fold them in so the trace is the single auditable record
        for n in getattr(self.policy, "notes", []):
            if n not in notes:
                notes.append(n)

        trace.selected = [cand_dicts[c.memory_id] for c in working if c.memory_id in cand_dicts]
        trace.iterations = len(trace.rounds)
        trace.stop_reason = stop_reason or "completed"
        trace.report = report.to_dict()
        trace.duration_ms = (time.perf_counter() - t0) * 1000.0
        # never claim evidence that does not exist
        if not records and not trace.external_evidence:
            trace.notes.append(
                "no active memories matched this question; the archive may not "
                "cover this topic — this is an explicit insufficient-evidence state")
        return trace

    def _run_external_phase(self, query: str, requirements: list,
                            report: Any, trace: RetrievalTrace,
                            local_records: list, t0: float) -> None:
        from intentgraph.websearch import NullWebSearchProvider, WebSearchError, \
            store_external_result

        provider = self.web_search
        external_ok = (provider is not None
                       and not isinstance(provider, NullWebSearchProvider)
                       and getattr(provider, "available", False)
                       and self.external_max_calls > 0)

        # 1. decide whether the QUESTION warrants external research.
        #    LLM policies may override via plan_next_action (validated,
        #    can only propose external when it is actually allowed).
        plan = self.policy.plan_next_action(
            query, {"missing": list(report.missing),
                    "coverage": {k: v["covered"]
                                 for k, v in report.coverage.items()},
                    "verdict": report.verdict},
            external_ok=external_ok, iteration=len(trace.rounds))
        if plan is not None:
            wants_external = plan["action"] == "external_search"
            ext_query = plan.get("external_query") or \
                self.policy.external_query(query)
            if plan.get("rationale"):
                trace.notes.append(f"llm action rationale: {plan['rationale']}")
        else:
            wants_external = self.policy.needs_external(query, report)
            ext_query = self.policy.external_query(query)

        if not wants_external:
            trace.external_status = "not_needed"
            return

        # 2. warranted but cannot run — say so, and keep it distinct from
        #    an evidence gap. These notes travel to the packet's trace.
        if provider is None or isinstance(provider, NullWebSearchProvider):
            trace.external_status = "disabled"
            trace.notes.append(
                "this question asks for current external information, but web "
                "search is disabled — enable web_search.provider (see "
                "docs/intentgraph.md) for fresh external evidence")
            return
        if not getattr(provider, "available", False):
            trace.external_status = "failed:auth"
            trace.notes.append(
                f"web search is configured but the {getattr(provider, 'name', 'web')} "
                "provider has no credentials (check the API key environment "
                "variable) — this is a provider failure, not an evidence gap")
            return
        if self.external_max_calls <= 0:
            trace.external_status = "disabled"
            trace.notes.append(
                "web search was warranted but external_max_calls is 0 "
                "(budget disabled)")
            return
        if (time.perf_counter() - t0) > self.timeout_s:
            trace.external_status = "failed:budget"
            trace.notes.append("web search skipped: retrieval time budget "
                               "already exhausted")
            return

        # 3. run it — bounded, failure-isolated
        try:
            results = provider.search(ext_query)
        except WebSearchError as exc:
            trace.external_status = f"failed:{exc.kind}"
            trace.notes.append(
                f"web search failed ({exc.kind}: {exc}) — local evidence is "
                "unchanged; treat the gap as local-only")
            return
        except Exception as exc:  # noqa: BLE001 — provider must never kill the loop
            trace.external_status = "failed:unknown"
            trace.notes.append(
                f"web search raised {type(exc).__name__} — local evidence "
                "is unchanged")
            return

        trace.external_calls += 1
        trace.external_evidence = [ev.to_dict() for ev in results]
        trace.external_status = "ok" if results else "ok_empty"
        if not results:
            trace.notes.append("web search returned no usable results "
                               "(results without a citable URL are dropped)")
            return

        # 4. freshness comparison: factual, date-based, never a fabricated
        #    conflict — if external results are newer than every matching
        #    local record, say so; the CONSUMER compares content. Only local
        #    records that carry a real event_time participate: an ingestion
        #    timestamp says when WE heard something, not when it happened,
        #    and comparing content freshness against it would be meaningless.
        #    Provider dates come in RFC-822 form (Tavily) and are parsed via
        #    the provider-date helper, not the ISO-only timestamp parser.
        if local_records:
            from intentgraph.models import parse_timestamp
            from intentgraph.websearch import parse_provider_date
            local_times = [parse_timestamp(r.event_time)
                           for r in local_records if r.event_time]
            local_times = [t for t in local_times if t is not None]
            ext_times = []
            for ev in results:
                dt = parse_provider_date(ev.published_date)
                if dt is not None:
                    ext_times.append(dt)
            if local_times and ext_times and \
                    max(ext_times) > max(local_times):
                trace.notes.append(
                    "external sources are more recent than the matching local "
                    "records — compare dates and content before treating the "
                    "local record as current")

        # 5. explicit, opt-in persistence (default off). ingest_hook is the
        #    IntentGraph instance itself (injected by the SDK at construction
        #    — no import cycle); store_external_result uses its .ingest.
        if self.external_store_results and self.ingest_hook is not None:
            stored = []
            for ev in results:
                rid = store_external_result(self.ingest_hook, ev,
                                            query=ext_query)
                if rid:
                    stored.append(rid)
            trace.notes.append(
                f"{len(stored)} external result(s) stored as memory records "
                "(web_search.store_results is enabled — they are ordinary "
                "records with full external provenance and can be deleted or "
                "superseded like any memory)")
