"""Evidence verification: provenance, validity, conflicts, coverage, gaps.

The retrieval layer answers "what is relevant?" — this module answers "is
what we found actually *evidence*, and does it cover the question?"

Principles encoded here (they are load-bearing, not decoration):

* **Similarity is not truth.** A high BM25/cosine score says a record talks
  about the topic; it says nothing about whether the record is current,
  sourced, or non-conflicting. Verification re-examines every admitted item
  with deterministic checks and records each result as a structured
  :class:`Check`, never a silent adjustment of the score.
* **Conflicts are surfaced, not adjudicated.** When two memories contradict,
  IntentGraph does not pick a winner — it reports both with provenance,
  timestamps, and the relation, and lets the consuming system (or human)
  decide. The same applies to a unique source that disagrees with several
  others: majority is not truth either.
* **Supersession demotes, never deletes.** A superseded record still carries
  its successor pointer; the packet includes the successor by default and
  mentions the superseded one in limitations.
* **An explicit insufficient-evidence state exists.** The report ends with a
  verdict — ``sufficient`` / ``partial`` / ``insufficient`` — plus the exact
  list of uncovered requirements, so a consuming model can abstain or ask
  instead of guessing.

All checks in this module are deterministic; no LLM is required or consulted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from intentgraph.models import MemoryRecord, parse_timestamp
from intentgraph.retrieve import Candidate

__all__ = [
    "Requirement",
    "analyze_requirements",
    "topical_terms",
    "Check",
    "EvidenceItem",
    "Conflict",
    "EvidenceReport",
    "verify_evidence",
    "superseded_successors",
    "group_duplicates",
]

# --- requirement analysis ----------------------------------------------------------
#
# The rule-based analyzer maps question intent onto evidence *requirements*.
# HEURISTIC by design (documented as such): a regex over the question's words.
# It is intentionally recall-oriented — a requirement only "absent" when no
# selected evidence matches it — because inventing coverage from nothing is
# the one error this system must never make.


@dataclass
class Requirement:
    """One kind of evidence the question needs (rule-derived)."""

    name: str
    description: str
    #: content keywords that satisfy this requirement (folded, substring)
    content_signals: tuple[str, ...] = ()
    #: record types that satisfy it
    record_types: tuple[str, ...] = ()

    def satisfied_by(self, record: MemoryRecord) -> bool:
        if self.record_types and record.type in self.record_types:
            return True
        content = record.content[:6000].lower()
        return any(sig in content for sig in self.content_signals)


_REQUIREMENT_RULES: tuple[tuple[Requirement, tuple[str, ...]], ...] = (
    (
        Requirement(
            "failure_evidence",
            "the original failure: error, crash, test output, or root cause",
            content_signals=("failed", "failure", "crash", "error", "exception",
                              "regression", "bug", "flaky", "root cause",
                              "stack trace", "timeout", "deadlock", "broken"),
            record_types=("experience",),
        ),
        ("fix the regression", "why did", "failed", "failure", "regression",
         "crash", "bug", "break", "broke", "error", "flaky", "outage",
         "root cause", "incident", "postmortem"),
    ),
    (
        Requirement(
            "attempted_fixes",
            "what was tried before and how it turned out",
            content_signals=("we tried", "attempted", "previous fix", "didn't work",
                              "did not work", "reverted", "rolled back",
                              "experiment", "attempt", "hotfix"),
        ),
        ("what we tried", "what did we try", "previous attempts", "tried before",
         "what we attempted", "earlier fix", "attempted", "have we tried",
         "last time we"),
    ),
    (
        Requirement(
            "attempt_failures",
            "why previous attempts failed",
            content_signals=("failed because", "still failing", "made it worse",
                              "didn't fix", "did not fix", "did not work",
                              "didn't work", "no effect", "because it",
                              "regressed further", "because of"),
        ),
        ("why did .* fail", "why .* failed", "why it failed",
         "why .* didn't work", "why it didn't work", "what went wrong"),
    ),
    (
        Requirement(
            "decision",
            "the decision and its rationale",
            content_signals=("we decided", "decided to", "decision", "adr",
                             "chose", "agreed", "rationale", "going with",
                             "architect"),
            record_types=("decision",),
        ),
        # "decide" as a trigger matches "decide", "decided", "decides" alike
        # (substring), so one entry covers all inflections without regex soup.
        ("decision", "decide", "architecture", "adr", "rationale",
         "what did we choose", "which .* did we (choose|pick)", "agreed"),
    ),
    (
        Requirement(
            "artifacts",
            "supporting artifacts: logs, test results, tool outputs",
            content_signals=("log", "output", "test result", "stdout", "stderr",
                             "exit code", "trace", "command"),
            record_types=("action",),
        ),
        ("logs?", "test results?", "output", "stack trace", "artifacts?",
         "evidence", "records of the run"),
    ),
    (
        Requirement(
            "temporal",
            "when things happened (ordering / recency)",
            content_signals=(),  # satisfied by having event_time at all
        ),
        ("when", "how recently", "latest", "most recent", "after we",
         "before we", "since when", "how long ago", "timeline", "order"),
    ),
)


def analyze_requirements(query: str) -> list[Requirement]:
    """Derive evidence requirements from the question (deterministic rules).

    The default analyzer returns a requirement for every rule whose trigger
    phrases appear in the question. An empty list means "general evidence" —
    the query asked for topical material without a specific evidential shape,
    so coverage is judged by whether anything relevant was found at all.
    """
    lowered = query.lower()
    found: list[Requirement] = []
    for req, triggers in _REQUIREMENT_RULES:
        for trig in triggers:
            if re.search(trig, lowered):
                if req.name not in {r.name for r in found}:
                    found.append(req)
                break
    return found


def topical_terms(query: str, requirements: Sequence[Requirement]) -> set[str]:
    """The query's content-bearing terms that are NOT requirement-signal words.

    "checkout regression" is a topic; "we decided" is an evidence-shape
    signal. Coverage checks require the topic to match too (see
    verify_evidence) — otherwise any question containing "decide" would be
    "answered" by the first unrelated decision record in the archive, which
    is exactly the false-positive an evidence system cannot afford.
    """
    from intentgraph.lexindex import tokenize, _STOPWORDS

    signal_words: set[str] = set()
    for req in requirements:
        for sig in req.content_signals:
            signal_words.update(tokenize(sig))
    out: set[str] = set()
    for tok in tokenize(query):
        if tok in _STOPWORDS or len(tok) < 4 or tok in signal_words:
            continue
        out.add(tok)
    return out


# --- checks -------------------------------------------------------------------------


@dataclass
class Check:
    """One verification result on one evidence item."""

    name: str
    status: str          # "pass" | "warn" | "fail"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def check_provenance(record: MemoryRecord) -> Check:
    """A memory with no source at all is hearsay — still usable, but flagged."""
    if record.source_id or record.provenance:
        return Check("provenance", "pass")
    return Check(
        "provenance", "warn",
        "no source_id and no provenance — cannot trace this memory to an origin",
    )


def check_validity(record: MemoryRecord, store: Any) -> list[Check]:
    """Validity: active / superseded (with successor) / deleted."""
    checks: list[Check] = []
    if record.is_deleted:
        checks.append(Check("validity", "fail", "record is deleted"))
        return checks
    if record.is_superseded:
        succ = record.superseded_by or "unknown successor"
        checks.append(Check(
            "validity", "warn",
            f"superseded by {succ} (valid_to={record.valid_to}); "
            "the successor is the current statement"))
    else:
        checks.append(Check("validity", "pass"))
    return checks


def check_temporal_edges(record: MemoryRecord, store: Any) -> list[Check]:
    """Temporal-edge consistency: occurred_before/after vs effective times.

    A record claiming to occur before another whose event_time is later is
    a data problem worth surfacing (it usually means one of the two
    event_time values was invented or wrong). Reported, never auto-repaired.
    """
    checks: list[Check] = []
    t_self = parse_timestamp(record.effective_time())
    if t_self is None:
        checks.append(Check("temporal", "warn",
                            "no event_time; ordering uses ingestion time"))
        return checks
    for edge in store.edges_for(record.id, relation="occurred_before") + \
            store.edges_for(record.id, relation="occurred_after"):
        other = store.get_record(edge.target if edge.source == record.id else edge.source)
        if other is None:
            continue
        t_other = parse_timestamp(other.effective_time())
        if t_other is None:
            continue
        before = (edge.source == record.id) == (edge.relation == "occurred_before")
        if before and t_self >= t_other:
            checks.append(Check(
                "temporal", "warn",
                f"claims to occur before {other.id} but its time {record.effective_time()} "
                f"is not earlier than {other.effective_time()}"))
        elif not before and t_self <= t_other:
            checks.append(Check(
                "temporal", "warn",
                f"claims to occur after {other.id} but its time {record.effective_time()} "
                f"is not later than {other.effective_time()}"))
    return checks


def check_support(record: MemoryRecord, store: Any) -> Check:
    """Explicitly supported vs merely asserted.

    A claim is *supported* when it links to validating artifacts
    (validated_by/mentions edges, or provenance refs). An unsupported claim
    is not discarded — it is labeled, so a consuming model knows which
    statements are corroborated.
    """
    edges = store.edges_for(record.id)
    supporting = [e for e in edges
                  if e.relation in ("validated_by", "derived_from", "mentions")
                  and e.source == record.id]
    if supporting or record.provenance.get("refs") or record.provenance.get("uri"):
        return Check("support", "pass",
                     f"{len(supporting)} linked support edge(s)" if supporting
                     else "provenance carries refs")
    return Check("support", "warn",
                 "claim is asserted without linked supporting records")


@dataclass
class EvidenceItem:
    """A selected evidence record with its verification results attached."""

    record: MemoryRecord
    candidate: dict[str, Any] | None = None
    checks: list[Check] = field(default_factory=list)

    @property
    def quality(self) -> float:
        """0..1 summary of check outcomes (pass=1, warn=0.5, fail=0).

        A ranking hint for the context compiler — never a truth value, and
        never used to drop an item.
        """
        if not self.checks:
            return 0.5
        return sum({"pass": 1.0, "warn": 0.5, "fail": 0.0}[c.status]
                   for c in self.checks) / len(self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.record.id,
            "type": self.record.type,
            "retrieval": self.candidate,
            "checks": [c.to_dict() for c in self.checks],
            "quality": round(self.quality, 3),
        }


@dataclass
class Conflict:
    """A surfaced contradiction between two selected records."""

    a: str
    b: str
    relation: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "relation": self.relation, "note": self.note}


def detect_conflicts(selected: Sequence[MemoryRecord], store: Any) -> list[Conflict]:
    """Contradiction edges among the selected set, with provenance context.

    Deliberately only reports *declared* contradictions (contradicts edges).
    Infering contradiction from content similarity alone is exactly the kind
    of guess this module refuses to make.
    """
    ids = {r.id for r in selected}
    out: list[Conflict] = []
    for rec in selected:
        for edge in store.edges_for(rec.id, relation="contradicts"):
            other_id = edge.target if edge.source == rec.id else edge.source
            if other_id not in ids:
                continue
            key = tuple(sorted((edge.source, edge.target)))
            if any((tuple(sorted((c.a, c.b))) == key) for c in out):
                continue
            other = store.get_record(other_id)
            note = (f"{rec.id} ({rec.effective_time()}) vs {other_id} "
                    f"({other.effective_time() if other else '?'}); basis: {edge.basis or 'user-declared'}")
            out.append(Conflict(edge.source, edge.target, "contradicts", note))
    return out


def superseded_successors(selected: Sequence[MemoryRecord], store: Any) -> dict[str, str]:
    """map superseded_id → current successor id for the selected set."""
    out: dict[str, str] = {}
    for rec in selected:
        if rec.is_superseded and rec.superseded_by:
            out[rec.id] = rec.superseded_by
    return out


def group_duplicates(selected: Sequence[MemoryRecord], store: Any) -> list[list[str]]:
    """Near-duplicate groups within the selected set (duplicate_of edges).

    Returns groups as lists of ids. Every member is kept in the group — the
    context compiler shows one representative and cites the others, but the
    data model never merges them silently.
    """
    ids = {r.id for r in selected}
    adjacency: dict[str, set[str]] = {}
    for rec in selected:
        for edge in store.edges_for(rec.id, relation="duplicate_of"):
            other = edge.target if edge.source == rec.id else edge.source
            if other in ids:
                adjacency.setdefault(rec.id, set()).add(other)
                adjacency.setdefault(other, set()).add(rec.id)
    groups: list[list[str]] = []
    seen: set[str] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, ()))
        seen |= component
        groups.append(sorted(component))
    return groups


# --- report -------------------------------------------------------------------------


@dataclass
class EvidenceReport:
    """Verification outcome for a selected evidence set."""

    verdict: str = "insufficient"          # sufficient | partial | insufficient
    items: list[EvidenceItem] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)
    successors: dict[str, str] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)   # requirement → status
    missing: list[str] = field(default_factory=list)         # unmet requirement names
    limitations: list[str] = field(default_factory=list)
    temporal_ambiguity: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "coverage": self.coverage,
            "missing": self.missing,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "duplicate_groups": self.duplicate_groups,
            "successors": self.successors,
            "limitations": self.limitations,
            "temporal_ambiguity": self.temporal_ambiguity,
            "items": [i.to_dict() for i in self.items],
        }


def verify_evidence(
    query: str,
    selected: Sequence[MemoryRecord],
    store: Any,
    *,
    candidates: dict[str, dict[str, Any]] | None = None,
    requirements: list[Requirement] | None = None,
    doc_filter: Callable[[str], bool] | None = None,
) -> EvidenceReport:
    """Full verification pass over a selected evidence set.

    ``candidates`` maps memory_id → retrieval transparency dict (methods,
    scores, graph_path) for attaching provenance-of-retrieval to each item.
    """
    candidates = candidates or {}
    requirements = requirements if requirements is not None else analyze_requirements(query)
    report = EvidenceReport()
    wants_temporal = any(r.name == "temporal" for r in requirements)

    # 1. per-item checks
    for rec in selected:
        checks = [check_provenance(rec)]
        checks.extend(check_validity(rec, store))
        checks.extend(check_temporal_edges(rec, store))
        checks.append(check_support(rec, store))
        item = EvidenceItem(record=rec, candidate=candidates.get(rec.id), checks=checks)
        report.items.append(item)
        # temporal ambiguity is judged against the requirement, not folded into
        # the item's quality score: a record without event_time is not a worse
        # record unless the question actually depends on ordering.
        if wants_temporal and rec.event_time is None:
            report.temporal_ambiguity.append(rec.id)

    # 2. cross-item structure
    report.conflicts = detect_conflicts(selected, store)
    report.duplicate_groups = group_duplicates(selected, store)
    report.successors = superseded_successors(selected, store)

    # 3. requirement coverage — shape match under a set-level topic anchor
    #
    # Two gates, in this order:
    #
    # * ANCHOR: does anything in the evidence set share topical terms with
    #   the question at all? A set anchored by one on-topic record lets the
    #   other records count toward requirement shapes without each of them
    #   individually repeating the question's words ("we tried bumping the
    #   linker version" need not contain "build" to answer "what did we try
    #   about the build failure"). A set with ZERO on-topic records covers
    #   nothing — this is what makes "what did we decide about postgres
    #   replication?" abstain on an archive whose decisions are about order
    #   amounts, no matter how many shape-matching records a follow-up round
    #   dredges up.
    # * SHAPE: with the anchor satisfied, a requirement is covered by records
    #   of its evidence shape (type or content signals).
    active = [i.record for i in report.items if not i.record.is_deleted]
    from intentgraph.lexindex import tokenize

    topics = topical_terms(query, requirements)
    rec_terms = {r.id: set(tokenize(r.search_text())) for r in active}
    anchored_by = [r.id for r in active if topics & rec_terms[r.id]]
    anchored = bool(anchored_by) if topics else True

    for req in requirements:
        if not anchored:
            hits: list[str] = []
        else:
            hits = [r.id for r in active if req.satisfied_by(r)]
            if req.name == "temporal" and not hits:
                # temporal needs any event-time-bearing record; shape-only
                # fallback because the temporal requirement has no signals
                hits = [r.id for r in active if r.event_time is not None]
        report.coverage[req.name] = {
            "description": req.description,
            "covered": bool(hits),
            "evidence_ids": hits,
            "anchored_by": anchored_by,
        }
        if not hits:
            report.missing.append(req.name)

    if topics and not anchored:
        report.limitations.append(
            "no retrieved record shares the question's subject matter — "
            "coverage cannot be claimed for evidence that is off-topic")

    # 4. verdict — three-way, based on what was asked and what was covered
    covered_any = any(v["covered"] for v in report.coverage.values())
    if not active:
        report.verdict = "insufficient"
    elif requirements and not covered_any:
        # the question named specific evidence needs and NONE of them were met:
        # the archive did not answer it. Partial (not insufficient) is reserved
        # for the case where some requirements were satisfied.
        report.verdict = "insufficient"
    elif requirements and report.missing:
        report.verdict = "partial"
    elif requirements and not report.missing:
        report.verdict = "sufficient"
    else:
        # no specific requirements were derivable; any verified evidence counts
        report.verdict = "sufficient" if active else "insufficient"

    # 5. limitations worth telling the consuming model about
    if report.conflicts:
        report.limitations.append(
            f"{len(report.conflicts)} contradiction(s) present in the evidence; "
            "both sides are included with provenance — resolve before relying on either")
    if report.successors:
        report.limitations.append(
            f"{len(report.successors)} selected record(s) are superseded; "
            "successors are included and marked current")
    if report.temporal_ambiguity:
        report.limitations.append(
            f"{len(report.temporal_ambiguity)} record(s) lack event times; "
            "temporal ordering for them is inferred from ingestion time")
    unsupported = [i.record.id for i in report.items
                   if any(c.name == "support" and c.status == "warn" for c in i.checks)]
    if unsupported:
        report.limitations.append(
            f"{len(unsupported)} claim(s) are asserted without linked supporting "
            "artifacts (marked unsupported): " + ", ".join(unsupported[:5]))
    for missing in report.missing:
        report.limitations.append(f"missing evidence: {missing}")
    return report
