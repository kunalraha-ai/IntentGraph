"""Unit tests for intentgraph.verify (evidence verification + coverage)."""
from __future__ import annotations

import pytest

from intentgraph.models import MemoryEdge, MemoryRecord
from intentgraph.store import MemoryStore
from intentgraph.verify import (
    analyze_requirements,
    check_provenance,
    group_duplicates,
    verify_evidence,
)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    yield s
    s.close()


def _add(store, content, **kw):
    return store.add_record(MemoryRecord(content=content, **kw))


class TestRequirementAnalysis:
    def test_regression_query_yields_failure_and_attempts(self):
        reqs = analyze_requirements(
            "Fix the checkout regression. Remember what we tried before, "
            "why it failed, and which decision we made afterward.")
        names = [r.name for r in reqs]
        assert "failure_evidence" in names
        assert "attempted_fixes" in names
        assert "decision" in names

    def test_why_failed_yields_attempt_failures(self):
        reqs = analyze_requirements("why did the previous attempt fail?")
        assert "attempt_failures" in [r.name for r in reqs]

    def test_when_yields_temporal(self):
        reqs = analyze_requirements("when did we decide this?")
        assert "temporal" in [r.name for r in reqs]

    def test_generic_query_yields_nothing(self):
        assert analyze_requirements("gateway configuration") == []


class TestChecks:
    def test_provenance_pass_and_warn(self, store):
        ok = _add(store, "sourced fact", source_id="docs")
        bare = _add(store, "unsourced claim")
        assert check_provenance(ok).status == "pass"
        assert check_provenance(bare).status == "warn"

    def test_superseded_record_flagged_with_successor(self, store):
        old = _add(store, "old statement")
        new = _add(store, "new statement")
        store.supersede(old.id, new.id)
        selected = [store.get_record(old.id)]
        report = verify_evidence("q", selected, store)
        item = report.items[0]
        validity = [c for c in item.checks if c.name == "validity"][0]
        assert validity.status == "warn"
        assert new.id in validity.detail

    def test_deleted_record_fails_validity(self, store):
        rec = _add(store, "to be deleted")
        store.delete_record(rec.id)
        report = verify_evidence("q", [store.get_record(rec.id)], store)
        item = report.items[0]
        assert [c for c in item.checks if c.name == "validity"][0].status == "fail"

    def test_support_check_uses_validated_by(self, store):
        claim = _add(store, "the fix works")
        evidence = _add(store, "test output proving it")
        store.add_edge(MemoryEdge(source=claim.id, target=evidence.id,
                                  relation="validated_by"))
        report = verify_evidence("q", [claim, evidence], store)
        by_id = {i.record.id: i for i in report.items}
        support = [c for c in by_id[claim.id].checks if c.name == "support"][0]
        assert support.status == "pass"

    def test_temporal_edge_inconsistency_surfaced(self, store):
        early = _add(store, "before the incident",
                     event_time="2026-01-01T00:00:00+00:00")
        late = _add(store, "after the incident",
                    event_time="2026-02-01T00:00:00+00:00")
        # early claims to occur BEFORE late, but its time is... consistent here.
        store.add_edge(MemoryEdge(source=early.id, target=late.id,
                                  relation="occurred_before"))
        report = verify_evidence("q", [early, late], store)
        warnings = [c for i in report.items for c in i.checks
                    if c.name == "temporal" and c.status == "warn"]
        assert warnings == []
        # now the inconsistent direction: early claims to occur AFTER late
        store.remove_edge(store.edges_for(early.id)[0].id)
        store.add_edge(MemoryEdge(source=early.id, target=late.id,
                                  relation="occurred_after"))
        report2 = verify_evidence("q", [early, late], store)
        warnings2 = [c for i in report2.items for c in i.checks
                     if c.name == "temporal" and c.status == "warn"]
        assert any("not later than" in w.detail for w in warnings2)


class TestConflictsAndDuplicates:
    def test_declared_contradiction_surfaced_with_both_sides(self, store):
        a = _add(store, "timeout is 5 seconds", source_id="docs")
        b = _add(store, "timeout is 30 seconds", source_id="patch-notes",
                 event_time="2026-06-20T00:00:00+00:00")
        store.add_edge(MemoryEdge(source=a.id, target=b.id, relation="contradicts",
                                  basis="doc vs patch notes"))
        report = verify_evidence("what is the gateway timeout", [a, b], store)
        assert len(report.conflicts) == 1
        assert report.conflicts[0].a == a.id and report.conflicts[0].b == b.id
        assert "contradiction(s)" in report.limitations[0]

    def test_unique_dissenting_source_not_dropped(self, store):
        # three records agree, one disagrees via declared contradicts edges —
        # the dissenter must survive in the evidence list.
        agree = [_add(store, f"metric is {v}") for v in (5, 5, 5)]
        dissenter = _add(store, "metric is 30")
        for a in agree:
            store.add_edge(MemoryEdge(source=a.id, target=dissenter.id,
                                      relation="contradicts"))
        selected = agree + [dissenter]
        report = verify_evidence("q", selected, store)
        assert len(report.items) == 4  # nothing dropped
        assert len(report.conflicts) == 3

    def test_duplicate_groups_keep_all_members(self, store):
        a = _add(store, "the checkout failed with INVALID_AMOUNT")
        b = _add(store, "checkout failed: INVALID_AMOUNT error seen")
        store.add_edge(MemoryEdge(source=b.id, target=a.id, relation="duplicate_of"))
        groups = group_duplicates([a, b], store)
        assert groups == [sorted([a.id, b.id])]


class TestCoverageAndVerdicts:
    def test_partial_when_requirement_unmet(self, store):
        reqs = analyze_requirements("fix the regression; what did we try before?")
        # archive has a decision but nothing failure-shaped
        _add(store, "decision: we will rewrite the module")
        selected = list(store.iter_all_records())
        report = verify_evidence("fix the regression; what did we try before?",
                                 selected, store, requirements=reqs)
        assert report.verdict in ("partial", "insufficient")
        assert "failure_evidence" in report.missing
        assert any("missing evidence" in lim for lim in report.limitations)

    def test_sufficient_when_all_requirements_met(self, store):
        failure = _add(store, "the build failed with a linker error", type="experience",
                       source_id="ci")
        attempt = _add(store, "we tried bumping the linker version; it did not work")
        decision = _add(store, "decision: we moved to a single shared object",
                        type="decision", source_id="adr-3")
        selected = [failure, attempt, decision]
        query = ("fix the build failure; remember what we tried before and "
                 "which decision we made")
        reqs = analyze_requirements(query)
        report = verify_evidence(query, selected, store, requirements=reqs)
        assert report.verdict == "sufficient", report.to_dict()
        assert report.missing == []

    def test_insufficient_when_nothing_relevant(self, store):
        _add(store, "preference: dark mode")
        report = verify_evidence(
            "what did we decide about postgres replication?",
            list(store.iter_all_records()), store)
        assert report.verdict == "insufficient"

    def test_temporal_ambiguity_flagged_when_event_times_missing(self, store):
        _add(store, "something happened but we did not record when")
        reqs = analyze_requirements("when did this happen?")
        report = verify_evidence("when did this happen?",
                                 list(store.iter_all_records()), store,
                                 requirements=reqs)
        assert report.temporal_ambiguity
        assert any("lack event times" in lim for lim in report.limitations)

    def test_candidates_attached_to_items(self, store):
        rec = _add(store, "evidence with retrieval metadata", source_id="s")
        cands = {rec.id: {"memory_id": rec.id, "methods": ["lexical"], "score": 1.0}}
        report = verify_evidence("q", [rec], store, candidates=cands)
        assert report.items[0].candidate == cands[rec.id]
        assert 0.0 <= report.items[0].quality <= 1.0
