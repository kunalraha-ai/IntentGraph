"""Unit tests for intentgraph.ingest (safe ingestion, inference, auto-links)."""
from __future__ import annotations

import pytest

from intentgraph.ingest import (
    Ingestor,
    infer_event_time,
    infer_type,
    ingest,
    ingest_dict,
    ingest_file,
)
from intentgraph.store import MemoryStore, StoreError


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


class TestTypeInference:
    def test_decision_signals(self):
        t, inferred = infer_type("We decided to use SQLite for the archive.")
        assert t == "decision" and inferred

    def test_experience_signals(self):
        t, _ = infer_type("The deploy failed because the migration locked the table.")
        assert t == "experience"

    def test_preference_signals(self):
        t, _ = infer_type("Convention: always use UTC timestamps in records.")
        assert t == "preference"

    def test_default_is_fact(self):
        assert infer_type("The port is 8080.")[0] == "fact"

    def test_explicit_type_not_overridden(self, store):
        ing = Ingestor(store)
        res = ing.ingest("We decided to ship.", type="task")
        assert res.record.type == "task"
        assert res.type_inferred is False


class TestEventTime:
    def test_iso_passthrough(self):
        assert infer_event_time("2026-03-14T09:00:00Z") == "2026-03-14T09:00:00+00:00"

    def test_date_only_expanded(self):
        assert infer_event_time("2026-03-14") == "2026-03-14T00:00:00+00:00"

    def test_none_stays_none(self):
        assert infer_event_time(None) is None
        assert infer_event_time("") is None

    def test_garbage_raises(self, store):
        with pytest.raises(StoreError):
            ingest(store, "x", event_time="last tuesday")


class TestIngestBasics:
    def test_ingest_returns_record_with_provenance(self, store):
        res = ingest(store, "The service crashed at 3am.",
                     source_type="tool", source_id="pagerduty")
        assert res.record.type == "fact"
        assert res.record.source_type == "tool"
        assert res.record.source_id == "pagerduty"
        assert store.get_record(res.record.id) is not None

    def test_empty_content_rejected(self, store):
        with pytest.raises(StoreError):
            ingest(store, "")
        with pytest.raises(StoreError):
            ingest(store, "   ")

    def test_exact_duplicate_short_circuits(self, store):
        r1 = ingest(store, "identical evidence", source_type="manual")
        r2 = ingest(store, "identical evidence", source_type="tool")
        assert r1.record.id == r2.record.id
        assert r2.duplicate_of is not None
        assert len(list(store.iter_all_records())) == 1

    def test_duplicate_allowed_when_requested(self, store):
        r1 = ingest(store, "identical evidence")
        r2 = ingest(store, "identical evidence", source_type="tool", allow_duplicate=True)
        assert r1.record.id != r2.record.id
        assert r2.duplicate_of is None

    def test_inferred_type_stamped_in_provenance(self, store):
        res = ingest(store, "We decided to adopt ADR-7.")
        assert res.record.provenance.get("type_inferred") is True


class TestMentionEdges:
    def test_mentions_create_inferred_edges(self, store):
        a = ingest(store, "the original failure: deadlock in the pool")
        res = ingest(store, f"Follow-up: {a.record.id} was resolved by bumping the pool size.")
        edges = store.edges_for(res.record.id)
        assert any(e.relation == "mentions" and e.target == a.record.id
                   and e.explicit is False for e in edges)

    def test_no_self_mention_edge(self, store):
        res = ingest(store, "note about itself: mem_0000000000000000")
        assert all(e.relation != "mentions" for e in store.edges_for(res.record.id))


class TestNearDuplicates:
    def test_near_dup_linked_not_merged(self, store):
        r1 = ingest(store, "The checkout flow fails on Safari when the payment token expires "
                            "mid-session and the user retries the purchase.")
        r2 = ingest(store, "The checkout flow fails on Safari when the payment token expires "
                           "mid-session and the user retries the purchase twice.")
        assert r1.record.id != r2.record.id  # kept distinct
        assert any(nd["id"] == r1.record.id for nd in r2.near_duplicates)
        edges = store.edges_for(r2.record.id)
        assert any(e.relation == "duplicate_of" for e in edges)

    def test_unrelated_content_not_flagged(self, store):
        r1 = ingest(store, "The build server ran out of disk space during the release.")
        r2 = ingest(store, "Users prefer dark mode in the analytics dashboard.")
        assert r2.near_duplicates == []


class TestIngestDict:
    def test_dict_shape(self, store):
        res = ingest_dict(store, {
            "content": "ADR-9: use SQLite.",
            "type": "decision",
            "tags": ["adr", "storage"],
            "source_type": "conversation",
            "source_id": "sess-1",
            "event_time": "2026-05-01",
        })
        assert res.record.type == "decision"
        assert res.record.event_time == "2026-05-01T00:00:00+00:00"

    def test_unknown_fields_rejected(self, store):
        with pytest.raises(StoreError, match="unknown ingest fields"):
            ingest_dict(store, {"content": "x", "tpe": "typo"})


class TestIngestFile:
    def test_ingest_local_file(self, store, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Notes\nthe regression was in auth middleware", encoding="utf-8")
        res = ingest_file(store, f)
        assert res.record.type == "document"
        assert res.record.source_type == "file"
        assert res.record.source_id.endswith("notes.md")

    def test_missing_file_rejected(self, store, tmp_path):
        with pytest.raises(StoreError):
            ingest_file(store, tmp_path / "nope.md")

    def test_oversized_file_rejected(self, store, tmp_path):
        big = tmp_path / "big.log"
        big.write_bytes(b"x" * (2_000_001))
        with pytest.raises(StoreError, match="too large"):
            ingest_file(store, big)
