"""Unit tests for intentgraph.models + intentgraph.store (P1 memory core)."""
from __future__ import annotations

import json

import pytest

from intentgraph.models import (
    MemoryEdge,
    MemoryRecord,
    Feedback,
    content_hash,
    parse_timestamp,
    utcnow,
    validate_edge,
    validate_record,
)
from intentgraph.store import MemoryStore, StoreError


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


class TestModels:
    def test_record_defaults_and_identity(self):
        r = MemoryRecord(content="hello")
        assert r.id.startswith("mem_")
        assert r.type == "fact"
        assert r.status == "active"
        assert r.is_active
        assert r.version == 1
        assert r.event_time is None

    def test_content_hash_stable_and_drift_insensitive(self):
        assert content_hash("a\nb") == content_hash("a\r\nb")
        assert content_hash("a\nb") != content_hash("a b")

    def test_timestamp_parsing(self):
        assert parse_timestamp("2026-01-02T03:04:05Z") is not None
        assert parse_timestamp("2026-01-02T03:04:05+0200") is not None
        assert parse_timestamp("2026-01-02T03:04:05+02:00") is not None
        assert parse_timestamp("not a date") is None
        assert parse_timestamp("2026-13-45T99:99:99Z") is None
        # all comparisons happen aware-vs-aware
        a = parse_timestamp("2026-01-02T03:04:05Z")
        b = parse_timestamp("2026-01-02T05:04:05+02:00")
        assert a == b

    def test_validate_record_rejects_garbage(self):
        errs = validate_record({"type": "wat", "content": 42, "source_type": "??",
                                "created_at": "yesterday"})
        assert any("unknown memory type" in e for e in errs)
        assert any("content must be a string" in e for e in errs)
        assert any("source_type" in e for e in errs)
        assert any("created_at" in e for e in errs)

    def test_validate_record_rejects_oversized_content(self):
        errs = validate_record({"content": "x" * 2_000_001})
        assert any("too large" in e for e in errs)

    def test_validate_edge(self):
        assert validate_edge({"source": "mem_a", "target": "mem_b"}) == []
        errs = validate_edge({"source": "mem_a", "target": "mem_a"})
        assert any("self-referencing" in e for e in errs)
        errs = validate_edge({"source": "mem_a", "target": "mem_b", "relation": "zaps"})
        assert any("unknown relation" in e for e in errs)

    def test_roundtrip_dicts(self):
        r = MemoryRecord(content="x", tags=["a"], provenance={"actor": "ci"})
        r2 = MemoryRecord.from_dict(json.loads(json.dumps(r.to_dict())))
        assert r2.to_dict() == r.to_dict()
        e = MemoryEdge(source="mem_" + "a" * 16, target="mem_" + "b" * 16)
        e2 = MemoryEdge.from_dict(json.loads(json.dumps(e.to_dict())))
        assert e2.to_dict() == e.to_dict()


class TestStoreRecords:
    def test_add_get_roundtrip(self, store):
        r = MemoryRecord(content="Use WAL for concurrent readers.",
                         type="decision", tags=["arch"], source_type="manual")
        store.add_record(r)
        got = store.get_record(r.id)
        assert got.content == r.content
        assert got.tags == ["arch"]
        assert got.content_hash == r.content_hash

    def test_add_rejects_invalid(self, store):
        with pytest.raises(StoreError):
            store.add_record(MemoryRecord(type="nonsense", content="x"))
        with pytest.raises(StoreError):
            store.add_record(MemoryRecord(content="y" * 2_000_001))

    def test_duplicate_id_is_an_error_not_overwrite(self, store):
        r = MemoryRecord(content="original evidence")
        store.add_record(r)
        with pytest.raises(StoreError):
            store.add_record(r)

    def test_update_appends_version_and_preserves_history(self, store):
        r = MemoryRecord(content="v1: the port is 8080")
        store.add_record(r)
        updated = store.update_record(r.id, content="v2: the port is 9090", note="fix")
        assert updated.version == 2
        v1 = store.get_record(r.id, version=1)
        assert v1.content == "v1: the port is 8080"
        assert store.get_record(r.id).content == "v2: the port is 9090"
        assert updated.metadata["update_note"] == "fix"
        hist = store.record_versions(r.id)
        assert hist[0]["current"] is True and hist[0]["version"] == 2
        assert hist[1]["version"] == 1

    def test_update_missing_record(self, store):
        with pytest.raises(StoreError):
            store.update_record("mem_" + "0" * 16, content="x")

    def test_supersede_sets_validity_and_edge(self, store):
        old = store.add_record(MemoryRecord(content="old fact"))
        new = store.add_record(MemoryRecord(content="new fact"))
        store.supersede(old.id, new.id, basis="postmortem revised the fact")
        got = store.get_record(old.id)
        assert got.status == "superseded"
        assert got.superseded_by == new.id
        assert got.valid_to is not None
        edges = store.edges_for(new.id)
        assert any(e.relation == "supersedes" and e.target == old.id for e in edges)
        assert not got.is_active

    def test_soft_delete_and_restore(self, store):
        r = store.add_record(MemoryRecord(content="delete me"))
        deleted = store.delete_record(r.id)
        assert deleted.status == "deleted"
        assert store.get_record(r.id).status == "deleted"
        restored = store.restore_record(r.id)
        assert restored.status == "active"
        assert restored.valid_to is None

    def test_purge_leaves_tombstone_and_cleans_edges(self, store):
        a = store.add_record(MemoryRecord(content="a"))
        b = store.add_record(MemoryRecord(content="b"))
        store.add_edge(MemoryEdge(source=a.id, target=b.id, relation="related_to"))
        store.add_feedback(Feedback(query="q", outcome="useful", memory_ids=[a.id]))
        assert store.purge_record(a.id) is True
        assert store.get_record(a.id) is None
        assert store.is_tombstoned(a.id)
        assert store.edges_for(b.id) == []
        # purging again is idempotent
        assert store.purge_record(a.id) is True

    def test_list_records_filters_and_pagination(self, store):
        for i in range(5):
            store.add_record(MemoryRecord(content=f"fact {i}", tags=["even"] if i % 2 == 0 else []))
        recs, total = store.list_records(limit=2, offset=0)
        assert total == 5 and len(recs) == 2
        # tag filter narrows at the SQL level, so the total reflects it
        recs, total = store.list_records(tag="even")
        assert total == 3
        assert all(r.tags == ["even"] for r in recs)
        # 'even' must not substring-match a longer tag like 'unevenly'
        store.add_record(MemoryRecord(content="x", tags=["unevenly"]))
        recs, total = store.list_records(tag="even")
        assert total == 3
        assert all(r.tags == ["even"] for r in recs)

    def test_get_records_by_ids(self, store):
        ids = [store.add_record(MemoryRecord(content=str(i))).id for i in range(3)]
        got = store.get_records_by_ids(ids[:2] + ["mem_" + "9" * 16])
        assert set(got) == set(ids[:2])


class TestStoreEdges:
    def test_edge_requires_endpoints(self, store):
        a = store.add_record(MemoryRecord(content="a"))
        with pytest.raises(StoreError):
            store.add_edge(MemoryEdge(source=a.id, target="mem_" + "0" * 16))
        with pytest.raises(StoreError):
            store.add_edge(MemoryEdge(source=a.id, target=a.id))

    def test_duplicate_edge_is_idempotent(self, store):
        a = store.add_record(MemoryRecord(content="a"))
        b = store.add_record(MemoryRecord(content="b"))
        e1 = store.add_edge(MemoryEdge(source=a.id, target=b.id, relation="related_to"))
        e2 = store.add_edge(MemoryEdge(source=a.id, target=b.id, relation="related_to"))
        assert e1.id == e2.id
        assert len(store.edges_for(a.id)) == 1

    def test_inferred_edges_flagged(self, store):
        a = store.add_record(MemoryRecord(content="a"))
        b = store.add_record(MemoryRecord(content="b"))
        store.add_edge(MemoryEdge(source=a.id, target=b.id, relation="mentions",
                                  explicit=False, basis="content reference"))
        e = store.edges_for(a.id)[0]
        assert e.explicit is False
        assert e.basis == "content reference"
        assert store.stats()["edges_inferred"] == 1


class TestExportImport:
    def test_roundtrip_preserves_records_edges_feedback_versions(self, store, tmp_path):
        a = store.add_record(MemoryRecord(content="v1", type="decision"))
        store.update_record(a.id, content="v2", note="revision")
        b = store.add_record(MemoryRecord(content="evidence"))
        store.add_edge(MemoryEdge(source=a.id, target=b.id, relation="validated_by"))
        store.add_feedback(Feedback(query="why", outcome="useful", memory_ids=[a.id]))
        path = tmp_path / "bundle.jsonl"
        report = store.export_bundle(path)
        assert report["records"] >= 3  # 2 live + 1 version snapshot

        other = MemoryStore(tmp_path / "imported.db")
        counts = other.import_bundle(path)
        assert counts["records"] >= 2
        # live row is the newest version
        assert other.get_record(a.id).content == "v2"
        # history survived
        assert other.get_record(a.id, version=1).content == "v1"
        assert other.edges_for(a.id)[0].relation == "validated_by"
        assert other.iter_feedback()[0].outcome == "useful"
        other.close()

    def test_import_is_idempotent(self, store, tmp_path):
        a = store.add_record(MemoryRecord(content="stable"))
        path = tmp_path / "b.jsonl"
        store.export_bundle(path)
        counts1 = store.import_bundle(path)
        assert counts1["skipped"] >= 1
        # re-import must not duplicate
        assert len(list(store.iter_all_records(include_deleted=True))) == 1

    def test_import_rejects_foreign_and_bad(self, store, tmp_path):
        not_bundle = tmp_path / "nb.jsonl"
        not_bundle.write_text('{"hello": "world"}\n', encoding="utf-8")
        with pytest.raises(StoreError):
            store.import_bundle(not_bundle)
        bad_json = tmp_path / "bad.jsonl"
        bad_json.write_text(
            json.dumps({"intentgraph_export": True, "schema_version": 1}) + "\n{oops\n",
            encoding="utf-8")
        with pytest.raises(StoreError):
            store.import_bundle(bad_json)
        wrong_version = tmp_path / "wv.jsonl"
        wrong_version.write_text(
            json.dumps({"intentgraph_export": True, "schema_version": 99}) + "\n",
            encoding="utf-8")
        with pytest.raises(StoreError):
            store.import_bundle(wrong_version)

    def test_import_skips_edge_to_missing_endpoint(self, store, tmp_path):
        store.add_record(MemoryRecord(content="a"))
        path = tmp_path / "partial.jsonl"
        lines = [
            json.dumps({"intentgraph_export": True, "schema_version": 1}),
            json.dumps({"kind": "edge", "edge": {
                "source": "mem_" + "a" * 16, "target": "mem_" + "b" * 16,
                "relation": "related_to"}}),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts = store.import_bundle(path)
        assert counts["edges"] == 0
        assert counts["skipped"] == 1


class TestStoreMisc:
    def test_stats_shape(self, store):
        store.add_record(MemoryRecord(content="a", type="decision"))
        s = store.stats()
        assert s["records"] == 1
        assert s["types"]["decision"] == 1
        assert s["edges"] == 0

    def test_search_text_includes_identity_fields(self):
        r = MemoryRecord(content="body", tags=["auth"], source_id="session-42")
        text = r.search_text()
        assert "auth" in text and "session-42" in text and "body" in text

    def test_effective_time_prefers_event_time(self):
        r = MemoryRecord(content="x", event_time="2026-01-01T00:00:00+00:00",
                         created_at="2026-06-01T00:00:00+00:00")
        assert r.effective_time().startswith("2026-01-01")

    def test_schema_version_recorded(self, store):
        assert store.schema_version >= 1

    def test_timestamp_now_sortable(self):
        assert utcnow() <= utcnow()
