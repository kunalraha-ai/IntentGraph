"""Persistent memory archive: SQLite storage with versioning and provenance.

Why SQLite: it is in the standard library, transactional, single-file, runs
anywhere Python runs, and is honestly sufficient for a local-first memory
archive of millions of records. A heavyweight distributed store would add an
operational dependency the core does not need; if a hosted deployment later
outgrows SQLite, the store is behind this one class and the on-disk bundle
format (see :meth:`MemoryStore.export_bundle`) is the portability contract.

Invariants this module guarantees:

* **Append-only history.** Updating a record copies the current version into
  ``record_versions`` before the main row changes; nothing is ever lost in
  place. Reading a specific version always returns the bytes as ingested.
* **Soft delete by default.** ``delete_record`` marks and timestamps; only
  ``purge_record`` (retention control) removes bytes, and even then leaves a
  tombstone (id + hash + time) so surviving records that reference the purged
  memory are detectable instead of silently dangling.
* **Derived data is rebuildable.** Embeddings and lexical indexes live in
  side tables / in memory, never in the record row. Wiping them loses nothing
  that cannot be recomputed from ``records``.
* **Parameterized SQL everywhere** — record content is untrusted input.

Concurrency: WAL + busy_timeout supports one writer with concurrent readers,
which is the CLI/local-API usage pattern. It is not a multi-writer server.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator

from intentgraph.models import (
    SCHEMA_VERSION,
    Feedback,
    MemoryEdge,
    MemoryRecord,
    utcnow,
    validate_edge,
    validate_record,
)

__all__ = ["MemoryStore", "StoreError"]


class StoreError(Exception):
    """Raised for store-level failures: bad input, missing records, conflicts."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id           TEXT PRIMARY KEY,
    type         TEXT NOT NULL,
    content      TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'text',
    source_type  TEXT NOT NULL,
    source_id    TEXT NOT NULL DEFAULT '',
    provenance   TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    event_time   TEXT,
    ingested_at  TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    tags         TEXT NOT NULL DEFAULT '[]',
    content_hash TEXT NOT NULL,
    version      INTEGER NOT NULL DEFAULT 1,
    valid_from   TEXT NOT NULL,
    valid_to     TEXT,
    superseded_by TEXT,
    status       TEXT NOT NULL DEFAULT 'active',
    graph_anchors TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_records_status   ON records(status);
CREATE INDEX IF NOT EXISTS idx_records_type     ON records(type);
CREATE INDEX IF NOT EXISTS idx_records_hash     ON records(content_hash);
CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at);

CREATE TABLE IF NOT EXISTS record_versions (
    id         TEXT NOT NULL,
    version    INTEGER NOT NULL,
    snapshot   TEXT NOT NULL,
    saved_at   TEXT NOT NULL,
    PRIMARY KEY (id, version)
);

CREATE TABLE IF NOT EXISTS edges (
    id         TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    target     TEXT NOT NULL,
    relation   TEXT NOT NULL,
    explicit   INTEGER NOT NULL DEFAULT 1,
    basis      TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_rel    ON edges(relation);

CREATE TABLE IF NOT EXISTS feedback (
    id         TEXT PRIMARY KEY,
    query      TEXT NOT NULL DEFAULT '',
    outcome    TEXT NOT NULL,
    memory_ids TEXT NOT NULL DEFAULT '[]',
    note       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS embeddings (
    record_id TEXT NOT NULL,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (record_id, model)
);

CREATE TABLE IF NOT EXISTS tombstones (
    id         TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    deleted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS code_links (
    id         TEXT PRIMARY KEY,   -- lnk_<16hex>
    record_id  TEXT NOT NULL,     -- mem_* the link is from
    node_id    TEXT NOT NULL,     -- code-graph node id the link points at
    link_type  TEXT NOT NULL,     -- explicit | extracted | candidate
    basis      TEXT NOT NULL DEFAULT '',
    method     TEXT NOT NULL DEFAULT '',
    match      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_code_links_record ON code_links(record_id);
CREATE INDEX IF NOT EXISTS idx_code_links_node ON code_links(node_id);
"""


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    d = dict(row)
    d["provenance"] = json.loads(d.get("provenance") or "{}")
    d["metadata"] = json.loads(d.get("metadata") or "{}")
    d["tags"] = json.loads(d.get("tags") or "[]")
    d["graph_anchors"] = json.loads(d.get("graph_anchors") or "[]")
    return MemoryRecord.from_dict(d)


class MemoryStore:
    """SQLite-backed archive of memory records, their edges, and feedback."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the local HTTP API serves requests from
        # worker threads against one store. CPython's sqlite3 is built in
        # serialized mode, so sharing the connection is memory-safe; the
        # caller owns the WRITE discipline (api.py serializes every mutating
        # endpoint under a lock — write-write interleaving is prevented
        # there, not here). Reads may observe at most one just-committed
        # transaction's difference, which is the intended freshness for a
        # memory archive, not a consistency bug.
        self._db = sqlite3.connect(str(self.path), timeout=30.0,
                                   check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # WAL gives concurrent readers during a write (the local API serves
        # queries while an ingest lands). busy_timeout absorbs short lock
        # contention instead of failing the read outright.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=10000")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    # --- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _migrate(self) -> None:
        self._db.executescript(_SCHEMA)
        self._db.execute(
            "INSERT OR REPLACE INTO kv(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        # Future schema versions: read PRAGMA user_version and step migrations
        # here. v1 stores SCHEMA_VERSION in kv so import can check it without
        # parsing DDL.
        self._db.commit()

    @property
    def schema_version(self) -> int:
        row = self._db.execute("SELECT value FROM kv WHERE key='schema_version'").fetchone()
        return int(row["value"]) if row else 0

    # --- records ---------------------------------------------------------------

    def add_record(self, record: MemoryRecord, *, replace: bool = False) -> MemoryRecord:
        """Insert a new record. ``replace`` allows re-adding a known id verbatim
        (used by import); without it a conflicting id is an error, never a
        silent overwrite — silent overwrite is exactly how memory archives rot."""
        errors = validate_record(record.to_dict())
        if errors:
            raise StoreError(f"invalid record {record.id}: {'; '.join(errors)}")
        existing = self._db.execute("SELECT version FROM records WHERE id=?", (record.id,)).fetchone()
        if existing is not None:
            if replace:
                self._db.execute("DELETE FROM records WHERE id=?", (record.id,))
            else:
                raise StoreError(f"record {record.id} already exists (use update_record)")
        self._db.execute(
            """INSERT INTO records
               (id,type,content,content_type,source_type,source_id,provenance,
                created_at,event_time,ingested_at,metadata,tags,content_hash,
                version,valid_from,valid_to,superseded_by,status,graph_anchors)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.id, record.type, record.content, record.content_type,
                record.source_type, record.source_id,
                json.dumps(record.provenance, ensure_ascii=False),
                record.created_at, record.event_time, record.ingested_at,
                json.dumps(record.metadata, ensure_ascii=False),
                json.dumps(record.tags, ensure_ascii=False), record.content_hash,
                record.version, record.valid_from, record.valid_to,
                record.superseded_by, record.status,
                json.dumps(record.graph_anchors, ensure_ascii=False),
            ),
        )
        self._db.commit()
        return record

    def get_record(self, record_id: str, version: int | None = None) -> MemoryRecord | None:
        """Current record, or a specific historical version.

        Version lookup goes to the snapshot table first, then the live row —
        so asking for the *current* version number works even though the
        live row is not (yet) duplicated into history.
        """
        if version is not None:
            row = self._db.execute(
                "SELECT snapshot FROM record_versions WHERE id=? AND version=?",
                (record_id, version),
            ).fetchone()
            if row is not None:
                return MemoryRecord.from_dict(json.loads(row["snapshot"]))
        row = self._db.execute("SELECT * FROM records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            return None
        rec = _row_to_record(row)
        if version is not None and rec.version != version:
            return None
        return rec

    def record_versions(self, record_id: str) -> list[dict[str, Any]]:
        """Version history metadata (no content), newest first."""
        rows = self._db.execute(
            "SELECT version, saved_at FROM record_versions WHERE id=? ORDER BY version DESC",
            (record_id,),
        ).fetchall()
        current = self.get_record(record_id)
        out = [{"version": r["version"], "saved_at": r["saved_at"], "current": False} for r in rows]
        if current is not None:
            out.insert(0, {"version": current.version, "saved_at": current.ingested_at, "current": True})
        return out

    def update_record(
        self,
        record_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        event_time: str | None = None,
        graph_anchors: list[str] | None = None,
        note: str = "",
    ) -> MemoryRecord:
        """Append a new version of a record. The previous version is snapshotted.

        Only content-bearing fields are updatable through this path. Identity,
        provenance and creation time are immutable — an update that claims a
        different origin is a different memory and must be ingested as one.
        """
        current = self.get_record(record_id)
        if current is None:
            raise StoreError(f"no such record: {record_id}")
        if current.status != "active":
            raise StoreError(f"record {record_id} is {current.status}; update refused")
        new = MemoryRecord.from_dict(current.to_dict())
        if content is not None:
            if not isinstance(content, str):
                raise StoreError("content must be a string")
            new.content = content
            from intentgraph.models import content_hash as _ch
            new.content_hash = _ch(content)
        if metadata is not None:
            new.metadata = {**current.metadata, **dict(metadata)}
        if tags is not None:
            new.tags = [str(t) for t in tags][:64]
        if event_time is not None:
            new.event_time = event_time
        if graph_anchors is not None:
            new.graph_anchors = [str(a) for a in graph_anchors][:64]
        new.version = current.version + 1
        new.ingested_at = utcnow()
        new.valid_from = new.ingested_at
        # The update note is provenance for THIS version — keep it on the new
        # version's metadata, never on the record row, so history stays a
        # faithful account of what each version said and why it changed.
        if note:
            new.metadata = {**new.metadata, "update_note": note}
        errors = validate_record(new.to_dict())
        if errors:
            raise StoreError(f"invalid update for {record_id}: {'; '.join(errors)}")
        # snapshot the outgoing version BEFORE the live row changes
        self._db.execute(
            "INSERT OR REPLACE INTO record_versions(id, version, snapshot, saved_at) VALUES (?,?,?,?)",
            (current.id, current.version,
             json.dumps(current.to_dict(), ensure_ascii=False), utcnow()),
        )
        self._db.execute(
            """UPDATE records SET content=?, metadata=?, tags=?, event_time=?,
               graph_anchors=?, content_hash=?, version=?, ingested_at=?, valid_from=?
               WHERE id=?""",
            (
                new.content, json.dumps(new.metadata, ensure_ascii=False),
                json.dumps(new.tags, ensure_ascii=False), new.event_time,
                json.dumps(new.graph_anchors, ensure_ascii=False),
                new.content_hash, new.version, new.ingested_at, new.valid_from,
                record_id,
            ),
        )
        self._db.commit()
        return new

    def supersede(self, old_id: str, new_id: str, *, basis: str = "") -> MemoryRecord | None:
        """Mark ``old_id`` replaced by ``new_id``; also records a supersedes edge.

        The denormalized ``superseded_by``/``valid_to`` fields are the fast
        path the verifier reads; the edge is the auditable relationship.
        Both are set by this one operation so they cannot drift apart.
        """
        old = self.get_record(old_id)
        new = self.get_record(new_id)
        if old is None or new is None:
            raise StoreError(f"supersede requires both records to exist ({old_id}, {new_id})")
        now = utcnow()
        self._db.execute(
            "UPDATE records SET status='superseded', superseded_by=?, valid_to=? WHERE id=?",
            (new_id, now, old_id),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO record_versions(id, version, snapshot, saved_at) VALUES (?,?,?,?)",
            (old.id, old.version, json.dumps(old.to_dict(), ensure_ascii=False), now),
        )
        edge = MemoryEdge(
            source=new_id, target=old_id, relation="supersedes",
            explicit=True, basis=basis or "supersede() via API",
        )
        self._db.execute(
            """INSERT OR REPLACE INTO edges
               (id,source,target,relation,explicit,basis,confidence,created_at,metadata)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (edge.id, edge.source, edge.target, edge.relation, 1, edge.basis,
             edge.confidence, edge.created_at, "{}"),
        )
        self._db.commit()
        return self.get_record(old_id)

    def delete_record(self, record_id: str) -> MemoryRecord | None:
        """Soft-delete: keep bytes, mark status + valid_to. Reversible by restore."""
        rec = self.get_record(record_id)
        if rec is None:
            return None
        self._db.execute(
            "UPDATE records SET status='deleted', valid_to=? WHERE id=?",
            (utcnow(), record_id),
        )
        self._db.commit()
        return self.get_record(record_id)

    def restore_record(self, record_id: str) -> MemoryRecord | None:
        """Undo a soft delete (valid_to cleared, back to active)."""
        rec = self.get_record(record_id)
        if rec is None:
            return None
        if rec.status != "deleted":
            return rec
        self._db.execute(
            "UPDATE records SET status='active', valid_to=NULL, superseded_by=NULL WHERE id=?",
            (record_id,),
        )
        self._db.commit()
        return self.get_record(record_id)

    def purge_record(self, record_id: str) -> bool:
        """Retention control: hard-delete rows, leave a tombstone.

        Surviving edges that touched the record are removed too (an edge to a
        nonexistent memory is noise, and its relation semantics are not
        safe to reinterpret). Code links follow the same rule: the record is
        gone, so nothing may claim it explains a code entity. The tombstone
        keeps the id + hash so later imports/audits can tell that this
        memory existed and was purged, which is the difference between
        "deleted" and "corrupted".
        """
        rec = self.get_record(record_id)
        if rec is None:
            # already purged?
            row = self._db.execute("SELECT id FROM tombstones WHERE id=?", (record_id,)).fetchone()
            return row is not None
        now = utcnow()
        self._db.execute("DELETE FROM records WHERE id=?", (record_id,))
        self._db.execute("DELETE FROM record_versions WHERE id=?", (record_id,))
        self._db.execute("DELETE FROM edges WHERE source=? OR target=?", (record_id, record_id))
        self._db.execute("DELETE FROM embeddings WHERE record_id=?", (record_id,))
        self._db.execute("DELETE FROM code_links WHERE record_id=?", (record_id,))
        self._db.execute(
            "INSERT OR REPLACE INTO tombstones(id, content_hash, deleted_at) VALUES (?,?,?)",
            (record_id, rec.content_hash, now),
        )
        self._db.commit()
        return True

    def is_tombstoned(self, record_id: str) -> bool:
        row = self._db.execute("SELECT id FROM tombstones WHERE id=?", (record_id,)).fetchone()
        return row is not None

    def list_records(
        self,
        *,
        status: str | None = None,
        type: str | None = None,
        tag: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "created_at",
    ) -> tuple[list[MemoryRecord], int]:
        """Filtered listing with a total count, for API pagination."""
        if order not in ("created_at", "ingested_at", "event_time"):
            order = "created_at"
        clauses, params = [], []
        if status:
            clauses.append("status=?")
            params.append(status)
        if type:
            clauses.append("type=?")
            params.append(type)
        if tag:
            clauses.append("tags LIKE ?")
            # Exact JSON element match: '["auth"]' inside the tags array. A bare
            # '%auth%' would also hit 'oauth'; the post-filter below enforces a
            # true element match for correctness on top of the SQL narrowing.
            params.append(f'%{json.dumps([tag], ensure_ascii=False)}%')
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total = self._db.execute(
            f"SELECT COUNT(*) AS n FROM records {where}", params
        ).fetchone()["n"]
        rows = self._db.execute(
            f"SELECT * FROM records {where} ORDER BY {order} DESC LIMIT ? OFFSET ?",
            (*params, max(0, int(limit)), max(0, int(offset))),
        ).fetchall()
        recs = [_row_to_record(r) for r in rows]
        if tag:  # post-filter the LIKE to a true element match
            recs = [r for r in recs if tag in r.tags]
        return recs, total

    def get_records_by_ids(self, ids: Iterable[str]) -> dict[str, MemoryRecord]:
        out: dict[str, MemoryRecord] = {}
        ids = list(ids)
        for i in range(0, len(ids), 500):  # bound the SQL variable count
            chunk = ids[i : i + 500]
            if not chunk:
                continue
            q = ",".join("?" * len(chunk))
            rows = self._db.execute(f"SELECT * FROM records WHERE id IN ({q})", chunk).fetchall()
            for r in rows:
                rec = _row_to_record(r)
                out[rec.id] = rec
        return out

    def find_by_content_hash(self, chash: str, *, include_deleted: bool = False) -> list[MemoryRecord]:
        rows = self._db.execute(
            "SELECT * FROM records WHERE content_hash=? ORDER BY ingested_at", (chash,)
        ).fetchall()
        recs = [_row_to_record(r) for r in rows]
        if not include_deleted:
            recs = [r for r in recs if r.status == "active"]
        return recs

    def iter_all_records(self, *, include_deleted: bool = False) -> Iterator[MemoryRecord]:
        """Every current record, oldest ingested first (index-friendly)."""
        rows = self._db.execute("SELECT * FROM records ORDER BY ingested_at, id").fetchall()
        for r in rows:
            rec = _row_to_record(r)
            if include_deleted or rec.status == "active":
                yield rec

    def stats(self) -> dict[str, Any]:
        def _count(sql: str, params: tuple = ()) -> int:
            return self._db.execute(sql, params).fetchone()["n"]

        out: dict[str, Any] = {
            "records": _count("SELECT COUNT(*) AS n FROM records"),
            "records_active": _count(
                "SELECT COUNT(*) AS n FROM records WHERE status='active'"),
            "records_superseded": _count(
                "SELECT COUNT(*) AS n FROM records WHERE status='superseded'"),
            "records_deleted": _count(
                "SELECT COUNT(*) AS n FROM records WHERE status='deleted'"),
            "versions": _count("SELECT COUNT(*) AS n FROM record_versions"),
            "edges": _count("SELECT COUNT(*) AS n FROM edges"),
            "edges_inferred": _count(
                "SELECT COUNT(*) AS n FROM edges WHERE explicit=0"),
            "feedback": _count("SELECT COUNT(*) AS n FROM feedback"),
            "tombstones": _count("SELECT COUNT(*) AS n FROM tombstones"),
            "types": {},
        }
        for row in self._db.execute(
            "SELECT type, COUNT(*) AS n FROM records GROUP BY type ORDER BY n DESC"
        ):
            out["types"][row["type"]] = row["n"]
        return out

    # --- edges -----------------------------------------------------------------

    def add_edge(self, edge: MemoryEdge, *, replace: bool = False) -> MemoryEdge:
        errors = validate_edge(edge.to_dict())
        if errors:
            raise StoreError(f"invalid edge: {'; '.join(errors)}")
        for rid in (edge.source, edge.target):
            if self.get_record(rid) is None:
                raise StoreError(f"edge endpoint {rid} does not exist")
        dup = self._db.execute(
            "SELECT id FROM edges WHERE source=? AND target=? AND relation=?",
            (edge.source, edge.target, edge.relation),
        ).fetchone()
        if dup is not None and not replace:
            # Same relationship twice is idempotent from the caller's view;
            # returning the existing edge keeps ingest retry-safe.
            row = self._db.execute("SELECT * FROM edges WHERE id=?", (dup["id"],)).fetchone()
            return MemoryEdge.from_dict(_edge_dict(row))
        self._db.execute(
            """INSERT OR REPLACE INTO edges
               (id,source,target,relation,explicit,basis,confidence,created_at,metadata)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (edge.id, edge.source, edge.target, edge.relation,
             1 if edge.explicit else 0, edge.basis, edge.confidence,
             edge.created_at, json.dumps(edge.metadata, ensure_ascii=False)),
        )
        self._db.commit()
        return edge

    def remove_edge(self, edge_id: str) -> bool:
        cur = self._db.execute("DELETE FROM edges WHERE id=?", (edge_id,))
        self._db.commit()
        return cur.rowcount > 0

    def get_edge(self, edge_id: str) -> MemoryEdge | None:
        row = self._db.execute("SELECT * FROM edges WHERE id=?", (edge_id,)).fetchone()
        return MemoryEdge.from_dict(_edge_dict(row)) if row else None

    def iter_edges(self) -> Iterator[MemoryEdge]:
        rows = self._db.execute("SELECT * FROM edges ORDER BY created_at, id").fetchall()
        for row in rows:
            yield MemoryEdge.from_dict(_edge_dict(row))

    def edges_for(self, record_id: str, *, relation: str | None = None) -> list[MemoryEdge]:
        """All edges touching a record, in either direction."""
        rows = self._db.execute(
            "SELECT * FROM edges WHERE source=? OR target=? ORDER BY created_at",
            (record_id, record_id),
        ).fetchall()
        out = [MemoryEdge.from_dict(_edge_dict(r)) for r in rows]
        if relation:
            out = [e for e in out if e.relation == relation]
        return out

    # --- feedback ----------------------------------------------------------------

    def add_feedback(self, fb: Feedback) -> Feedback:
        if fb.outcome not in ("useful", "missing", "incorrect"):
            raise StoreError(f"invalid feedback outcome: {fb.outcome!r}")
        self._db.execute(
            """INSERT OR REPLACE INTO feedback
               (id,query,outcome,memory_ids,note,created_at,metadata)
               VALUES (?,?,?,?,?,?,?)""",
            (fb.id, fb.query, fb.outcome, json.dumps(fb.memory_ids, ensure_ascii=False),
             fb.note, fb.created_at, json.dumps(fb.metadata, ensure_ascii=False)),
        )
        self._db.commit()
        return fb

    def iter_feedback(self, limit: int = 100) -> list[Feedback]:
        rows = self._db.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (max(0, limit),)
        ).fetchall()
        out = []
        for r in rows:
            d = _edge_dict(r)
            d["memory_ids"] = json.loads(d.get("memory_ids") or "[]")
            d["outcome"] = d.get("outcome") or "useful"
            out.append(Feedback(**{k: v for k, v in d.items() if k in Feedback.__slots__}))
        return out

    # --- code links (code-graph ↔ memory; historical evidence never
    # rewritten by graph rebuilds — links live in their own table) --------------

    def add_code_link(self, record_id: str, node_id: str, link_type: str, *,
                      basis: str = "", method: str = "",
                      match: str = "") -> dict[str, Any]:
        """Attach a memory record to a code-graph node.

        ``link_type`` is the verification tier: ``explicit`` (user asserted),
        ``extracted`` (a deterministic matcher found a real file/symbol
        reference in the record text), ``candidate`` (similarity suggestion —
        NOT verified evidence). Idempotent per (record, node, type): re-adding
        refreshes the basis, never duplicates.
        """
        if link_type not in ("explicit", "extracted", "candidate"):
            raise StoreError(f"unknown link type {link_type!r}")
        if self.get_record(record_id) is None:
            raise StoreError(f"link failed: no such record {record_id}")
        existing = self._db.execute(
            "SELECT id FROM code_links WHERE record_id=? AND node_id=? AND link_type=?",
            (record_id, node_id, link_type)).fetchone()
        if existing is not None:
            self._db.execute(
                "UPDATE code_links SET basis=?, method=?, match=? WHERE id=?",
                (basis, method, match, existing["id"]))
            self._db.commit()
            out = self.get_code_link(existing["id"])
            assert out is not None  # just updated
            return out
        link_id = "lnk_" + uuid.uuid4().hex[:16]
        self._db.execute(
            """INSERT INTO code_links(id, record_id, node_id, link_type,
                                      basis, method, match, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (link_id, record_id, node_id, link_type, basis, method, match,
             utcnow()))
        self._db.commit()
        out = self.get_code_link(link_id)
        assert out is not None  # just inserted
        return out

    def get_code_link(self, link_id: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM code_links WHERE id=?",
                               (link_id,)).fetchone()
        return dict(row) if row else None

    def remove_code_link(self, link_id: str) -> bool:
        cur = self._db.execute("DELETE FROM code_links WHERE id=?", (link_id,))
        self._db.commit()
        return cur.rowcount > 0

    def remove_code_links(self, *, record_id: str | None = None,
                          node_id: str | None = None,
                          link_type: str | None = None) -> int:
        """Bulk removal for the refresh path (e.g. stale extracted links)."""
        clauses, params = [], []
        if record_id is not None:
            clauses.append("record_id=?"); params.append(record_id)
        if node_id is not None:
            clauses.append("node_id=?"); params.append(node_id)
        if link_type is not None:
            clauses.append("link_type=?"); params.append(link_type)
        if not clauses:
            raise StoreError("remove_code_links needs at least one filter")
        sql = "DELETE FROM code_links WHERE " + " AND ".join(clauses)
        cur = self._db.execute(sql, params)
        self._db.commit()
        return cur.rowcount

    def iter_code_links(self, *, record_id: str | None = None,
                        node_id: str | None = None,
                        link_type: str | None = None) -> list[dict[str, Any]]:
        """Links in stable (node_id, record_id) order — deterministic listing."""
        clauses, params = [], []
        if record_id is not None:
            clauses.append("record_id=?"); params.append(record_id)
        if node_id is not None:
            clauses.append("node_id=?"); params.append(node_id)
        if link_type is not None:
            clauses.append("link_type=?"); params.append(link_type)
        sql = "SELECT * FROM code_links"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY node_id, record_id, link_type"
        return [dict(r) for r in self._db.execute(sql, params)]

    def code_links_count(self) -> dict[str, int]:
        out = {"explicit": 0, "extracted": 0, "candidate": 0}
        for r in self._db.execute(
                "SELECT link_type, COUNT(*) AS n FROM code_links GROUP BY link_type"):
            out[r["link_type"]] = r["n"]
        return out

    # --- embeddings (derived, rebuildable) ----------------------------------------

    def set_embedding(self, record_id: str, model: str, vector: Any) -> None:
        import struct

        blob = struct.pack(f"<{len(vector)}f", *vector)
        self._db.execute(
            """INSERT OR REPLACE INTO embeddings(record_id, model, dim, vector, updated_at)
               VALUES (?,?,?,?,?)""",
            (record_id, model, len(vector), blob, utcnow()),
        )
        self._db.commit()

    def load_embeddings(self, model: str) -> tuple[list[str], Any]:
        """All stored vectors for a model → (record_ids, numpy matrix).

        Returns an empty matrix when numpy is unavailable — the semantic
        retriever checks and disables itself rather than half-working.
        """
        import struct

        rows = self._db.execute(
            "SELECT record_id, dim, vector FROM embeddings WHERE model=? ORDER BY record_id",
            (model,),
        ).fetchall()
        if not rows:
            return [], None
        try:
            import numpy as np
        except ImportError:
            return [], None
        ids = [r["record_id"] for r in rows]
        mat = np.empty((len(rows), rows[0]["dim"]), dtype=np.float32)
        for i, r in enumerate(rows):
            mat[i] = np.frombuffer(r["vector"], dtype=np.float32)
        return ids, mat

    def stored_embedding_model(self) -> str | None:
        row = self._db.execute("SELECT DISTINCT model FROM embeddings LIMIT 1").fetchone()
        return row["model"] if row else None

    def drop_embeddings(self) -> int:
        cur = self._db.execute("DELETE FROM embeddings")
        self._db.commit()
        return cur.rowcount

    # --- export / import -----------------------------------------------------------

    def export_bundle(self, path: str | Path, *, include_deleted: bool = True,
                      include_feedback: bool = True) -> dict[str, Any]:
        """Write a portable JSONL bundle: one header line, then records/edges.

        The bundle is the portability contract (same format locally and for a
        future hosted import): stable ids, full provenance, versions, and
        relationship endpoints. Indexes and embeddings are excluded — they are
        derived and rebuildable.
        """
        path = Path(path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        n_records = n_edges = n_feedback = n_links = 0
        with path.open("w", encoding="utf-8") as fh:
            header = {
                "intentgraph_export": True,
                "schema_version": SCHEMA_VERSION,
                "exported_at": utcnow(),
                "counts": {},
            }
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            for rec in self._db.execute("SELECT * FROM records ORDER BY id"):
                if rec["status"] == "deleted" and not include_deleted:
                    continue
                fh.write(json.dumps({"kind": "record", "record": _row_to_record(rec).to_dict()},
                                   ensure_ascii=False) + "\n")
                n_records += 1
            for row in self._db.execute("SELECT * FROM record_versions ORDER BY id, version"):
                fh.write(json.dumps({"kind": "record_version", "id": row["id"],
                                     "version": row["version"], "snapshot": json.loads(row["snapshot"]),
                                     "saved_at": row["saved_at"]}, ensure_ascii=False) + "\n")
                n_records += 1
            for row in self._db.execute("SELECT * FROM edges ORDER BY id"):
                fh.write(json.dumps({"kind": "edge", "edge": _edge_dict(row)}, ensure_ascii=False) + "\n")
                n_edges += 1
            # code links are part of the portable archive: they are separate
            # from record content by design (graph rebuilds never rewrite
            # historical evidence), but they ARE the archive's evidence wiring
            for row in self._db.execute("SELECT * FROM code_links ORDER BY id"):
                fh.write(json.dumps({"kind": "code_link", "link": dict(row)},
                                   ensure_ascii=False) + "\n")
                n_links += 1
            if include_feedback:
                for row in self._db.execute("SELECT * FROM feedback ORDER BY id"):
                    d = _edge_dict(row)
                    d["memory_ids"] = json.loads(d.get("memory_ids") or "[]")
                    fh.write(json.dumps({"kind": "feedback", "feedback": d},
                                        ensure_ascii=False) + "\n")
                    n_feedback += 1
        return {"records": n_records, "edges": n_edges, "feedback": n_feedback,
                "code_links": n_links,
                "path": str(path), "schema_version": SCHEMA_VERSION}

    def import_bundle(self, path: str | Path, *, replace: bool = True) -> dict[str, Any]:
        """Import a bundle written by :meth:`export_bundle`.

        Idempotent: a record whose id+version already exists is skipped, so a
        re-import after a partial failure cannot double-apply. ``replace``
        lets a same-id bundle entry overwrite the live row (restore use case).
        """
        path = Path(path)
        counts = {"records": 0, "record_versions": 0, "edges": 0, "feedback": 0,
                  "code_links": 0, "skipped": 0}
        with path.open("r", encoding="utf-8") as fh:
            header_seen = False
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StoreError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
                if not header_seen:
                    if not entry.get("intentgraph_export"):
                        raise StoreError(f"{path}: not an IntentGraph export bundle")
                    bundle_version = int(entry.get("schema_version", 0))
                    if bundle_version > SCHEMA_VERSION:
                        # newer than this build understands: refuse, never
                        # silently reinterpret (older bundles import fine —
                        # additions like code_links are optional kinds)
                        raise StoreError(
                            f"{path}: bundle schema version {bundle_version} "
                            f"is newer than supported {SCHEMA_VERSION}")
                    header_seen = True
                    continue
                kind = entry.get("kind")
                if kind == "record":
                    rec = MemoryRecord.from_dict(entry["record"])
                    existing = self.get_record(rec.id)
                    if existing is not None and existing.version >= rec.version and not replace:
                        counts["skipped"] += 1
                        continue
                    if existing is not None and existing.version == rec.version \
                            and existing.content_hash == rec.content_hash:
                        counts["skipped"] += 1
                        continue
                    self.add_record(rec, replace=replace)
                    counts["records"] += 1
                elif kind == "record_version":
                    snap = MemoryRecord.from_dict(entry["snapshot"])
                    self._db.execute(
                        "INSERT OR REPLACE INTO record_versions(id, version, snapshot, saved_at)"
                        " VALUES (?,?,?,?)",
                        (entry["id"], entry["version"],
                         json.dumps(entry["snapshot"], ensure_ascii=False), entry["saved_at"]),
                    )
                    counts["record_versions"] += 1
                elif kind == "edge":
                    edge = MemoryEdge.from_dict(entry["edge"])
                    for rid in (edge.source, edge.target):
                        if self.get_record(rid) is None and not self.is_tombstoned(rid):
                            # An endpoint missing from the bundle AND not a
                            # known purge would be a dangling edge; skip the
                            # edge and count it so the report shows the loss.
                            counts["skipped"] += 1
                            edge = None
                            break
                    if edge is not None:
                        self._db.execute(
                            """INSERT OR REPLACE INTO edges
                               (id,source,target,relation,explicit,basis,confidence,
                                created_at,metadata) VALUES (?,?,?,?,?,?,?,?,?)""",
                            (edge.id, edge.source, edge.target, edge.relation,
                             1 if edge.explicit else 0, edge.basis, edge.confidence,
                             edge.created_at,
                             json.dumps(edge.metadata, ensure_ascii=False)),
                        )
                        counts["edges"] += 1
                elif kind == "feedback":
                    fb = Feedback(**{k: v for k, v in entry["feedback"].items()
                                     if k in Feedback.__slots__})
                    self.add_feedback(fb)
                    counts["feedback"] += 1
                elif kind == "code_link":
                    # additive kind: bundles without it (schema v1) import
                    # unchanged; links whose record is missing AND not a
                    # known purge are skipped and counted, like edges
                    link = entry["link"]
                    rid = link.get("record_id", "")
                    if self.get_record(rid) is None and not self.is_tombstoned(rid):
                        counts["skipped"] += 1
                        continue
                    self._db.execute(
                        """INSERT OR REPLACE INTO code_links
                           (id, record_id, node_id, link_type, basis, method,
                            match, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                        (link.get("id") or "lnk_" + uuid.uuid4().hex[:16],
                         rid, link.get("node_id", ""),
                         link.get("link_type", "explicit"),
                         link.get("basis", ""), link.get("method", ""),
                         link.get("match", ""), link.get("created_at", utcnow())))
                    counts["code_links"] += 1
        self._db.commit()
        return counts


def _edge_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if "metadata" in d:
        d["metadata"] = json.loads(d.get("metadata") or "{}")
    if "explicit" in d:
        d["explicit"] = bool(d["explicit"])
    return d
