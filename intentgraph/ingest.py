"""Safe ingestion: validation, type inference, auto-linking, near-dup detection.

The ingest path is where untrusted content enters the archive, so this module
is deliberately the only place raw text becomes a record:

* Content is newline-normalized and size-capped (models.validate_record).
* ``type`` may be omitted; a documented keyword heuristic infers it. The
  inference is a *convenience default*, not a claim — the record's data is
  what it is, and every inferred value is recorded in
  ``provenance.type_inferred`` so a consumer can distrust it.
* **Exact duplicates** (same content hash, same active status) are not
  re-ingested: the existing record is returned, flagged, and a
  ``duplicate_of`` edge is offered. Identical bytes are one memory.
* **Near-duplicates** (MinHash Jaccard >= threshold) ARE ingested — near-dup
  records often carry distinct provenance (different session, different
  source) and the master prompt's honesty rule is to keep them and *link*
  them, not silently merge. The MinHash sketch reuses graphify's
  ``graphify._minhash.MinHash`` (datasketch-compatible, no scipy).
* **Mention edges**: content that references other record ids
  (``mem_<16hex>``) gets ``mentions`` edges, marked inferred with the exact
  basis so the association is auditable.

All heuristics here are documented as heuristics — none are learned weights.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from intentgraph.models import (
    MemoryEdge,
    MemoryRecord,
    parse_timestamp,
)
from intentgraph.store import MemoryStore, StoreError

# Reused from graphify's dedup machinery (datasketch-compatible, no scipy).
# numpy is a hard dependency of graphify, so this import is unconditional —
# the same reasoning as graphify/dedup.py importing it directly.
from graphify._minhash import MinHash

__all__ = ["IngestResult", "Ingestor", "ingest", "ingest_file", "ingest_dict"]

# --- type-inference heuristic ----------------------------------------------------
#
# Ordered: first match wins. Deliberately keyword-based and conservative — a
# wrong type is worse than a generic one, so only strong signals map away from
# the default ("fact"). This is a heuristic, not a classifier.

_TYPE_SIGNALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("decision", (
        "we decided", "decided to", "decision:", "adr-", "architecture decision",
        "we chose", "chosen approach", "agreed to", "going with", "rationale:",
    )),
    ("experience", (
        "failed because", "we tried", "attempted fix", "did not work",
        "lesson learned", "root cause", "postmortem", "regression",
        "broke the build", "flaky test", "outage",
    )),
    ("preference", (
        "always use", "never use", "prefer", "please use", "convention:",
        "style guide", "must be", "should always",
    )),
    ("task", ("todo", "task:", "follow-up", "action item", "needs to be done")),
    ("action", ("ran the", "executed", "deployed", "merged", "reverted", "pushed")),
    ("conversation", ("user asked", "the user said", "assistant replied", "in chat")),
)


def infer_type(content: str) -> tuple[str, bool]:
    """Return (type, was_inferred). Heuristic keyword match, first-wins."""
    lowered = content[:4000].lower()  # signals live at the start of real records
    for mtype, signals in _TYPE_SIGNALS:
        for sig in signals:
            if sig in lowered:
                return mtype, True
    return "fact", True


def infer_event_time(value: str | None) -> str | None:
    """Parse an explicit event-time string; None stays None.

    Accepted: full ISO-8601, and date-only (``YYYY-MM-DD``) which is a very
    common form in decision records and logs. Date-only is expanded to
    midnight UTC — ordering matters more than the hour for memory recall.
    """
    if not value:
        return None
    v = value.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
        v = v + "T00:00:00+00:00"
    dt = parse_timestamp(v)
    return dt.isoformat() if dt else None


_MENTION_RE = re.compile(r"\bmem_[0-9a-f]{16}\b")

# Word 4-gram shingles: whitespace-stable and edit-tolerant where char k-grams
# would over-fragment on formatting drift. Words keep meaning-bearing units.
_SHINGLE_WORDS = 4
_NEAR_DUP_JACCARD = 0.72   # above: near-duplicate; below: merely related
_NUM_PERM = 128


def _word_shingles(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < _SHINGLE_WORDS:
        return [" ".join(words)] if words else []
    return [
        " ".join(words[i : i + _SHINGLE_WORDS])
        for i in range(len(words) - _SHINGLE_WORDS + 1)
    ]


def _sketch(text: str):
    m = MinHash(num_perm=_NUM_PERM)
    for sh in _word_shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def _jaccard(a, b) -> float:
    """Jaccard over two MinHash sketches (estimate from signature agreement)."""
    if a is None or b is None:
        return 0.0
    agree = int((a.hashvalues == b.hashvalues).sum())
    return agree / float(a.num_perm)


class IngestResult:
    """What an ingest produced — the record plus everything derived around it."""

    __slots__ = ("record", "duplicate_of", "near_duplicates", "auto_edges", "type_inferred")

    def __init__(self, record: MemoryRecord, *, duplicate_of: MemoryRecord | None = None,
                 near_duplicates: list[dict[str, Any]] | None = None,
                 auto_edges: list[MemoryEdge] | None = None, type_inferred: bool = False):
        self.record = record
        self.duplicate_of = duplicate_of
        self.near_duplicates = near_duplicates or []
        self.auto_edges = auto_edges or []
        self.type_inferred = type_inferred

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record.id,
            "type": self.record.type,
            "type_inferred": self.type_inferred,
            "duplicate_of": self.duplicate_of.id if self.duplicate_of else None,
            "near_duplicates": self.near_duplicates,
            "auto_edges": [e.to_dict() for e in self.auto_edges],
        }


class Ingestor:
    """Stateful ingest helper bound to one store.

    Keeps an in-memory MinHash cache of active records so repeated ingests do
    not re-hash the archive each time. The cache is a performance layer only —
    correctness never depends on it (near-dup edges are advisory links).
    """

    def __init__(self, store: MemoryStore):
        self.store = store
        self._sketches: dict[str, Any] = {}

    def _sketch_for(self, record_id: str, content: str):
        if record_id not in self._sketches:
            self._sketches[record_id] = _sketch(content)
        return self._sketches[record_id]

    def ingest(
        self,
        content: str,
        *,
        type: str | None = None,
        tags: list[str] | None = None,
        source_type: str = "manual",
        source_id: str = "",
        provenance: dict[str, Any] | None = None,
        event_time: str | None = None,
        metadata: dict[str, Any] | None = None,
        graph_anchors: list[str] | None = None,
        content_type: str = "text",
        id: str | None = None,
        created_at: str | None = None,
        allow_duplicate: bool = False,
        link_mentions: bool = True,
        link_near_duplicates: bool = True,
    ) -> IngestResult:
        if not isinstance(content, str) or not content.strip():
            raise StoreError("content must be a non-empty string")
        type_inferred = False
        if type is None:
            type, type_inferred = infer_type(content)
        if type_inferred:
            provenance = dict(provenance or {})
            provenance["type_inferred"] = True
        ev = infer_event_time(event_time)
        if event_time and ev is None:
            raise StoreError(f"event_time is not a parseable timestamp: {event_time!r}")
        rec = MemoryRecord(
            id=id, type=type, content=content, content_type=content_type,
            source_type=source_type, source_id=source_id, provenance=provenance,
            event_time=ev, tags=tags, metadata=metadata,
            graph_anchors=graph_anchors, created_at=created_at,
        )

        # 1. exact duplicate short-circuit
        dup = self.store.find_by_content_hash(rec.content_hash)
        if dup and not allow_duplicate:
            existing = dup[0]
            return IngestResult(existing, duplicate_of=existing)

        # 2. store first — every derived link needs the record to exist
        self.store.add_record(rec)

        auto_edges: list[MemoryEdge] = []
        near_dups: list[dict[str, Any]] = []

        # 3. mention edges (inferred, auditable)
        if link_mentions:
            for other_id in sorted(set(_MENTION_RE.findall(content))):
                if other_id == rec.id:
                    continue
                other = self.store.get_record(other_id)
                if other is None:
                    continue
                edge = self.store.add_edge(MemoryEdge(
                    source=rec.id, target=other_id, relation="mentions",
                    explicit=False, basis="record id referenced in content",
                    confidence=1.0,
                ))
                auto_edges.append(edge)

        # 4. near-duplicate links (inferred, similarity recorded)
        if link_near_duplicates:
            mine = self._sketch_for(rec.id, rec.content)
            if mine is not None:
                for other in self.store.iter_all_records():
                    if other.id == rec.id or other.content_hash == rec.content_hash:
                        continue
                    theirs = self._sketch_for(other.id, other.content)
                    j = _jaccard(mine, theirs)
                    if j >= _NEAR_DUP_JACCARD:
                        near_dups.append({"id": other.id, "jaccard": round(j, 3)})
                for nd in near_dups:
                    edge = self.store.add_edge(MemoryEdge(
                        source=rec.id, target=nd["id"], relation="duplicate_of",
                        explicit=False,
                        basis=f"minhash jaccard {nd['jaccard']} >= {_NEAR_DUP_JACCARD}",
                        confidence=nd["jaccard"],
                    ))
                    auto_edges.append(edge)

        return IngestResult(
            rec, auto_edges=auto_edges, near_duplicates=near_dups,
            type_inferred=type_inferred,
        )

    def ingest_dict(self, data: dict[str, Any], **overrides: Any) -> IngestResult:
        """Ingest from a plain dict (the API's wire shape).

        Unknown keys are rejected loudly (a typo'd field silently dropping
        provenance is exactly the kind of rot a memory archive cannot afford).
        """
        allowed = {
            "content", "type", "tags", "source_type", "source_id", "provenance",
            "event_time", "metadata", "graph_anchors", "content_type", "id",
            "created_at", "allow_duplicate", "link_mentions", "link_near_duplicates",
        }
        unknown = set(data) - allowed
        if unknown:
            raise StoreError(f"unknown ingest fields: {sorted(unknown)}")
        kwargs = {**{k: v for k, v in data.items() if k not in ("content",)},
                  **overrides}
        return self.ingest(data.get("content", ""), **kwargs)

    def ingest_file(self, path: str | Path, **kwargs: Any) -> IngestResult:
        """Ingest one local file as a document record.

        Size guard: files above the record content cap are rejected with a
        clear message rather than read and then failed — a 2GB log dragged
        into memory first is the DoS, not the rejection.
        """
        from intentgraph.models import _MAX_CONTENT_CHARS

        p = Path(path)
        if not p.is_file():
            raise StoreError(f"not a file: {p}")
        size = p.stat().st_size
        if size > 2_000_000:
            raise StoreError(
                f"file too large to ingest as one record ({size} bytes; "
                f"cap 2,000,000). Split or excerpt it first.")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise StoreError(f"cannot read {p}: {exc}") from exc
        kwargs.setdefault("source_type", "file")
        kwargs.setdefault("source_id", str(p))
        kwargs.setdefault("type", "document")
        return self.ingest(text, **kwargs)


def ingest(store: MemoryStore, content: str, **kwargs: Any) -> IngestResult:
    """One-shot ingest on a fresh Ingestor (cache-free path)."""
    return Ingestor(store).ingest(content, **kwargs)


def ingest_file(store: MemoryStore, path: str | Path, **kwargs: Any) -> IngestResult:
    return Ingestor(store).ingest_file(path, **kwargs)


def ingest_dict(store: MemoryStore, data: dict[str, Any], **overrides: Any) -> IngestResult:
    return Ingestor(store).ingest_dict(data, **overrides)
