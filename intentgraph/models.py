"""IntentGraph data model: memory records, edges, relations, provenance.

Design decisions, recorded where they live:

* **Original evidence is the record.** Derived artifacts (BM25 index,
  embeddings, extracted facts, summaries) are always rebuildable from the
  stored content. A record's ``content`` is never rewritten in place — updates
  append a new version and the previous version stays readable
  (:class:`MemoryStore.record_versions`).
* **Stable identity.** A record id (``mem_<16hex>``) is assigned at first
  ingest and never changes across updates. ``content_hash`` is separate, so
  an edit produces a new version of the *same* memory rather than a new one.
* **Explicit vs inferred.** Every edge carries ``explicit`` and ``basis`` so a
  consumer can tell a user-declared relationship from one the system inferred
  (and why) — the EXTRACTED/INFERRED distinction rendered on the graph.
* **Validity is data, not deletion.** Supersession and deletion set
  ``valid_to``/``status``; rows are kept (soft delete) so history and audit
  remain intact.

All timestamps are ISO-8601 UTC strings ("YYYY-MM-DDTHH:MM:SS.ffffff+00:00").
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "MEMORY_TYPES",
    "RELATIONS",
    "RELATIONS_TEMPORAL",
    "RELATION_HELP",
    "SOURCE_TYPES",
    "FEEDBACK_OUTCOMES",
    "SCHEMA_VERSION",
    "utcnow",
    "parse_timestamp",
    "content_hash",
    "new_record_id",
    "new_edge_id",
    "canonicalize_content",
    "MemoryRecord",
    "MemoryEdge",
    "Feedback",
    "validate_record",
    "validate_edge",
]


# --- schema -------------------------------------------------------------------

#: Bumped when the on-disk record/edge shape changes in a way old readers would
#: misinterpret. Export bundles carry it; import refuses mismatched majors.
SCHEMA_VERSION = 1

#: Memory types the ingest API understands. Deliberately open vocabulary for
#: ``metadata`` but closed for ``type``: retrieval heuristics and the evidence
#: verifier branch on type, and a typo there would silently opt a record out of
#: checks. New types are a schema change.
MEMORY_TYPES = (
    "conversation",   # an interaction episode, turn, or dialogue excerpt
    "fact",           # a stated fact or entity description
    "decision",        # a decision, with its rationale in content
    "action",          # something that was done; a tool call, a change, a step
    "experience",      # an outcome: a failure, a success, a lesson learned
    "document",       # external research, docs, a paper, a note
    "code_reference",  # knowledge anchored to a file/symbol in a Graphify graph
    "task",           # a task, plan, or project state item
    "preference",      # a user/system preference or constraint
    "project_state",   # a snapshot of project/system state
)

#: Edge relation vocabulary. ``related_to`` is the generic fallback; the rest
#: carry retrieval meaning (the graph expander prefers high-signal relations
#: like ``caused_by``/``failed_because_of`` when following evidence trails).
RELATIONS = (
    "related_to",
    "derived_from",
    "caused_by",
    "contradicts",
    "supersedes",
    "depends_on",
    "occurred_before",
    "occurred_after",
    "validated_by",
    "failed_because_of",
    "belongs_to",
    "duplicate_of",
    "mentions",
)
#: Relations whose meaning is temporal ordering; the verifier checks these
#: against ``event_time`` for consistency (see verify.py).
RELATIONS_TEMPORAL = ("occurred_before", "occurred_after")

RELATION_HELP = {
    "related_to": "Generic association; the weakest, always-safe edge.",
    "derived_from": "Target record is a source this one was derived from.",
    "caused_by": "This record's event was caused by the target record.",
    "contradicts": "The two records make conflicting claims (surfaced, not resolved).",
    "supersedes": "This record replaces the target record.",
    "depends_on": "This record depends on the target being true/available.",
    "occurred_before": "Temporal ordering: source happened before target.",
    "occurred_after": "Temporal ordering: source happened after target.",
    "validated_by": "Target record provides validation/evidence for this one.",
    "failed_because_of": "This failure happened because of the target record.",
    "belongs_to": "Same project, entity, or episode grouping.",
    "duplicate_of": "Near-identical content stored separately (kept distinct).",
    "mentions": "Source content references the target memory.",
}

#: Where a record came from. Used for provenance checks before evidence is
#: admitted into a context packet.
SOURCE_TYPES = (
    "conversation",  # a chat/agent session
    "file",          # a local file that was ingested
    "tool",          # a tool result (test output, command output, log)
    "agent",         # an AI agent's note
    "api",           # the public API
    "import",        # an import bundle
    "manual",        # a human writing into the archive
    "url",           # fetched external content
)

#: Feedback outcomes recorded by consuming systems (POST /v1/feedback).
FEEDBACK_OUTCOMES = ("useful", "missing", "incorrect")


# --- primitives ----------------------------------------------------------------

_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$"
)


def utcnow() -> str:
    """Current UTC time as a sortable ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp leniently; ``None`` when unparseable.

    Accepts a trailing ``Z`` (common in JSON) and ``+hhmm`` without a colon.
    Returns an aware datetime in UTC so temporal comparisons are never done
    between naive and aware values (which raises TypeError mid-verification).
    """
    if not isinstance(value, str) or not _TIMESTAMP_RE.match(value):
        return None
    v = value[:-1] + "+00:00" if value.endswith("Z") else value
    if re.search(r"[+-]\d{4}$", v):
        v = v[:-5] + v[-5:-2] + ":" + v[-2:]
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def canonicalize_content(text: str) -> str:
    """Normalize newlines only — NOT whitespace-collapsing.

    A content hash must change when a *semantic* edit happens but stay stable
    across line-ending drift (CRLF vs LF), which is the one transformation
    machines routinely make behind a user's back. Collapsing all whitespace
    would make ``"fix it"`` and ``"fix   it"`` the same memory.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def content_hash(text: str) -> str:
    """sha256 of canonicalized content, hex, 16 chars.

    16 hex chars (64 bits) is far beyond any realistic archive's birthday
    bound (a billion records gives ~10^-11 collision odds); the id is an
    integrity/dedup key, not a security boundary.
    """
    return hashlib.sha256(canonicalize_content(text).encode("utf-8", "replace")).hexdigest()[:16]


def new_record_id() -> str:
    return f"mem_{uuid.uuid4().hex[:16]}"


def new_edge_id() -> str:
    return f"edg_{uuid.uuid4().hex[:16]}"


def _cap(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# --- records --------------------------------------------------------------------

class MemoryRecord:
    """One remembered unit with provenance and validity.

    ``content`` is the original evidence. It is stored verbatim (only
    newline-normalized on hash/id computation) and is never the only copy of
    derived data — see :mod:`intentgraph.store` for versioning.
    """

    __slots__ = (
        "id", "type", "content", "content_type", "source_type", "source_id",
        "provenance", "created_at", "event_time", "ingested_at",
        "metadata", "tags", "content_hash", "version", "valid_from",
        "valid_to", "superseded_by", "status", "graph_anchors",
    )

    def __init__(
        self,
        *,
        id: str | None = None,
        type: str = "fact",
        content: str = "",
        content_type: str = "text",
        source_type: str = "manual",
        source_id: str = "",
        provenance: dict[str, Any] | None = None,
        created_at: str | None = None,
        event_time: str | None = None,
        ingested_at: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        version: int = 1,
        valid_from: str | None = None,
        valid_to: str | None = None,
        superseded_by: str | None = None,
        status: str = "active",
        graph_anchors: list[str] | None = None,
        content_hash: str | None = None,
    ):
        self.id = id or new_record_id()
        self.type = type
        self.content = canonicalize_content(content)
        self.content_type = content_type
        self.source_type = source_type
        self.source_id = _cap(str(source_id or ""), 512)
        self.provenance = dict(provenance or {})
        self.created_at = created_at or utcnow()
        self.event_time = event_time
        self.ingested_at = ingested_at or utcnow()
        self.metadata = dict(metadata or {})
        self.tags = [str(t) for t in (tags or [])][:64]
        # The `content_hash` parameter shadows the module-level function of
        # the same name inside this scope, so compute the fallback via _hash.
        self.content_hash = content_hash or _hash(self.content)
        self.version = int(version)
        self.valid_from = valid_from or self.created_at
        self.valid_to = valid_to
        self.superseded_by = superseded_by
        self.status = status
        self.graph_anchors = [str(a) for a in (graph_anchors or [])][:64]

    # -- derived properties ---------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True when this version is the current, non-deleted one."""
        return self.status == "active" and self.valid_to is None

    @property
    def is_deleted(self) -> bool:
        return self.status == "deleted"

    @property
    def is_superseded(self) -> bool:
        return self.status == "superseded" or self.superseded_by is not None

    def effective_time(self) -> str:
        """Event time when known, otherwise created_at.

        Retrieval recency and temporal verification use the time the memory
        *describes*, not the (often much later) ingestion time.
        """
        return self.event_time or self.created_at

    def search_text(self) -> str:
        """Text the lexical index and embedder see.

        Content plus the human-relevant identity fields (tags, source) so a
        query for ``adr-007`` or ``auth`` can hit records whose body never
        spells the word. Bounded: metadata is excluded because it is often
        large, machine-generated JSON whose tokens would drown the body.
        """
        parts = [self.content]
        if self.tags:
            parts.append(" ".join(self.tags))
        if self.source_id:
            parts.append(self.source_id)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "content_type": self.content_type,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "provenance": self.provenance,
            "created_at": self.created_at,
            "event_time": self.event_time,
            "ingested_at": self.ingested_at,
            "metadata": self.metadata,
            "tags": self.tags,
            "content_hash": self.content_hash,
            "version": self.version,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "superseded_by": self.superseded_by,
            "status": self.status,
            "graph_anchors": self.graph_anchors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        known = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return f"<MemoryRecord {self.id} type={self.type} v{self.version} {self.status}>"


# `content_hash` is both a module function and a record field; keep the call
# unambiguous inside __init__ (the parameter shadows the function there).
def _hash(text: str) -> str:
    return content_hash(text)


class MemoryEdge:
    """A typed, directed relationship between two memory records."""

    __slots__ = ("id", "source", "target", "relation", "explicit", "basis",
                 "confidence", "created_at", "metadata")

    def __init__(
        self,
        *,
        id: str | None = None,
        source: str = "",
        target: str = "",
        relation: str = "related_to",
        explicit: bool = True,
        basis: str = "",
        confidence: float = 1.0,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.id = id or new_edge_id()
        self.source = source
        self.target = target
        self.relation = relation
        self.explicit = bool(explicit)
        # basis: WHY the edge exists — "user declared", "mentioned mem_x in
        # content", "extracted from import". Preserved so inferred edges stay
        # auditable (the same honesty rule as the EXTRACTED/INFERRED split).
        self.basis = _cap(str(basis or ""), 2048)
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.created_at = created_at or utcnow()
        self.metadata = dict(metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "explicit": self.explicit,
            "basis": self.basis,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEdge":
        known = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __repr__(self) -> str:  # pragma: no cover — debug helper
        return f"<MemoryEdge {self.source} -{self.relation}-> {self.target}>"


class Feedback:
    """A consuming system's report on how a retrieval/context turned out."""

    __slots__ = ("id", "query", "outcome", "memory_ids", "note", "created_at", "metadata")

    def __init__(
        self,
        *,
        id: str | None = None,
        query: str = "",
        outcome: str = "useful",
        memory_ids: list[str] | None = None,
        note: str = "",
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.id = id or f"fb_{uuid.uuid4().hex[:16]}"
        self.query = query
        self.outcome = outcome
        self.memory_ids = [str(m) for m in (memory_ids or [])][:256]
        self.note = _cap(str(note or ""), 2048)
        self.created_at = created_at or utcnow()
        self.metadata = dict(metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "outcome": self.outcome,
            "memory_ids": self.memory_ids,
            "note": self.note,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


# --- validation ----------------------------------------------------------------
#
# Validation errors are returned as strings (caller decides to raise).
# Malformed records must
# be rejected before they land in the store: one bad row would otherwise ride
# along in every export and poison every rebuild.

_MAX_CONTENT_CHARS = 2_000_000   # 2 MB of text per record; larger must be chunked
_MAX_PROV_DEPTH = 8_000          # provenance/metadata JSON cap


def validate_record(data: dict[str, Any]) -> list[str]:
    """Return a list of schema problems with a record-shaped dict (empty = ok)."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["record must be a dict"]
    rid = data.get("id")
    if rid is not None and not (isinstance(rid, str) and re.match(r"^mem_[0-9a-f]{16}$", rid)):
        errors.append(f"bad id: {rid!r} (expected mem_<16hex>)")
    rtype = data.get("type", "fact")
    if rtype not in MEMORY_TYPES:
        errors.append(f"unknown memory type: {rtype!r}")
    content = data.get("content", "")
    if not isinstance(content, str):
        errors.append("content must be a string")
    elif len(content) > _MAX_CONTENT_CHARS:
        errors.append(f"content too large ({len(content)} > {_MAX_CONTENT_CHARS} chars)")
    if data.get("content_type") not in (None, "text", "markdown", "json", "code"):
        errors.append(f"unknown content_type: {data.get('content_type')!r}")
    stype = data.get("source_type", "manual")
    if stype not in SOURCE_TYPES:
        errors.append(f"unknown source_type: {stype!r}")
    for field in ("created_at", "ingested_at", "valid_from", "valid_to", "event_time"):
        v = data.get(field)
        if v is not None and (not isinstance(v, str) or parse_timestamp(v) is None):
            errors.append(f"{field} is not a valid ISO-8601 timestamp: {v!r}")
    for field in ("provenance", "metadata"):
        v = data.get(field)
        if v is not None and not isinstance(v, dict):
            errors.append(f"{field} must be a dict")
        elif isinstance(v, dict) and len(json.dumps(v, default=str)) > _MAX_PROV_DEPTH:
            errors.append(f"{field} too large (>{_MAX_PROV_DEPTH} chars of JSON)")
    tags = data.get("tags")
    if tags is not None and not isinstance(tags, list):
        errors.append("tags must be a list")
    anchors = data.get("graph_anchors")
    if anchors is not None and not isinstance(anchors, list):
        errors.append("graph_anchors must be a list")
    return errors


def validate_edge(data: dict[str, Any]) -> list[str]:
    """Return a list of schema problems with an edge-shaped dict (empty = ok)."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["edge must be a dict"]
    for field in ("source", "target"):
        v = data.get(field)
        if not isinstance(v, str) or not v:
            errors.append(f"edge.{field} must be a non-empty string")
    relation = data.get("relation", "related_to")
    if relation not in RELATIONS:
        errors.append(f"unknown relation: {relation!r}")
    if data.get("source") and data.get("source") == data.get("target"):
        errors.append("self-referencing edge")
    confidence = data.get("confidence", 1.0)
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"confidence out of range: {confidence!r}")
    return errors
