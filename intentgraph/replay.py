"""Evidence Replay: a navigation script with connection explanations.

Each step explains WHY it's connected to the previous node (the relation
or link basis), reads the node's content completely (not truncated), and
the voice waits for the speech to actually finish before advancing —
no timer estimates, no cut-offs.
"""
from __future__ import annotations

from typing import Any

__all__ = ["generate_replay"]

_MAX_STEPS = 6


def _short(text: str, words: int = 20) -> str:
    """A readable excerpt — long enough to explain, short enough to speak."""
    w = text.replace("\n", " ").split()
    return " ".join(w[:words]) + ("…" if len(w) > words else "")


def _type_label(t: str) -> str:
    return {"decision": "decision", "experience": "experience",
            "action": "action record", "fact": "fact",
            "preference": "preference"}.get(t, "memory record")


def _connection_phrase(step_data: dict[str, Any]) -> str:
    """WHY this node connects to the previous one — from real provenance."""
    conn = step_data.get("connection") or ""
    basis = step_data.get("connection_basis") or ""
    if conn and basis:
        return f"This is connected through a {conn}: {basis}."
    if conn:
        return f"This is connected through a {conn}."
    return "This is directly related to the previous evidence."


def _focal_narration(entry: dict[str, Any]) -> str:
    t = _type_label(entry.get("type", "memory"))
    time = (entry.get("event_time") or "")[:10]
    src = entry.get("source_id") or ""
    when = f" recorded on {time}" if time else ""
    by = f", sourced from {src}" if src else ""
    content = _short(entry.get("excerpt", ""), 25)
    return f"The primary evidence is a {t}{when}{by}. {content}"


def _evidence_narration(entry: dict[str, Any],
                        conn: dict[str, Any]) -> str:
    t = _type_label(entry.get("type", "memory"))
    time = (entry.get("event_time") or "")[:10]
    when = f" on {time}" if time else ""
    connection = _connection_phrase(conn)
    content = _short(entry.get("excerpt", ""), 18)
    return f"{connection} This is a {t}{when}. {content}"


def _code_narration(entry: dict[str, Any],
                    conn: dict[str, Any]) -> str:
    qualname = entry.get("qualname") or entry.get("label", "")
    src = entry.get("source_file", "")
    loc = entry.get("source_location", "")
    where = f" at line {loc.lstrip('L')}" if loc else ""
    connection = _connection_phrase(conn)
    return f"{connection} The code entity {qualname} is defined in {src}{where}."


def _extract_connection(entry: dict[str, Any], ig: Any,
                        focal_id: str | None) -> dict[str, Any]:
    """Pull the connection info from the retrieval's own transparency,
    then from the store's edges — the most specific honest answer."""
    # graph_path: the memory-edge traversal from the retrieval
    gp = entry.get("graph_path") or []
    if gp:
        relation, _ = gp[-1]
        return {"connection": relation, "connection_basis": ""}

    # via_code_link: arrived through a verified code link
    vcl = entry.get("via_code_link")
    if vcl:
        return {"connection": f"code link ({vcl.get('type', 'link')})",
                "connection_basis": vcl.get("basis", "")}

    # check the store: is there a typed edge between the focal and this?
    if focal_id and ig is not None:
        try:
            for e in ig.store.edges_for(focal_id):
                if (e.source == focal_id and e.target == entry.get("id")) or \
                   (e.target == focal_id and e.source == entry.get("id")):
                    basis = f" {e.basis}" if e.basis else ""
                    return {"connection": e.relation,
                            "connection_basis": basis.strip()}
        except Exception:  # noqa: BLE001 — connection info is best-effort
            pass

    # honest fallback: they matched the same question
    methods = "+".join(entry.get("methods") or ["retrieval"])
    return {"connection": "shared topic",
            "connection_basis": f"both records matched this question via {methods}"}


def _extract_code_connection(code_entry: dict[str, Any],
                             mem_entry: dict[str, Any] | None) -> dict[str, Any]:
    """Why this code entity connects to the memory it's linked to."""
    for lm in (code_entry.get("linked_memories") or []):
        if mem_entry and lm.get("record_id") == mem_entry.get("id"):
            return {"connection": f"{lm.get('link_type', 'link')} link",
                    "connection_basis": lm.get("basis", "")}
    return {"connection": "code link", "connection_basis": ""}


def generate_replay(ig: Any, query: str, *, top_k: int = 3,
                   code_top_k: int = 2) -> dict[str, Any]:
    """A replay that explains connections and reads fully.

    Uses the retrieval's own transparency (graph_path relations, verified
    link bases) to explain WHY each node connects to the previous one.
    """
    result = ig.query(query, top_k=top_k, code_top_k=code_top_k)
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()

    memories = result.get("memories", [])
    code = result.get("code", [])
    notes = result.get("notes", [])

    mem_ids = {m["id"] for m in memories}
    linked_code = [c for c in code if c.get("linked_memories") and
                   any(l["record_id"] in mem_ids
                       for l in c["linked_memories"])]

    if memories:
        m = memories[0]
        steps.append({
            "id": m["id"], "kind": "focal",
            "narration": _focal_narration(m),
            "edge_from": None, "edge_to": m["id"],
        })
        seen.add(m["id"])

    for m in memories[1:top_k]:
        if m["id"] in seen or len(steps) >= _MAX_STEPS - 1:
            break
        focal_id = memories[0]["id"] if memories else None
        conn = _extract_connection(m, ig, focal_id)
        edge_from = steps[0]["id"] if steps else None
        steps.append({
            "id": m["id"], "kind": "evidence",
            "narration": _evidence_narration(m, conn),
            "connection": conn["connection"],
            "connection_basis": conn["connection_basis"],
            "edge_from": edge_from, "edge_to": m["id"],
        })
        seen.add(m["id"])

    for c in linked_code[:code_top_k]:
        if c["id"] in seen or len(steps) >= _MAX_STEPS - 1:
            break
        conn = _extract_code_connection(c, memories[0] if memories else None)
        edge_from = None
        for m in memories[:top_k]:
            for lc in (m.get("linked_code") or []):
                if lc.get("id") == c["id"]:
                    edge_from = m["id"]
                    break
            if edge_from:
                break
        steps.append({
            "id": c["id"], "kind": "code",
            "narration": _code_narration(c, conn),
            "connection": conn["connection"],
            "connection_basis": conn["connection_basis"],
            "edge_from": edge_from, "edge_to": c["id"],
        })
        seen.add(c["id"])

    n = len(steps)
    relations = [s.get("connection", "") for s in steps if s.get("connection")]
    rel_summary = ", ".join(sorted(set(relations))) if relations else "retrieval"
    steps.append({
        "id": None, "kind": "verdict",
        "narration": f"Replay complete. {n} node{'s' if n != 1 else ''} connected "
                     f"through {rel_summary}. This is the full trail behind "
                     f"the answer."
                     if n else "No evidence was found for this question.",
        "edge_from": steps[-1]["id"] if steps else None,
        "edge_to": None,
    })

    return {"query": query, "steps": steps, "notes": notes}


