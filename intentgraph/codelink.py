"""Memory ↔ code-graph links: explicit, evidence-backed, and candidate.

Three verification tiers, deliberately distinct:

* **explicit** — a human or tool asserted the link (``codelink add``).
* **extracted** — a deterministic matcher found a real repository file or
  symbol reference in the record's own text: backtick-quoted identifiers,
  dotted chains that resolve to a symbol, node ids, and file paths. Every
  extracted link records the matched string and where it came from.
* **candidate** — semantic similarity suggested a relationship. Candidates
  are labelled and are NEVER followed as retrieval evidence; embedding
  similarity is a hint, not proof that a memory explains code.

Links live in the archive's own ``code_links`` table — separate from record
content by design: refreshing the code graph can never rewrite historical
evidence, and refreshing the LINKS recomputes only the extracted tier
(explicit and candidate tiers are untouched).

Extraction precision rule: bare single words never link (a memory saying
"run the tests" must not link to every symbol named ``run``); only
backtick-quoted spans, dotted chains, node ids, and file paths do.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = ["add_explicit_link", "extract_links_for_record", "refresh_links",
           "suggest_links", "LINK_TYPES"]

LINK_TYPES = ("explicit", "extracted", "candidate")

_BACKTICK = re.compile(r"`([^`\n]{2,200})`")
_DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_FILEISH = re.compile(r"\b[A-Za-z0-9_][\w./-]*\.[A-Za-z]{1,4}\b")
#: fileish tokens must have a real path shape — not English sentences that
#: happen to end in a dot
_PATHISH = re.compile(r"^[\w][\w./-]*$")
_MIN_EXTRACT_TOKEN = 4


def add_explicit_link(ig: Any, record_id: str, node_id: str,
                      basis: str = "") -> dict[str, Any]:
    """User-asserted link. Validates the record exists and the node exists
    in the project's code graph — a link to a ghost node is a typo, not
    evidence."""
    from intentgraph.store import StoreError

    index = getattr(ig, "code_index", None)
    if index is None:
        raise StoreError("no code graph for this project — run "
                         "`intentgraph codegraph build .` first")
    if index.node(node_id) is None:
        raise StoreError(f"no such code-graph node: {node_id}")
    return ig.store.add_code_link(
        record_id, node_id, "explicit",
        basis=basis or "explicitly asserted",
        method="user", match="")


def _resolve_token(index: Any, token: str) -> list[str]:
    """Code-graph node ids a written token resolves to (best matches only)."""
    token = token.strip().strip("`\"'")
    if not token:
        return []
    hits = index.resolve(token)
    if not hits:
        return []
    best = hits[0]["score"]
    if best < 55:  # substring/label noise never becomes a link
        return []
    return [h["id"] for h in hits if h["score"] >= best]


def _matched_nodes(index: Any, rec: Any) -> list[tuple[str, str, str]]:
    """Deterministic matcher: code-graph nodes this record's text really
    references. Pure — no store writes, no knowledge of existing links."""
    text = rec.content
    tokens: list[tuple[str, str]] = []     # (token, how it was found)

    for m in _BACKTICK.finditer(text):
        inner = m.group(1).strip()
        if inner:
            tokens.append((inner, "backtick-quoted"))
    covered: list[tuple[int, int]] = []
    # file-ish paths BEFORE dotted chains: "app/api.py" must not be shadowed
    # by its own "api.py" fragment (longer, slash-bearing matches win)
    for m in _FILEISH.finditer(text):
        covered.append((m.start(), m.end()))
        if not _PATHISH.match(m.group(0)):
            continue
        tokens.append((m.group(0), "file path"))
    for m in _DOTTED.finditer(text):
        s, e = m.start(), m.end()
        if any(s < ce and e > cs for cs, ce in covered):
            continue
        tokens.append((m.group(0), "dotted chain"))

    nodes: list[tuple[str, str, str]] = []   # (node id, token, how)
    seen: set[str] = set()
    for token, how in tokens:
        if len(token) < _MIN_EXTRACT_TOKEN:
            continue
        for nid in _resolve_token(index, token):
            if nid in seen or nid == rec.id:
                continue
            seen.add(nid)
            nodes.append((nid, token, how))
    return nodes


def extract_links_for_record(ig: Any, rec: Any) -> list[dict[str, Any]]:
    """Create ``extracted`` links for ONE record's text (the ingest hook).

    Matches: backtick-quoted identifiers, dotted chains, node ids, and
    file-ish paths — each resolved against the code graph. Bare single
    words never match (precision rule above). Idempotent: existing links
    (any tier) are never duplicated, and explicit links are never touched.
    """
    index = getattr(ig, "code_index", None)
    if index is None:
        return []
    existing = {l["node_id"] for l in ig.store.iter_code_links(
        record_id=rec.id)}
    created: list[dict[str, Any]] = []
    for nid, token, how in _matched_nodes(index, rec):
        if nid in existing:
            continue
        created.append(ig.store.add_code_link(
            rec.id, nid, "extracted",
            basis=f"{how} reference {token!r} in record text",
            method="deterministic-text-match", match=token))
        existing.add(nid)
    return created


def refresh_links(ig: Any) -> dict[str, Any]:
    """Recompute the extracted tier for the whole archive: creates missing
    matches and removes stale ones (the text changed since the link was
    extracted). Explicit and candidate links are untouched."""
    index = getattr(ig, "code_index", None)
    if index is None:
        return {"available": False,
                "reason": "no code graph for this project"}
    records = [r for r in ig.store.iter_all_records() if r.is_active]
    extracted_in_store = {(l["record_id"], l["node_id"])
                          for l in ig.store.iter_code_links(
                              link_type="extracted")}

    expected: dict[tuple[str, str], tuple[str, str]] = {}
    for rec in records:
        for nid, token, how in _matched_nodes(index, rec):
            expected[(rec.id, nid)] = (token, how)
    created = 0
    for key in sorted(set(expected) - extracted_in_store):
        token, how = expected[key]
        ig.store.add_code_link(
            key[0], key[1], "extracted",
            basis=f"{how} reference {token!r} in record text",
            method="deterministic-text-match", match=token)
        created += 1
    removed = 0
    for record_id, node_id in sorted(extracted_in_store - set(expected)):
        ig.store.remove_code_links(record_id=record_id, node_id=node_id,
                                   link_type="extracted")
        removed += 1
    return {
        "available": True,
        "records_scanned": len(records),
        "created": created,
        "removed": removed,
        "kept": len(extracted_in_store & set(expected)),
        "explicit_untouched": True,
        "candidates_untouched": True,
    }


def suggest_links(ig: Any, *, threshold: float = 0.62, limit: int = 20,
                  max_nodes: int = 400, apply: bool = False
                  ) -> dict[str, Any]:
    """Candidate links via semantic similarity — suggestions ONLY.

    Embeds bounded symbol summaries (classes and functions first, then
    methods, then modules — deterministic order, capped at ``max_nodes``)
    and compares them with the archive's STORED record vectors. Requires
    the embedding provider AND stored record vectors; otherwise reports
    unavailable honestly. ``apply=True`` persists the top pairs as
    ``candidate`` links; candidates are never verified evidence and never
    affect retrieval.
    """
    import numpy as np

    index = getattr(ig, "code_index", None)
    if index is None:
        return {"available": False,
                "reason": "no code graph for this project",
                "suggestions": []}
    if not ig.embedding_provider.is_configured():
        return {"available": False,
                "reason": "no embedding provider configured (opt-in)",
                "suggestions": []}
    if not ig.semantic_index.ready:
        return {"available": False,
                "reason": "no stored record vectors — run `intentgraph "
                          "embed` first",
                "suggestions": []}

    # deterministic, useful-first node selection: classes + functions, then
    # methods, then modules — never the whole graph indiscriminately
    wanted: list[str] = []
    for type_order in (("class", "function"), ("method",), ("module",)):
        wanted.extend(nid for nid in sorted(index.G.nodes)
                      if index.node(nid) and
                      index.node(nid).get("type") in type_order)
        if len(wanted) >= max_nodes:
            break
    wanted = wanted[:max_nodes]
    if not wanted:
        return {"available": False, "reason": "no symbol nodes in the graph",
                "suggestions": []}

    summaries = []
    for nid in wanted:
        d = index.node(nid) or {}
        summaries.append(
            f"{d.get('label', nid)} ({d.get('type', '?')}) defined in "
            f"{d.get('source_file', '')} {d.get('source_location', '')}")
    vecs = np.asarray(ig.embedding_provider.embed_texts(summaries),
                      dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms == 0, 1.0, norms)

    rec_ids = ig.semantic_index.ids
    rec_vecs = np.asarray(ig.semantic_index.matrix, dtype=np.float32)
    rec_norms = np.linalg.norm(rec_vecs, axis=1, keepdims=True)
    rec_vecs = rec_vecs / np.where(rec_norms == 0, 1.0, rec_norms)

    sims = vecs @ rec_vecs.T  # (symbols × records)
    pairs: list[dict[str, Any]] = []
    for si, sv in enumerate(sims):
        for ri in np.argsort(-sv)[:5]:
            score = float(sv[ri])
            if score < threshold:
                continue
            pairs.append({
                "node_id": wanted[si],
                "node_label": (index.node(wanted[si]) or {}).get("label",
                                                                  wanted[si]),
                "source_file": (index.node(wanted[si]) or {}).get(
                    "source_file", ""),
                "record_id": rec_ids[ri],
                "similarity": round(score, 4),
                "basis": f"semantic similarity {score:.3f} (candidate — "
                         "verify before trusting)",
            })
    pairs.sort(key=lambda p: (-p["similarity"], p["node_id"], p["record_id"]))
    # never suggest a pair that already has a VERIFIED link; candidates are
    # for discovery, not for duplicating evidence
    existing_verified = {(l["record_id"], l["node_id"])
                         for l in ig.store.iter_code_links()
                         if l["link_type"] in ("explicit", "extracted")}
    pairs = [p for p in pairs
             if (p["record_id"], p["node_id"]) not in existing_verified]
    pairs = pairs[:limit]

    applied = 0
    if apply:
        for p in pairs:
            ig.store.add_code_link(
                p["record_id"], p["node_id"], "candidate",
                basis=p["basis"], method="semantic-suggestion",
                match=f"{p['similarity']}")
            applied += 1
    return {"available": True, "threshold": threshold,
            "nodes_embedded": len(summaries), "suggestions": pairs,
            "applied": applied, "apply": apply}
