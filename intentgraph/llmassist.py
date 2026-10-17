"""Optional LLM-assisted retrieval planning (Nebius Token Factory or any
OpenAI-compatible chat endpoint), with the deterministic policy as fallback.

What the LLM is allowed to do here — and what it is structurally prevented
from doing:

* **Query decomposition** into evidence requirements (``requirements()``)
  and **retrieval-action selection** (``plan_next_action()``: local
  follow-up vs external search vs stop). Both produce STRICT, schema-validated
  JSON: unknown fields are rejected, names are sanitized to identifiers,
  strings are length-capped, and any validation failure falls back to the
  deterministic :class:`RuleBasedPolicy` answer with a recorded note.
* It **cannot invent memories, source IDs, or factual claims**: the output
  schemas contain only requirement *shapes* and *queries* — no evidence
  fields exist in the schema, so a hallucinated "found record mem_…" cannot
  validate. Evidence admission stays entirely in the deterministic
  verify/compile pipeline.
* **Gap assessment stays deterministic**: whether evidence covers the
  question is decided by verify.py's coverage rules, never by the model.
  The model only chooses the next *action*; the controller still stops on
  its own bounded conditions.

Failure handling is total: missing credentials, unreachable endpoint, rate
limits, malformed/JSON-less output, schema violations — every path yields
the deterministic fallback plus an honest note in the trace, never a crash
and never a fabricated plan.

Credential conventions match the embeddings module: for the ``nebius``
provider, ``NEBIUS_API_KEY`` → ``NEBIUS_TOKEN_FACTORY_KEY`` →
``INTENTGRAPH_LLM_API_KEY``; keys are never logged or included in errors.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from intentgraph.controller import ActivationPolicy, RuleBasedPolicy
from intentgraph.embeddings import NEBIUS_DEFAULT_BASE_URL, nebius_api_key_from_env
from intentgraph.models import MEMORY_TYPES
from intentgraph.verify import Requirement

__all__ = [
    "LLMAssistError",
    "LLMClient",
    "LLMPolicy",
    "llm_policy_from_config",
]

_MAX_NAME_LEN = 64
_MAX_DESC_LEN = 400
_MAX_SIGNALS = 12
_MAX_QUERY_LEN = 600


class LLMAssistError(Exception):
    """LLM assist failure — always recoverable (caller falls back)."""


class LLMClient:
    """Minimal OpenAI-compatible chat/completions client (stdlib urllib).

    One client per policy; no new HTTP abstraction (the embeddings module's
    transport conventions are followed exactly). Bounded retries on 429/5xx
    only; JSON responses only; usage metadata captured without content.
    """

    def __init__(self, *, base_url: str, model: str, api_key: str,
                 timeout_s: float = 30.0, retries: int = 2,
                 max_output_tokens: int = 1024):
        if not model:
            raise ValueError("llm provider requires a model — set "
                             "INTENTGRAPH_LLM_MODEL or the llm.model config")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.retries = max(0, int(retries))
        self.max_output_tokens = max(64, int(max_output_tokens))
        self.last_usage: dict[str, Any] = {}
        self.calls_made = 0

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """One chat call expecting a single JSON object in the response.

        Returns the parsed object. Raises :class:`LLMAssistError` on any
        failure mode — transport, auth, non-JSON, or non-object payload.
        """
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        self.calls_made += 1
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                if isinstance(body.get("usage"), dict):
                    self.last_usage = dict(body["usage"])
                choices = body.get("choices") or []
                if not choices:
                    raise LLMAssistError("llm returned no choices")
                text = choices[0].get("message", {}).get("content") or ""
                parsed = _parse_json_object(text)
                if parsed is None:
                    raise LLMAssistError("llm response was not a JSON object")
                return parsed
            except LLMAssistError:
                raise
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise LLMAssistError(
                        f"llm endpoint rejected credentials (HTTP {exc.code})") from exc
                if exc.code == 404:
                    raise LLMAssistError(
                        f"llm model or endpoint not found (HTTP 404) — "
                        "check the model id and account availability") from exc
                if exc.code == 422:
                    raise LLMAssistError(
                        "llm endpoint rejected the request (HTTP 422) — "
                        "the model may not support this request shape") from exc
                last = exc
                if exc.code in (429, 500, 502, 503) and attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise LLMAssistError(
                    f"llm request failed (HTTP {exc.code})") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise LLMAssistError(
                    f"llm endpoint unreachable: {type(exc).__name__}") from exc
            except json.JSONDecodeError as exc:
                raise LLMAssistError(f"llm returned invalid JSON: {exc}") from exc
        raise LLMAssistError(f"llm request failed after retries: {last}")


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse one JSON object from model output, tolerating markdown fences.

    Models wrap JSON in ```json fences often enough that rejecting it would
    make the feature flaky; stripping KNOWN wrappers is safe because the
    schema validation afterwards is strict about content.
    """
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    try:
        parsed = json.loads(t)
    except json.JSONDecodeError:
        # last resort: first {...} balanced block
        start = t.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None
    return parsed if isinstance(parsed, dict) else None


# --- schemas -------------------------------------------------------------------------
#
# Strict allow-lists: unknown top-level keys and unknown requirement keys
# are REJECTED (not ignored) — a model drifting off-schema yields the
# deterministic fallback instead of a half-trusted plan.

_REQUIREMENT_KEYS = {"name", "description", "content_signals", "record_types"}
_PLAN_KEYS = {"action", "follow_up_query", "external_query", "rationale"}
_PLAN_ACTIONS = {"local_followup", "external_search", "stop"}

_REQUIREMENTS_SYSTEM = (
    "You analyze a question for a memory-retrieval system. Decompose it "
    "into the kinds of EVIDENCE a complete answer needs. Respond with ONE "
    "JSON object, no prose: "
    '{"requirements": [{"name": "snake_case_name", "description": "what '
    'kind of evidence", "content_signals": ["short keyword phrases that '
    'satisfy it"], "record_types": ["memory types from the allowed list"]}]} '
    "Rules: names are snake_case; 1-6 requirements; do not include fields "
    "beyond the schema; do not claim any evidence exists — you are only "
    "describing what kinds would be needed. Allowed memory types: "
    + ", ".join(MEMORY_TYPES) + "."
)

_PLAN_SYSTEM = (
    "You are the action selector for an iterative memory-retrieval loop. "
    "Given the question, the requirement coverage report (which evidence "
    "kinds were found and which are missing), and config limits, choose "
    "ONE next action. Respond with ONE JSON object, no prose: "
    '{"action": "local_followup" | "external_search" | "stop", '
    '"follow_up_query": "...", "external_query": "...", "rationale": "..."} '
    "Rules: local_followup needs follow_up_query (a sharper query for local "
    "memory); external_search needs external_query AND is ONLY allowed when "
    "external_ok is true; stop when coverage looks sufficient or further "
    "retrieval will not help. Never invent record ids or evidence. Keep "
    "queries under 600 characters."
)


def _validate_requirements(parsed: dict[str, Any]) -> list[Requirement] | None:
    """Schema-validate a requirements plan; None on any violation."""
    if set(parsed) != {"requirements"}:
        return None
    reqs_raw = parsed["requirements"]
    if not isinstance(reqs_raw, list) or not 1 <= len(reqs_raw) <= 6:
        return None
    out: list[Requirement] = []
    seen_names: set[str] = set()
    for item in reqs_raw:
        if not isinstance(item, dict) or set(item) - _REQUIREMENT_KEYS:
            return None
        name = item.get("name")
        if not isinstance(name, str):
            return None
        clean = re.sub(r"[^a-z0-9_]+", "_", name.lower().strip()).strip("_")
        if not clean or len(clean) > _MAX_NAME_LEN or clean in seen_names:
            return None
        seen_names.add(clean)
        desc = item.get("description")
        if not isinstance(desc, str) or not desc.strip() \
                or len(desc) > _MAX_DESC_LEN:
            return None
        signals_raw = item.get("content_signals", [])
        if not isinstance(signals_raw, list) \
                or len(signals_raw) > _MAX_SIGNALS \
                or any(not isinstance(s, str) or not s.strip()
                       or len(s) > 80 for s in signals_raw):
            return None
        types_raw = item.get("record_types", [])
        if not isinstance(types_raw, list) \
                or any(t not in MEMORY_TYPES for t in types_raw):
            return None
        out.append(Requirement(
            name=clean,
            description=desc.strip()[:_MAX_DESC_LEN],
            content_signals=tuple(s.strip().lower() for s in signals_raw),
            record_types=tuple(types_raw),
        ))
    return out


def _validate_plan(parsed: dict[str, Any], *, external_ok: bool) -> dict[str, Any] | None:
    """Schema-validate an action plan; None on any violation."""
    if set(parsed) - _PLAN_KEYS:
        return None
    action = parsed.get("action")
    if action not in _PLAN_ACTIONS:
        return None
    if action == "external_search" and not external_ok:
        return None  # model proposed a disallowed action → deterministic path
    plan: dict[str, Any] = {"action": action, "rationale": ""}
    rationale = parsed.get("rationale")
    if isinstance(rationale, str):
        plan["rationale"] = rationale.strip()[:_MAX_DESC_LEN]
    for key in ("follow_up_query", "external_query"):
        q = parsed.get(key)
        needed = (action == "local_followup" and key == "follow_up_query") or \
                 (action == "external_search" and key == "external_query")
        if needed:
            if not isinstance(q, str) or not q.strip() or len(q) > _MAX_QUERY_LEN:
                return None
            plan[key] = q.strip()
        elif q is not None and not isinstance(q, str):
            return None
    return plan


# --- the policy ------------------------------------------------------------------------


class LLMPolicy(ActivationPolicy):
    """LLM-backed planning with deterministic fallback on every failure.

    Keeps a :class:`RuleBasedPolicy` internally: every LLM failure mode
    (transport, auth, schema, rate limit) returns the rule-based answer and
    records an honest note. Notes are exposed on ``self.notes`` and merged
    into the retrieval trace by the controller.
    """

    def __init__(self, client: LLMClient, *, external_enabled: bool = False):
        self.client = client
        self.fallback = RuleBasedPolicy()
        self.external_enabled = external_enabled
        self.notes: list[str] = []
        self.llm_calls = 0
        self.fallbacks = 0

    def _note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    # -- ActivationPolicy interface ---------------------------------------------

    def requirements(self, query: str) -> list[Requirement]:
        try:
            self.llm_calls += 1
            parsed = self.client.complete_json(
                _REQUIREMENTS_SYSTEM,
                f"Question: {query}",
            )
            reqs = _validate_requirements(parsed)
            if reqs is None:
                self.fallbacks += 1
                self._note("llm requirement plan failed schema validation; "
                           "fell back to deterministic rules")
                return self.fallback.requirements(query)
            self._note("llm-assisted requirement decomposition "
                       f"({len(reqs)} requirements)")
            return reqs
        except LLMAssistError as exc:
            self.fallbacks += 1
            self._note(f"llm assist unavailable ({exc}); deterministic rules used")
            return self.fallback.requirements(query)

    def follow_up_query(self, query: str, missing: Requirement) -> str:
        try:
            self.llm_calls += 1
            parsed = self.client.complete_json(
                _PLAN_SYSTEM,
                f"Question: {query}\nMissing evidence kind: "
                f"{missing.name} — {missing.description}\n"
                "action: propose the local follow-up query.",
            )
            plan = _validate_plan(parsed, external_ok=False)
            if plan and plan["action"] == "local_followup":
                self._note("llm-planned local follow-up")
                return plan["follow_up_query"]
            self.fallbacks += 1
            self._note("llm follow-up plan unusable; deterministic sharpening")
            return self.fallback.follow_up_query(query, missing)
        except LLMAssistError as exc:
            self.fallbacks += 1
            self._note(f"llm assist unavailable ({exc}); deterministic sharpening")
            return self.fallback.follow_up_query(query, missing)

    # -- optional action-selection hook (controller uses it when present) --------

    def plan_next_action(self, query: str, report_summary: dict[str, Any],
                         *, external_ok: bool, iteration: int) -> dict[str, Any] | None:
        """LLM evidence-gap assessment + retrieval action selection.

        Returns a validated plan dict (action/queries/rationale) or None
        (→ controller falls back to its deterministic action logic). The
        coverage REPORT the model sees is deterministic output from
        verify.py — the model never judges evidence truth, only which
        retrieval action helps next.
        """
        try:
            self.llm_calls += 1
            parsed = self.client.complete_json(
                _PLAN_SYSTEM,
                f"Question: {query}\nIteration: {iteration}\n"
                f"External search available: {external_ok}\n"
                f"Coverage report: {json.dumps(report_summary, default=str)[:1500]}",
            )
            plan = _validate_plan(parsed, external_ok=external_ok)
            if plan is None:
                self.fallbacks += 1
                self._note("llm action plan failed validation; deterministic "
                           "action selection used")
            else:
                self._note(f"llm-planned action: {plan['action']}")
            return plan
        except LLMAssistError as exc:
            self.fallbacks += 1
            self._note(f"llm assist unavailable during action selection ({exc})")
            return None


def llm_policy_from_config(cfg: dict[str, Any] | None,
                           *, local_only: bool = False,
                           external_enabled: bool = False) -> LLMPolicy | None:
    """Build the configured LLM policy; None when disabled (the default).

    ``nebius`` uses the Token Factory chat endpoint with the documented
    credential precedence; any other OpenAI-compatible endpoint works via
    ``provider: "openai"`` + ``base_url``. ``local_only`` refuses both —
    planning assistance is a network provider like any other.
    """
    cfg = dict(cfg or {})
    kind = str(cfg.get("provider", "none")).strip().lower()
    if kind in ("", "none", "null"):
        return None
    if kind == "nebius":
        base = cfg.get("base_url") or os.environ.get(
            "INTENTGRAPH_LLM_BASE_URL", "") or NEBIUS_DEFAULT_BASE_URL
        model = cfg.get("model") or os.environ.get("INTENTGRAPH_LLM_MODEL", "")
        key = (cfg.get("api_key") or nebius_api_key_from_env()
               or os.environ.get("INTENTGRAPH_LLM_API_KEY", ""))
    elif kind in ("openai", "openai_compatible", "openai-compatible"):
        base = cfg.get("base_url") or os.environ.get(
            "INTENTGRAPH_LLM_BASE_URL", "") or "https://api.openai.com/v1"
        model = cfg.get("model") or os.environ.get("INTENTGRAPH_LLM_MODEL", "")
        key = (cfg.get("api_key") or os.environ.get("INTENTGRAPH_LLM_API_KEY")
               or os.environ.get("OPENAI_API_KEY", ""))
    else:
        raise ValueError(f"unknown llm provider: {kind!r} (none|nebius|openai)")
    if local_only:
        raise PermissionError(
            f"privacy.local_only is set but an llm provider is configured "
            f"({kind}) — planning assistance would send text off-machine")
    client = LLMClient(
        base_url=base, model=model, api_key=key,
        timeout_s=float(cfg.get("timeout_s", 30)),
        retries=int(cfg.get("retries", 2)),
        max_output_tokens=int(cfg.get("max_output_tokens", 1024)),
    )
    return LLMPolicy(client, external_enabled=external_enabled)
