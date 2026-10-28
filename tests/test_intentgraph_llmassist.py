"""Tests for the optional LLM-assisted planning policy (all mocked).

Anti-fabrication is structural: the output schemas contain only
requirement shapes and queries — there is no field through which a model
could claim evidence exists, and any off-schema output falls back to the
deterministic policy.
"""
from __future__ import annotations

import json
import urllib.error

import pytest

from intentgraph.controller import RuleBasedPolicy
from intentgraph.llmassist import (
    LLMAssistError,
    LLMClient,
    LLMPolicy,
    _validate_plan,
    _validate_requirements,
    llm_policy_from_config,
)


class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _chat_response(content: str, usage=None):
    payload = {"choices": [{"message": {"content": content}}],
               "usage": usage or {"prompt_tokens": 9}}
    return _FakeResponse(payload)


def _mock_chat(monkeypatch, responses, capture=None):
    calls = capture if capture is not None else []
    state = {"i": 0}

    def fake(req, timeout=None):
        calls.append({
            "url": req.full_url,
            "auth": req.get_header("Authorization"),
            "body": json.loads(req.data.decode()),
        })
        item = responses[state["i"]]
        state["i"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("intentgraph.llmassist.urllib.request.urlopen", fake)
    return calls


def _http_error(code):
    import io
    try:
        raise urllib.error.HTTPError("https://x/v1/chat/completions", code,
                                     "e", {}, io.BytesIO(b"{}"))
    except urllib.error.HTTPError as e:
        return e


def _policy(monkeypatch, responses, external_enabled=False, capture=None):
    calls = _mock_chat(monkeypatch, responses, capture)
    client = LLMClient(base_url="https://api.tokenfactory.nebius.com/v1/",
                       model="test/llm", api_key="k")
    return LLMPolicy(client, external_enabled=external_enabled), calls


REQUIREMENTS_JSON = json.dumps({
    "requirements": [{
        "name": "failure_evidence",
        "description": "the original failure and its error output",
        "content_signals": ["failed", "error", "regression"],
        "record_types": ["experience"],
    }],
})


class TestRequirementsPlanning:
    def test_valid_plan_accepted(self, monkeypatch):
        p, calls = _policy(monkeypatch, [_chat_response(REQUIREMENTS_JSON)])
        reqs = p.requirements("why did the build fail?")
        assert len(reqs) == 1
        assert reqs[0].name == "failure_evidence"
        assert reqs[0].content_signals == ("failed", "error", "regression")
        assert p.notes and "llm-assisted" in p.notes[0]

    def test_names_sanitized_to_identifiers(self, monkeypatch):
        bad = json.dumps({"requirements": [{
            "name": "Why Did It FAIL?!",
            "description": "d",
        }]})
        p, _ = _policy(monkeypatch, [_chat_response(bad)])
        reqs = p.requirements("q")
        assert reqs[0].name == "why_did_it_fail"

    def test_off_schema_output_falls_back(self, monkeypatch):
        # extra/unknown fields are REJECTED, not ignored
        off_schema = json.dumps({"requirements": [{
            "name": "r", "description": "d", "invented_evidence": "mem_abc"}]})
        p, _ = _policy(monkeypatch, [_chat_response(off_schema)])
        reqs = p.requirements("why did the build fail?")
        assert reqs == RuleBasedPolicy().requirements("why did the build fail?")
        assert any("failed schema validation" in n for n in p.notes)
        assert p.fallbacks == 1

    def test_transport_failure_falls_back_with_note(self, monkeypatch):
        p, _ = _policy(monkeypatch, [_http_error(429), _http_error(429),
                                     _http_error(429)])
        reqs = p.requirements("why did the build fail?")
        assert reqs  # deterministic answer
        assert any("llm assist unavailable" in n for n in p.notes)

    def test_non_json_output_falls_back(self, monkeypatch):
        p, _ = _policy(monkeypatch, [_chat_response("I think you need logs!")])
        reqs = p.requirements("why did the build fail?")
        assert reqs == RuleBasedPolicy().requirements("why did the build fail?")

    def test_markdown_fenced_json_accepted(self, monkeypatch):
        p, _ = _policy(monkeypatch, [
            _chat_response(f"```json\n{REQUIREMENTS_JSON}\n```")])
        assert p.requirements("q")[0].name == "failure_evidence"

    def test_request_shape(self, monkeypatch):
        p, calls = _policy(monkeypatch, [_chat_response(REQUIREMENTS_JSON)])
        p.requirements("the build failed")
        req = calls[0]
        assert req["url"].endswith("/chat/completions")
        assert req["auth"] == "Bearer k"
        assert req["body"]["model"] == "test/llm"
        assert req["body"]["temperature"] == 0
        sys_msg = req["body"]["messages"][0]["content"]
        assert "memory-retrieval" in sys_msg
        assert "mem_" not in sys_msg  # no fabricated id template


class TestActionPlanning:
    def test_external_action_allowed_when_enabled(self, monkeypatch):
        plan = {"action": "external_search", "external_query": "latest docs",
                "rationale": "question needs current information"}
        p, _ = _policy(monkeypatch, [_chat_response(json.dumps(plan))],
                       external_enabled=True)
        out = p.plan_next_action("q", {}, external_ok=True, iteration=1)
        assert out["action"] == "external_search"
        assert out["external_query"] == "latest docs"

    def test_external_action_rejected_when_not_allowed(self):
        # validation alone must veto: no network needed for this assertion
        plan = {"action": "external_search", "external_query": "q"}
        assert _validate_plan(plan, external_ok=False) is None
        assert _validate_plan(plan, external_ok=True)["action"] == "external_search"

    def test_local_followup_needs_query(self):
        assert _validate_plan({"action": "local_followup"}, external_ok=True) is None
        ok = _validate_plan({"action": "local_followup",
                             "follow_up_query": "sharpened"}, external_ok=True)
        assert ok["follow_up_query"] == "sharpened"

    def test_unknown_action_rejected(self):
        assert _validate_plan({"action": "fabricate_evidence"}, external_ok=True) is None
        assert _validate_plan({"action": "stop", "surprise": 1},
                              external_ok=True) is None

    def test_llm_plan_failure_returns_none_for_controller_fallback(self, monkeypatch):
        p, _ = _policy(monkeypatch, [_http_error(500), _http_error(500),
                                     _http_error(500)])
        out = p.plan_next_action("q", {}, external_ok=True, iteration=2)
        assert out is None  # controller falls back to its deterministic logic


class TestFollowUpPlanning:
    def test_valid_follow_up_used(self, monkeypatch):
        plan = json.dumps({"action": "local_followup",
                           "follow_up_query": "ci logs failure INVALID_AMOUNT"})
        p, _ = _policy(monkeypatch, [_chat_response(plan)])
        missing = RuleBasedPolicy().requirements("why did it fail")[0]
        q = p.follow_up_query("why did the build fail", missing)
        assert q == "ci logs failure INVALID_AMOUNT"

    def test_invalid_follow_up_falls_back_to_rule_based(self, monkeypatch):
        p, _ = _policy(monkeypatch, [_chat_response("no query here")])
        missing = RuleBasedPolicy().requirements("why did it fail")[0]
        expected = RuleBasedPolicy().follow_up_query("why did the build fail", missing)
        assert p.follow_up_query("why did the build fail", missing) == expected


class TestConfig:
    def test_policy_from_config_none(self):
        assert llm_policy_from_config(None) is None
        assert llm_policy_from_config({"provider": "none"}) is None

    def test_nebius_policy_built(self, monkeypatch):
        monkeypatch.setenv("NEBIUS_API_KEY", "k")
        monkeypatch.setenv("INTENTGRAPH_LLM_MODEL", "some/chat-model")
        pol = llm_policy_from_config({"provider": "nebius"})
        assert isinstance(pol, LLMPolicy)
        assert pol.client.model == "some/chat-model"
        assert "tokenfactory.nebius" in pol.client.base_url

    def test_local_only_refuses_llm(self):
        with pytest.raises(PermissionError, match="local_only"):
            llm_policy_from_config({"provider": "nebius", "model": "m"},
                                   local_only=True)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown llm provider"):
            llm_policy_from_config({"provider": "oracle"})


class TestSchemaHelpers:
    def test_requirement_validation_rejects_types_outside_vocabulary(self):
        bad = json.dumps({"requirements": [{
            "name": "x", "description": "d", "record_types": ["vibes"]}]})
        assert _validate_requirements(json.loads(bad)) is None

    def test_requirement_validation_caps_signals(self):
        reqs = {"requirements": [{"name": "x", "description": "d",
                                  "content_signals": [f"s{i}" for i in range(20)]}]}
        assert _validate_requirements(reqs) is None
