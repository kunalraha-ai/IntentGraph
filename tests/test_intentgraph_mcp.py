"""Tests for intentgraph.mcp (native agent tools) + the serve.py MCP wiring.

Three layers:

1. Pure tool functions — no mcp import needed, runs everywhere.
2. The end-to-end session regression scenario (Step-4 spec): a session that
   remembers a failure trail, then a *fresh* session that must recover it by
   record id and content — not by plausible-looking text.
3. intentgraph.serve wiring — skipped when the optional mcp extra is absent.
"""
from __future__ import annotations

import json

import pytest

from intentgraph import mcp as igm
from intentgraph.mcp import (
    IntentGraphSessions,
    MCPToolError,
    tool_context,
    tool_feedback,
    tool_recall,
    tool_remember,
)


@pytest.fixture
def project(tmp_path):
    p = tmp_path / "project"
    p.mkdir()
    return p


@pytest.fixture
def sessions(project):
    s = IntentGraphSessions()
    yield s
    s.close()


def _remember(sessions, project, content, **kw):
    out = json.loads(tool_remember(sessions, {"content": content,
                                             "project_path": str(project), **kw}))
    assert out["id"].startswith("mem_")
    return out


# --- session resolution -----------------------------------------------------------


class TestSessions:
    def test_project_must_exist(self, sessions, tmp_path):
        with pytest.raises(MCPToolError, match="not an existing directory"):
            sessions.get(str(tmp_path / "nope"))

    def test_default_is_cwd(self, sessions, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ig = sessions.get(None)
        assert ig.config.db_path.parent == tmp_path / "intentgraph-data"

    def test_archive_is_project_local(self, sessions, project):
        ig = sessions.get(str(project))
        assert ig.config.db_path.parent == project / "intentgraph-data"

    def test_home_override_wins(self, sessions, project, tmp_path, monkeypatch):
        home = tmp_path / "shared-memory"
        home.mkdir()
        monkeypatch.setenv("INTENTGRAPH_HOME", str(home))
        ig = sessions.get(str(project))
        assert ig.config.db_path.parent == home

    def test_lru_eviction_bounds_warm_instances(self, tmp_path):
        s = IntentGraphSessions(max_sessions=2)
        for i in range(4):
            p = tmp_path / f"p{i}"
            p.mkdir()
            s.get(str(p))
            assert len(s) <= 2
        # the two most recent projects are warm
        s.get(str(tmp_path / "p2"))
        assert len(s) == 2
        s.close()

    def test_bridge_auto_enabled_when_graph_exists(self, sessions, project):
        (project / "graphify-out").mkdir()
        (project / "graphify-out" / "graph.json").write_text(json.dumps(
            {"directed": False, "nodes": [{"id": "a", "label": "A"}], "links": []}))
        ig = sessions.get(str(project))
        assert ig.bridge.available


# --- tool: remember -----------------------------------------------------------------


class TestRemember:
    def test_happy_path_returns_actionable_message(self, sessions, project):
        out = _remember(sessions, project,
                        "Decision: use integer minor units; the gateway API is "
                        "integer-only.", type="decision", source_id="sess-1")
        assert "Cite this id" in out["message"]
        assert out["type"] == "decision"
        assert out["duplicate_of"] is None

    def test_links_create_explicit_edges(self, sessions, project):
        attempt = _remember(sessions, project,
                            "We tried rounding to cents. It did not work: the "
                            "gateway still rejected the amounts.")
        decision = _remember(
            sessions, project, "ADR-14: integer minor units everywhere.",
            type="decision",
            links=[{"id": attempt["id"], "relation": "derived_from",
                   "basis": "the decision came from the failed attempt"}])
        assert decision["links_created"][0]["relation"] == "derived_from"

    def test_supersedes_link_flips_validity_not_just_edges(self, sessions, project):
        old = _remember(sessions, project,
                        "ADR-11: store amounts as float dollars.",
                        type="decision", event_time="2026-06-28")
        new = _remember(sessions, project,
                        "ADR-14: integer minor units.", type="decision",
                        links=[{"id": old["id"], "relation": "supersedes",
                                "basis": "replaced after the regression"}])
        ig = sessions.get(str(project))
        rec = ig.store.get_record(old["id"])
        edge_note = [l for l in new["links_created"]
                     if l["relation"] == "supersedes"][0]["note"]
        assert rec.status == "superseded"
        assert rec.superseded_by == new["id"]
        assert "superseded by this record" in edge_note

    def test_duplicate_content_is_reported_not_resaved(self, sessions, project):
        first = _remember(sessions, project, "identical evidence", type="fact")
        second = json.loads(tool_remember(
            sessions, {"content": "identical evidence", "type": "fact",
                       "project_path": str(project)}))
        assert second["id"] == first["id"]
        assert "Duplicate" in second["message"]

    def test_validation_errors_are_actionable(self, sessions, project):
        with pytest.raises(MCPToolError, match="content"):
            tool_remember(sessions, {"project_path": str(project)})
        with pytest.raises(MCPToolError, match="unknown type"):
            tool_remember(sessions, {"content": "x", "type": "vibes",
                                     "project_path": str(project)})
        with pytest.raises(MCPToolError, match="unknown argument"):
            tool_remember(sessions, {"content": "x", "flavor": "sour",
                                     "project_path": str(project)})
        with pytest.raises(MCPToolError, match="too large"):
            tool_remember(sessions, {"content": "x" * 2_000_001,
                                     "project_path": str(project)})
        with pytest.raises(MCPToolError, match="unknown relation"):
            tool_remember(sessions, {"content": "x",
                                     "links": [{"id": "mem_" + "0" * 16,
                                                "relation": "zaps"}],
                                     "project_path": str(project)})
        with pytest.raises(MCPToolError, match="link id"):
            tool_remember(sessions, {"content": "x",
                                     "links": [{"id": "not-an-id",
                                                "relation": "related_to"}],
                                     "project_path": str(project)})

    def test_link_to_unknown_memory_is_an_error(self, sessions, project):
        with pytest.raises(MCPToolError, match="link failed"):
            tool_remember(sessions, {
                "content": "claims a relationship to a ghost",
                "links": [{"id": "mem_" + "0" * 16,
                           "relation": "derived_from"}],
                "project_path": str(project)})


# --- tool: recall --------------------------------------------------------------------


class TestRecall:
    def test_finds_memories_with_provenance_and_excerpt(self, sessions, project):
        _remember(sessions, project,
                  "The checkout flow failed with INVALID_AMOUNT for orders "
                  "under one dollar after the currency refactor.",
                  type="experience", source_id="pagerduty-4471")
        _remember(sessions, project,
                  "Preference: dark mode in the dashboard.", type="preference")
        out = json.loads(tool_recall(sessions, {
            "query": "why did checkout fail for small orders",
            "project_path": str(project)}))
        assert out["results"], out
        top = out["results"][0]
        assert top["source_id"] == "pagerduty-4471"
        assert "INVALID_AMOUNT" in top["excerpt"]
        assert "lexical" in top["methods"]

    def test_mode_and_top_k_validation(self, sessions, project):
        _remember(sessions, project, "something to find", type="fact")
        with pytest.raises(MCPToolError, match="unknown mode"):
            tool_recall(sessions, {"query": "x", "mode": "psychic",
                                   "project_path": str(project)})
        with pytest.raises(MCPToolError, match="out of range"):
            tool_recall(sessions, {"query": "x", "top_k": 500,
                                   "project_path": str(project)})
        out = json.loads(tool_recall(sessions, {"query": "something",
                                                "top_k": 1,
                                                "project_path": str(project)}))
        assert len(out["results"]) <= 1

    def test_truncation_is_flagged_never_silent(self, sessions, project, monkeypatch):
        for i in range(6):
            _remember(sessions, project,
                      f"long memory {i} " + "gateway checkout amounts " * 20,
                      type="experience")
        full = json.loads(tool_recall(sessions, {"query": "gateway amounts",
                                                "project_path": str(project)}))
        # shrink the budget until the output must cut candidates; the exact
        # surviving count varies with content, so assert the RELATION: fewer
        # results than the full run, plus an explicit truncated count
        monkeypatch.setattr(igm, "_RECALL_TOKEN_BUDGET", 100)
        out = json.loads(tool_recall(sessions, {"query": "gateway amounts",
                                               "project_path": str(project)}))
        assert out["truncated"] >= 1
        assert len(out["results"]) < len(full["results"])


# --- tool: context / feedback ---------------------------------------------------------


class TestContextAndFeedback:
    def test_context_returns_packet_text_with_fences(self, sessions, project):
        _remember(sessions, project,
                  "We tried rounding to cents before validation. It did not "
                  "work: the gateway expects integer minor units.",
                  type="experience", source_id="war-room-0712")
        text = tool_context(sessions, {
            "task": "fix the gateway rounding failure; what did we try "
                    "before and why did it fail?",
            "project_path": str(project)})
        assert "# IntentGraph evidence packet" in text
        assert "<untrusted_evidence id=" in text
        assert "war-room-0712" in text
        assert "Verdict:" in text

    def test_budget_clamps(self, sessions, project):
        with pytest.raises(MCPToolError, match="out of range"):
            tool_context(sessions, {"task": "t", "budget_tokens": 10,
                                    "project_path": str(project)})
        with pytest.raises(MCPToolError, match="out of range"):
            tool_context(sessions, {"task": "t", "budget_tokens": 999_999,
                                    "project_path": str(project)})

    def test_feedback_round_trip(self, sessions, project):
        rec = _remember(sessions, project, "some evidence", type="fact")
        out = json.loads(tool_feedback(sessions, {
            "query": "find evidence", "outcome": "useful",
            "memory_ids": [rec["id"]], "note": "found the failed attempt",
            "project_path": str(project)}))
        assert out["outcome"] == "useful"
        with pytest.raises(MCPToolError, match="outcome must be one of"):
            tool_feedback(sessions, {"query": "q", "outcome": "meh",
                                     "project_path": str(project)})


# --- Step 4: the end-to-end session regression scenario ---------------------------------


class TestSessionRegressionScenario:
    """Agent session 1 hits the regression and remembers it; agent session 2
    (a *fresh* session object — no shared state, same archive) must recover
    the exact records and their content, and must NOT treat the superseded
    decision as current."""

    def _session_one_remembers(self, project) -> dict[str, str]:
        sessions = IntentGraphSessions()
        try:
            ids = {}
            out = json.loads(tool_remember(sessions, {
                "content": "Attempted fix: we tried rounding the amount to "
                            "cents before validation. It did not work: the "
                            "payment gateway still rejected the amounts "
                            "because it expects integer minor units, and "
                            "rounding never changed the serialized type.",
                "type": "experience", "tags": ["attempt", "checkout"],
                "source_id": "opencode session 1, war room",
                "event_time": "2026-07-12T16:40:00+00:00",
                "project_path": str(project)}))
            ids["attempt"] = out["id"]

            out = json.loads(tool_remember(sessions, {
                "content": "Test output: test_checkout_under_one_dollar FAILED — "
                           "PaymentGatewayError: INVALID_AMOUNT for amount=0.87 "
                           "(expected integer minor units, got float) at "
                           "gateway/client.py L88, exit code 1.",
                "type": "action", "source_id": "ci/build-9931",
                "event_time": "2026-07-12T15:02:00+00:00",
                "project_path": str(project)}))
            ids["test_output"] = out["id"]

            out = json.loads(tool_remember(sessions, {
                "content": "ADR-14 decision: order amounts are stored as integer "
                           "minor units; the validation layer never converts "
                           "through float. Rationale: the payment gateway API "
                           "is integer-only and every float conversion "
                           "reintroduces the INVALID_AMOUNT failure.",
                "type": "decision", "tags": ["adr", "checkout"],
                "source_id": "opencode session 1, ADR review",
                "event_time": "2026-07-14T09:30:00+00:00",
                "links": [
                    {"id": ids["attempt"], "relation": "derived_from",
                     "basis": "decision taken from the failed attempt"},
                    {"id": ids["test_output"], "relation": "validated_by",
                     "basis": "test output proves the failure mode"},
                ],
                "project_path": str(project)}))
            ids["decision"] = out["id"]

            out = json.loads(tool_remember(sessions, {
                "content": "ADR-11 decision (superseded): store order amounts "
                           "as float dollars and round to cents on read.",
                "type": "decision", "source_id": "older ADR session",
                "event_time": "2026-06-28T10:00:00+00:00",
                "project_path": str(project)}))
            ids["decision_old"] = out["id"]

            out = json.loads(tool_remember(sessions, {
                "content": "ADR-14 replaces ADR-11: integer minor units after "
                           "the checkout regression.",
                "type": "decision", "source_id": "opencode session 1",
                "links": [{"id": ids["decision_old"], "relation": "supersedes",
                           "basis": "ADR-14 supersedes ADR-11"}],
                "project_path": str(project)}))
            ids["decision_supersedes_marker"] = out["id"]
            return ids
        finally:
            sessions.close()

    def test_fresh_session_recovers_the_failure_trail_by_id(self, project):
        ids = self._session_one_remembers(project)
        # fresh session: new sessions object, nothing shared but the archive
        fresh = IntentGraphSessions()
        try:
            out = json.loads(tool_recall(fresh, {
                "query": "fix the checkout INVALID_AMOUNT regression; what did "
                         "we try and why did it fail?",
                "project_path": str(project)}))
            found = {r["id"]: r for r in out["results"]}
            assert ids["attempt"] in found
            assert "It did not work" in found[ids["attempt"]]["excerpt"]
            assert ids["decision"] in found
            assert ids["test_output"] in found
            # provenance travelled: the CI run is attributed to its build
            assert found[ids["test_output"]]["source_id"] == "ci/build-9931"
        finally:
            fresh.close()

    def test_fresh_session_context_packet_carries_exact_evidence(self, project):
        ids = self._session_one_remembers(project)
        fresh = IntentGraphSessions()
        try:
            text = tool_context(fresh, {
                "task": "Fix the checkout regression. Remember what we tried "
                        "before, why it failed, and which architecture "
                        "decision we made afterward.",
                "project_path": str(project)})
            # the exact records, by id, with their content
            assert ids["attempt"] in text
            assert "rounding the amount to cents" in text
            assert ids["decision"] in text
            assert "integer minor units" in text
            assert "Verdict:" in text
        finally:
            fresh.close()

    def test_stale_decision_is_not_presented_as_current(self, project):
        ids = self._session_one_remembers(project)
        fresh = IntentGraphSessions()
        try:
            # default (agent) view: the CURRENT decision is what surfaces
            text = tool_context(fresh, {
                "task": "what did we decide about order amounts?",
                "project_path": str(project)})
            assert "integer minor units" in text  # the current rationale
            # the stale record may legitimately be absent entirely (it lost
            # ranking to the current decision) — but if it EVER appears in a
            # packet, that packet must say it is superseded
            if ids["decision_old"] in text:
                assert "superseded" in text
            # the store-level invariant the packets rest on:
            ig = fresh.get(str(project))
            old = ig.store.get_record(ids["decision_old"])
            assert old.status == "superseded"
            assert old.superseded_by == ids["decision_supersedes_marker"]
            # audit view: the stale record IS reachable when explicitly
            # requested — and its status travels with it, so an agent can
            # never mistake it for current
            out = json.loads(tool_recall(fresh, {
                "query": "order amounts decision",
                "include_inactive": True,
                "project_path": str(project)}))
            statuses = {r["id"]: r["status"] for r in out["results"]}
            assert statuses.get(ids["decision_old"]) == "superseded"
        finally:
            fresh.close()

    def test_contradictory_memories_surface_both_sides(self, project):
        sessions = IntentGraphSessions()
        try:
            a = json.loads(tool_remember(sessions, {
                "content": "The gateway connection timeout is 5 seconds.",
                "type": "fact", "source_id": "gateway-api-docs.md",
                "project_path": str(project)}))
            b = json.loads(tool_remember(sessions, {
                "content": "The gateway connection timeout is 30 seconds "
                           "after the 2026-06 reliability patch.",
                "type": "fact", "source_id": "reliability-patch-notes",
                "links": [{"id": a["id"], "relation": "contradicts",
                           "basis": "doc says 5s, patch notes say 30s"}],
                "project_path": str(project)}))
        finally:
            sessions.close()
        fresh = IntentGraphSessions()
        try:
            text = tool_context(fresh, {
                "task": "what is the gateway connection timeout?",
                "project_path": str(project)})
            assert a["id"] in text and b["id"] in text
            assert "Conflicting evidence" in text
        finally:
            fresh.close()

    def test_missing_evidence_is_reported_not_fabricated(self, project):
        self._session_one_remembers(project)
        fresh = IntentGraphSessions()
        try:
            text = tool_context(fresh, {
                "task": "what did we decide about the postgres replication "
                        "topology?",
                "project_path": str(project)})
            assert "verdict: insufficient" in text.lower()
        finally:
            fresh.close()


# --- Step 3 wiring: the serve.py MCP surface (skip without the mcp extra) ---------------


# mcp 1.x names it isError, 2.x is_error — read whichever the installed
# major version models, the assertion is about the semantics.
def _is_error(result) -> bool:
    return bool(getattr(result, "isError", getattr(result, "is_error", False)))


class TestIntentGraphServe:
    """intentgraph.serve — IntentGraph's own MCP server (memory tools only).

    Same tool implementations (intentgraph.mcp), own registration: slim
    server, memory tools, usage instructions; no graph dependencies.
    Driven synchronously via asyncio.run on the registered request
    handlers, dual-compat across the mcp 1.x registry (request_handlers,
    request-object args) and the 2.x API (get_request_handler, (ctx,
    params) args) — the same two surfaces intentgraph/serve.py registers.
    """

    @pytest.fixture(autouse=True)
    def _needs_mcp(self):
        pytest.importorskip("mcp")

    def _server(self):
        from intentgraph import serve as serve_mod
        return serve_mod._build_server()

    def _call(self, server, name, arguments):
        import asyncio
        from mcp import types as mcp_types
        params = mcp_types.CallToolRequestParams(name=name,
                                                 arguments=arguments)

        if hasattr(server, "request_handlers"):  # mcp 1.x registry
            async def run():
                handler = server.request_handlers[mcp_types.CallToolRequest]
                result = await handler(mcp_types.CallToolRequest(params=params))
                return result.root if hasattr(result, "root") else result
        else:                                    # mcp 2.x registry
            async def run():
                entry = server.get_request_handler("tools/call")
                result = await entry.handler(None, params)
                return result.root if hasattr(result, "root") else result
        return asyncio.run(run())

    def _list(self, server):
        import asyncio
        from mcp import types as mcp_types

        if hasattr(server, "request_handlers"):  # mcp 1.x registry
            async def run():
                handler = server.request_handlers[mcp_types.ListToolsRequest]
                return (await handler(None)).root.tools
        else:                                    # mcp 2.x registry
            async def run():
                entry = server.get_request_handler("tools/list")
                result = await entry.handler(None, None)
                return (result.root if hasattr(result, "root") else result).tools
        return asyncio.run(run())

    def test_exactly_the_four_memory_tools(self):
        server = self._server()
        names = {t.name for t in self._list(server)}
        assert names == {"remember", "recall", "context", "feedback"}

    def test_instructions_travel_with_the_server(self):
        server = self._server()
        assert server.instructions
        assert "intentgraph_recall" in server.instructions
        assert "Never store secrets" in server.instructions

    def test_remember_recall_round_trip_without_any_graph(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        server = self._server()
        result = self._call(server, "remember", {
            "content": "the archive must survive a server swap",
            "type": "fact", "project_path": str(project)})
        assert not _is_error(result)
        saved = json.loads(result.content[0].text)
        result = self._call(server, "recall", {
            "query": "server swap survival", "project_path": str(project)})
        payload = json.loads(result.content[0].text)
        assert saved["id"] in [r["id"] for r in payload["results"]]

    def test_validation_failure_is_an_iserror_result(self, tmp_path):
        server = self._server()
        result = self._call(server, "remember", {
            "content": "   ", "project_path": str(tmp_path)})
        assert _is_error(result)
        assert "content" in result.content[0].text

    def test_entry_point_module_resolves(self):
        # the console script target imports and is callable
        from intentgraph import serve as serve_mod
        assert callable(serve_mod.main)

    def test_unknown_tool_reported_not_crashed(self):
        server = self._server()
        result = self._call(server, "nonexistent_tool", {})
        assert _is_error(result)
        assert "Unknown tool" in result.content[0].text

    def test_project_scoping_across_tools(self, tmp_path):
        # one server, many projects: a memory saved under project A is never
        # found by a recall aimed at project B (per-call project_path)
        project = tmp_path / "project"
        project.mkdir()
        other = tmp_path / "other-project"
        other.mkdir()
        server = self._server()
        saved = json.loads(self._call(server, "remember", {
            "content": "belongs to project A",
            "project_path": str(project)}).content[0].text)
        result = self._call(server, "recall", {
            "query": "belongs to project", "project_path": str(other)})
        assert not _is_error(result)
        payload = json.loads(result.content[0].text)
        assert saved["id"] not in [r["id"] for r in payload["results"]]
