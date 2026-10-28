"""Integration tests for the intentgraph local HTTP API (real server, loopback)."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from intentgraph import api as api_module
from intentgraph.api import MAX_BODY_BYTES, make_server
from intentgraph.config import load_config
from tests.intentgraph_scenario import seed


@pytest.fixture
def server(tmp_path):
    cfg = load_config(overrides={"data_dir": str(tmp_path / "archive")})
    srv = make_server("127.0.0.1", 0, "", cfg)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    srv.api.ig.close()


def _base(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def _request(base, method: str, path: str, body: dict | None = None,
             headers: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


class TestHealthAndListing:
    def test_health(self, server):
        status, body = _request(_base(server), "GET", "/health")
        assert status == 200
        assert body["status"] == "ok"
        assert "records" in body["archive"]

    def test_ingest_and_get_roundtrip(self, server):
        base = _base(server)
        status, created = _request(base, "POST", "/v1/memories", {
            "content": "We decided to adopt WAL mode for the archive.",
            "type": "decision", "tags": ["arch"],
            "source_id": "test-session",
        })
        assert status == 200
        assert created["id"].startswith("mem_")

        status, fetched = _request(base, "GET", f"/v1/memories/{created['id']}")
        assert status == 200
        assert fetched["content"].startswith("We decided")
        assert fetched["source_id"] == "test-session"

    def test_get_malformed_id_is_400(self, server):
        status, body = _request(_base(server), "GET", "/v1/memories/not-an-id")
        assert status == 400
        assert "malformed" in body["error"]

    def test_get_missing_is_404(self, server):
        status, body = _request(_base(server), "GET",
                                f"/v1/memories/mem_{'0' * 16}")
        assert status == 404

    def test_list_pagination_and_filters(self, server):
        base = _base(server)
        for i in range(3):
            _request(base, "POST", "/v1/memories", {"content": f"note {i}",
                                                    "tags": ["bench"]})
        status, body = _request(base, "GET", "/v1/memories?limit=2&tag=bench")
        assert status == 200
        assert body["total"] == 3
        assert len(body["records"]) == 2


class TestValidationAndErrors:
    def test_ingest_requires_content(self, server):
        status, body = _request(_base(server), "POST", "/v1/memories",
                                {"type": "fact"})
        assert status == 400
        assert "content" in body["error"]

    def test_invalid_json_is_400(self, server):
        req = urllib.request.Request(_base(server) + "/v1/memories",
                                     data=b"{not json",
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 400

    def test_oversized_body_is_413(self, server):
        status, body = _request(_base(server), "POST", "/v1/memories",
                                {"content": "x" * (MAX_BODY_BYTES + 10)})
        assert status == 413

    def test_unknown_endpoint_is_404(self, server):
        status, body = _request(_base(server), "GET", "/v1/nope")
        assert status == 404
        status, _ = _request(_base(server), "POST", "/v1/nope", {"a": 1})
        assert status == 404

    def test_bad_event_time_is_400(self, server):
        status, body = _request(_base(server), "POST", "/v1/memories",
                                {"content": "x", "event_time": "last tuesday"})
        assert status == 400


class TestLifecycle:
    def test_update_delete_restore_cycle(self, server):
        base = _base(server)
        _, created = _request(base, "POST", "/v1/memories", {"content": "v1"})
        rid = created["id"]

        status, updated = _request(base, "POST", f"/v1/memories/{rid}/update",
                                   {"content": "v2", "note": "revised"})
        assert status == 200 and updated["version"] == 2

        status, hist = _request(base, "GET", f"/v1/memories/{rid}/versions")
        assert status == 200
        assert len(hist["versions"]) == 2

        status, deleted = _request(base, "DELETE", f"/v1/memories/{rid}")
        assert status == 200 and deleted["status"] == "deleted"

        status, restored = _request(base, "POST", f"/v1/memories/{rid}/restore")
        assert status == 200 and restored["status"] == "active"

    def test_link_requires_relation(self, server):
        base = _base(server)
        _, a = _request(base, "POST", "/v1/memories", {"content": "a"})
        _, b = _request(base, "POST", "/v1/memories", {"content": "b"})
        status, body = _request(base, "POST", "/v1/links",
                                {"source": a["id"], "target": b["id"]})
        assert status == 400
        status, body = _request(base, "POST", "/v1/links",
                                {"source": a["id"], "target": b["id"],
                                 "relation": "validated_by"})
        assert status == 200


class TestRetrievalEndpoints:
    def test_retrieve_and_context(self, server):
        # seed the full scenario through the API's own SDK
        ig = server.api.ig
        ids = seed(ig)
        base = _base(server)

        status, out = _request(base, "POST", "/v1/retrieve",
                               {"query": "checkout regression", "mode": "hybrid"})
        assert status == 200
        found = {c["memory_id"] for c in out["candidates"]}
        assert ids["failure"] in found

        status, ctx = _request(base, "POST", "/v1/context",
                               {"query": "why did checkout fail and what did we try?",
                                "budget_tokens": 800})
        assert status == 200
        packet = ctx["packet"]
        assert packet["verdict"] in ("sufficient", "partial")
        assert packet["token_usage"] <= 800 * 1.1
        # the packet must carry the attempted-fixes evidence AND the failure
        # evidence (in at least one form — incident report, near-duplicate
        # retelling, or the CI test output that reproduces it)
        evidence = {e["memory_id"] for e in packet["evidence"]}
        assert any(ids[n] in evidence for n in ("attempt_1", "attempt_2"))
        assert any(ids[n] in evidence for n in
                   ("failure", "failure_near_dup", "test_output"))

    def test_feedback_endpoint(self, server):
        status, fb = _request(_base(server), "POST", "/v1/feedback",
                              {"query": "q", "outcome": "useful",
                               "note": "found it"})
        assert status == 200 and fb["outcome"] == "useful"
        status, bad = _request(_base(server), "POST", "/v1/feedback",
                              {"query": "q", "outcome": "meh"})
        assert status == 400

    def test_export_import_endpoints(self, server):
        base = _base(server)
        _request(base, "POST", "/v1/memories", {"content": "portable memory"})
        # bundle paths are constrained to the archive's data directory —
        # relative names resolve inside it
        status, report = _request(base, "POST", "/v1/export",
                                  {"path": "api-bundle.jsonl"})
        assert status == 200 and report["records"] == 1

        status, counts = _request(base, "POST", "/v1/import",
                                  {"path": "api-bundle.jsonl"})
        assert status == 200
        assert counts["skipped"] >= 1  # idempotent re-import

    def test_export_import_paths_cannot_escape_data_dir(self, server, tmp_path):
        base = _base(server)
        for bad in (str(tmp_path / "escape.jsonl"), "../../escape.jsonl"):
            status, body = _request(base, "POST", "/v1/export", {"path": bad})
            assert status == 400, bad
            assert "data directory" in body["error"]
        status, _ = _request(base, "POST", "/v1/import",
                             {"path": "/etc/passwd"})
        assert status == 400


class TestAuth:
    def test_api_key_required_and_enforced(self, tmp_path):
        cfg = load_config(overrides={"data_dir": str(tmp_path / "auth-archive")})
        srv = make_server("127.0.0.1", 0, "secret-token", cfg)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            status, body = _request(base, "GET", "/health")
            assert status == 401
            status, body = _request(base, "GET", "/health",
                                    headers={"Authorization": "Bearer wrong"})
            assert status == 401
            status, body = _request(base, "GET", "/health",
                                    headers={"Authorization": "Bearer secret-token"})
            assert status == 200
            status, body = _request(base, "GET", "/health",
                                    headers={"X-API-Key": "secret-token"})
            assert status == 200
        finally:
            srv.shutdown()
            srv.server_close()
            srv.api.ig.close()


def _get_raw(base: str, path: str) -> tuple[int, bytes, str]:
    req = urllib.request.Request(base + path)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


class TestGraphViews:
    """The web UI: dynamically generated knowledge graphs with a
    view-switcher dropdown, served at / (memory-code default)."""

    def test_root_serves_memory_code_graph(self, server):
        status, body, ctype = _get_raw(_base(server), "/")
        assert status == 200 and "text/html" in ctype
        html = body.decode("utf-8")
        # the graph renders (vis-network, sidebar, nodes)
        assert "vis-network" in html and 'id="sidebar"' in html
        # the view-switcher dropdown is present and pre-selects memory-code
        assert 'value="memory-code" selected' in html
        assert 'value="memory"' in html and 'value="code"' in html
        assert "Memory + Code graph" in html

    def test_memory_graph_view(self, server):
        status, body, ctype = _get_raw(_base(server), "/graph/memory")
        assert status == 200 and "text/html" in ctype
        html = body.decode("utf-8")
        assert 'value="memory" selected' in html

    def test_code_graph_view_404_without_graph(self, server):
        # no codegraph-out/graph.json in the fixture → honest 404
        status, body, _ = _get_raw(_base(server), "/graph/code")
        assert status == 404
        assert "no code graph" in json.loads(body.decode())["error"]

    def test_graph_view_with_code_graph(self, server, tmp_path):
        from intentgraph.codegraph.build import BuildOptions, build_graph
        from intentgraph.codegraph.export import write_json
        _write_test_repo(tmp_path)
        graph = write_json(build_graph(BuildOptions(root=tmp_path)),
                           tmp_path / "codegraph-out" / "graph.json")
        server.api.ig.config.graphify["graph_path"] = str(graph)
        # re-wire the bridge
        from intentgraph.graphbridge import GraphifyBridge
        server.api.ig.bridge = GraphifyBridge(graph)
        server.api.ig._code_index = None
        server.api.ig._code_index_built = False
        status, body, ctype = _get_raw(_base(server), "/graph/code")
        assert status == 200 and "text/html" in ctype
        html = body.decode("utf-8")
        assert 'value="code" selected' in html

    def test_graph_auth_via_query_param(self, tmp_path):
        # when a key is set, graph views accept ?key= (the dropdown
        # navigates via location.href, which can't set headers)
        cfg = load_config(overrides={"data_dir": str(tmp_path / "auth-archive")})
        srv = make_server("127.0.0.1", 0, "secret-token", cfg)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            status, _, _ = _get_raw(base, "/")
            assert status == 401
            status, _, _ = _get_raw(base, "/?key=secret-token")
            assert status == 200
            status, _, _ = _get_raw(base, "/?key=wrong")
            assert status == 401
        finally:
            srv.shutdown()
            srv.server_close()
            srv.api.ig.close()

    def test_unknown_graph_view_is_404(self, server):
        status, body, _ = _get_raw(_base(server), "/graph/nope")
        assert status == 404


def _write_test_repo(root):
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "mod.py").write_text(
        "class Thing:\n"
        "    def do(self):\n"
        "        return 1\n"
    )


class TestStatusAndCodeLinks:
    def test_status_reports_archive_bridge_and_links(self, server):
        base = _base(server)
        _request(base, "POST", "/v1/memories",
                 {"content": "decision about `app/core.py`"})
        status, body = _request(base, "GET", "/v1/status")
        assert status == 200
        assert body["records"] == 1
        assert "graphify_bridge" in body and "code_links" in body
        assert "lexical_index" not in body  # index internals are not API data

    def test_code_links_list_and_type_filter(self, server):
        base = _base(server)
        created = _request(base, "POST", "/v1/memories",
                           {"content": "fact for linking"})[1]
        link = server.api.ig.store.add_code_link(
            created["id"], "some_node_id", "explicit", basis="test link")
        assert link["id"].startswith("lnk_")
        status, body = _request(base, "GET", "/v1/code-links")
        assert status == 200 and body["count"] == 1
        assert body["links"][0]["node_id"] == "some_node_id"
        status, body = _request(base, "GET", "/v1/code-links?type=candidate")
        assert status == 200 and body["count"] == 0
        status, body = _request(base, "GET", "/v1/code-links?type=vibes")
        assert status == 400


class TestQueryEndpoint:
    def test_query_returns_memories_and_honest_notes(self, server):
        base = _base(server)
        _request(base, "POST", "/v1/memories", {
            "content": "Decision: the checkout timeout stays at 30 seconds "
                       "because the reliability patch made it safe.",
            "type": "decision"})
        status, body = _request(base, "POST", "/v1/query",
                                {"query": "why is the checkout timeout 30 seconds"})
        assert status == 200
        assert any(m["id"].startswith("mem_") for m in body["memories"])
        assert body["code"] == []  # no code graph wired in this fixture
        assert any("no code graph" in n for n in body["notes"])

    def test_query_requires_query_and_validates_numbers(self, server):
        base = _base(server)
        status, body = _request(base, "POST", "/v1/query", {"query": "   "})
        assert status == 400
        status, body = _request(base, "POST", "/v1/query",
                                {"query": "x", "top_k": "many"})
        assert status == 400
        status, body = _request(base, "POST", "/v1/query",
                                {"query": "x", "top_k": 3})
        assert status == 200


class TestReplayEndpoint:
    def test_replay_returns_ordered_steps(self, server):
        base = _base(server)
        d = _request(base, "POST", "/v1/memories", {
            "content": "Decision: the checkout timeout stays at 30 seconds "
                       "because the reliability patch made it safe.",
            "type": "decision", "source_id": "ADR-14",
            "event_time": "2026-10-20"})[1]
        f = _request(base, "POST", "/v1/memories", {
            "content": "Failed attempt: direct execution without the retry "
                       "wrapper produced double execution in the checkout.",
            "type": "experience", "source_id": "war-room-1",
            "event_time": "2026-10-19"})[1]
        # link them: the decision was derived from the failed attempt —
        # hybrid retrieval's graph expansion then finds both from the query
        _request(base, "POST", "/v1/links", {
            "source": d["id"], "target": f["id"],
            "relation": "derived_from", "basis": "the decision followed the failure"})
        status, body = _request(base, "POST", "/v1/replay",
                                {"query": "why is the checkout timeout 30 seconds"})
        assert status == 200
        assert body["query"].startswith("why is")
        assert len(body["steps"]) >= 3
        # step 0 is the focal node (the top memory)
        assert body["steps"][0]["kind"] == "focal"
        assert body["steps"][0]["id"].startswith("mem_")
        assert "decision" in body["steps"][0]["narration"].lower()
        assert "2026-10-20" in body["steps"][0]["narration"]
        # the supporting evidence follows the graph edge
        assert body["steps"][1]["kind"] == "evidence"
        assert body["steps"][1]["edge_from"] == body["steps"][0]["id"]
        # last step is the verdict
        assert body["steps"][-1]["kind"] == "verdict"
        assert "complete" in body["steps"][-1]["narration"].lower()
        # every step has narration
        for s in body["steps"]:
            assert s["narration"], s

    def test_replay_requires_query(self, server):
        status, body = _request(_base(server), "POST", "/v1/replay", {})
        assert status == 400

    def test_replay_honest_when_no_evidence(self, server):
        status, body = _request(_base(server), "POST", "/v1/replay",
                                {"query": "quantum teleportation"})
        assert status == 200
        assert body["steps"][-1]["kind"] == "verdict"
        assert "no evidence" in body["steps"][-1]["narration"].lower()

    def test_graph_view_includes_replay_engine(self, server):
        _, body, _ = _get_raw(_base(server), "/")
        html = body.decode("utf-8")
        assert "Replay.start" in html           # the engine is injected
        assert "speechSynthesis" in html         # voice guide
        assert "replay-timeline" in html        # playback controls


class TestVoiceEndpoint:
    def test_voice_reports_unconfigured_without_key(self, server, monkeypatch):
        monkeypatch.delenv("GNANI_API_KEY", raising=False)
        status, body = _request(_base(server), "POST", "/v1/voice",
                                {"text": "hello"})
        assert status == 503
        assert "GNANI_API_KEY" in body["error"]

    def test_voice_requires_text(self, server, monkeypatch):
        monkeypatch.setenv("GNANI_API_KEY", "vach_dummy")
        status, body = _request(_base(server), "POST", "/v1/voice",
                                {"text": "  "})
        assert status == 400

    def test_status_reports_voice_provider(self, server, monkeypatch):
        monkeypatch.setenv("GNANI_API_KEY", "vach_dummy")
        status, body = _request(_base(server), "GET", "/v1/status")
        assert body["voice"]["available"] is True
        assert body["voice"]["provider"] == "gnani"
        monkeypatch.delenv("GNANI_API_KEY", raising=False)
        status, body = _request(_base(server), "GET", "/v1/status")
        assert body["voice"]["available"] is False
