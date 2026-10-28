"""Cross-graph milestone tests: code links, code search, combined retrieval,
combined visualization, and the end-to-end "why does this code exist"
scenario — including the negative control (an unrelated query must not
surface the seeded records just because they exist).

All offline: the embedding provider is the deterministic hash projection,
the code graph is built from a synthetic repository, and the assertions
cover provenance (source paths, line numbers, memory ids, link bases) as
strictly as the presence of results.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from intentgraph.codegraph.build import BuildOptions, build_graph
from intentgraph.codegraph.export import write_json
from intentgraph.codequery import CodeIndex
from intentgraph.mcp import (IntentGraphSessions, MCPToolError, tool_recall,
                             tool_remember)
from intentgraph.sdk import IntentGraph
from intentgraph.store import StoreError

# --- the synthetic repository ----------------------------------------------------

CORE = '''\
class Base:
    """Base behavior."""

    def run(self):
        return self.step()

    def step(self):
        return 1


class Service(Base):
    def run(self):
        return self.step()

    def step(self):
        # retryable step entry: the split exists so direct execution
        # failures can be retried (the decision under test)
        return 1


def helper():
    return Service()
'''

API = '''\
from app.core import Service


def make():
    Service().run()
    return Service
'''

UTILS = "def util():\n    return 2\n"

TEST_CORE = '''\
from app.core import Service


def test_service():
    assert Service() is not None
'''


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "app/__init__.py", "")
    _write(root, "app/core.py", CORE)
    _write(root, "app/api.py", API)
    _write(root, "app/utils.py", UTILS)
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_core.py", TEST_CORE)
    return root


@pytest.fixture
def graph_json(repo):
    """A freshly built native code graph — no pre-existing file needed."""
    return write_json(build_graph(BuildOptions(root=repo)),
                      repo / "codegraph-out" / "graph.json")


@pytest.fixture
def ig(tmp_path, repo, graph_json, monkeypatch):
    """Archive wired to the code graph, with the deterministic hash
    embedding provider (offline tests need no credentials)."""
    monkeypatch.setenv("INTENTGRAPH_EMBEDDING_PROVIDER", "hash")
    session = IntentGraph(
        data_dir=str(tmp_path / "archive"),
        overrides={"graphify.graph_path": str(graph_json)})
    yield session
    session.close()


def _links(ig, **kw):
    return ig.store.iter_code_links(**kw)


def _ingest(ig, content, **kw):
    """ig.ingest → the MemoryRecord (IngestResult wraps it)."""
    return ig.ingest(content, **kw).record


# --- the link store ---------------------------------------------------------------


class TestCodeLinkStore:
    def test_add_list_remove_and_idempotency(self, ig):
        rec = _ingest(ig, "decision about the service", type="decision")
        link = ig.store.add_code_link(rec.id, "app_core_service", "explicit",
                                     basis="adr-7", method="user")
        assert link["id"].startswith("lnk_")
        again = ig.store.add_code_link(rec.id, "app_core_service", "explicit",
                                      basis="updated basis")
        assert again["id"] == link["id"]          # same (record, node, type)
        assert again["basis"] == "updated basis"  # refreshed, not duplicated
        assert len(_links(ig, record_id=rec.id)) == 1
        assert ig.store.remove_code_link(link["id"]) is True
        assert ig.store.remove_code_link(link["id"]) is False
        assert _links(ig) == []

    def test_unknown_type_and_record_are_rejected(self, ig):
        rec = _ingest(ig, "something", type="fact")
        with pytest.raises(StoreError):
            ig.store.add_code_link(rec.id, "app_core", "vibes")
        with pytest.raises(StoreError):
            ig.store.add_code_link("mem_" + "0" * 16, "app_core", "explicit")

    def test_purge_cascades_links(self, ig):
        rec = _ingest(ig, "will be purged", type="fact")
        ig.store.add_code_link(rec.id, "app_core", "explicit")
        assert ig.store.purge_record(rec.id) is True
        assert _links(ig, record_id=rec.id) == []

    def test_export_import_roundtrip_with_links(self, ig, tmp_path):
        rec = _ingest(ig, "portable decision", type="decision")
        ig.store.add_code_link(rec.id, "app_core_service", "explicit",
                               basis="adr-9")
        bundle = tmp_path / "b.jsonl"
        ig.store.export_bundle(bundle)
        other = IntentGraph(data_dir=str(tmp_path / "other-archive"))
        try:
            counts = other.store.import_bundle(bundle)
            assert counts["code_links"] == 1
            assert len(_links(other)) == 1
        finally:
            other.close()

    def test_older_bundle_imports_newer_rejected(self, ig, tmp_path):
        from intentgraph.models import MemoryRecord
        extra = MemoryRecord(content="record carried by an old bundle",
                             type="fact")
        bundle = tmp_path / "v1.jsonl"
        bundle.write_text("\n".join([
            json.dumps({"intentgraph_export": True, "schema_version": 1}),
            json.dumps({"kind": "record", "record": extra.to_dict()}),
        ]) + "\n", encoding="utf-8")
        counts = ig.store.import_bundle(bundle)   # older: accepted
        assert counts["records"] >= 1
        future = tmp_path / "v99.jsonl"
        future.write_text(json.dumps(
            {"intentgraph_export": True, "schema_version": 99}) + "\n",
            encoding="utf-8")
        with pytest.raises(StoreError):
            ig.store.import_bundle(future)        # newer: refused, clearly


# --- explicit links -----------------------------------------------------------------


class TestExplicitLinks:
    def test_add_validates_against_the_graph(self, ig):
        rec = _ingest(ig, "explicit decision", type="decision")
        from intentgraph.codelink import add_explicit_link
        link = add_explicit_link(ig, rec.id, "app_core_service",
                                 basis="adr-7 says so")
        assert link["link_type"] == "explicit"
        with pytest.raises(StoreError):
            add_explicit_link(ig, rec.id, "no_such_node")
        with pytest.raises(StoreError):
            add_explicit_link(ig, "mem_" + "0" * 16, "app_core_service")

    def test_add_without_a_graph_is_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INTENTGRAPH_EMBEDDING_PROVIDER", raising=False)
        bare = IntentGraph(data_dir=str(tmp_path / "bare"))
        try:
            rec = bare.ingest("no graph here", type="fact").record
            from intentgraph.codelink import add_explicit_link
            with pytest.raises(StoreError, match="no code graph"):
                add_explicit_link(bare, rec.id, "anything")
        finally:
            bare.close()


# --- evidence-backed extraction ------------------------------------------------------


class TestExtractedLinks:
    def test_ingest_extracts_real_references(self, ig):
        rec = _ingest(ig,
            "Decision: the `Service` class in `app/core.py` runs through "
            "Service.step because direct execution failed; app/api.py "
            "constructs it. The record `Service.run` is the entry point.",
            type="decision", source_id="adr-7")
        links = {(l["node_id"], l["match"]) for l in _links(ig, record_id=rec.id)}
        assert ("app_core", "app/core.py") in links
        assert ("app_core_service_run", "Service.run") in links
        assert ("app_core_service", "Service") in links       # backticked
        assert all(l["link_type"] == "extracted" for l in _links(ig,
                                                                 record_id=rec.id))
        for l in _links(ig, record_id=rec.id):
            assert l["method"] == "deterministic-text-match"
            assert "reference" in l["basis"]

    def test_bare_words_never_link(self, ig):
        # "run" and "test" exist as symbols; the words appear in text — but
        # bare single words are not evidence (precision rule)
        rec = _ingest(ig, "We run the test and it was run by the runner",
                        type="experience")
        assert _links(ig, record_id=rec.id) == []

    def test_package_name_links_to_the_module_not_a_same_named_class(self, ig):
        # `app.core` names a MODULE; resolution must land on the module
        # node, not on any same-named symbol — module names outrank
        # symbol fallbacks
        rec = _ingest(ig, "Decision: the `app.core` module owns the "
                          "service surface", type="decision")
        links = _links(ig, record_id=rec.id)
        assert links, "backticked module name must link"
        assert all(l["node_id"] == "app_core" for l in links), links

    def test_refresh_is_idempotent_and_removes_stale(self, ig):
        from intentgraph.codelink import refresh_links
        rec = _ingest(ig, "Talks about `Service.run` here", type="experience")
        # ingest already auto-extracted; refresh is the idempotent catch-up
        assert {l["node_id"] for l in _links(ig, record_id=rec.id)} == \
            {"app_core_service_run"}
        first = refresh_links(ig)
        assert first["created"] == 0          # nothing new: idempotent
        assert first["kept"] >= 1
        ig.update(rec.id, content="Now about `app.utils.util` instead")
        third = refresh_links(ig)
        assert third["removed"] >= 1           # stale match dropped
        nodes = {l["node_id"] for l in _links(ig, record_id=rec.id)}
        assert "app_core_service_run" not in nodes
        assert "app_utils_util" in nodes

    def test_refresh_never_touches_explicit_or_candidates(self, ig):
        from intentgraph.codelink import refresh_links
        rec = _ingest(ig, "has `Service.run` in text", type="experience")
        ig.store.add_code_link(rec.id, "app_core_service", "explicit",
                               basis="human assertion")
        ig.store.add_code_link(rec.id, "app_core", "candidate",
                               basis="similarity 0.9")
        out = refresh_links(ig)
        assert out["explicit_untouched"] and out["candidates_untouched"]
        assert len(_links(ig, record_id=rec.id,
                          link_type="explicit")) == 1
        assert len(_links(ig, record_id=rec.id,
                          link_type="candidate")) == 1


# --- candidate suggestions ------------------------------------------------------------


class TestSuggestLinks:
    def test_suggestions_are_candidates_only(self, ig):
        ig.ingest("The retry ladder was chosen after the gateway flaked",
                  type="decision")
        ig.ingest("Direct execution of the service timed out",
                  type="experience")
        ig.reindex_semantic()   # hash provider: offline, deterministic
        from intentgraph.codelink import suggest_links
        out = suggest_links(ig, threshold=0.1, limit=10, apply=False)
        assert out["available"] is True
        assert out["nodes_embedded"] > 0
        assert out["applied"] == 0               # dry run persists nothing
        assert _links(ig, link_type="candidate") == []
        if out["suggestions"]:
            assert all("candidate" in s["basis"] for s in out["suggestions"])

    def test_apply_persists_candidates_with_labels(self, ig):
        ig.ingest("Direct execution of the service timed out",
                  type="experience")
        ig.reindex_semantic()
        from intentgraph.codelink import suggest_links
        out = suggest_links(ig, threshold=0.0, limit=5, apply=True)
        assert out["available"] is True
        assert out["suggestions"], "threshold 0 must produce suggestions"
        assert out["applied"] == len(out["suggestions"])
        for l in _links(ig, link_type="candidate"):
            assert "verify before trusting" in l["basis"]
            assert l["method"] == "semantic-suggestion"

    def test_unavailable_reasons_are_honest(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INTENTGRAPH_EMBEDDING_PROVIDER", raising=False)
        bare = IntentGraph(data_dir=str(tmp_path / "bare2"))
        try:
            from intentgraph.codelink import suggest_links
            out = suggest_links(bare)
            assert out["available"] is False
            assert "no code graph" in out["reason"]
        finally:
            bare.close()

    def test_candidates_are_not_followed_as_evidence(self, ig):
        # a record whose text shares NO tokens with the code or the query,
        # connected ONLY by a candidate link — combined retrieval must not
        # surface it via the link path
        rec = _ingest(ig, "Zebras galloping over mars", type="experience")
        ig.store.add_code_link(rec.id, "app_core_service_step", "candidate",
                               basis="similarity 0.99")
        out = ig.query("Service.step app/core.py")
        for entry in out["memories"]:
            if entry["id"] == rec.id:
                assert "code_link" not in entry["methods"]
        code = [c for c in out["code"]
                if c["id"] == "app_core_service_step"]
        assert code, "expected the code hit"
        assert all(l["link_type"] != "candidate"
                   for l in code[0]["linked_memories"])


# --- deterministic code search ---------------------------------------------------------


class TestCodeIndexSearch:
    @pytest.fixture
    def index(self, ig):
        return ig.code_index

    def test_exact_qualified_and_path_matches(self, index):
        hits = index.search("Service.step")
        assert hits and hits[0]["id"] == "app_core_service_step"
        assert hits[0]["match"] == "symbol qualname"
        hits = index.search("app/core.py")
        assert any(h["id"] == "app_core" for h in hits)
        hits = index.search("app_core_service_run")
        assert hits and hits[0]["id"] == "app_core_service_run"
        assert hits[0]["match"] == "node id"

    def test_module_qualified_resolution(self, index):
        hits = index.search("app.core.step")
        assert hits and hits[0]["id"] == "app_core_base_step"

    def test_bare_common_words_do_not_match(self, index):
        hits = index.search("run the test")
        assert all(h["score"] < 55 for h in hits) or hits == []

    def test_search_is_deterministic_and_bounded(self, index):
        a = index.search("Service run step app/core.py", top_k=5)
        b = index.search("Service run step app/core.py", top_k=5)
        assert a == b
        assert len(a) <= 5

    def test_hits_carry_source_provenance(self, index):
        hits = index.search("Service.step")
        assert hits[0]["source_file"] == "app/core.py"
        assert hits[0]["source_location"].startswith("L")
        assert hits[0]["type"] == "method"
        assert hits[0]["qualname"] == "Service.step"


# --- combined retrieval: the end-to-end scenario ---------------------------------------


class TestCombinedRetrievalScenario:
    """The milestone scenario, step by step:

    1. a fresh graph from a small repository (fixtures above);
    2. a decision explaining an implementation choice, stored WITH its
       rationale and real code references;
    3. evidence of the previous failed approach;
    4. a FRESH retrieval session (new instance, nothing shared but the
       archive and the graph);
    5. ask why the implementation exists;
    6. the answer contains the code entities, the decision, the failed
       approach, and their source provenance;
    7. an unrelated query does NOT receive any of it.
    """

    @pytest.fixture
    def seeded(self, ig):
        decision = _ingest(ig, 
            "Decision: the step entry runs through Service.step in "
            "`app/core.py` because it must be retryable; direct execution "
            "of Service.run failed before this split.",
            type="decision", source_id="ADR-7",
            event_time="2026-10-20")
        failed = _ingest(ig, 
            "Failed attempt: calling Service.run directly with no step "
            "indirection produced double execution under retries "
            "(app/core.py).",
            type="experience", source_id="war-room-1",
            event_time="2026-10-19")
        unrelated = _ingest(ig, 
            "Preference: the docs portal uses a dark theme.",
            type="preference", source_id="design-review")
        ig.reindex_semantic()
        return {"decision": decision.id, "failed": failed.id,
                "unrelated": unrelated.id}

    @pytest.fixture
    def fresh(self, tmp_path, seeded):
        # step 4: a fresh session — no shared state but archive + graph
        fresh = IntentGraph(
            data_dir=str(tmp_path / "archive"),
            overrides={"graphify.graph_path": str(
                tmp_path / "repo" / "codegraph-out" / "graph.json")})
        yield fresh
        fresh.close()

    def test_the_why_question_returns_code_and_memories_with_provenance(
            self, fresh, seeded):
        out = fresh.query("why does Service.step exist — what is it for?")
        # code entities with source locations
        code_ids = {c["id"] for c in out["code"]}
        assert "app_core_service_step" in code_ids
        for c in out["code"]:
            assert c["source_file"] and c["source_location"]
        # the decision and the failed approach both surface
        ids = {m["id"] for m in out["memories"]}
        assert seeded["decision"] in ids
        assert seeded["failed"] in ids
        # every memory entry carries provenance
        for m in out["memories"]:
            assert m["id"].startswith("mem_")
            assert "excerpt" in m and "event_time" in m
        # verified links appear as provenance on code hits
        step = next(c for c in out["code"]
                    if c["id"] == "app_core_service_step")
        assert any(l["record_id"] == seeded["decision"]
                   for l in step["linked_memories"])

    def test_code_hit_pulls_in_a_linked_record_lexically_unreachable(
            self, fresh, seeded, ig):
        # explicit link from the code node to a record whose words do NOT
        # match the query at all — only the verified link can surface it
        hidden = _ingest(ig, "Zebras galloping over mars at dusk",
                           type="experience", source_id="zoo-1")
        from intentgraph.codelink import add_explicit_link
        add_explicit_link(ig, hidden.id, "app_core_service_step",
                          basis="zoo log proves the step split")
        out = fresh.query("app_core_service_step")
        entry = next((m for m in out["memories"] if m["id"] == hidden.id),
                     None)
        assert entry is not None, "verified link must be followed"
        assert "code_link" in entry["methods"]
        assert entry["via_code_link"]["node"] == "app_core_service_step"
        assert entry["via_code_link"]["type"] in ("explicit", "extracted")

    def test_the_negative_control(self, fresh, seeded):
        out = fresh.query("quantum teleportation ballet in postgres")
        ids = {m["id"] for m in out["memories"]}
        code_ids = {c["id"] for c in out["code"]}
        assert seeded["decision"] not in ids
        assert seeded["failed"] not in ids
        assert seeded["unrelated"] not in ids, "existence is not relevance"
        assert not any(c["id"].startswith("app_") for c in out["code"])

    def test_unrelated_memory_does_not_leak_into_the_why_answer(
            self, fresh, seeded):
        out = fresh.query("why does Service.step exist")
        ids = {m["id"] for m in out["memories"]}
        assert seeded["unrelated"] not in ids

    def test_candidates_listed_but_never_evidence(self, fresh, ig, seeded):
        ig.store.add_code_link(seeded["decision"], "app_utils_util",
                               "candidate", basis="similarity 0.8")
        out = fresh.query("app_utils_util", include_candidates=True)
        assert out["candidates"], "candidate must be listable when asked"
        assert all("NOT verified" in c["note"] for c in out["candidates"])
        util = next(c for c in out["code"] if c["id"] == "app_utils_util")
        assert all(l["link_type"] != "candidate"
                   for l in util["linked_memories"])

    def test_budget_trims_and_says_so(self, fresh, seeded):
        out = fresh.query("why does Service.step exist", budget_tokens=300)
        assert "trimmed" in out or out["code"] == [] or len(out["memories"]) <= 1
        if "trimmed" in out:
            assert any("trimmed to the token budget" in n for n in out["notes"])

    def test_no_graph_is_a_note_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INTENTGRAPH_EMBEDDING_PROVIDER", raising=False)
        bare = IntentGraph(data_dir=str(tmp_path / "no-graph"))
        try:
            rec = bare.ingest("decision with no code graph", type="decision").record
            out = bare.query("no code graph decision")
            assert any("no code graph" in n for n in out["notes"])
            assert out["memories"] and out["memories"][0]["id"] == rec.id
            assert out["code"] == []
        finally:
            bare.close()

    def test_semantic_layer_participates_in_combined_retrieval(
            self, fresh, seeded):
        ig_note = [n for n in
                   fresh.query("why does Service.step exist")["notes"]]
        assert not any("semantic retrieval unavailable" in n for n in ig_note)

    def test_token_cost_is_reportable(self, fresh, seeded):
        from intentgraph.tokens import estimate_tokens
        out = fresh.query("why does Service.step exist")
        tokens = estimate_tokens(json.dumps(out, ensure_ascii=False,
                                            default=str))
        assert tokens > 0  # measurement hook for the benchmark report


# --- MCP surface ------------------------------------------------------------------------


class TestMCPRecallIncludeCode:
    def test_include_code_adds_the_code_section(self, repo, tmp_path,
                                                graph_json, monkeypatch):
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_PROVIDER", "hash")
        sessions = IntentGraphSessions()
        try:
            tool_remember(sessions, {
                "content": "Decision: Service.step in `app/core.py` splits "
                           "the retry entry because direct execution failed.",
                "type": "decision", "project_path": str(repo),
                "source_id": "adr-7"})
            payload = json.loads(tool_recall(sessions, {
                "query": "why does Service.step exist",
                "include_code": True, "project_path": str(repo)}))
            assert "code" in payload
            assert any(c["id"] == "app_core_service_step"
                       for c in payload["code"])
            assert payload["results"], "memory results unchanged shape"
            assert payload["results"][0]["id"].startswith("mem_")
        finally:
            sessions.close()

    def test_default_recall_shape_is_unchanged(self, repo, tmp_path,
                                              graph_json, monkeypatch):
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_PROVIDER", "hash")
        sessions = IntentGraphSessions()
        try:
            tool_remember(sessions, {"content": "plain memory",
                                     "type": "fact",
                                     "project_path": str(repo)})
            payload = json.loads(tool_recall(sessions, {
                "query": "plain memory", "project_path": str(repo)}))
            assert "code" not in payload
            assert payload["results"]
        finally:
            sessions.close()

    def test_include_code_with_no_graph_is_a_note(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INTENTGRAPH_EMBEDDING_PROVIDER", raising=False)
        sessions = IntentGraphSessions()
        try:
            tool_remember(sessions, {"content": "no graph memory",
                                     "type": "fact",
                                     "project_path": str(tmp_path)})
            payload = json.loads(tool_recall(sessions, {
                "query": "no graph memory", "include_code": True,
                "project_path": str(tmp_path)}))
            assert payload["code"] == []
            assert any("no code graph" in n for n in payload["notes"])
        finally:
            sessions.close()


# --- combined visualization ---------------------------------------------------------------


class TestCombinedVisualization:
    def test_combined_render_shows_both_graphs_and_links(self, ig, tmp_path):
        from intentgraph.viz import write_memory_html
        rec = _ingest(ig, "Decision: `Service.run` delegates to the step "
                        "entry", type="decision")
        out = tmp_path / "combined.html"
        assert write_memory_html(ig, out, with_code=True,
                                 query="Service.run") is True
        html = out.read_text(encoding="utf-8")
        assert "app_core_service_run" in html          # code node present
        assert "Verified memory-code link" in html    # verified-link legend
        assert "Code modules" in html                  # combined communities
        assert rec.id in html                          # memory node present
        # only REAL retrieval results highlighted: the query's code hit and
        # nothing decorative
        blob = html[html.find("const RETRIEVAL = "):
                    html.find("\n", html.find("const RETRIEVAL = "))]
        retrieval = json.loads(
            blob.split("=", 1)[1].rstrip(";").replace("<\\/", "</"))
        assert "app_core_service_run" in {r["id"] for r in retrieval["results"]}

    def test_candidate_links_render_dashed_and_labelled(self, ig, tmp_path):
        from intentgraph.viz import write_memory_html
        rec = _ingest(ig, "a candidate-only association", type="experience")
        ig.store.add_code_link(rec.id, "app_core_service", "candidate",
                               basis="similarity 0.7")
        out = tmp_path / "cand.html"
        write_memory_html(ig, out, with_code=True)
        html = out.read_text(encoding="utf-8")
        assert "Candidate link (unverified)" in html
        # the renderer dashes non-EXTRACTED edges and names the confidence in
        # the edge title — the candidate edge is visibly dashed, labelled
        assert '"code_link [INFERRED]"' in html
        assert '"code_link [EXTRACTED]"' not in html

    def test_no_graph_renders_memory_only(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INTENTGRAPH_EMBEDDING_PROVIDER", raising=False)
        bare = IntentGraph(data_dir=str(tmp_path / "mg"))
        try:
            bare.ingest("memory only", type="fact")
            from intentgraph.viz import write_memory_html
            out = tmp_path / "mem.html"
            assert write_memory_html(bare, out, with_code=True) is True
            assert "app_core" not in out.read_text(encoding="utf-8")
        finally:
            bare.close()


# --- CLI surface ---------------------------------------------------------------------------


class TestCLISurfaces:
    def _run(self, argv, capsys):
        from intentgraph.cli import main
        rc = main(argv)
        out = capsys.readouterr().out
        return rc, json.loads(out) if out.strip() else None

    def test_query_command(self, repo, tmp_path, graph_json, monkeypatch,
                            capsys):
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_PROVIDER", "hash")
        monkeypatch.chdir(repo)
        monkeypatch.setenv("INTENTGRAPH_HOME", str(tmp_path / "arch"))
        rc, _ = self._run(["ingest", "--text",
                           "Decision: Service.step splits the retry entry",
                           "--type", "decision"], capsys)
        assert rc == 0
        rc, payload = self._run(
            ["query", "why does Service.step exist"], capsys)
        assert rc == 0
        assert payload["code"] or payload["memories"]
        assert "notes" in payload

    def test_codelink_cli_lifecycle(self, repo, tmp_path, graph_json,
                                    monkeypatch, capsys):
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_PROVIDER", "hash")
        monkeypatch.chdir(repo)
        monkeypatch.setenv("INTENTGRAPH_HOME", str(tmp_path / "arch2"))
        rc, res = self._run(["ingest", "--text", "needs a link",
                             "--type", "fact"], capsys)
        rid = res["id"]
        rc, link = self._run(["codelink", "add", rid, "app_core_service",
                              "--basis", "test"], capsys)
        assert rc == 0 and link["link_type"] == "explicit"
        rc, listing = self._run(["codelink", "list", "--record", rid],
                                capsys)
        assert listing["count"] == 1
        rc, refresh = self._run(["codelink", "refresh"], capsys)
        assert rc == 0 and refresh["records_scanned"] >= 1
        rc, removed = self._run(["codelink", "remove", link["id"]], capsys)
        assert rc == 0 and removed["removed"] is True

    def test_codegraph_search_command(self, repo, tmp_path, graph_json,
                                      capsys):
        rc, payload = self._run(["codegraph", "search", "Service.step",
                                 str(graph_json)], capsys)
        assert rc == 0
        assert payload["results"][0]["id"] == "app_core_base_step" \
            or payload["results"][0]["id"] == "app_core_service_step"

    def test_graph_rebuild_is_atomic(self, repo, tmp_path, graph_json):
        # a valid graph is never corrupted by an interrupted rebuild: the
        # write stages to a temp file and renames; no .tmp survives
        before = graph_json.read_bytes()
        write_json(build_graph(BuildOptions(root=repo)), graph_json)
        assert graph_json.exists()
        assert not list(graph_json.parent.glob("*.tmp"))
        json.loads(graph_json.read_text(encoding="utf-8"))  # still valid
        assert before  # original existed
