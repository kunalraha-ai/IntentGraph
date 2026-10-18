"""Tests for the IntentGraph visualization adapter + renderer integration.

Two sides, per the extension contract:
- Graphify side: `to_html` with the new optional kwargs defaults to the
  pre-feature behavior (byte-level markers absent, RAW_NODES shape
  unchanged), and every overlay block is strictly opt-in.
- IntentGraph side: the adapter converts ONLY real archive data; ids,
  provenance, status, conflicts, and retrieval facts survive; missing
  metadata renders as "not recorded"; empty/insufficient states are honest.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from intentgraph.sdk import IntentGraph
from intentgraph.viz import (
    build_memory_graph,
    retrieval_overlay,
    write_memory_html,
    CONFLICT_RING,
    SUPERSEDED_RING,
    RETRIEVAL_DIRECT,
    RETRIEVAL_EXPANDED,
)
from tests.intentgraph_scenario import build

from graphify.export import to_html


@pytest.fixture
def scenario(tmp_path):
    ig, ids = build(tmp_path)
    yield ig, ids
    ig.close()


# --- Graphify side: the default render is unchanged (opt-in only) -------------


class TestRendererBackwardCompatibility:
    def _plain_graph(self):
        G = nx.Graph()
        G.add_node("a", label="A", source_file="a.py", file_type="code")
        G.add_node("b", label="B", source_file="b.py", file_type="code")
        G.add_edge("a", "b", relation="calls", confidence="EXTRACTED")
        return G

    def test_default_render_has_no_overlay_blocks(self, tmp_path):
        out = tmp_path / "g.html"
        to_html(self._plain_graph(), {0: ["a", "b"]}, str(out))
        html = out.read_text(encoding="utf-8")
        for absent in ("retrieval-hud", "legend-extra", "RETRIEVAL", '"extra_fields"',
                       '"viz_ring"', '"shapeProperties"', "#22d3ee"):
            assert absent not in html, f"overlay marker leaked: {absent}"

    def test_default_raw_nodes_shape_unchanged(self, tmp_path):
        out = tmp_path / "g.html"
        to_html(self._plain_graph(), {0: ["a", "b"]}, str(out))
        html = out.read_text(encoding="utf-8")
        m = re.search(r"const RAW_NODES = (\[.*?\]);\n", html, re.DOTALL)
        nodes = json.loads(m.group(1).replace("<\\/", "</"))
        assert set(nodes[0].keys()) == {
            "id", "label", "color", "size", "font", "title", "community",
            "community_name", "source_file", "file_type", "degree"}

    def test_legacy_graph_json_still_renders(self, tmp_path):
        # a graph.json from before IntentGraph existed: node-link JSON with
        # none of the viz_* fields — must render exactly as before
        legacy = {"directed": False,
                  "nodes": [{"id": "n1", "label": "Legacy", "source_file": "x.py"},
                            {"id": "n2", "label": "Node2", "source_file": "y.py"}],
                  "links": [{"source": "n1", "target": "n2", "relation": "imports",
                             "confidence": "EXTRACTED"}]}
        gp = tmp_path / "legacy.json"
        gp.write_text(json.dumps(legacy), encoding="utf-8")
        raw = json.loads(gp.read_text(encoding="utf-8"))
        G = nx.node_link_graph(raw, edges="links")
        out = tmp_path / "legacy.html"
        to_html(G, {0: list(G.nodes())}, str(out))
        html = out.read_text(encoding="utf-8")
        assert "Legacy" in html and "vis-network" in html
        assert "retrieval-hud" not in html

    def test_existing_interactions_remain_in_output(self, tmp_path):
        out = tmp_path / "g.html"
        to_html(self._plain_graph(), {0: ["a", "b"]}, str(out))
        html = out.read_text(encoding="utf-8")
        # search, community filter, node info panel, neighbors, select-all
        for marker in ('id="search"', "showInfo", "toggleAllCommunities",
                       "neighbor-link", 'id="select-all-cb"', "Communities"):
            assert marker in html, f"interaction lost: {marker}"

    def test_overlay_render_adds_hud_and_reset(self, tmp_path):
        out = tmp_path / "g.html"
        to_html(self._plain_graph(), {0: ["a", "b"]}, str(out),
                retrieval={"query": "why", "results": [{"id": "a", "tier": "direct"}]},
                legend_extra=[{"color": "#f59e0b", "label": "Contradiction"}])
        html = out.read_text(encoding="utf-8")
        assert "retrieval-hud" in html and "Reset highlighting" in html
        assert "legend-extra" in html

    def test_viz_attrs_reach_the_render(self, tmp_path):
        G = nx.Graph()
        G.add_node("m", label="Memory", viz_ring=CONFLICT_RING,
                   viz_shape="diamond", viz_dashes=True,
                   viz_fields=[("Memory type", "fact"), ("Status", "active")])
        out = tmp_path / "g.html"
        to_html(G, {0: ["m"]}, str(out))
        html = out.read_text(encoding="utf-8")
        assert '"shape": "diamond"' in html
        assert '"extra_fields": [["Memory type", "fact"], ["Status", "active"]]' in html
        assert '"borderDashes"' in html

    def test_viz_values_cannot_break_out_of_script_tag(self, tmp_path):
        # The #1838/#3664 boundary: RAW_* JSON is embedded inside a
        # <script> block. _js_safe escapes "</" so a hostile field value
        # cannot terminate the script tag; the panel then renders it via
        # esc() so it cannot inject markup. Assert both halves.
        G = nx.Graph()
        G.add_node("m", label="X",
                   viz_fields=[("Evil", '</script><script>alert(1)</script>')],
                   viz_ring='<img src=x onerror=alert(1)>')
        out = tmp_path / "g.html"
        to_html(G, {0: ["m"]}, str(out))
        html = out.read_text(encoding="utf-8")
        # (a) no unescaped </script> inside the RAW_NODES JSON blob:
        blob_start = html.find("const RAW_NODES = ")
        blob_end = html.find(";\n", blob_start)
        assert "</script>" not in html[blob_start:blob_end]
        # (b) the neutralized form is present (JSON "\/" escape)
        assert "<\\/script>" in html[blob_start:blob_end]
        # (c) the panel renders via esc() (structural: extra_fields only
        # ever flows through esc(pair[...]) in showInfo)
        assert "n.extra_fields.forEach(pair => {" in html


# --- Adapter: real data, stable ids, honest metadata ---------------------------


class TestMemoryGraphAdapter:
    def test_records_become_nodes_with_stable_ids(self, scenario):
        ig, ids = scenario
        G, labels = build_memory_graph(ig), None
        assert set(ids.values()) <= set(G.nodes())
        assert G.nodes[ids["decision_new"]]["viz_record_type"] == "decision"

    def test_typed_edges_preserved_with_direction(self, scenario):
        ig, ids = scenario
        G = build_memory_graph(ig)
        assert G.has_edge(ids["failure"], ids["test_output"])
        d = G[ids["failure"]][ids["test_output"]]
        assert d["relation"] == "validated_by"
        assert d["_src"] == ids["failure"] and d["_tgt"] == ids["test_output"]

    def test_explicit_vs_inferred_confidence_mapping(self, scenario):
        ig, ids = scenario
        G = build_memory_graph(ig)
        # the store is ground truth: every edge's confidence mirrors the
        # stored explicit flag exactly (EXTRACTED ↔ explicit, INFERRED ↔
        # inferred) — the renderer dashes INFERRED edges for free
        store_flags = {(e.source, e.target, e.relation): e.explicit
                       for e in ig.store.iter_edges()}
        for u, v, d in G.edges(data=True):
            if d["relation"] == "occurred_before":
                continue  # derived, covered by the conversation test
            key = (u, v, d["relation"])
            explicit = store_flags.get(key, store_flags.get((v, u, d["relation"])))
            assert d["confidence"] == ("EXTRACTED" if explicit else "INFERRED")

    def test_provenance_and_times_in_panel_fields(self, scenario):
        ig, ids = scenario
        G = build_memory_graph(ig)
        fields = dict(G.nodes[ids["failure"]]["viz_fields"])
        assert fields["Record ID"] == ids["failure"]
        assert fields["Event time"].startswith("2026-07-12")
        assert fields["Source"] == "tool: pagerduty-4471"
        assert fields["Status"] == "active (current)"

    def test_missing_metadata_renders_as_not_recorded(self, tmp_path):
        ig = IntentGraph(data_dir=str(tmp_path / "ig"))
        try:
            ig.ingest("bare memory with no source", type="fact")
            G = build_memory_graph(ig)
            nid = list(G.nodes())[0]
            fields = dict(G.nodes[nid]["viz_fields"])
            assert fields["Source"] == "not recorded"  # no source_id
            assert fields["Event time"] == "not recorded"  # no event_time
        finally:
            ig.close()

    def test_superseded_records_visible_gray_dashed_not_removed(self, scenario):
        ig, ids = scenario
        G = build_memory_graph(ig)  # include_superseded default True
        assert ids["decision_old"] in G.nodes()
        d = G.nodes[ids["decision_old"]]
        assert d["viz_ring"] == SUPERSEDED_RING
        assert d["viz_dashes"] is True
        fields = dict(d["viz_fields"])
        assert fields["Status"] == f"superseded (superseded by {ids['decision_new']})"

    def test_conflicts_marked_on_both_endpoints_with_ids(self, scenario):
        ig, ids = scenario
        G = build_memory_graph(ig)
        for nid in (ids["timeout_a"], ids["timeout_b"]):
            assert G.nodes[nid]["viz_ring"] == CONFLICT_RING
            other = ids["timeout_b"] if nid == ids["timeout_a"] else ids["timeout_a"]
            assert other in dict(G.nodes[nid]["viz_fields"])["Conflicts with"]
        assert G[ids["timeout_a"]][ids["timeout_b"]]["viz_color"] == CONFLICT_RING

    def test_conversation_events_diamond_and_labeled_not_facts(self, tmp_path):
        ig = IntentGraph(data_dir=str(tmp_path / "ig"))
        try:
            ig.ingest("User asked: why is it slow?", type="conversation",
                      source_id="sess-1", event_time="2026-07-12T10:00:00+00:00")
            ig.ingest("We measured the retry loop.", type="conversation",
                      source_id="sess-1", event_time="2026-07-12T10:01:00+00:00")
            G = build_memory_graph(ig)
            conv = [n for n, d in G.nodes(data=True)
                    if d.get("viz_shape") == "diamond"]
            assert len(conv) == 2
            fields = dict(G.nodes[conv[0]]["viz_fields"])
            assert "not a verified fact" in fields["Layer"]
            # chronological edge derived honestly: INFERRED, labeled
            a, b = conv
            assert G.has_edge(a, b)
            assert G[a][b]["relation"] == "occurred_before"
            assert G[a][b]["confidence"] == "INFERRED"
            # conversation excluded when asked
            G2 = build_memory_graph(ig, include_conversation=False)
            assert all(d.get("viz_shape") != "diamond"
                       for _, d in G2.nodes(data=True))
            assert not any(d["relation"] == "occurred_before"
                           for *_, d in G2.edges(data=True))
        finally:
            ig.close()

    def test_no_fake_connections(self, scenario):
        ig, ids = scenario
        G = build_memory_graph(ig)
        # every non-occurred_before edge exists in the store
        store_edges = {(e.source, e.target, e.relation)
                       for e in ig.store.iter_edges()}
        for u, v, d in G.edges(data=True):
            if d["relation"] == "occurred_before":
                continue
            assert (u, v, d["relation"]) in store_edges or \
                   (v, u, d["relation"]) in store_edges

    def test_empty_archive_yields_empty_graph_and_still_renders(self, tmp_path):
        ig = IntentGraph(data_dir=str(tmp_path / "ig"))
        try:
            G = build_memory_graph(ig)
            assert G.number_of_nodes() == 0
            out = tmp_path / "empty.html"
            assert write_memory_html(ig, out) is True
            html = out.read_text(encoding="utf-8")
            assert "0 nodes" in html  # stats line honest
            assert "retrieval-hud" not in html  # no query ran
        finally:
            ig.close()

    def test_malformed_optional_metadata_does_not_crash(self, tmp_path):
        ig = IntentGraph(data_dir=str(tmp_path / "ig"))
        try:
            ig.ingest("fine", type="fact")
            # corrupt optional fields on a node dict after build
            G = build_memory_graph(ig)
            nid = list(G.nodes())[0]
            G.nodes[nid]["viz_fields"] = "not-a-list"   # wrong type
            G.nodes[nid]["viz_ring"] = 42               # wrong type
            from graphify.export import to_html
            out = tmp_path / "m.html"
            to_html(G, {0: [nid]}, str(out))   # must not raise
            assert out.exists()
        finally:
            ig.close()


# --- Retrieval highlighting uses REAL retrieval output ------------------------


class TestRetrievalOverlay:
    def test_highlight_uses_actual_retrieve_results(self, scenario):
        ig, ids = scenario
        overlay = retrieval_overlay(ig, "checkout regression failure")
        assert overlay["query"] == "checkout regression failure"
        hit_ids = {r["id"] for r in overlay["results"]}
        assert hit_ids  # non-empty for a covered topic
        # every id is a real record id that exists in the archive
        for rid in hit_ids:
            assert ig.store.get_record(rid) is not None
        # ranks are 1..n and methods recorded
        ranks = [r["rank"] for r in overlay["results"]]
        assert ranks == list(range(1, len(ranks) + 1))
        assert all(r.get("methods") for r in overlay["results"])

    def test_direct_vs_expanded_tiers_from_real_methods(self, scenario):
        ig, ids = scenario
        # query that triggers graph expansion: attempt_1's failed_because_of
        # pulls the gateway fact in as supporting context
        overlay = retrieval_overlay(ig, "why did the checkout attempts fail")
        tiers = {r["id"]: r["tier"] for r in overlay["results"]}
        assert any(t == "direct" for t in tiers.values())
        # expanded candidates must have come from the graph path in the
        # real retrieval output — spot-check one expanded id
        result = ig.retrieve("why did the checkout attempts fail", mode="hybrid")
        for cand in result["candidates"]:
            if cand.get("graph_path"):
                assert tiers[cand["memory_id"]] == "expanded"
                break

    def test_empty_results_are_honest(self, scenario):
        ig, ids = scenario
        overlay = retrieval_overlay(ig, "quantum teleportation ballet in postgres")
        assert overlay["results"] == []
        assert "no evidence" in overlay["note"]

    def test_rank_facts_attached_to_highlighted_nodes_only(self, scenario):
        ig, ids = scenario
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.html"
            write_memory_html(ig, out, query="checkout regression")
            html = out.read_text(encoding="utf-8")
            assert '"Retrieval rank"' in html
            # panel facts are JSON pairs rendered as label: value in the JS
            # — the pair itself proves the fact survived the pipeline
            assert '"Retrieval rank", "#1' in html

    def test_score_is_labeled_as_retrieval_not_truth(self, scenario):
        ig, ids = scenario
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.html"
            write_memory_html(ig, out, query="checkout regression")
            html = out.read_text(encoding="utf-8")
            # the only place a score appears is as a retrieval-rank fact
            m = re.search(r'"Retrieval rank", "#(\d+) \(score ([0-9.]+), via ([a-z+]+)\)"', html)
            assert m  # format: rank, score, methods — no truth language
            assert "verified" not in html.lower() or "not a verified fact" in html

    def test_no_secrets_in_generated_html(self, scenario):
        ig, ids = scenario
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.html"
            write_memory_html(ig, out, query="checkout")
            html = out.read_text(encoding="utf-8")
            for absent in ("nvapi-", "tvly-", "NEBIUS_API_KEY", "TAVILY_API_KEY",
                           "NVIDIA_API_KEY"):
                assert absent not in html

    def test_insufficient_evidence_renders_notice_without_fabrication(self, scenario):
        ig, ids = scenario
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.html"
            write_memory_html(ig, out,
                              query="quantum teleportation ballet in postgres")
            html = out.read_text(encoding="utf-8")
            assert "No evidence returned" in html
            assert '"results": []' in html.replace("<\\/", "</")
            # nothing was highlighted: no rank facts fabricated
            assert '"Retrieval rank"' not in html

    def test_legend_reflects_real_render_content(self, scenario):
        ig, ids = scenario
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.html"
            write_memory_html(ig, out, query="checkout regression amounts")
            html = out.read_text(encoding="utf-8")
            assert "Retrieved evidence (direct)" in html
            assert "Contradiction" in html          # scenario has a contradicts edge
            assert "Superseded / historical" in html  # scenario has a superseded decision
            # direct color appears in the legend entries
            assert RETRIEVAL_DIRECT in html


# --- CLI surface ---------------------------------------------------------------


class TestVizCLI:
    def test_viz_command_writes_html(self, tmp_path):
        import subprocess, sys
        home = tmp_path / "ig-home"
        # seed via the SDK, then render via the CLI
        ig = IntentGraph(data_dir=str(home))
        try:
            ig.ingest("Decision: amounts are integer minor units.",
                      type="decision", source_id="ADR-14")
        finally:
            ig.close()
        out = tmp_path / "viz.html"
        r = subprocess.run(
            [sys.executable, "-m", "intentgraph", "--home", str(home),
             "viz", "--query", "amounts decision", "--out", str(out)],
            capture_output=True, text=True, timeout=120,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                 "INTENTGRAPH_DISABLE_DOTENV": "1"})
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert payload["written"] is True
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "integer minor units" in html
        assert "retrieval-hud" in html
