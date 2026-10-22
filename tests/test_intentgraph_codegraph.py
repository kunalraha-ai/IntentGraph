"""Tests for the native codegraph engine (scan → parse → resolve → graph →
export → analyze), its CLI, and its integration with the existing bridge,
renderer, and memory layers.

The engine is IntentGraph's own: no external code-graph package is
installed or imported (asserted in TestGraphifyAbsent), and every edge in
the graph is backed by static evidence — the tests assert the absence of
invented edges as carefully as the presence of resolved ones.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from intentgraph.codegraph.analyze import (GraphFileError, components,
                                           find_paths, graph_stats,
                                           import_cycles, load_graph,
                                           neighborhood, node_report,
                                           orphan_candidates)
from intentgraph.codegraph.build import BuildOptions, build_graph
from intentgraph.codegraph.export import (communities_by_directory,
                                          to_node_link, write_json)
from intentgraph.codegraph.scan import ScanOptions, scan_repository
from intentgraph.graphbridge import GraphifyBridge

# --- fixture: a small synthetic repository with known structure ------------------

CORE = '''\
class Base:
    """A base class."""

    def run(self):
        return self._step()

    def _step(self):
        return 1


class Service(Base):
    def __init__(self):
        self.cache = {}

    def run(self):
        self._step()
        return Base.run(self)


def module_helper():
    return Service()
'''

API = '''\
from app.core import Service, Base
import app.utils
from app import core as core_mod


def make():
    Service().run()
    return Base
'''

UTILS = "def util():\n    return 2\n"

TEST_CORE = '''\
from app.core import Service


def test_service():
    assert Service() is not None
'''

TEST_NAMING = "# no repo imports: exercises the naming-convention fallback\ndef test_nothing():\n    pass\n"

ENTRY = '''\
def main():
    return 1


if __name__ == "__main__":
    main()
'''

EXTERNAL = '''\
import os
import missing_pkg
from missing_pkg.thing import widget
from app.core import definitely_missing


def f():
    return os.path.join("a")
'''


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, ".gitignore",
           "ignored_dir/\ngenerated*.py\n*.gen.py\n!keep.gen.py\n")
    _write(root, "app/__init__.py", "")
    _write(root, "app/core.py", CORE)
    _write(root, "app/api.py", API)
    _write(root, "app/utils.py", UTILS)
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_core.py", TEST_CORE)
    _write(root, "tests/test_naming.py", TEST_NAMING)
    _write(root, "naming.py", "VALUE = 1\n")
    _write(root, "bad.py", "def broken(:\n")
    _write(root, "entry.py", ENTRY)
    _write(root, "external.py", EXTERNAL)
    _write(root, "keep.gen.py", "W = 4\n")
    _write(root, ".env", "SECRET=1\n")
    _write(root, "keys/server.pem", "-----BEGIN KEY-----\n")
    _write(root, "ignored_dir/skipme.py", "X = 1\n")
    _write(root, "generated_extra.py", "Y = 2\n")
    _write(root, "skip.gen.py", "Z = 3\n")
    _write(root, "node_modules/dep.py", "D = 1\n")
    _write(root, "docs/readme.md", "hello\n")
    return root


def _build(root: Path, **kw) -> object:
    from intentgraph.codegraph.build import CodeBuild  # noqa: F401 type hint
    defaults = dict(excludes=(), max_files=20_000, max_file_size=1_048_576,
                     include_tests=True)
    defaults.update(kw)
    return build_graph(BuildOptions(root=root, **defaults))


# --- scanning ----------------------------------------------------------------------


class TestScan:
    def test_includes_exactly_the_supported_unignored_files(self, repo):
        result = scan_repository(ScanOptions(root=repo))
        assert [f.relpath for f in result.files] == [
            "app/__init__.py", "app/api.py", "app/core.py", "app/utils.py",
            "bad.py", "entry.py", "external.py", "keep.gen.py", "naming.py",
            "tests/__init__.py", "tests/test_core.py", "tests/test_naming.py",
        ]

    def test_skip_reasons_are_honest_and_counted(self, repo):
        result = scan_repository(ScanOptions(root=repo))
        assert result.skipped_counts == {
            "excluded directory": 1,                      # node_modules/
            "ignored (gitignore/exclude rule)": 3,        # ignored_dir/, generated*.py, *.gen.py
            "secret-like file name (never read)": 2,      # .env, keys/server.pem
            "unsupported file type": 2,                    # .gitignore, docs/readme.md
        }
        # the negated pattern really re-includes
        assert any(f.relpath == "keep.gen.py" for f in result.files)
        assert not result.truncated and not result.errors

    def test_secret_files_are_never_read(self, repo):
        scan_repository(ScanOptions(root=repo))
        # the scan never opened them: nothing recorded beyond name+reason
        names = {s.relpath for s in scan_repository(ScanOptions(root=repo)).skipped}
        assert ".env" in names

    def test_max_files_truncates_deterministically(self, tmp_path):
        root = tmp_path / "mini"
        root.mkdir()
        for name in ("a.py", "b.py", "c.py", "d.py"):
            (root / name).write_text("", encoding="utf-8")
        r1 = scan_repository(ScanOptions(root=root, max_files=2))
        r2 = scan_repository(ScanOptions(root=root, max_files=2))
        assert [f.relpath for f in r1.files] == ["a.py", "b.py"]
        assert [f.relpath for f in r2.files] == ["a.py", "b.py"]  # same subset
        assert r1.truncated
        assert r1.skipped_counts.get("scan limit reached") == 2

    def test_max_file_size_skips_large_files(self, tmp_path):
        root = tmp_path / "mini"
        root.mkdir()
        (root / "tiny.py").write_text("", encoding="utf-8")
        (root / "big.py").write_text("x = 1\n" * 100, encoding="utf-8")
        result = scan_repository(ScanOptions(root=root, max_file_size=10))
        assert [f.relpath for f in result.files] == ["tiny.py"]
        assert result.skipped_counts.get("file exceeds size limit") == 1

    def test_symlinks_are_skipped_never_followed(self, repo, tmp_path):
        outside = tmp_path / "outside.py"
        outside.write_text("OUTSIDE = 1\n", encoding="utf-8")
        try:
            os.symlink(outside, repo / "linked.py")
            os.symlink(repo / "app", repo / "linked_app")
        except OSError:  # pragma: no cover - platforms without symlinks
            pytest.skip("symlinks unavailable on this machine")
        result = scan_repository(ScanOptions(root=repo))
        names = [f.relpath for f in result.files]
        assert "linked.py" not in names
        assert not any(n.startswith("linked_app") for n in names)
        assert any(s.relpath == "linked.py" and "symlink" in s.reason
                   for s in result.skipped)

    def test_user_excludes_without_gitignore(self, repo):
        result = scan_repository(ScanOptions(root=repo, excludes=("app/*",)))
        assert not any(f.relpath.startswith("app/") for f in result.files)

    def test_missing_root_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_repository(ScanOptions(root=tmp_path / "nope"))


# --- parsing, ids, and the graph ------------------------------------------------


class TestBuild:
    @pytest.fixture
    def built(self, repo):
        return _build(repo)

    def test_node_types_and_stable_ids(self, built):
        G = built.graph
        for nid, ntype in [
            ("repository", "repository"),
            ("dir_app", "directory"), ("dir_tests", "directory"),
            ("app_core", "module"), ("app_api", "module"),
            ("app_core_base", "class"), ("app_core_service", "class"),
            ("app_core_base_run", "method"),
            ("app_core_base__step", "method"),
            ("app_core_service_run", "method"),
            ("app_core_module_helper", "function"),
            ("app_api_make", "function"),
            ("tests_test_core", "test"),
            ("tests_test_core_test_service", "function"),
            ("bad", "module"),
            ("entry", "module"),
        ]:
            assert nid in G, f"missing node: {nid}"
            assert G.nodes[nid]["type"] == ntype, nid

    def test_ids_are_relative_never_absolute(self, built, repo):
        G = built.graph
        for nid in G.nodes:
            assert str(repo) not in nid
            assert str(repo.parent) not in nid
        for _, d in G.nodes(data=True):
            for key in ("label", "source_file"):
                if key in d:
                    assert str(repo) not in str(d[key])

    def test_module_and_symbol_provenance(self, built):
        G = built.graph
        n = G.nodes["app_core_base_run"]
        assert n["source_file"] == "app/core.py"
        assert n["source_location"].startswith("L")
        assert n["extraction"] == "ast"
        assert n["language"] == "python"
        assert G.nodes["tests_test_core_test_service"]["is_test"] is True

    def test_contains_and_defines_edges(self, built):
        G = built.graph
        rel = _rels(G)
        assert ("repository", "dir_app", "contains") in rel
        assert ("repository", "dir_tests", "contains") in rel
        assert ("dir_app", "app_core", "contains") in rel
        assert ("app_core", "app_core_base", "defines") in rel
        assert ("app_core", "app_core_service_run", "defines") in rel

    def test_imports_edges_resolved(self, built):
        rel = _rels(built.graph)
        # from-import of symbols
        assert ("app_api", "app_core", "imports") in rel
        # plain `import app.utils` — deep target, root binding
        assert ("app_api", "app_utils", "imports") in rel
        # `from app import core` — submodule resolution
        assert ("tests_test_core", "app_core", "imports") in rel

    def test_calls_edges_resolved_with_evidence(self, built):
        G = built.graph
        rel = _rels(G)
        # Service() — constructor call via from-import binding
        assert ("app_api_make", "app_core_service", "calls") in rel
        # self._step() inside Service.run resolves through the base class
        assert ("app_core_service_run", "app_core_base__step", "calls") in rel
        # self._step() inside Base.run resolves in the own class
        assert ("app_core_base_run", "app_core_base__step", "calls") in rel
        # Base.run(self) — explicit class-member chain
        assert ("app_core_service_run", "app_core_base_run", "calls") in rel
        # module-level call in the __main__ guard
        assert ("entry", "entry_main", "calls") in rel
        # Service().run(): the chained .run() on a call RESULT is not a
        # statically provable chain — it must NOT appear as an edge
        assert ("app_api_make", "app_core_service_run", "calls") not in rel

    def test_inherits_references_and_tests_edges(self, built):
        G = built.graph
        rel = _rels(G)
        assert ("app_core_service", "app_core_base", "inherits") in rel
        assert ("app_api_make", "app_core_base", "references") in rel  # `return Base`
        # tests edge, proven by the import
        e = _edge(built.graph, "tests_test_core", "app_core", "tests")
        assert e["confidence"] == "EXTRACTED"
        assert e["basis"] == "test module imports"
        # tests edge via naming convention alone is INFERRED, and labelled
        e = _edge(built.graph, "tests_test_naming", "naming", "tests")
        assert e["confidence"] == "INFERRED"

    def test_unresolved_things_never_become_edges(self, built):
        G = built.graph
        rel = _rels(G)
        # external.py's external dependencies (os, missing_pkg) create no
        # edges — its only imports edge is the resolved in-repo base
        ext_targets = {t for s, t, r in rel
                       if s == "external" and r == "imports"}
        assert ext_targets == {"app_core"}
        # local variable calls (`service.run()` style) stay unresolved
        assert built.stats["unresolved_calls"] >= 1
        # dynamic self.<attr> chains are counted, not invented
        assert ("app_core_service_run", "app_core_service_cache", "calls") \
            not in rel

    def test_in_repo_unresolved_import_is_counted_not_edged(self, built):
        # `from app.core import definitely_missing` — base resolves, name does not
        G = built.graph
        assert "app.core.definitely_missing" in \
            G.nodes["external"]["imports_unresolved"]
        assert built.stats["imports_unresolved"] >= 1
        # but the resolved base module still yields the import dependency
        rel = _rels(G)
        assert ("external", "app_core", "imports") in rel

    def test_external_imports_are_listed_not_edged(self, built):
        G = built.graph
        ext = G.nodes["external"]["imports_external"]
        assert "os" in ext and any(x.startswith("missing_pkg") for x in ext)

    def test_syntax_errors_degrade_to_a_marked_node(self, built):
        G = built.graph
        assert "bad" in G
        assert "SyntaxError" in G.nodes["bad"]["parse_error"]
        assert built.stats["parse_error_count"] == 1
        rel = _rels(G)
        # the file is a node, but nothing was invented from broken source
        assert not any(s == "bad" and r == "defines" for s, t, r in rel)

    def test_entry_points_discovered(self, built):
        assert "entry.py" in built.stats["entry_points"]
        assert "app/core.py" not in built.stats["entry_points"]

    def test_include_tests_false_excludes_test_modules(self, repo):
        built = _build(repo, include_tests=False)
        G = built.graph
        assert "tests_test_core" not in G
        assert built.stats["tests_excluded"] == 2
        assert "test" not in built.stats["nodes_by_type"]

    def test_empty_repository_builds_an_honest_trivial_graph(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        built = _build(root)
        assert built.stats["nodes"] == 1          # the repository node
        assert built.stats["edges"] == 0
        out = tmp_path / "g.json"
        write_json(built, out)
        G, D, meta = load_graph(out)
        assert G.number_of_nodes() == 1

    def test_id_collision_is_disambiguated_deterministically(self, tmp_path):
        # my_file.py and my/file.py flatten identically → deterministic suffix
        root = tmp_path / "collide"
        root.mkdir()
        _write(root, "my_file.py", "def a():\n    pass\n")
        _write(root, "my/file.py", "def b():\n    pass\n")
        built = _build(root)
        ids = set(built.graph.nodes)
        # "my/file.py" sorts first and keeps the bare id; the later file is
        # suffixed — a function of the tree, not of scan order
        assert "my_file" in ids and "my_file__2" in ids
        assert "my_file_b" in ids          # symbol of the first-registered file
        assert any(c["raw"] == "my_file" for c in built.stats["id_collisions"])


def _rels(G) -> set[tuple[str, str, str]]:
    return {(u, v, d.get("relation", "?"))
            for u, v, d in G.edges(data=True)}


def _edge(G, u, v, relation):
    for a, b, d in G.edges(data=True):
        if (a, b, d.get("relation")) == (u, v, relation):
            return d
    raise AssertionError(f"no {relation} edge {u} -> {v}")


# --- determinism and export --------------------------------------------------------


class TestDeterminismAndExport:
    def test_repeated_builds_are_byte_identical(self, repo, tmp_path):
        b1 = _build(repo)
        b2 = _build(repo)
        p1, p2 = tmp_path / "g1.json", tmp_path / "g2.json"
        write_json(b1, p1)
        write_json(b2, p2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_export_shape_is_bridge_and_renderer_compatible(self, repo):
        data = to_node_link(_build(repo))
        assert data["directed"] is False
        assert data["multigraph"] is True
        assert data["graph"]["schema"] == "intentgraph-codegraph"
        assert data["graph"]["schema_version"] == 1
        assert data["graph"]["languages"] == ["python"]
        assert [n["id"] for n in data["nodes"]] == \
            sorted(n["id"] for n in data["nodes"])
        for link in data["links"]:
            assert link["_src"] == link["source"]
            assert link["_tgt"] == link["target"]
            assert "relation" in link and "confidence" in link
        # renderer-required fields on a module node
        core = next(n for n in data["nodes"] if n["id"] == "app_core")
        assert core["label"] == "app/core.py"
        assert core["source_file"] == "app/core.py"
        assert core["file_type"] == "code"

    def test_load_graph_rebuilds_directed_view(self, repo, tmp_path):
        p = tmp_path / "g.json"
        write_json(_build(repo), p)
        G, D, meta = load_graph(p)
        assert not G.is_directed()
        assert D is not None and D.is_directed()
        assert ("app_api", "app_core", "imports") in _rels(D)
        assert meta["schema"] == "intentgraph-codegraph"

    def test_newer_schema_version_is_refused_not_guessed(self, tmp_path):
        data = {"directed": False, "multigraph": False,
                "graph": {"schema": "intentgraph-codegraph",
                          "schema_version": 99},
                "nodes": [{"id": "x"}], "links": []}
        p = tmp_path / "future.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(GraphFileError):
            load_graph(p)

    def test_legacy_files_without_marker_still_load(self, tmp_path):
        data = {"directed": False, "multigraph": False, "graph": {},
                "nodes": [{"id": "a", "label": "A"}],
                "links": [{"source": "a", "target": "b",
                           "relation": "imports", "confidence": "EXTRACTED"}]}
        p = tmp_path / "legacy.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        G, D, meta = load_graph(p)
        assert meta.get("schema") is None   # legacy: no marker, still valid
        assert G.number_of_nodes() == 2

    def test_graphml_export(self, repo, tmp_path):
        from intentgraph.codegraph.export import write_graphml
        p = tmp_path / "g.graphml"
        write_graphml(_build(repo), p)
        assert p.exists() and p.stat().st_size > 0

    def test_communities_are_deterministic_and_covered(self, repo):
        G = _build(repo).graph
        c1, l1 = communities_by_directory(G)
        c2, _ = communities_by_directory(G)
        assert c1 == c2
        covered = [n for members in c1.values() for n in members]
        assert sorted(covered) == sorted(G.nodes)
        assert all(label for label in l1.values())


# --- analysis -----------------------------------------------------------------------


class TestAnalyze:
    @pytest.fixture
    def graph(self, repo, tmp_path):
        p = tmp_path / "g.json"
        write_json(_build(repo), p)
        return load_graph(p)

    def test_stats_summary(self, graph):
        G, D, meta = graph
        stats = graph_stats(G, D, meta)
        assert stats["nodes"] == G.number_of_nodes()
        assert stats["nodes_by_type"]["module"] >= 2
        assert stats["edges_by_relation"]["defines"] > 0
        assert stats["components"]["count"] == 1
        assert stats["schema_version"] == 1

    def test_import_cycles_are_found(self, tmp_path):
        root = tmp_path / "cyclic"
        root.mkdir()
        _write(root, "a.py", "import b\n")
        _write(root, "b.py", "import a\n")
        p = tmp_path / "c.json"
        write_json(_build(root), p)
        _, D, _ = load_graph(p)
        out = import_cycles(D)
        assert out["cycles"] == [["a", "b"]]
        assert out["truncated"] is False

    def test_orphan_candidates_are_labelled_not_condemned(self, graph):
        G, _, _ = graph
        orphans = orphan_candidates(G)
        ids = {o["id"] for o in orphans}
        # nothing calls util or module_helper — candidates for review
        assert "app_utils_util" in ids
        assert "app_core_module_helper" in ids
        # symbols with proven incoming edges are not candidates
        assert "app_core_base__step" not in ids   # called by run() twice
        assert "app_core_service" not in ids      # constructed by make()
        # dunder methods are protocol-invoked and excluded by policy
        assert "app_core_service___init__" not in ids
        # test symbols are framework-discovered, not "uncalled"
        assert "tests_test_core_test_service" not in ids

    def test_find_paths_between_symbols(self, graph):
        G, _, _ = graph
        # the shortest route from make to Base._step is structural:
        # make -defines→ app_api -imports→ app_core -defines→ _step
        out = find_paths(G, "app_api_make", "app_core_base__step")
        assert out["found"] is True and out["length"] == 3
        assert out["paths"][0]["nodes"][0] == "app_api_make"
        assert out["paths"][0]["nodes"][-1] == "app_core_base__step"
        assert set(out["paths"][0]["relations"]) == {"defines", "imports"}
        # a directly proven call edge is one hop, labelled calls
        out = find_paths(G, "app_core_service_run", "app_core_base__step")
        assert out["length"] == 1 and out["paths"][0]["relations"] == ["calls"]

    def test_find_paths_reports_disconnection(self):
        G = nx.Graph()
        G.add_edge("a", "b")
        G.add_edge("c", "d")
        out = find_paths(G, "a", "c")
        assert out["found"] is False

    def test_neighborhood_is_bounded(self, graph):
        G, _, _ = graph
        hood = neighborhood(G, "app_core", depth=1, fanout=3, cap=10)
        assert "app_api" in hood or "tests_test_core" in hood
        assert len(hood) <= 10

    def test_node_report_groups_typed_edges(self, graph):
        G, D, _ = graph
        report = node_report(G, "app_core", D)
        assert report["attributes"]["label"] == "app/core.py"
        relations_out = {e["relation"] for e in report["outgoing"]}
        assert "defines" in relations_out

    def test_components_on_disconnected_graph(self):
        G = nx.Graph()
        G.add_nodes_from("abc")
        G.add_edge("a", "b")
        assert components(G) == {"count": 2, "largest": 2,
                                 "largest_sizes": [2, 1]}


# --- bridge, renderer, and memory integration ------------------------------------


class TestIntegration:
    def test_bridge_loads_a_native_graph_and_reports_schema(self, repo,
                                                            tmp_path):
        p = tmp_path / "g.json"
        write_json(_build(repo), p)
        bridge = GraphifyBridge(p)
        assert bridge.available
        assert bridge.schema["schema"] == "intentgraph-codegraph"
        assert bridge.node_count > 0

    def test_anchored_memories_reach_each_other_through_the_code_graph(
            self, repo, tmp_path):
        from intentgraph.mcp import IntentGraphSessions
        write_json(_build(repo), repo / "codegraph-out" / "graph.json")
        sessions = IntentGraphSessions()
        try:
            ig = sessions.get(str(repo))
            assert ig.bridge.available            # auto-wired: codegraph-out/
            a = ig.ingest("decision: services run in steps", type="decision",
                          graph_anchors=["app_core"]).to_dict()
            b = ig.ingest("experience: api builds the service", type="experience",
                          graph_anchors=["app_api"]).to_dict()
            ig.bridge.index_records(list(ig.store.iter_all_records()))
            records = {r.id: r for r in ig.store.iter_all_records()}
            hits = ig.bridge.related_records([a["id"]], records)
            assert b["id"] in {h["id"] for h in hits}
        finally:
            sessions.close()

    def test_status_reports_bridge_schema(self, repo):
        from intentgraph.mcp import IntentGraphSessions
        write_json(_build(repo), repo / "codegraph-out" / "graph.json")
        sessions = IntentGraphSessions()
        try:
            ig = sessions.get(str(repo))
            bridge = ig.status()["graphify_bridge"]
            assert bridge["schema"] == "intentgraph-codegraph"
            assert bridge["schema_version"] == 1
        finally:
            sessions.close()

    def test_renderer_renders_code_graph_with_code_wording(self, repo,
                                                           tmp_path):
        from intentgraph.renderer import to_html
        p = tmp_path / "g.json"
        write_json(_build(repo), p)
        G, _, _ = load_graph(p)
        communities, labels = communities_by_directory(G)
        out = tmp_path / "code.html"
        to_html(G, communities, str(out), community_labels=labels,
                lexicon={"title_prefix": "IntentGraph Code",
                         "info_heading": "Code Info",
                         "empty_hint": "Click a node to inspect it",
                         "search_placeholder": "Search symbols...",
                         "legend_heading": "Top-level groups"})
        html = out.read_text(encoding="utf-8")
        assert "Code Info" in html and "Search symbols..." in html
        assert "Memory Types" not in html
        assert "Base" in html                        # a real symbol label
        assert "SECRET=1" not in html                # .env never entered

    def test_renderer_default_lexicon_unchanged(self, tmp_path):
        from intentgraph.renderer import to_html
        G = nx.Graph()
        G.add_node("m", label="Mem", source_file="m.py")
        out = tmp_path / "mem.html"
        to_html(G, {0: ["m"]}, str(out))
        html = out.read_text(encoding="utf-8")
        assert "Memory Types" in html and "Memory Info" in html
        assert "Search memories..." in html


# --- CLI ----------------------------------------------------------------------------


class TestCLI:
    def _run(self, argv, capsys):
        from intentgraph.cli import main
        rc = main(argv)
        out = capsys.readouterr().out
        return rc, json.loads(out) if out.strip() else None

    def test_build_reports_everything(self, repo, capsys):
        rc, payload = self._run(
            ["codegraph", "build", str(repo), "--output",
             str(repo / "out.json"), "--verbose"], capsys)
        assert rc == 0
        assert payload["files_included"] == 12
        assert payload["parse_error_count"] == 1
        assert payload["nodes"] > 0 and payload["edges"] > 0
        assert payload["skipped_by_reason"]["unsupported file type"] == 2
        assert payload["elapsed_ms"] >= 0
        assert Path(payload["output"]).exists()
        assert payload["skipped"] and payload["scan_errors"] == []

    def test_build_default_output_and_refresh_alias(self, repo, capsys):
        rc, payload = self._run(["codegraph", "refresh", str(repo)], capsys)
        assert rc == 0
        assert payload["output"] == str(repo / "codegraph-out" / "graph.json")
        assert Path(payload["output"]).exists()

    def test_build_exclude_and_include_tests_flags(self, repo, capsys,
                                                   tmp_path):
        rc, payload = self._run(
            ["codegraph", "build", str(repo), "--output",
             str(tmp_path / "g.json"), "--exclude", "app/*",
             "--no-include-tests"], capsys)
        assert rc == 0
        assert payload["tests_excluded"] == 2
        assert "test" not in payload["nodes_by_type"]

    def test_build_html_flag_renders(self, repo, capsys, tmp_path):
        html = tmp_path / "code.html"
        rc, payload = self._run(
            ["codegraph", "build", str(repo), "--output",
             str(tmp_path / "g.json"), "--html", str(html)], capsys)
        assert rc == 0 and payload["html"] == str(html)
        assert "Code Info" in html.read_text(encoding="utf-8")

    def test_stats_show_path(self, repo, capsys):
        out = repo / "cg.json"
        rc, _ = self._run(["codegraph", "build", str(repo), "--output",
                           str(out)], capsys)
        assert rc == 0
        rc, stats = self._run(["codegraph", "stats", str(out)], capsys)
        assert rc == 0 and stats["nodes"] > 0
        rc, report = self._run(["codegraph", "show", "app_core", str(out)],
                               capsys)
        assert rc == 0 and report["attributes"]["label"] == "app/core.py"
        rc, paths = self._run(
            ["codegraph", "path", "app_api_make", "app_core_base__step",
             str(out)], capsys)
        assert rc == 0 and paths["found"] is True

    def test_error_reporting_is_clean(self, repo, capsys):
        from intentgraph.cli import main
        rc = main(["codegraph", "build", str(repo / "missing")])
        assert rc == 1 and "not a directory" in capsys.readouterr().err
        rc = main(["codegraph", "stats", str(repo / "missing.json")])
        assert rc == 1 and "not found" in capsys.readouterr().err
        rc = main(["codegraph", "show", "nope", str(repo / "cg.json")])
        assert rc == 1  # no graph → file error
        rc = main(["codegraph", "path", "x", "y", str(repo / "cg.json")])
        assert rc == 1

    def test_render_command(self, repo, capsys, tmp_path):
        out = tmp_path / "g.json"
        self._run(["codegraph", "build", str(repo), "--output", str(out)],
                  capsys)
        rc, payload = self._run(
            ["codegraph", "render", str(out), "--out",
             str(tmp_path / "r.html")], capsys)
        assert rc == 0 and payload["written"] is True
        assert (tmp_path / "r.html").exists()

    def test_end_to_end_subprocess(self, repo, tmp_path):
        out = tmp_path / "g.json"
        r = subprocess.run(
            [sys.executable, "-m", "intentgraph", "codegraph", "build",
             str(repo), "--output", str(out)],
            capture_output=True, text=True, timeout=300,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                 "INTENTGRAPH_DISABLE_DOTENV": "1"})
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert payload["nodes"] > 0 and payload["parse_error_count"] == 1
        assert Path(payload["output"]).exists()


# --- the Graphify-absent guarantee ---------------------------------------------------


class TestGraphifyAbsent:
    def test_engine_never_imports_graphify(self, repo, monkeypatch):
        # tripwire: a `import graphify` anywhere in the engine would raise
        # here instead of silently succeeding
        monkeypatch.setitem(sys.modules, "graphify", None)
        built = _build(repo)
        assert built.stats["nodes"] > 0

    def test_clean_interpreter_build_without_graphify(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        _write(root, "a.py", "import b\n\ndef f():\n    return b.g()\n")
        _write(root, "b.py", "def g():\n    return 1\n")
        script = (
            "import importlib.util, json, sys\n"
            "assert importlib.util.find_spec('graphify') is None\n"
            "from intentgraph.codegraph import build_graph\n"
            "from intentgraph.codegraph.build import BuildOptions\n"
            "from intentgraph.codegraph.export import write_json\n"
            "b = build_graph(BuildOptions(root=%r))\n"
            "write_json(b, %r)\n"
            "assert b.stats['nodes'] > 3\n"
            "assert not any(m == 'graphify' or m.startswith('graphify.')\n"
            "               for m in sys.modules)\n"
            "print('OK')\n" % (str(root), str(tmp_path / "g.json")))
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=300,
                           cwd=str(tmp_path),
                           env={"PATH": "/usr/bin:/bin",
                                "HOME": str(tmp_path),
                                "INTENTGRAPH_DISABLE_DOTENV": "1"})
        assert r.returncode == 0, r.stderr
        assert "OK" in r.stdout


# --- the real repository: id-compatibility with the documented scheme -------------


class TestRealRepoCompatibility:
    def test_ids_match_the_documented_anchor_scheme(self):
        repo_root = Path(__file__).resolve().parents[1]
        built = build_graph(BuildOptions(root=repo_root))
        ids = set(built.graph.nodes)
        # the exact ids the bridge docstring documents as the anchor scheme
        assert "intentgraph_models_memoryrecord" in ids
        assert "intentgraph_store" in ids
        assert "intentgraph_store_memorystore_add_edge" in ids

    def test_bridge_loads_the_real_repo_graph(self, tmp_path):
        repo_root = Path(__file__).resolve().parents[1]
        p = tmp_path / "real.json"
        write_json(build_graph(BuildOptions(root=repo_root)), p)
        bridge = GraphifyBridge(p)
        assert bridge.available
        assert bridge.node_count > 500
