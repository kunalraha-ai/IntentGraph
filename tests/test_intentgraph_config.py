"""Tests for intentgraph.config and intentgraph.graphbridge."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from intentgraph.config import Config, load_config
from intentgraph.graphbridge import GraphifyBridge
from intentgraph.sdk import IntentGraph
from tests.intentgraph_scenario import seed


class TestConfigResolution:
    def test_defaults_are_local_and_private(self):
        cfg = load_config(data_dir="/tmp/whatever-x")
        assert cfg.embedding["provider"] == "none"
        assert cfg.api["host"] == "127.0.0.1"
        assert cfg.privacy["local_only"] is False

    def test_env_overrides_win(self, monkeypatch, tmp_path):
        monkeypatch.setenv("INTENTGRAPH_HOME", str(tmp_path / "env-home"))
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_PROVIDER", "hash")
        cfg = load_config()
        assert cfg.data_dir == str(tmp_path / "env-home")
        assert cfg.embedding["provider"] == "hash"

    def test_config_file_roundtrip(self, tmp_path):
        home = tmp_path / "cfg-home"
        home.mkdir()
        (home / "intentgraph.json").write_text(json.dumps({
            "retrieval": {"max_iterations": 5, "context_budget_tokens": 2048},
            "embedding": {"provider": "hash", "dim": 128},
        }), encoding="utf-8")
        cfg = load_config(data_dir=str(home))
        assert cfg.retrieval["max_iterations"] == 5
        assert cfg.retrieval["context_budget_tokens"] == 2048
        assert cfg.embedding["dim"] == 128

    def test_env_beats_config_file(self, tmp_path, monkeypatch):
        home = tmp_path / "cfg-home2"
        home.mkdir()
        (home / "intentgraph.json").write_text(json.dumps({
            "embedding": {"provider": "none"}}), encoding="utf-8")
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_PROVIDER", "hash")
        cfg = load_config(data_dir=str(home))
        assert cfg.embedding["provider"] == "hash"

    def test_explicit_config_path(self, tmp_path):
        cf = tmp_path / "custom.json"
        cf.write_text(json.dumps({"privacy": {"local_only": True}}), encoding="utf-8")
        cfg = load_config(data_dir=str(tmp_path / "d"), config_file=cf)
        assert cfg.privacy["local_only"] is True

    def test_malformed_config_file_raises(self, tmp_path):
        home = tmp_path / "bad-home"
        home.mkdir()
        (home / "intentgraph.json").write_text("{oops", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot read config file"):
            load_config(data_dir=str(home))

    def test_local_only_refuses_remote_embedding(self, tmp_path):
        with pytest.raises(PermissionError, match="local_only"):
            load_config(data_dir=str(tmp_path / "d"),
                       overrides={
                           "privacy.local_only": True,
                           "embedding.provider": "openai",
                           "embedding.base_url": "https://api.example.com/v1",
                           "embedding.model": "m",
                       })

    def test_local_only_allows_localhost_embedding(self, tmp_path):
        cfg = load_config(data_dir=str(tmp_path / "d"),
                          overrides={
                              "privacy.local_only": True,
                              "embedding.provider": "openai",
                              "embedding.base_url": "http://localhost:11434/v1",
                              "embedding.model": "m",
                          })
        assert cfg.embedding["provider"] == "openai"

    def test_caller_overrides_win_over_everything(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INTENTGRAPH_RETRIEVAL_BUDGET", "500")
        cfg = load_config(data_dir=str(tmp_path / "d"),
                          overrides={"retrieval.context_budget_tokens": 999})
        assert cfg.retrieval["context_budget_tokens"] == 999

    def test_db_path_derived(self, tmp_path):
        cfg = load_config(data_dir=str(tmp_path / "home3"))
        assert cfg.db_path.name == "memory.db"
        assert str(cfg.db_path.parent) == str(tmp_path / "home3")


class TestGraphbridge:
    def _mini_graph(self, tmp_path) -> str:
        # a tiny node-link graph: auth.py --imports--> models.py
        data = {
            "directed": False,
            "multigraph": False,
            "nodes": [
                {"id": "auth.py", "label": "auth.py"},
                {"id": "models.py", "label": "models.py"},
                {"id": "routes.py", "label": "routes.py"},
            ],
            "links": [
                {"source": "auth.py", "target": "models.py", "relation": "imports"},
                {"source": "routes.py", "target": "auth.py", "relation": "imports"},
            ],
        }
        path = tmp_path / "graph.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def test_missing_graph_degrades_gracefully(self):
        b = GraphifyBridge(None)
        assert not b.available
        assert b.related_records(["x"], {}) == []

    def test_corrupt_graph_reports_error_not_crash(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{oops", encoding="utf-8")
        b = GraphifyBridge(p)
        assert not b.available
        assert "could not load" in b.load_error

    def test_records_link_through_code_neighborhood(self, tmp_path):
        graph_path = self._mini_graph(tmp_path)
        ig = IntentGraph(data_dir=str(tmp_path / "arch"),
                         overrides={"graphify.graph_path": graph_path})
        ids = seed(ig)
        # anchor the decision to routes.py and the timeout fact to models.py:
        # 2 hops apart in the code graph, no memory edge between them
        ig.update(ids["decision_new"], graph_anchors=["routes.py"])
        ig.update(ids["gateway_fact"], graph_anchors=["models.py"])
        ig.bridge.index_records(list(ig.store.iter_all_records()))

        records = {r.id: r for r in ig.store.iter_all_records()}
        hits = ig.bridge.related_records([ids["decision_new"]], records, depth=2)
        hit_ids = {h["id"] for h in hits}
        assert ids["gateway_fact"] in hit_ids
        by_id = {h["id"]: h for h in hits}
        assert "hop" in by_id[ids["gateway_fact"]]["via"]
        ig.close()

    def test_bridge_inactive_without_anchors(self, tmp_path):
        graph_path = self._mini_graph(tmp_path)
        ig = IntentGraph(data_dir=str(tmp_path / "arch2"),
                         overrides={"graphify.graph_path": graph_path})
        ids = seed(ig)  # scenario records carry no anchors
        records = {r.id: r for r in ig.store.iter_all_records()}
        assert ig.bridge.related_records([ids["failure"]], records) == []
        ig.close()

    def test_bridge_participates_in_hybrid_retrieval(self, tmp_path):
        graph_path = self._mini_graph(tmp_path)
        ig = IntentGraph(data_dir=str(tmp_path / "arch3"),
                         overrides={"graphify.graph_path": graph_path})
        ids = seed(ig)
        ig.update(ids["decision_new"], graph_anchors=["routes.py"])
        ig.update(ids["gateway_fact"], graph_anchors=["models.py"])
        ig.bridge.index_records(list(ig.store.iter_all_records()))
        result = ig.retrieve("amounts decision", mode="hybrid")
        methods_seen = {c.get("retriever", "") for c in result["candidates"]}
        assert any("graphify-bridge" in m for m in methods_seen)
        ig.close()
