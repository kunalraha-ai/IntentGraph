"""Tests for the intentgraph CLI (in-process subcommand dispatch)."""
from __future__ import annotations

import json

import pytest

from intentgraph.cli import main


@pytest.fixture
def home(tmp_path):
    return str(tmp_path / "archive")


def _run(*args: str) -> int:
    return main([*args])


def _last_json(capsys) -> dict:
    out = capsys.readouterr().out
    return json.loads(out)


class TestIngestGetList:
    def test_ingest_text_and_get(self, home, capsys):
        rc = _run("--home", home, "ingest", "--text",
                  "We decided to use SQLite for the archive.",
                  "--type", "decision", "--tags", "arch,storage",
                  "--source-id", "adr-1")
        assert rc == 0
        res = _last_json(capsys)
        assert res["type"] == "decision"

        rc = _run("--home", home, "get", res["id"])
        assert rc == 0
        rec = _last_json(capsys)
        assert rec["content"].startswith("We decided")
        assert rec["tags"] == ["arch", "storage"]

    def test_ingest_type_inference_via_cli(self, home, capsys):
        rc = _run("--home", home, "ingest", "--text",
                  "The deploy failed because the migration locked the table.")
        res = _last_json(capsys)
        assert res["type"] == "experience"

    def test_get_missing_id_exits_1(self, home, capsys):
        rc = _run("--home", home, "get", "mem_" + "0" * 16)
        assert rc == 1
        assert "no such record" in capsys.readouterr().err

    def test_list_with_filters(self, home, capsys):
        _run("--home", home, "ingest", "--text", "fact one", "--type", "fact")
        capsys.readouterr()
        _run("--home", home, "ingest", "--text", "decision one", "--type", "decision")
        capsys.readouterr()
        rc = _run("--home", home, "list", "--type", "fact")
        out = _last_json(capsys)
        assert rc == 0
        assert out["total"] == 1
        assert out["records"][0]["type"] == "fact"

    def test_status_reports_archive(self, home, capsys):
        _run("--home", home, "ingest", "--text", "hello")
        capsys.readouterr()
        rc = _run("--home", home, "status")
        out = _last_json(capsys)
        assert rc == 0
        assert out["records"] == 1
        assert out["embedding"]["provider"] == "null"
        assert out["graphify_bridge"]["available"] is False


class TestRelationships:
    def _two_ids(self, home, capsys) -> tuple[str, str]:
        _run("--home", home, "ingest", "--text", "record a")
        a = _last_json(capsys)["id"]
        _run("--home", home, "ingest", "--text", "record b")
        b = _last_json(capsys)["id"]
        capsys.readouterr()
        return a, b

    def test_link_and_delete(self, home, capsys):
        a, b = self._two_ids(home, capsys)
        rc = _run("--home", home, "link", a, b, "--relation", "validated_by",
                  "--basis", "test")
        assert rc == 0
        edge = _last_json(capsys)
        assert edge["relation"] == "validated_by"
        assert edge["explicit"] is True

        rc = _run("--home", home, "delete", a)
        assert rc == 0
        deleted = _last_json(capsys)
        assert deleted["status"] == "deleted"

    def test_link_unknown_relation_fails_cleanly(self, home, capsys):
        a, b = self._two_ids(home, capsys)
        capsys.readouterr()
        rc = _run("--home", home, "link", a, b, "--relation", "zaps")
        assert rc == 1
        assert "invalid edge" in capsys.readouterr().err

    def test_update_appends_version(self, home, capsys):
        a, _ = self._two_ids(home, capsys)
        rc = _run("--home", home, "update", a, "--text", "record a, revised",
                  "--note", "typo fix")
        assert rc == 0
        rec = _last_json(capsys)
        assert rec["version"] == 2
        assert rec["metadata"]["update_note"] == "typo fix"

    def test_supersede(self, home, capsys):
        a, b = self._two_ids(home, capsys)
        capsys.readouterr()
        rc = _run("--home", home, "supersede", a, b, "--basis", "replaced")
        assert rc == 0
        out = _last_json(capsys)
        assert out["status"] == "superseded"


class TestRetrievalCommands:
    def test_retrieve_then_remember(self, home, capsys):
        _run("--home", home, "ingest", "--text",
             "the checkout flow failed with INVALID_AMOUNT for small orders")
        capsys.readouterr()
        rc = _run("--home", home, "retrieve", "checkout failure", "--mode", "lexical")
        assert rc == 0
        out = _last_json(capsys)
        assert out["candidates"], out
        assert out["records"]

        rc = _run("--home", home, "remember", "why did checkout fail?")
        assert rc == 0
        result = _last_json(capsys)
        assert "packet" in result and "text" in result
        assert result["packet"]["verdict"] in ("sufficient", "partial", "insufficient")

    def test_feedback_recorded(self, home, capsys):
        _run("--home", home, "ingest", "--text", "something")
        rid = _last_json(capsys)["id"]
        assert rid.startswith("mem_")
        capsys.readouterr()
        rc = _run("--home", home, "feedback", "find the thing", "--outcome",
                  "useful", "--ids", rid)
        assert rc == 0
        fb = _last_json(capsys)
        assert fb["outcome"] == "useful"


class TestPortabilityCommands:
    def test_export_import_roundtrip(self, home, tmp_path, capsys):
        _run("--home", home, "ingest", "--text", "exported memory",
             "--tags", "port")
        rid = _last_json(capsys)["id"]
        capsys.readouterr()
        assert rid.startswith("mem_")
        bundle = tmp_path / "bundle.jsonl"
        rc = _run("--home", home, "export", str(bundle))
        assert rc == 0
        capsys.readouterr()

        other_home = str(tmp_path / "other-archive")
        rc = _run("--home", other_home, "import", str(bundle))
        assert rc == 0
        counts = _last_json(capsys)
        assert counts["records"] == 1

        rc = _run("--home", other_home, "get", rid)
        assert rc == 0

    def test_doctor_reports_configuration(self, home, capsys):
        rc = _run("--home", home, "doctor")
        assert rc == 0
        text = capsys.readouterr().out
        assert "embedding" in text and "code-graph bridge" in text


class TestProviderFlags:
    def test_embed_without_provider_reports_honestly(self, home, capsys):
        _run("--home", home, "ingest", "--text", "some memory")
        capsys.readouterr()
        rc = _run("--home", home, "embed")
        assert rc == 0
        out = _last_json(capsys)
        assert out["embedded"] == 0
        assert "no provider configured" in out["note"]

    def test_embed_with_hash_provider(self, home, capsys, monkeypatch):
        monkeypatch.setenv("INTENTGRAPH_EMBEDDING_PROVIDER", "hash")
        _run("--home", home, "ingest", "--text", "some memory")
        capsys.readouterr()
        rc = _run("--home", home, "embed")
        assert rc == 0
        out = _last_json(capsys)
        assert out["embedded"] == 1
        assert out["provider"]["provider"] == "hash-384"
