"""IntentGraph CLI — `intentgraph <command>` (and `graphify intent <command>`).

Conventions mirror the graphify CLI deliberately: subcommand dispatch on
sys.argv, exit code 1 with a stderr line on usage errors, `--json` for
machine-readable output, and no interactive prompts (an agent must be able
to drive every command).

The `--home` flag / INTENTGRAPH_HOME env decide where the archive lives
(default ./intentgraph-data). Configuration follows config.py: env + file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from intentgraph.config import load_config
from intentgraph.sdk import IntentGraph
from intentgraph.store import StoreError

__all__ = ["main", "build_parser"]


def _json_out(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _open(config_overrides: dict[str, Any]) -> IntentGraph:
    return IntentGraph(overrides=config_overrides)


def _parse_tags(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()] or None


def _parse_kv(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    out: dict[str, str] = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="intentgraph",
        description="IntentGraph — persistent memory and retrieval intelligence "
                    "for AI systems (built on Graphify).")
    p.add_argument("--home", help="data directory (default: ./intentgraph-data; "
                                  "env: INTENTGRAPH_HOME)")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="ingest a memory")
    ing.add_argument("--text", help="memory content (stdin if omitted)")
    ing.add_argument("--file", help="ingest a local file's contents")
    ing.add_argument("--type", choices=("conversation", "fact", "decision", "action",
                                        "experience", "document", "code_reference",
                                        "task", "preference", "project_state"),
                    help="memory type (inferred when omitted)")
    ing.add_argument("--tags", help="comma-separated tags")
    ing.add_argument("--source-type", default="manual",
                    help="conversation|file|tool|agent|api|import|manual|url")
    ing.add_argument("--source-id", help="where it came from (session id, path, ...)")
    ing.add_argument("--event-time", help="when it happened, if known (ISO or YYYY-MM-DD)")
    ing.add_argument("--anchors", help="comma-separated Graphify graph.json node ids")

    getp = sub.add_parser("get", help="fetch a memory by id")
    getp.add_argument("id")
    getp.add_argument("--version", type=int)

    sub.add_parser("status", help="archive stats, providers, bridge state")

    lst = sub.add_parser("list", help="list memories")
    lst.add_argument("--type")
    lst.add_argument("--tag")
    lst.add_argument("--status")
    lst.add_argument("--limit", type=int, default=20)
    lst.add_argument("--offset", type=int, default=0)

    lnk = sub.add_parser("link", help="create a typed relationship between memories")
    lnk.add_argument("source")
    lnk.add_argument("target")
    lnk.add_argument("--relation", required=True,
                     help="related_to|derived_from|caused_by|contradicts|supersedes|"
                          "depends_on|occurred_before|occurred_after|validated_by|"
                          "failed_because_of|belongs_to")
    lnk.add_argument("--basis", default="", help="why this edge exists")
    lnk.add_argument("--inferred", action="store_true",
                     help="mark the edge as inferred rather than explicit")

    upd = sub.add_parser("update", help="append a new version of a memory")
    upd.add_argument("id")
    upd.add_argument("--text", help="new content (stdin if omitted)")
    upd.add_argument("--file", help="new content from a local file")
    upd.add_argument("--note", default="", help="why this version changed")

    sup = sub.add_parser("supersede", help="mark one memory replaced by another")
    sup.add_argument("old_id")
    sup.add_argument("new_id")
    sup.add_argument("--basis", default="")

    dele = sub.add_parser("delete", help="soft-delete a memory (reversible)")
    dele.add_argument("id")
    dele.add_argument("--purge", action="store_true",
                      help="hard-delete (retention control; leaves a tombstone)")

    ret = sub.add_parser("retrieve", help="run a retrieval round")
    ret.add_argument("query")
    ret.add_argument("--mode", default="hybrid",
                     choices=("lexical", "semantic", "graph", "hybrid"))
    ret.add_argument("--top-k", type=int, default=10)

    rem = sub.add_parser("remember", help="full adaptive retrieval → context packet")
    rem.add_argument("query")
    rem.add_argument("--budget", type=int, default=None, help="token budget")
    rem.add_argument("--trace", action="store_true", help="include the full trace")

    fb = sub.add_parser("feedback", help="record retrieval feedback")
    fb.add_argument("query")
    fb.add_argument("--outcome", required=True, choices=("useful", "missing", "incorrect"))
    fb.add_argument("--ids", help="comma-separated memory ids")
    fb.add_argument("--note", default="")

    exp = sub.add_parser("export", help="export a portable JSONL bundle")
    exp.add_argument("path")
    imp = sub.add_parser("import", help="import a bundle (idempotent)")
    imp.add_argument("path")

    emb = sub.add_parser("embed", help="(re)compute embeddings via the configured provider")
    emb.add_argument("--list-models", action="store_true",
                     help="list the models the configured embedding endpoint "
                          "serves THIS account (one network call; verify "
                          "before picking embedding.model)")
    web = sub.add_parser("websearch", help="explicit external web search "
                        "(opt-in; requires web_search provider + TAVILY_API_KEY)")
    web.add_argument("query")
    web.add_argument("--max-results", type=int, default=None,
                     help="override web_search.max_results for this call")
    api = sub.add_parser("api", help="start the local HTTP API")
    api.add_argument("--host", default=None)
    api.add_argument("--port", type=int, default=None)
    api.add_argument("--api-key", default=None,
                     help="require this bearer token (env: INTENTGRAPH_API_KEY)")

    bench = sub.add_parser("benchmark", help="run the reproducible retrieval benchmark")
    bench.add_argument("--out", help="write JSON results to this path")
    bench.add_argument("--agent", action="store_true",
                       help="run the agent-session benchmark instead: same task "
                            "with and without previous-session memory, through "
                            "the actual MCP tool path")
    bench.add_argument("--providers", action="store_true",
                       help="run the provider comparison: hash baseline vs the "
                            "configured real embedding provider (labeled, "
                            "skipped-with-reason when not configured), and "
                            "adaptive + web search arms")

    doc = sub.add_parser("doctor", help="diagnose configuration and archive health")
    return p


def _read_text_arg(args: argparse.Namespace) -> str:
    if getattr(args, "text", None):
        return args.text
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    raise SystemExit("error: provide --text, --file, or stdin content")


def _hoist_global_flags(argv: list[str]) -> list[str]:
    """Move ``--home``/``--json`` to the front so they work anywhere.

    argparse only accepts subparser-level globals BEFORE the subcommand,
    but agents and humans write ``ingest --text x --home DIR``. Hoisting
    only exact option tokens (never positional values) makes both orders
    work without changing the subcommand parsers.
    """
    hoisted: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--home" and i + 1 < len(argv):
            hoisted.extend(["--home", argv[i + 1]])
            i += 2
            continue
        if a.startswith("--home="):
            hoisted.append(a)
            i += 1
            continue
        if a == "--json":
            hoisted.append(a)
            i += 1
            continue
        rest.append(a)
        i += 1
    return hoisted + rest


def main(argv: Sequence[str] | None = None) -> int:
    argv = _hoist_global_flags(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.home:
        overrides["data_dir"] = args.home

    try:
        return _dispatch(args, overrides)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace, overrides: dict[str, Any]) -> int:
    cmd = args.command

    if cmd == "ingest":
        ig = _open(overrides)
        content = _read_text_arg(args)
        res = ig.ingest(
            content, type=args.type, tags=_parse_tags(args.tags),
            source_type=args.source_type, source_id=args.source_id or "",
            event_time=args.event_time,
            graph_anchors=_parse_tags(args.anchors) or None)
        _json_out(res.to_dict())
        return 0

    if cmd == "get":
        ig = _open(overrides)
        rec = ig.get(args.id, version=args.version)
        if rec is None:
            print(f"error: no such record: {args.id}", file=sys.stderr)
            return 1
        _json_out(rec)
        return 0

    if cmd == "status":
        ig = _open(overrides)
        _json_out(ig.status())
        return 0

    if cmd == "list":
        ig = _open(overrides)
        records, total = ig.list(type=args.type, tag=args.tag, status=args.status,
                                 limit=args.limit, offset=args.offset)
        _json_out({"total": total, "records": records})
        return 0

    if cmd == "link":
        ig = _open(overrides)
        edge = ig.link(args.source, args.target, args.relation,
                       explicit=not args.inferred, basis=args.basis)
        _json_out(edge)
        return 0

    if cmd == "update":
        ig = _open(overrides)
        content = _read_text_arg(args)
        rec = ig.update(args.id, content=content, note=args.note)
        _json_out(rec)
        return 0

    if cmd == "supersede":
        ig = _open(overrides)
        _json_out(ig.supersede(args.old_id, args.new_id, basis=args.basis))
        return 0

    if cmd == "delete":
        ig = _open(overrides)
        if args.purge:
            ok = ig.purge(args.id)
            _json_out({"id": args.id, "purged": ok})
        else:
            _json_out(ig.delete(args.id))
        return 0

    if cmd == "retrieve":
        ig = _open(overrides)
        result = ig.retrieve(args.query, mode=args.mode, top_k=args.top_k)
        _json_out(result)
        return 0

    if cmd == "remember":
        ig = _open(overrides)
        result = ig.remember(args.query, budget_tokens=args.budget)
        out = {
            "packet": result["packet"],
            "text": result["text"],
        }
        if args.trace:
            out["trace"] = result["trace"]
        _json_out(out)
        return 0

    if cmd == "feedback":
        ig = _open(overrides)
        fb = ig.feedback(args.query, args.outcome,
                         memory_ids=_parse_tags(args.ids) or None, note=args.note)
        _json_out(fb)
        return 0

    if cmd == "export":
        ig = _open(overrides)
        _json_out(ig.export(args.path))
        return 0

    if cmd == "import":
        ig = _open(overrides)
        _json_out(ig.import_(args.path))
        return 0

    if cmd == "embed":
        ig = _open(overrides)
        try:
            if args.list_models:
                models = ig.list_embedding_models()
                _json_out({"models": models} if not isinstance(models, dict)
                          else models)
                return 0
            _json_out(ig.reindex_semantic())
        finally:
            ig.close()
        return 0

    if cmd == "websearch":
        ig = _open(overrides)
        try:
            if args.max_results is not None:
                from intentgraph.websearch import TavilySearchProvider
                if isinstance(ig.web_search, TavilySearchProvider):
                    ig.web_search.max_results = max(
                        1, min(20, int(args.max_results)))
            results = ig.search_web(args.query)
            _json_out({"query": args.query,
                       "provider": ig.web_search.name,
                       "results": results})
        except Exception as exc:  # noqa: BLE001 — CLI surface: honest exit
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        finally:
            ig.close()
        return 0

    if cmd == "api":
        from intentgraph.api import serve
        return serve(overrides=overrides,
                     host_override=args.host, port_override=args.port,
                     api_key_override=args.api_key)

    if cmd == "benchmark":
        from intentgraph.eval import (run_benchmark, run_provider_benchmark,
                                       run_session_benchmark)
        if args.providers:
            results = run_provider_benchmark()
        elif args.agent:
            results = run_session_benchmark()
        else:
            results = run_benchmark()
        _json_out(results)
        if args.out:
            Path(args.out).write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    if cmd == "doctor":
        return _doctor(overrides)

    parser_error = f"unknown command: {cmd}"
    raise SystemExit(parser_error)


def _doctor(overrides: dict[str, Any]) -> int:
    cfg = load_config(overrides=overrides or None)
    print(cfg.describe())
    ig = IntentGraph(overrides=overrides or None)
    try:
        status = ig.status()
        print(f"archive: {status['records']} records "
              f"({status['records_active']} active, "
              f"{status['records_superseded']} superseded, "
              f"{status['records_deleted']} deleted), "
              f"{status['versions']} historical versions, "
              f"{status['edges']} edges ({status['edges_inferred']} inferred)")
        print(f"lexical index: {status['lexical_index']['docs']} docs, "
              f"{status['lexical_index']['vocabulary']} terms")
        emb = status["embedding"]
        if emb["provider"] == "null":
            print("embedding: NOT CONFIGURED — semantic retrieval disabled "
                  "(lexical + graph + adaptive all fully functional).")
        elif str(emb["provider"]).startswith("hash"):
            print("embedding: hash provider (deterministic lexical-overlap "
                  "projection — a TEST/FALLBACK path, NOT real semantic "
                  "embeddings; configure nebius/openai for those).")
        elif emb.get("api_key_present") is False:
            print(f"embedding: {emb['provider']} dim={emb['dim']} — CREDENTIALS "
                  "MISSING: no API key resolved; embedding requests will fail "
                  "honestly until the provider's env var is set "
                  "(see docs/intentgraph.md)")
        else:
            print(f"embedding: {emb['provider']} dim={emb['dim']} "
                  f"configured={emb['configured']}")
        stale = emb.get("stale_model")
        if stale:
            print(f"  STALE MODEL: {stale}")
        llm = status.get("llm", {})
        if llm.get("active"):
            print(f"llm assist: ACTIVE ({llm.get('provider')}), "
                  f"calls={llm.get('llm_calls', 0)}, "
                  f"fallbacks={llm.get('fallbacks', 0)} — deterministic "
                  "policy is the fallback on every failure")
        else:
            print(f"llm assist: off (provider {llm.get('provider', 'none')}) — "
                  "deterministic controller in use")
        rr = status.get("reranker", {})
        if rr.get("provider", "none") != "none":
            print(f"reranker: {rr['provider']} model={rr.get('model')} "
                  f"available={rr.get('available')} (availability is "
                  "account-dependent — verify with the live smoke test)")
        else:
            print("reranker: off — fused order stands")
        ws = status.get("web_search", {})
        if ws.get("provider", "none") == "none":
            print("web search: NOT CONFIGURED — zero external requests possible")
        else:
            print(f"web search: {ws['provider']} enabled={ws.get('enabled')} "
                  f"available={ws.get('available')} "
                  f"max_results={ws.get('max_results')} "
                  f"store_results={ig.config.web_search.get('store_results')}")
            if not ws.get("enabled"):
                print("  (configured but NOT enabled — the controller will not "
                      "call it automatically; `intentgraph websearch` remains "
                      "the explicit manual search)")
            if ws.get("provider") == "tavily" and not ws.get("available"):
                print("  TAVILY_API_KEY is not set — search would fail; "
                      "export it or remove the provider")
        bridge = status["graphify_bridge"]
        if bridge["available"]:
            print(f"graphify bridge: ACTIVE ({bridge['nodes']} nodes from {bridge['path']})")
        else:
            print(f"graphify bridge: off"
                  + (f" ({bridge['error']})" if bridge["error"] else ""))
        return 0
    finally:
        ig.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
