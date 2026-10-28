"""CLI surface for the codegraph engine: ``intentgraph codegraph <cmd>``.

Commands (JSON on stdout, agent-drivable, no interactive prompts):

* ``build [ROOT]``   — scan + resolve + write graph.json (default output:
  ``<root>/codegraph-out/graph.json``). Re-running it IS the refresh.
* ``refresh [ROOT]`` — alias for build with the default output.
* ``stats [GRAPH]``  — summary of a saved graph (types, relations,
  components, hubs, import cycles, orphan candidates).
* ``show NODE``      — one node's attributes + bounded typed relationships.
* ``path A B``       — deterministic shortest paths between two nodes.
* ``render [GRAPH]`` — interactive graph.html through IntentGraph's renderer.

Build reports scanned/included/parsed/erroring files, skipped files (by
reason, with limits), nodes and edges by type and relation, unresolved and
external imports, entry points, output path, and elapsed time. Secret-like
files are never read and their names never carry content.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["register", "run"]

#: Renderer wording for a CODE graph (the memory renderer's defaults stay
#: untouched when this lexicon is not passed).
_CODE_LEXICON = {
    "title_prefix": "IntentGraph Code",
    "info_heading": "Code Info",
    "empty_hint": "Click a node to inspect it",
    "search_placeholder": "Search symbols...",
    "legend_heading": "Top-level groups",
}

_DEFAULT_OUTPUT_DIR = "codegraph-out"


def _json_out(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _default_output(root: Path) -> Path:
    return root / _DEFAULT_OUTPUT_DIR / "graph.json"


def _find_graph(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    cwd = Path.cwd()
    return cwd / _DEFAULT_OUTPUT_DIR / "graph.json"


# -- subcommand registration -----------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    cg = sub.add_parser(
        "codegraph",
        help="native codebase graphs: scan a repository, build a typed, "
             "evidence-backed graph.json")
    cgsub = cg.add_subparsers(dest="codegraph_command", required=True)

    def add_build_options(p: argparse.ArgumentParser,
                          with_output: bool = True) -> None:
        p.add_argument("root", nargs="?", default=".",
                       help="repository root to scan (default: current dir)")
        if with_output:
            p.add_argument("--output", default=None,
                           help=f"output path (default: <root>/"
                                f"{_DEFAULT_OUTPUT_DIR}/graph.json)")
        p.add_argument("--exclude", action="append", default=[],
                       metavar="GLOB",
                       help="gitignore-style exclusion (repeatable)")
        p.add_argument("--include-tests", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="include test modules and tests edges "
                            "(default: included)")
        p.add_argument("--max-files", type=int, default=20_000,
                       help="stop including files after this many "
                            "(default: 20000)")
        p.add_argument("--max-file-size", type=int, default=1_048_576,
                       help="skip files larger than this many bytes "
                            "(default: 1 MiB)")
        p.add_argument("--format", choices=("json", "graphml"),
                       default="json",
                       help="graph.json (bridge/renderer-compatible) or "
                            "GraphML")
        p.add_argument("--html", default=None, metavar="PATH",
                       help="also render an interactive graph.html here")
        p.add_argument("--verbose", action="store_true",
                      help="progress on stderr + full skip/error records "
                           "in the JSON payload")

    add_build_options(cgsub.add_parser(
        "build", help="scan a repository and write a code graph"))
    add_build_options(cgsub.add_parser(
        "refresh", help="rebuild the graph at the default output path"))

    st = cgsub.add_parser("stats", help="summary statistics for a code graph")
    st.add_argument("graph", nargs="?", default=None,
                    help="graph.json path (default: ./codegraph-out/graph.json)")

    se = cgsub.add_parser("search", help="deterministic symbol/file search "
                          "over a code graph")
    se.add_argument("query", help="symbols, dotted names, and/or file paths")
    se.add_argument("graph", nargs="?", default=None,
                    help="graph.json path (default: ./codegraph-out/graph.json)")
    se.add_argument("--top-k", type=int, default=10)

    sh = cgsub.add_parser("show", help="inspect one node: attributes + edges")
    sh.add_argument("node_id")
    sh.add_argument("graph", nargs="?", default=None,
                    help="graph.json path (default: ./codegraph-out/graph.json)")
    sh.add_argument("--depth", type=int, default=1,
                    help="also report this many BFS hops of neighborhood")

    pa = cgsub.add_parser("path", help="shortest paths between two nodes")
    pa.add_argument("a")
    pa.add_argument("b")
    pa.add_argument("graph", nargs="?", default=None,
                    help="graph.json path (default: ./codegraph-out/graph.json)")
    pa.add_argument("--max-paths", type=int, default=5)

    re = cgsub.add_parser("render", help="render a code graph as graph.html")
    re.add_argument("graph", nargs="?", default=None,
                    help="graph.json path (default: ./codegraph-out/graph.json)")
    re.add_argument("--out", default=None,
                    help="output path (default: alongside the graph.json)")


# -- handlers ----------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    cmd = args.codegraph_command
    if cmd in ("build", "refresh"):
        return _cmd_build(args)
    if cmd == "stats":
        return _cmd_stats(args)
    if cmd == "search":
        return _cmd_search(args)
    if cmd == "show":
        return _cmd_show(args)
    if cmd == "path":
        return _cmd_path(args)
    if cmd == "render":
        return _cmd_render(args)
    _err(f"unknown codegraph command: {cmd}")  # pragma: no cover
    return 1


def _cmd_search(args: argparse.Namespace) -> int:
    from intentgraph.codegraph.analyze import GraphFileError, load_graph
    from intentgraph.codequery import CodeIndex
    try:
        G, _, _ = load_graph(_find_graph(getattr(args, "graph", None)))
    except (FileNotFoundError, GraphFileError) as exc:
        _err(str(exc))
        return 1
    index = CodeIndex(G)
    hits = index.search(args.query, top_k=max(1, args.top_k))
    _json_out({
        "query": args.query,
        "count": len(hits),
        "results": hits,
        "note": "" if hits else
                "no code entity matched — search is exact/structural "
                "(symbols, dotted names, file paths), not fuzzy",
    })
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from intentgraph.codegraph.build import BuildOptions, build_graph
    from intentgraph.codegraph.export import (communities_by_directory,
                                              write_graphml, write_json)

    root = Path(args.root).resolve()
    if not root.is_dir():
        _err(f"scan root is not a directory: {args.root}")
        return 1
    output = Path(args.output) if args.output else _default_output(root)
    verbose = bool(args.verbose)

    def note(msg: str) -> None:
        if verbose:
            print(f"[codegraph] {msg}", file=sys.stderr)

    note(f"scanning {root} ...")
    started = time.perf_counter()
    build = build_graph(BuildOptions(
        root=root, excludes=tuple(args.exclude),
        max_files=args.max_files, max_file_size=args.max_file_size,
        include_tests=args.include_tests))
    note(f"parsed {build.stats['parsed']} module(s) "
         f"({build.stats['parse_error_count']} parse error(s))")

    if args.format == "graphml":
        written = write_graphml(build, output)
    else:
        written = write_json(build, output)
    note(f"wrote {written}")

    html_path: str | None = None
    if args.html:
        from intentgraph.renderer import to_html
        G = build.graph
        communities, labels = communities_by_directory(G)
        to_html(G.to_undirected(), communities, args.html,
                community_labels=labels, lexicon=_CODE_LEXICON)
        html_path = args.html
        note(f"rendered {args.html}")

    payload: dict[str, Any] = {
        "command": "build",
        "root": str(root),
        "output": str(written),
        "format": args.format,
        "elapsed_ms": build.stats["elapsed_ms"],
        "files_included": build.stats["files_included"],
        "parsed": build.stats["parsed"],
        "parse_error_count": build.stats["parse_error_count"],
        "parse_errors": build.stats["parse_errors"],
        "nodes": build.stats["nodes"],
        "edges": build.stats["edges"],
        "nodes_by_type": build.stats["nodes_by_type"],
        "edges_by_relation": build.stats["edges_by_relation"],
        "imports_unresolved": build.stats["imports_unresolved"],
        "imports_external": build.stats["imports_external"],
        "unresolved_calls": build.stats["unresolved_calls"],
        "unresolved_bases": build.stats["unresolved_bases"],
        "entry_points": build.stats["entry_points"],
        "truncated": build.scan.truncated,
        "skipped_by_reason": dict(sorted(build.scan.skipped_counts.items())),
        "tests_excluded": build.stats["tests_excluded"],
        "html": html_path,
        "total_elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
    if args.verbose:
        payload["skipped"] = [f"{s.relpath} ({s.reason})"
                              for s in build.scan.skipped]
        payload["scan_errors"] = build.scan.errors
        payload["id_collisions"] = build.stats["id_collisions"]
    _json_out(payload)
    return 0


def _loaded(args: argparse.Namespace):
    from intentgraph.codegraph.analyze import load_graph
    return load_graph(_find_graph(getattr(args, "graph", None)))


def _cmd_stats(args: argparse.Namespace) -> int:
    from intentgraph.codegraph.analyze import GraphFileError, graph_stats
    try:
        G, D, meta = _loaded(args)
    except (FileNotFoundError, GraphFileError) as exc:
        _err(str(exc))
        return 1
    _json_out(graph_stats(G, D, meta))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from intentgraph.codegraph.analyze import (GraphFileError, neighborhood,
                                               node_report)
    try:
        G, D, _ = _loaded(args)
    except (FileNotFoundError, GraphFileError) as exc:
        _err(str(exc))
        return 1
    try:
        report = node_report(G, args.node_id, D)
    except KeyError as exc:
        _err(str(exc).strip("'"))
        return 1
    depth = max(0, min(5, args.depth))
    if depth:
        try:
            report["neighborhood"] = neighborhood(G, args.node_id, depth=depth)
        except KeyError:  # pragma: no cover - report already validated node
            pass
    _json_out(report)
    return 0


def _cmd_path(args: argparse.Namespace) -> int:
    from intentgraph.codegraph.analyze import GraphFileError, find_paths
    try:
        G, _, _ = _loaded(args)
    except (FileNotFoundError, GraphFileError) as exc:
        _err(str(exc))
        return 1
    try:
        result = find_paths(G, args.a, args.b, max_paths=args.max_paths)
    except KeyError as exc:
        _err(str(exc).strip("'"))
        return 1
    _json_out({"from": args.a, "to": args.b, **result})
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from intentgraph.codegraph.analyze import GraphFileError, load_graph
    from intentgraph.codegraph.export import communities_by_directory
    from intentgraph.renderer import to_html
    graph_path = _find_graph(args.graph)
    try:
        G, _, _ = load_graph(graph_path)
    except (FileNotFoundError, GraphFileError) as exc:
        _err(str(exc))
        return 1
    out = args.out or str(graph_path.with_name("graph.html"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    communities, labels = communities_by_directory(G)
    to_html(G, communities, out, community_labels=labels,
            lexicon=_CODE_LEXICON)
    _json_out({"written": True, "graph": str(graph_path), "html": out,
               "nodes": G.number_of_nodes()})
    return 0
