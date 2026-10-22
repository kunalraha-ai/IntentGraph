"""Codegraph builder: scan → parse → index → resolve → typed graph.

Two passes over the scanned tree:

1. **Index pass** — parse every scanned file, build the module index
   (dotted module name → repository-relative path) and per-file symbol
   tables, and create the graph's nodes (repository, directories,
   modules/tests, classes/functions/methods) with stable ids.
2. **Resolution pass** — per module (in sorted order), resolve import
   statements, call sites, name uses, and base-class expressions against
   the indexes. Resolved relationships become typed edges; anything static
   analysis cannot prove is recorded as per-module metadata or counters —
   never as an invented edge.

Evidence labels follow the memory graph's discipline:

* ``EXTRACTED`` — statically proven (the import statement exists, the call
  chain binds to a repo entity, the base class resolves).
* ``INFERRED`` — derived by a documented heuristic (currently: a method
  reached through an ambiguous base-class set, and test→code links matched
  by naming convention alone).

Import accounting distinguishes three states, honestly:

* resolved in-repo → ``imports`` edge;
* resolved as **external** (stdlib / site-packages / missing dependency) —
  recorded on the node (``imports_external``) and counted, never an edge;
* **unresolved** — the base module IS in the repo but the imported name is
  not among its top-level symbols (re-export indirection, ``__all__``
  tricks): recorded as ``imports_unresolved`` — genuinely uncertain.

Resolution is lexical and best-effort by design: dynamic attributes,
monkey-patching, and runtime rebinding are invisible to it, and the engine
reports that with counts rather than guessing. Analyzed code is never
executed.
"""
from __future__ import annotations

import builtins
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import networkx as nx

from intentgraph.codegraph.python_parser import (
    LanguageParser, ParsedModule, is_test_path, module_name_for)
from intentgraph.codegraph.scan import (
    ScanOptions, ScanResult, ScannedFile, scan_repository)
from intentgraph.codegraph.schema import (
    EXTRACTED, INFERRED, CodeEdge, IdRegistry, directory_id, file_id,
    repository_id, symbol_id_part)

__all__ = ["CodeBuild", "BuildOptions", "build_graph"]

_BUILTIN_NAMES = frozenset(dir(builtins)) | {
    "self", "cls", "super", "__name__", "__file__", "print", "len",
}
#: resolution sentinel: the chain roots in something provably outside the
#: repository (builtin, or a module binding that does not resolve in-repo).
_EXTERNAL = "external"

_MAX_RECORDED_ERRORS = 100
_MAX_LISTED = 200


@dataclass(frozen=True)
class BuildOptions:
    root: Path
    excludes: tuple[str, ...] = ()
    max_files: int = 20_000
    max_file_size: int = 1_048_576
    include_tests: bool = True


@dataclass
class CodeBuild:
    graph: nx.DiGraph
    scan: ScanResult
    options: BuildOptions
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Binding:
    kind: str        # "local" | "module" | "symbol"
    a: str           # local: qualname · module: dotted · symbol: module dotted
    b: str = ""      # symbol binding: imported symbol name


@dataclass
class _ModCtx:
    relpath: str
    module: str                       # dotted module name
    parsed: ParsedModule
    node_id: str
    ids: dict[str, str] = field(default_factory=dict)   # qualname → node id
    bindings: dict[str, _Binding] = field(default_factory=dict)
    imported: set[str] = field(default_factory=set)     # imported relpaths


class _Builder:
    def __init__(self, options: BuildOptions) -> None:
        self.options = options
        self.registry = IdRegistry()
        # keyed by relation: a module can both define and reference a symbol,
        # and neither relation may silently replace the other
        self.graph = nx.MultiDiGraph()
        self.ctx_by_relpath: dict[str, _ModCtx] = {}
        self.ctxs: list[_ModCtx] = []
        self.parse_errors: list[dict[str, str]] = []
        self.module_index: dict[str, str] = {}     # dotted → relpath
        self.top_symbols: dict[str, dict[str, tuple[str, str]]] = {}
        # module dotted → {name → (relpath, qualname)}
        self.class_members: dict[tuple[str, str], dict[str, str]] = {}
        self.classes_by_relpath: dict[str, list[str]] = {}
        self.bases: dict[tuple[str, str], list[list[str]]] = {}
        self.edges: dict[tuple[str, str, str], CodeEdge] = {}
        self.unresolved_calls = 0
        self.unresolved_uses = 0
        self.unresolved_bases = 0
        self.unresolved_imports = 0
        self.external_imports: set[str] = set()

    # -- driver -------------------------------------------------------------------

    def run(self) -> CodeBuild:
        started = time.perf_counter()
        scan = scan_repository(ScanOptions(
            root=self.options.root, excludes=self.options.excludes,
            max_files=self.options.max_files,
            max_file_size=self.options.max_file_size))

        tests_excluded = 0
        files: list[ScannedFile] = []
        for f in scan.files:
            if not self.options.include_tests and is_test_path(f.relpath):
                tests_excluded += 1
                scan.note_skip(f.relpath, "tests excluded (--no-include-tests)")
                continue
            files.append(f)
        scan.files = files

        self._parse_all(files)
        self._index_modules(files)
        self._create_nodes(scan, files)
        for ctx in self.ctxs:
            self._resolve_module(ctx)

        G = self.graph
        for (src, tgt, rel), edge in sorted(self.edges.items()):
            G.add_edge(src, tgt, key=rel, relation=rel,
                       confidence=edge.confidence,
                       _src=src, _tgt=tgt,
                       source_file=edge.source_file,
                       source_location=edge.source_location,
                       extraction=edge.extraction, basis=edge.basis,
                       locations=sorted(edge.locations))

        stats = {
            "root_label": scan.root.name,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "files_included": len(files),
            "parsed": sum(1 for c in self.ctxs if c.parsed.error is None),
            "parse_errors": self.parse_errors[:_MAX_RECORDED_ERRORS],
            "parse_error_count": len(self.parse_errors),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "nodes_by_type": self._count_by_type(),
            "edges_by_relation": self._count_by_relation(),
            "imports_unresolved": self.unresolved_imports,
            "imports_external": len(self.external_imports),
            "unresolved_calls": self.unresolved_calls,
            "unresolved_uses": self.unresolved_uses,
            "unresolved_bases": self.unresolved_bases,
            "entry_points": sorted(
                c.relpath for c in self.ctxs if c.parsed.entry_point),
            "tests_excluded": tests_excluded,
            "id_collisions": self.registry.collisions,
        }
        return CodeBuild(graph=G, scan=scan, options=self.options, stats=stats)

    # -- pass 1: parse, index, nodes ------------------------------------------------

    def _parse_all(self, files: list[ScannedFile]) -> None:
        for f in files:
            parser = _parser_for(f.language)
            if parser is None:  # pragma: no cover - registry guarantees this
                continue
            try:
                text = (self.options.root / f.relpath).read_text(
                    encoding="utf-8", errors="replace")
            except OSError as exc:
                self.parse_errors.append(
                    {"file": f.relpath, "error": f"read failed: {exc}"})
                continue
            parsed = parser.parse(text, f.relpath)
            ctx = _ModCtx(relpath=f.relpath, module=f.module, parsed=parsed,
                          node_id="")
            self.ctxs.append(ctx)
            self.ctx_by_relpath[f.relpath] = ctx
            if parsed.error:
                self.parse_errors.append(
                    {"file": f.relpath, "error": parsed.error})

    def _index_modules(self, files: list[ScannedFile]) -> None:
        # a package (a/__init__.py) wins over a same-named module (a.py),
        # mirroring Python's regular-package precedence
        for f in sorted(files, key=lambda x: x.relpath):
            if f.relpath.endswith("__init__.py"):
                self.module_index[f.module] = f.relpath
        for f in files:
            if not f.relpath.endswith("__init__.py"):
                self.module_index.setdefault(f.module, f.relpath)

    def _create_nodes(self, scan: ScanResult, files: list[ScannedFile]) -> None:
        G = self.graph
        repo = self.registry.register(repository_id, scan.root.name)
        G.add_node(repo, label=scan.root.name, type="repository",
                   extraction="scan")

        dirs: set[str] = set()
        for f in files:
            parts = f.relpath.split("/")
            for i in range(1, len(parts)):
                dirs.add("/".join(parts[:i]))
        dir_ids: dict[str, str] = {}
        for d in sorted(dirs):
            did = self.registry.register(directory_id(d), d)
            dir_ids[d] = did
            G.add_node(did, label=d, type="directory", source_file=d,
                       extraction="scan")
            # structural containment: repository → top dirs, dir → subdirs
            if "/" in d:
                parent = "/".join(d.split("/")[:-1])
                self._edge(dir_ids[parent], did, "contains", EXTRACTED,
                           source_file=d, location="")
            else:
                self._edge(repo, did, "contains", EXTRACTED,
                           source_file=d, location="")

        def container(relpath: str) -> str:
            if "/" not in relpath:
                return repo
            return dir_ids["/".join(relpath.split("/")[:-1])]

        for f in sorted(files, key=lambda x: x.relpath):
            fid = self.registry.register(file_id(f.relpath), f.relpath)
            is_test = is_test_path(f.relpath)
            ctx = self.ctx_by_relpath[f.relpath]
            attrs: dict[str, Any] = {
                "label": f.relpath, "type": "test" if is_test else "module",
                "source_file": f.relpath, "language": f.language,
                "file_type": "code", "extraction": "ast",
            }
            if ctx.parsed.entry_point:
                attrs["entry_point"] = True
            if ctx.parsed.error:
                attrs["parse_error"] = ctx.parsed.error
            G.add_node(fid, **attrs)
            ctx.node_id = fid
            self._edge(container(f.relpath), fid, "contains", EXTRACTED,
                       source_file=f.relpath, location="")
            self._index_symbols(ctx)

    def _index_symbols(self, ctx: _ModCtx) -> None:
        G = self.graph
        mod_id = ctx.node_id
        for d in ctx.parsed.definitions:
            raw = f"{mod_id}_{symbol_id_part(d.qualname)}"
            sid = self.registry.register(raw, f"{ctx.relpath}:{d.qualname}")
            ctx.ids[d.qualname] = sid
            label = d.name if d.kind != "method" else f".{d.name}()"
            attrs: dict[str, Any] = {
                "label": label, "type": d.kind, "source_file": ctx.relpath,
                "source_location": f"L{d.line}", "language": "python",
                "extraction": "ast",
            }
            if d.is_test:
                attrs["is_test"] = True
            G.add_node(sid, **attrs)
            self._edge(mod_id, sid, "defines", EXTRACTED,
                       source_file=ctx.relpath, location=f"L{d.line}")

            parts = d.qualname.split(".")
            if len(parts) == 1:
                self.top_symbols.setdefault(ctx.module, {})[d.name] = (
                    ctx.relpath, d.qualname)
            if d.kind == "class":
                self.class_members.setdefault((ctx.relpath, d.qualname), {})
                self.bases[(ctx.relpath, d.qualname)] = list(d.bases)
                self.classes_by_relpath.setdefault(ctx.relpath, []).append(
                    d.qualname)
            elif d.kind == "method" and len(parts) == 2:
                members = self.class_members.setdefault(
                    (ctx.relpath, parts[0]), {})
                members[d.name] = d.qualname

    # -- pass 2: resolution ------------------------------------------------------

    def _package_of(self, relpath: str) -> str:
        posix = relpath.replace("\\", "/")
        dirname = posix.rsplit("/", 1)[0] if "/" in posix else ""
        return module_name_for(dirname + "/__init__.py") if dirname else ""

    def _resolve_module_path(self, dotted: str) -> str | None:
        if not dotted:
            return None
        return self.module_index.get(dotted)

    def _target_module_for_import(self, ctx: _ModCtx, stmt_module: str,
                                  level: int) -> str | None:
        if level == 0:
            return self._resolve_module_path(stmt_module)
        base = self._package_of(ctx.relpath)
        parts = [p for p in base.split(".") if p]
        if level - 1 > len(parts):
            return None
        if level - 1:
            parts = parts[: len(parts) - (level - 1)]
        target = ".".join(parts + ([stmt_module] if stmt_module else []))
        return self._resolve_module_path(target)

    def _resolve_module_imports(self, ctx: _ModCtx) -> None:
        for stmt in ctx.parsed.imports:
            if stmt.level == 0 and not stmt.module:
                # `import a.b.c [as m]` — names carry the full dotted paths.
                # Without an alias, the bound name is the ROOT package (a),
                # matching Python's actual binding semantics.
                for imported, bound in stmt.names:
                    target = self._resolve_module_path(imported)
                    if target is not None:
                        self._add_import_edge(ctx, target, stmt.line)
                        if bound == imported.split(".")[0]:
                            ctx.bindings.setdefault(
                                bound, _Binding("module", bound))
                        else:
                            ctx.bindings.setdefault(
                                bound, _Binding("module", imported))
                    else:
                        self._record_import(ctx, imported, external=True)
                continue
            # from-import: resolve the base module once
            base_relpath = self._target_module_for_import(ctx, stmt.module,
                                                           stmt.level)
            for imported, bound in stmt.names:
                if imported == "*":
                    if base_relpath is not None:
                        self._add_import_edge(ctx, base_relpath, stmt.line)
                    continue
                if base_relpath is not None:
                    base_dotted = module_name_for(base_relpath)
                    if imported in self.top_symbols.get(base_dotted, {}):
                        # a top-level symbol of a repo module
                        self._add_import_edge(ctx, base_relpath, stmt.line)
                        ctx.bindings.setdefault(bound, _Binding(
                            "symbol", base_dotted, imported))
                        continue
                    if stmt.level == 0:
                        submodule = self._resolve_module_path(
                            f"{stmt.module}.{imported}")
                        if submodule is not None:
                            self._add_import_edge(ctx, submodule, stmt.line)
                            ctx.bindings.setdefault(bound, _Binding(
                                "module", f"{stmt.module}.{imported}"))
                            continue
                    # base resolves, the name does not: genuinely unresolved
                    self._add_import_edge(ctx, base_relpath, stmt.line)
                    self._record_import(ctx, f"{base_dotted}.{imported}",
                                        external=False)
                    continue
                # base module is not in the repo: external dependency
                dotted = (f"{stmt.module}.{imported}" if stmt.module
                          else imported)
                self._record_import(ctx, dotted, external=True)

    def _add_import_edge(self, ctx: _ModCtx, target_relpath: str,
                         line: int) -> None:
        if target_relpath == ctx.relpath:
            return
        ctx.imported.add(target_relpath)
        target_id = self.ctx_by_relpath[target_relpath].node_id
        self._edge(ctx.node_id, target_id, "imports", EXTRACTED,
                   source_file=ctx.relpath, location=f"L{line}")

    def _record_import(self, ctx: _ModCtx, dotted: str, *,
                       external: bool) -> None:
        node = self.graph.nodes[ctx.node_id]
        key = "imports_external" if external else "imports_unresolved"
        listed = node.get(key, [])
        if dotted not in listed and len(listed) < _MAX_LISTED:
            listed.append(dotted)
            node[key] = sorted(listed)
        if not external:
            self.unresolved_imports += 1
        else:
            self.external_imports.add(dotted.split(".")[0])

    def _enclosing_class(self, ctx: _ModCtx, scope: str) -> str:
        if not scope:
            return ""
        parts = scope.split(".")
        for i in range(len(parts), 0, -1):
            cand = ".".join(parts[:i])
            if (ctx.relpath, cand) in self.class_members:
                return cand
        return ""

    def _resolve_local_or_bound(self, ctx: _ModCtx, chain: list[str]
                                ) -> tuple[str, str] | str | None:
        """Resolve a chain rooted at a binding → (relpath, qualname),
        ``_EXTERNAL``, or None (unresolvable)."""
        binding = ctx.bindings.get(chain[0])
        if binding is None:
            return None
        if binding.kind == "local":
            qualname = binding.a
            if len(chain) == 1:
                return (ctx.relpath, qualname)
            if (ctx.relpath, qualname) in self.class_members:
                member = self.class_members[(ctx.relpath, qualname)].get(
                    chain[1])
                if member:
                    return (ctx.relpath, member)
            return None
        if binding.kind == "symbol":
            table = self.top_symbols.get(binding.a, {})
            hit = table.get(binding.b)
            if hit is None:
                return None
            relpath, qualname = hit
            if len(chain) == 1:
                return (relpath, qualname)
            if (relpath, qualname) in self.class_members:
                member = self.class_members[(relpath, qualname)].get(chain[1])
                if member:
                    return (relpath, member)
            return None
        # module binding: longest repo-module prefix, remainder = symbol path
        for i in range(len(chain), 0, -1):
            dotted = ".".join([binding.a] + chain[1:i])
            relpath = self._resolve_module_path(dotted)
            if relpath is None:
                continue
            rest = chain[i:]
            if not rest:
                return None  # a bare module reference, not a symbol
            mod_dotted = module_name_for(relpath)
            table = self.top_symbols.get(mod_dotted, {})
            if len(rest) == 1:
                return table.get(rest[0])  # type: ignore[return-value]
            if len(rest) == 2:
                cls = table.get(rest[0])
                if cls and (relpath, cls[1]) in self.class_members:
                    member = self.class_members[(relpath, cls[1])].get(rest[1])
                    if member:
                        return (relpath, member)
            return None
        if self._resolve_module_path(binding.a) is None:
            return _EXTERNAL  # bound to an outside-repo module
        return None

    def _resolve_via_bases(self, ctx: _ModCtx, class_qualname: str,
                           method: str, seen: set[tuple[str, str]] | None = None
                           ) -> list[tuple[str, str]]:
        """Repo classes in the base closure defining ``method``."""
        if seen is None:
            seen = set()
        if (ctx.relpath, class_qualname) in seen:
            return []
        seen.add((ctx.relpath, class_qualname))
        found: list[tuple[str, str]] = []
        for chain in self.bases.get((ctx.relpath, class_qualname), []):
            hit = self._resolve_local_or_bound(ctx, chain)
            if not isinstance(hit, tuple):
                continue
            b_relpath, b_qualname = hit
            members = self.class_members.get((b_relpath, b_qualname), {})
            if method in members:
                found.append((b_relpath, members[method]))
                continue
            b_ctx = self.ctx_by_relpath.get(b_relpath)
            if b_ctx is not None:
                found.extend(self._resolve_via_bases(b_ctx, b_qualname, method,
                                                     seen))
        return found

    def _resolve_chain(self, ctx: _ModCtx, chain: list[str], scope: str
                       ) -> tuple[tuple[str, str], str] | str | None:
        """Resolve a written chain → ((relpath, qualname), confidence),
        ``_EXTERNAL``, or None."""
        root = chain[0]
        cls_q = self._enclosing_class(ctx, scope)
        if root in ("self", "cls", "super"):
            if not cls_q or len(chain) < 2:
                return None
            if root != "super":
                members = self.class_members.get((ctx.relpath, cls_q), {})
                if chain[1] in members:
                    return ((ctx.relpath, members[chain[1]]), EXTRACTED)
            hits = self._resolve_via_bases(ctx, cls_q, chain[1])
            if len(hits) == 1:
                return (hits[0], EXTRACTED)
            if len(hits) > 1:
                return (sorted(hits)[0], INFERRED)
            return None
        hit = self._resolve_local_or_bound(ctx, chain)
        if isinstance(hit, tuple):
            return (hit, EXTRACTED)
        if hit == _EXTERNAL:
            return _EXTERNAL
        if root in _BUILTIN_NAMES:
            return _EXTERNAL
        return None

    def _origin_id(self, ctx: _ModCtx, scope: str) -> str:
        if not scope:
            return ctx.node_id
        return ctx.ids.get(scope, ctx.node_id)

    def _resolve_module(self, ctx: _ModCtx) -> None:
        if ctx.parsed.error:
            return
        # module-level definitions are bindings for same-module calls;
        # on a name collision with an import, the local definition wins
        # (deterministic approximation)
        for d in ctx.parsed.definitions:
            if "." not in d.qualname:
                ctx.bindings.setdefault(d.name, _Binding("local", d.qualname))
        self._resolve_module_imports(ctx)
        for call in ctx.parsed.calls:
            resolved = self._resolve_chain(ctx, call.chain, call.scope)
            if not isinstance(resolved, tuple):
                if resolved == _EXTERNAL or call.chain[0] in _BUILTIN_NAMES:
                    continue
                self.unresolved_calls += 1
                continue
            hit, confidence = resolved
            sid = self._symbol_node_id(hit)
            if sid is None:
                self.unresolved_calls += 1
                continue
            self._edge(self._origin_id(ctx, call.scope), sid, "calls",
                       confidence, source_file=ctx.relpath,
                       location=f"L{call.line}")
        for use in ctx.parsed.name_uses:
            resolved = self._resolve_chain(ctx, use.chain, use.scope)
            if not isinstance(resolved, tuple):
                if resolved == _EXTERNAL:
                    continue
                self.unresolved_uses += 1
                continue
            hit, confidence = resolved
            sid = self._symbol_node_id(hit)
            if sid is None:
                self.unresolved_uses += 1
                continue
            self._edge(self._origin_id(ctx, use.scope), sid, "references",
                       confidence, source_file=ctx.relpath,
                       location=f"L{use.line}")
        for cls_q in sorted(self.classes_by_relpath.get(ctx.relpath, [])):
            for chain in self.bases.get((ctx.relpath, cls_q), []):
                hit = self._resolve_local_or_bound(ctx, chain)
                if not isinstance(hit, tuple):
                    self.unresolved_bases += 1
                    continue
                sid = self._symbol_node_id(hit)
                if sid is None:
                    self.unresolved_bases += 1
                    continue
                self._edge(ctx.ids.get(cls_q, ctx.node_id), sid, "inherits",
                           EXTRACTED, source_file=ctx.relpath, location="")
        if is_test_path(ctx.relpath):
            self._emit_tests_edges(ctx)

    def _emit_tests_edges(self, ctx: _ModCtx) -> None:
        tested = {t for t in ctx.imported if not is_test_path(t)}
        for t in sorted(tested):
            tid = self.ctx_by_relpath[t].node_id
            self._edge(ctx.node_id, tid, "tests", EXTRACTED,
                       source_file=ctx.relpath, location="",
                       basis="test module imports")
        if tested:
            return
        # naming-convention fallback: tests/test_<name>.py ↔ <name>.py
        base = ctx.relpath.rsplit("/", 1)[-1]
        if not (base.startswith("test_") and base.endswith(".py")):
            return
        stem = base[len("test_"): -len(".py")]
        if not stem:
            return
        own_dir = ctx.relpath.rsplit("/", 1)[0] if "/" in ctx.relpath else ""
        for cand in (f"{stem}.py",
                     f"{own_dir}/{stem}.py" if own_dir else None):
            if cand and cand in self.ctx_by_relpath:
                self._edge(ctx.node_id, self.ctx_by_relpath[cand].node_id,
                           "tests", INFERRED, source_file=ctx.relpath,
                           location="",
                           basis="naming convention (test_<name> ↔ <name>.py)")
                return

    def _symbol_node_id(self, hit: tuple[str, str]) -> str | None:
        relpath, qualname = hit
        ctx = self.ctx_by_relpath.get(relpath)
        if ctx is None:
            return None
        return ctx.ids.get(qualname)

    # -- shared helpers -----------------------------------------------------------

    def _edge(self, source: str, target: str, relation: str,
              confidence: str, *, source_file: str, location: str,
              basis: str = "") -> None:
        key = (source, target, relation)
        existing = self.edges.get(key)
        if existing is None:
            self.edges[key] = CodeEdge(
                source=source, target=target, relation=relation,
                confidence=confidence, source_file=source_file,
                source_location=location, basis=basis,
                locations=[location] if location else [])
            return
        existing.merge_location(location, confidence)

    def _count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            counts[d.get("type", "?")] = counts.get(d.get("type", "?"), 0) + 1
        return dict(sorted(counts.items()))

    def _count_by_relation(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            counts[d["relation"]] = counts.get(d["relation"], 0) + 1
        return dict(sorted(counts.items()))


def _parser_for(language: str) -> LanguageParser | None:
    from intentgraph.codegraph.python_parser import PARSERS
    return PARSERS.get(language)


def build_graph(options: BuildOptions) -> CodeBuild:
    """Scan and resolve ``options.root`` into a typed, evidence-backed graph."""
    if not isinstance(options.root, Path):
        options = replace(options, root=Path(options.root))
    return _Builder(options).run()
