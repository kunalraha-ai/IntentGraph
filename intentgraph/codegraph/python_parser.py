"""Language parsers for the codegraph engine.

``PythonParser`` extracts structure from Python source with the standard
library ``ast`` module — modules, classes, functions, methods, imports,
statically visible calls and name uses, base classes, decorators, and the
``if __name__ == "__main__":`` entry-point guard. It never executes analyzed
code, never evaluates anything, and never extracts string-literal contents
(only identifiers and import paths — so file content beyond symbol names
cannot end up in graph labels).

The ``LanguageParser`` protocol is the extension point: a new language is a
class with a ``language`` name, its ``extensions``, and a ``parse`` returning
the same ``ParsedModule`` shape. The engine (scan → build → analyze → export)
never imports a parser directly — it goes through ``PARSERS``, so adding a
language does not touch the engine.

Resolution honesty: everything collected here is a *fact about the source
text* (an import statement exists at a line, a call chain is written here).
Whether a chain resolves to a repo entity is decided by ``build.py``; when
it cannot, no edge is invented.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["Definition", "ImportStmt", "CallSite", "NameUse", "ParsedModule",
           "LanguageParser", "PythonParser", "PARSERS", "module_name_for"]


@dataclass
class Definition:
    kind: str            # "class" | "function" | "method"
    name: str
    qualname: str        # dotted within the module
    line: int
    end_line: int
    bases: list[list[str]] = field(default_factory=list)   # source-level chains
    is_test: bool = False


@dataclass
class ImportStmt:
    module: str          # dotted module as written ("" for `from . import x`)
    level: int           # relative-import depth (0 = absolute)
    names: list[tuple[str, str]] = field(default_factory=list)
    # names: (imported path or symbol, locally bound name). For plain
    # `import a.b` the binding is the root name "a".
    line: int = 0


@dataclass
class CallSite:
    chain: list[str]     # dotted chain as written: ["self", "run"], ["Service"], ...
    line: int
    scope: str          # qualname of the enclosing symbol ("" = module level)


@dataclass
class NameUse:
    chain: list[str]
    line: int
    scope: str


@dataclass
class ParsedModule:
    relpath: str
    module: str          # dotted module name
    language: str
    definitions: list[Definition] = field(default_factory=list)
    imports: list[ImportStmt] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    name_uses: list[NameUse] = field(default_factory=list)
    entry_point: bool = False
    error: str | None = None


@runtime_checkable
class LanguageParser(Protocol):
    language: str
    extensions: tuple[str, ...]

    def parse(self, text: str, relpath: str) -> ParsedModule: ...


def module_name_for(relpath: str) -> str:
    """Dotted module name implied by a repository-relative path."""
    posix = relpath.replace("\\", "/")
    parts = [p for p in posix.split("/") if p]
    if not parts:
        return ""
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def is_test_path(relpath: str) -> bool:
    """Test-module convention: basename ``test_*.py`` or ``*_test.py``."""
    base = relpath.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base.startswith("test_") or (base.endswith("_test.py") and len(base) > len("_test.py"))


class PythonParser:
    """Structure extraction over Python's standard AST — no execution."""

    language = "python"
    extensions: tuple[str, ...] = (".py",)

    def parse(self, text: str, relpath: str) -> ParsedModule:
        module = module_name_for(relpath)
        out = ParsedModule(relpath=relpath, module=module, language=self.language)
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            out.error = f"SyntaxError: {exc.msg} (line {exc.lineno or 0})"
            return out
        except (ValueError, RecursionError) as exc:  # null bytes, deep nesting
            out.error = f"{type(exc).__name__}: {exc}"
            return out
        v = _Visitor(out)
        v.visit_module(tree)
        return out


class _Visitor:
    """Scope-tracking walker. Scope entries are ``(name, kind)`` pairs so a
    definition knows whether its parent is a class (→ method) or a
    function/module (→ function)."""

    def __init__(self, out: ParsedModule) -> None:
        self.out = out

    # -- module ----------------------------------------------------------------

    def visit_module(self, tree: ast.Module) -> None:
        self._body(tree.body, scope=[])
        for stmt in tree.body:
            if isinstance(stmt, ast.If) and self._is_main_guard(stmt.test):
                self.out.entry_point = True

    @staticmethod
    def _is_main_guard(test: ast.expr) -> bool:
        if not isinstance(test, ast.Compare):
            return False
        name_ok = (isinstance(test.left, ast.Name) and test.left.id == "__name__"
                   and any(isinstance(c, ast.Constant) and c.value == "__main__"
                           for c in test.comparators))
        rev_ok = (any(isinstance(c, ast.Constant) and c.value == "__main__"
                      for c in test.comparators)
                  and isinstance(test.left, ast.Constant)
                  and test.left.value == "__main__")
        return bool(name_ok or rev_ok)

    # -- statements --------------------------------------------------------------

    def _body(self, body: list[ast.stmt], scope: list[tuple[str, str]]) -> None:
        for stmt in body:
            self._stmt(stmt, scope)

    def _stmt(self, node: ast.stmt, scope: list[tuple[str, str]]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._def(node, scope)
            return
        if isinstance(node, ast.ClassDef):
            self._class(node, scope)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self.out.imports.append(self._import(node))
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self._stmt(child, scope)
            elif isinstance(child, ast.expr):
                self._expr(child, self._qual(scope))

    @staticmethod
    def _qual(scope: list[tuple[str, str]]) -> str:
        return ".".join(name for name, _ in scope)

    def _def(self, node: ast.FunctionDef | ast.AsyncFunctionDef,
             scope: list[tuple[str, str]]) -> None:
        qualname = ".".join([n for n, _ in scope] + [node.name])
        parent_is_class = bool(scope) and scope[-1][1] == "class"
        kind = "method" if parent_is_class else "function"
        self.out.definitions.append(Definition(
            kind=kind, name=node.name, qualname=qualname, line=node.lineno,
            end_line=getattr(node, "end_lineno", None) or node.lineno,
            is_test=node.name.startswith("test_")))
        # signature references (annotations/defaults) + decorators are
        # attributed to the defined symbol — they are part of its definition
        for dec in node.decorator_list:
            self._expr(dec, qualname)
        a = node.args
        for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
            if arg.annotation:
                self._expr(arg.annotation, qualname)
        for arg in (a.vararg, a.kwarg):
            if arg is not None and arg.annotation:
                self._expr(arg.annotation, qualname)
        for default in list(a.defaults) + [d for d in a.kw_defaults if d]:
            self._expr(default, qualname)
        if node.returns is not None:
            self._expr(node.returns, qualname)
        self._body(node.body, scope + [(node.name, "function")])

    def _class(self, node: ast.ClassDef, scope: list[tuple[str, str]]) -> None:
        qualname = ".".join([n for n, _ in scope] + [node.name])
        bases: list[list[str]] = []
        for base in list(node.bases) + [kw.value for kw in node.keywords]:
            expr = base.func if isinstance(base, ast.Call) else base
            chain = self._chain_of(expr)
            if chain:
                bases.append(chain)
            else:  # pragma: no cover - exotic base expression
                self._expr(base, qualname)
        self.out.definitions.append(Definition(
            kind="class", name=node.name, qualname=qualname, line=node.lineno,
            end_line=getattr(node, "end_lineno", None) or node.lineno,
            bases=bases))
        for dec in node.decorator_list:
            self._expr(dec, qualname)
        self._body(node.body, scope + [(node.name, "class")])

    def _import(self, node: ast.Import | ast.ImportFrom) -> ImportStmt:
        if isinstance(node, ast.Import):
            names = [(a.name, a.asname or a.name.split(".")[0])
                     for a in node.names]
            return ImportStmt(module="", level=0, names=names, line=node.lineno)
        module = node.module or ""
        names = [(a.name, a.asname or a.name) for a in node.names]
        return ImportStmt(module=module, level=node.level or 0, names=names,
                          line=node.lineno)

    # -- expressions ---------------------------------------------------------------

    @staticmethod
    def _chain_of(e: ast.expr) -> list[str] | None:
        parts: list[str] = []
        cur: ast.expr = e
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return list(reversed(parts))
        return None

    def _expr(self, e: ast.expr, scope: str) -> None:
        if isinstance(e, ast.Call):
            chain = self._chain_of(e.func)
            if chain:
                self.out.calls.append(CallSite(chain=chain, line=e.lineno,
                                              scope=scope))
            else:
                self._expr(e.func, scope)
            for arg in e.args:
                self._expr(arg, scope)
            for kw in e.keywords:
                self._expr(kw.value, scope)
            return
        if isinstance(e, ast.Attribute):
            if isinstance(e.ctx, ast.Load):
                chain = self._chain_of(e)
                if chain:
                    self.out.name_uses.append(NameUse(chain=chain, line=e.lineno,
                                                      scope=scope))
                    return
            self._generic(e, scope)
            return
        if isinstance(e, ast.Name):
            if isinstance(e.ctx, ast.Load):
                self.out.name_uses.append(NameUse(chain=[e.id], line=e.lineno,
                                                   scope=scope))
            return
        self._generic(e, scope)

    def _generic(self, e: ast.expr, scope: str) -> None:
        # comprehensions: `elt`/`key`/`value` live on the comprehension
        # expression itself; the `comprehension` clause carries iter/ifs
        if isinstance(e, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            self._expr(e.elt, scope)
            for gen in e.generators:
                self._comp(gen, scope)
            return
        if isinstance(e, ast.DictComp):
            self._expr(e.key, scope)
            self._expr(e.value, scope)
            for gen in e.generators:
                self._comp(gen, scope)
            return
        for child in ast.iter_child_nodes(e):
            if isinstance(child, ast.expr):
                self._expr(child, scope)
            elif isinstance(child, ast.comprehension):
                self._comp(child, scope)

    def _comp(self, gen: ast.comprehension, scope: str) -> None:
        self._expr(gen.iter, scope)
        for cond in gen.ifs:
            self._expr(cond, scope)


#: Registry — the engine's only parser entry point. Adding a language = adding
#: an entry here plus (optionally) scan-time file-type support.
PARSERS: dict[str, LanguageParser] = {"python": PythonParser()}
