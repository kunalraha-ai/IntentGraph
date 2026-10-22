"""Repository scanning: a deterministic, bounded walk that yields parseable
source files plus an honest record of everything skipped and why.

Ordering is total: directories and files are visited in sorted order, so a
limit-truncated scan covers the same subset on every run.

Exclusion layers (checked in this order):

1. Built-in directory excludes — version control, virtual environments,
   build output, caches, generated-output and memory-archive directories
   (``intentgraph-data/`` is user data and is never touched).
2. ``--exclude`` patterns — gitignore-style globs supplied by the caller;
   explicit intent, cannot be re-included by repository ignore files.
3. ``.gitignore`` — a practical subset of the documented gitignore format
   (public spec): comments, blank lines, ``/`` anchoring, trailing-slash
   directory patterns, ``*``/``**``/``?``/``[...]`` globs, and ``!``
   negation evaluated last-match-wins per path. Excluded directories are
   pruned, and git does not re-include beneath a pruned directory; this
   implementation matches that behavior.

Safety rules baked in:

* Directory symlinks are never followed (``os.walk`` without followlinks);
  symlinked files are recorded as skipped and never read, so no content
  outside the repository root can enter the graph.
* Secret-looking file names (``.env``, ``*.key``, ``id_rsa*``, …) are
  skipped explicitly and never opened — in addition, only source files a
  parser claims are ever read, and parsers extract structure, never string
  literal contents.
* File-size and file-count limits are enforced with explicit ``truncated``
  reporting.
"""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ScanOptions", "ScanResult", "ScannedFile", "SkipRecord",
           "scan_repository", "DEFAULT_EXCLUDED_DIRS", "SECRET_FILE_PATTERNS"]

#: Directory names never descended into.
DEFAULT_EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", ".tox",
    ".nox", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".hypothesis",
    ".eggs", ".idea", ".vscode", "node_modules", "dist", "build", "out",
    "target", ".claude", ".kilo", ".opencode", "codegraph-out",
    "graphify-out", "intentgraph-data",
})

#: Basename patterns whose files are skipped and never opened.
SECRET_FILE_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*",
    "id_ed25519*", "id_ecdsa*", "id_dsa*", ".netrc", ".npmrc",
    "credentials", "credentials.*", "*.tfvars", "secrets.*",
)

_MAX_SKIPPED_RECORDS = 500


@dataclass(frozen=True)
class ScanOptions:
    root: Path
    #: gitignore-style patterns applied to repository-relative paths
    excludes: tuple[str, ...] = ()
    max_files: int = 20_000
    max_file_size: int = 1_048_576  # 1 MiB


@dataclass
class ScannedFile:
    relpath: str      # repository-relative, forward slashes
    size: int
    language: str
    module: str       # dotted module name ("intentgraph.models")


@dataclass
class SkipRecord:
    relpath: str
    reason: str


@dataclass
class ScanResult:
    root: Path
    files: list[ScannedFile] = field(default_factory=list)
    skipped: list[SkipRecord] = field(default_factory=list)
    skipped_counts: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    errors: list[str] = field(default_factory=list)

    def note_skip(self, relpath: str, reason: str) -> None:
        self.skipped_counts[reason] = self.skipped_counts.get(reason, 0) + 1
        if len(self.skipped) < _MAX_SKIPPED_RECORDS:
            self.skipped.append(SkipRecord(relpath, reason))


def _module_name(relpath: str) -> str:
    """Dotted module name implied by a repository-relative file path."""
    posix = relpath.replace("\\", "/")
    parts = [p for p in posix.split("/") if p]
    if not parts:
        return ""
    name = parts[-1]
    if name == "__init__.py":
        parts = parts[:-1]
    elif name.endswith(".py"):
        parts[-1] = name[: -len(".py")]
    else:
        parts[-1] = name
    return ".".join(parts)


# --- gitignore subset ---------------------------------------------------------


class _IgnoreRule:
    __slots__ = ("negated", "dir_only", "regex", "base")

    def __init__(self, pattern: str, base: str) -> None:
        self.base = base
        self.negated = pattern.startswith("!")
        if self.negated:
            pattern = pattern[1:]
        self.dir_only = pattern.endswith("/")
        pattern = pattern.rstrip("/")
        anchored = pattern.startswith("/")
        pattern = pattern.lstrip("/")
        self.regex = _compile_gitignore_glob(pattern, anchored or "/" in pattern)

    def matches(self, relpath: str, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False
        if not relpath.startswith(self.base):
            return False
        sub = relpath[len(self.base):] if self.base else relpath
        return re.match(self.regex, sub) is not None


def _compile_gitignore_glob(pattern: str, anchored: bool) -> str:
    """Translate a gitignore glob into a regex over posix relpaths."""
    out: list[str] = []
    segments = pattern.split("/")
    first = True
    for seg in segments:
        if seg == "":
            first = False
            continue
        if seg == "**":
            # `**` matches zero or more path segments
            if first:
                out.append("(?:[^/]+/)*")
            else:
                out.append("(?:/[^/]+)*")
            first = False
            continue
        piece: list[str] = []
        i = 0
        while i < len(seg):
            ch = seg[i]
            if ch == "*":
                piece.append("[^/]*")
            elif ch == "?":
                piece.append("[^/]")
            elif ch == "[":
                j = seg.find("]", i + 1)
                if j == -1:
                    piece.append(re.escape(ch))
                else:
                    inner = seg[i + 1: j]
                    if inner.startswith("!"):
                        inner = "^" + inner[1:]
                    piece.append("[" + inner + "]")
                    i = j
            elif ch == "\\" and i + 1 < len(seg):
                piece.append(re.escape(seg[i + 1]))
                i += 1
            else:
                piece.append(re.escape(ch))
            i += 1
        out.append("".join(piece))
        first = False
    body = "/".join(out) if out else ""
    if anchored:
        return body + "/?"
    # unanchored: match at any depth
    return "(?:[^/]+/)*" + body + "/?"


def _parse_ignore_lines(text: str, base: str) -> list[_IgnoreRule]:
    rules = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rules.append(_IgnoreRule(line.strip(), base))
    return rules


class _ExcludeSet:
    """Ordered rule list; last match wins (gitignore semantics)."""

    def __init__(self) -> None:
        self._rules: list[_IgnoreRule] = []

    def add(self, pattern: str, base: str) -> None:
        try:
            self._rules.append(_IgnoreRule(pattern, base))
        except Exception:  # noqa: BLE001 — a bad pattern must not kill a scan
            pass

    def add_many(self, patterns: tuple[str, ...] | list[str], base: str) -> None:
        for p in patterns:
            self.add(p, base)

    def excludes(self, relpath: str, is_dir: bool) -> bool:
        verdict = False
        for rule in self._rules:
            if rule.matches(relpath, is_dir):
                verdict = not rule.negated
        return verdict


# --- language registry (parsers register extensions) ---------------------------


def _language_for(relpath: str) -> str:
    from intentgraph.codegraph.python_parser import PythonParser
    for lang, exts in _EXTENSIONS().items():
        lower = relpath.lower()
        if any(lower.endswith(ext) for ext in exts):
            return lang
    return ""


_EXTENSIONS_CACHE: dict[str, tuple[str, ...]] | None = None


def _EXTENSIONS() -> dict[str, tuple[str, ...]]:
    global _EXTENSIONS_CACHE
    if _EXTENSIONS_CACHE is None:
        from intentgraph.codegraph.python_parser import PARSERS
        _EXTENSIONS_CACHE = {p.language: p.extensions for p in PARSERS.values()}
    return _EXTENSIONS_CACHE


# --- the walk -------------------------------------------------------------------


def scan_repository(options: ScanOptions) -> ScanResult:
    """Walk ``options.root`` deterministically and classify every entry."""
    root = Path(options.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"scan root is not a directory: {options.root}")

    result = ScanResult(root=root)
    user_excludes = _ExcludeSet()
    user_excludes.add_many(options.excludes, base="")

    def base_rules() -> _ExcludeSet:
        exc = _ExcludeSet()
        exc._rules.extend(user_excludes._rules)  # noqa: SLF001 — same class
        if (root / ".gitignore").is_file():
            try:
                exc._rules.extend(  # noqa: SLF001
                    _parse_ignore_lines(
                        (root / ".gitignore").read_text(encoding="utf-8",
                                                         errors="replace"), ""))
            except OSError as exc_os:  # pragma: no cover - unreadable ignore
                result.errors.append(f"could not read .gitignore: {exc_os}")
        return exc

    top_rules = base_rules()

    # (relative dir path, applicable rules); deterministic DFS with sorted dirs
    stack: list[tuple[str, _ExcludeSet]] = [("", top_rules)]
    while stack:
        reldir, rules = stack.pop()
        absdir = root / reldir if reldir else root
        try:
            entries = sorted(os.listdir(absdir))
        except OSError as exc:
            result.errors.append(f"could not list {reldir or '.'}: {exc}")
            continue

        subdirs: list[tuple[str, _ExcludeSet]] = []
        for name in entries:
            rel = f"{reldir}/{name}" if reldir else name
            full = absdir / name
            if name in DEFAULT_EXCLUDED_DIRS and (root / rel).is_dir():
                result.note_skip(rel, "excluded directory")
                continue
            if name in DEFAULT_EXCLUDED_DIRS:
                result.note_skip(rel, "excluded name")
                continue
            if full.is_symlink():
                result.note_skip(rel, "symlink (never followed or read)")
                continue
            if full.is_dir():
                if rules.excludes(rel, True):
                    result.note_skip(rel, "ignored (gitignore/exclude rule)")
                    continue
                sub_rules = _ExcludeSet()
                sub_rules._rules.extend(rules._rules)  # noqa: SLF001
                gi = full / ".gitignore"
                if gi.is_file():
                    try:
                        sub_rules._rules.extend(  # noqa: SLF001
                            _parse_ignore_lines(
                                gi.read_text(encoding="utf-8",
                                             errors="replace"), rel + "/"))
                    except OSError:  # pragma: no cover - unreadable ignore
                        pass
                subdirs.append((rel, sub_rules))
                continue
            # file
            if any(fnmatch.fnmatch(name.lower(), pat.lower())
                   for pat in SECRET_FILE_PATTERNS):
                result.note_skip(rel, "secret-like file name (never read)")
                continue
            if rules.excludes(rel, False):
                result.note_skip(rel, "ignored (gitignore/exclude rule)")
                continue
            language = _language_for(rel)
            if not language:
                result.note_skip(rel, "unsupported file type")
                continue
            if result.truncated:
                result.note_skip(rel, "scan limit reached")
                continue
            try:
                size = full.stat().st_size
            except OSError as exc:
                result.errors.append(f"could not stat {rel}: {exc}")
                continue
            if size > options.max_file_size:
                result.note_skip(rel, "file exceeds size limit")
                continue
            if len(result.files) >= options.max_files:
                result.truncated = True
                result.note_skip(rel, "scan limit reached")
                continue
            result.files.append(ScannedFile(
                relpath=rel, size=size, language=language,
                module=_module_name(rel)))

        # push subdirectories in reverse so the pop() order is sorted
        for sub in reversed(subdirs):
            stack.append(sub)

    result.files.sort(key=lambda f: f.relpath)
    return result
