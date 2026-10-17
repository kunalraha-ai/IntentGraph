"""IntentGraph configuration: defaults < config file < environment.

Resolution order (later wins):

1. Built-in defaults — local-first, zero-network, deterministic.
2. Config file — JSON at ``$INTENTGRAPH_CONFIG``, else
   ``<data_dir>/intentgraph.json`` if present.
3. Environment variables (see :data:`ENV_OVERRIDES`).

Privacy is structural, not aspirational: with the default config, no
embedding/LLM provider is configured, and nothing in this package makes a
network call. A provider only exists if the user explicitly configured one,
and ``privacy.local_only=true`` hard-refuses any provider with a non-local
endpoint so a copy-pasted config can never quietly leak memories off-box.
What would leave the machine is exactly: record text sent to the configured
embedding endpoint (and nothing else — no ids, no provenance, no metadata).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_DATA_DIR", "Config", "load_config", "config_to_dict",
           "load_env_file"]

DEFAULT_DATA_DIR = "intentgraph-data"

#: env var → dotted config path
ENV_OVERRIDES: dict[str, str] = {
    "INTENTGRAPH_HOME": "data_dir",
    "INTENTGRAPH_EMBEDDING_PROVIDER": "embedding.provider",
    "INTENTGRAPH_EMBEDDING_BASE_URL": "embedding.base_url",
    "INTENTGRAPH_EMBEDDING_MODEL": "embedding.model",
    "INTENTGRAPH_EMBEDDING_DIM": "embedding.dim",
    "INTENTGRAPH_EMBEDDING_BATCH": "embedding.batch",
    "INTENTGRAPH_EMBEDDING_TIMEOUT": "embedding.timeout_s",
    "INTENTGRAPH_LLM_PROVIDER": "llm.provider",
    "INTENTGRAPH_LLM_MODEL": "llm.model",
    "INTENTGRAPH_LLM_BASE_URL": "llm.base_url",
    "INTENTGRAPH_RERANKER_PROVIDER": "reranker.provider",
    "INTENTGRAPH_RERANKER_MODEL": "reranker.model",
    "INTENTGRAPH_RERANKER_TOP_N": "reranker.top_n",
    "INTENTGRAPH_WEB_SEARCH_PROVIDER": "web_search.provider",
    "INTENTGRAPH_WEB_SEARCH_ENABLED": "web_search.enabled",
    "INTENTGRAPH_WEB_SEARCH_MAX_RESULTS": "web_search.max_results",
    "INTENTGRAPH_WEB_SEARCH_TIMEOUT": "web_search.timeout_s",
    "INTENTGRAPH_GRAPHIFY_GRAPH": "graphify.graph_path",
    "INTENTGRAPH_RETRIEVAL_MAX_ITERATIONS": "retrieval.max_iterations",
    "INTENTGRAPH_RETRIEVAL_BUDGET": "retrieval.context_budget_tokens",
    "INTENTGRAPH_API_HOST": "api.host",
    "INTENTGRAPH_API_PORT": "api.port",
    "INTENTGRAPH_API_KEY": "api.api_key",
    "INTENTGRAPH_LOCAL_ONLY": "privacy.local_only",
}

# env values that need coercion beyond strings
_ENV_BOOL = {"INTENTGRAPH_LOCAL_ONLY", "INTENTGRAPH_WEB_SEARCH_ENABLED"}
_ENV_INT = {
    "INTENTGRAPH_EMBEDDING_DIM", "INTENTGRAPH_EMBEDDING_BATCH",
    "INTENTGRAPH_EMBEDDING_TIMEOUT", "INTENTGRAPH_RETRIEVAL_MAX_ITERATIONS",
    "INTENTGRAPH_RETRIEVAL_BUDGET", "INTENTGRAPH_API_PORT",
    "INTENTGRAPH_WEB_SEARCH_MAX_RESULTS", "INTENTGRAPH_WEB_SEARCH_TIMEOUT",
}


@dataclass
class Config:
    """Resolved configuration for one IntentGraph instance."""

    data_dir: str = DEFAULT_DATA_DIR
    #: semantic provider settings — provider "none"|"hash"|"nebius"|"openai"
    embedding: dict[str, Any] = field(default_factory=lambda: {
        "provider": "none",
    })
    #: optional LLM-assisted planning (query decomposition + action
    #: selection) — provider "none"|"nebius"|"openai"; deterministic policy
    #: is the default and always the fallback
    llm: dict[str, Any] = field(default_factory=lambda: {
        "provider": "none",
    })
    #: optional remote reranking — provider "none"|"nebius"; model must be
    #: explicitly configured (account-dependent availability)
    reranker: dict[str, Any] = field(default_factory=lambda: {
        "provider": "none",
        "top_n": 10,
    })
    #: optional external web search — provider "none"|"tavily"; off by
    #: default; store_results persists findings as memory records ONLY
    #: when explicitly true
    web_search: dict[str, Any] = field(default_factory=lambda: {
        "provider": "none",
        "enabled": False,
        "max_results": 5,
        "max_calls_per_query": 1,
        "search_depth": "basic",
        "topic": "general",
        "time_range": "",
        "include_domains": [],
        "exclude_domains": [],
        "timeout_s": 15,
        "retries": 2,
        "content_max_chars": 2000,
        "store_results": False,
    })
    #: optional Graphify graph.json to bridge code structure into retrieval
    graphify: dict[str, Any] = field(default_factory=lambda: {
        "graph_path": "",   # e.g. graphify-out/graph.json
        "hop_depth": 2,
    })
    retrieval: dict[str, Any] = field(default_factory=lambda: {
        "mode": "adaptive",
        "max_iterations": 3,
        "max_working_set": 24,
        "max_candidates": 50,
        "graph_depth": 2,
        "context_budget_tokens": 4000,
        "timeout_s": 30.0,
        "weights": {"exact": 1.4, "lexical": 1.0, "semantic": 1.0, "graph": 0.7},
    })
    api: dict[str, Any] = field(default_factory=lambda: {
        "host": "127.0.0.1",
        "port": 8710,
        "api_key": "",       # required for non-loopback binds
    })
    privacy: dict[str, Any] = field(default_factory=lambda: {
        "local_only": False,
    })

    # -- derived paths --------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "memory.db"

    @property
    def config_path(self) -> Path:
        return Path(self.data_dir) / "intentgraph.json"

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    # -- validation ----------------------------------------------------------

    def enforce_privacy(self) -> None:
        """Refuse ANY configured network provider under local_only.

        Raised before any provider is constructed — the earlier a privacy
        violation is refused, the fewer code paths can forget to check.
        Covers all four provider classes that would send text off-machine:
        embeddings, LLM planning, reranking, and web search. Localhost
        endpoints are allowed (an Ollama-style local server is local).
        """
        if not self.privacy.get("local_only"):
            return

        def _is_remote(base: str) -> bool:
            base = str(base or "")
            return bool(base) and "localhost" not in base \
                and "127.0.0.1" not in base

        emb = str(self.embedding.get("provider", "none"))
        # any remote embedding provider (nebius/nvidia/openai) is refused
        # under local_only unless pointed at a loopback endpoint — the
        # provider list will keep growing, so the rule is "remote unless
        # none/hash/local", not an enumeration
        if emb not in ("none", "null", "", "hash") and \
                (_is_remote(self.embedding.get("base_url", "")) or emb != "openai"):
            raise PermissionError(
                "privacy.local_only is set but an external embedding endpoint "
                f"is configured ({emb}). Remove the endpoint or unset "
                "local_only.")
        llm = str(self.llm.get("provider", "none"))
        if llm != "none":
            raise PermissionError(
                "privacy.local_only is set but an llm provider is configured "
                f"({llm}) — planning assistance would send text off-machine. "
                "Remove the provider or unset local_only.")
        rr = str(self.reranker.get("provider", "none"))
        if rr != "none":
            raise PermissionError(
                "privacy.local_only is set but a reranker is configured "
                f"({rr}) — reranking would send record excerpts off-machine. "
                "Remove the provider or unset local_only.")
        ws = str(self.web_search.get("provider", "none"))
        if ws != "none" and self.web_search.get("enabled"):
            raise PermissionError(
                "privacy.local_only is set but web search is enabled "
                f"({ws}) — search would send the query off-machine. "
                "Remove the provider or unset local_only.")

    def describe(self) -> str:
        lines = [
            f"data_dir: {self.data_dir}",
            f"embedding.provider: {self.embedding.get('provider', 'none')}",
            f"llm.provider: {self.llm.get('provider', 'none')}",
            f"reranker.provider: {self.reranker.get('provider', 'none')}",
            f"web_search: {self.web_search.get('provider', 'none')} "
            f"(enabled={self.web_search.get('enabled', False)})",
            f"graphify.graph_path: {self.graphify.get('graph_path') or '(none)'}",
            f"retrieval.mode: {self.retrieval.get('mode', 'adaptive')}",
            f"privacy.local_only: {self.privacy.get('local_only', False)}",
        ]
        return "\n".join(lines)


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


# --- dotenv (.env) support ------------------------------------------------------------
#
# A deliberate, minimal, stdlib-only loader — NOT python-dotenv:
#   * looks in: $INTENTGRAPH_ENV_FILE, then ./.env, then each parent directory
#     (≤6 levels) so one file at the workspace root serves the repo, the CLI,
#     and the OpenCode MCP server regardless of process CWD;
#   * sets ONLY variables not already present in the environment — real env
#     always wins, so CI/shell overrides stay authoritative;
#   * supports KEY=VALUE, quoted values, comments; never evaluates anything;
#   * fully disableable (INTENTGRAPH_DISABLE_DOTENV=1), which the offline
#     test suite does (tests/conftest.py) to stay credential-hermetic: the
#     default suite must neither require keys nor make network calls, even
#     on a developer machine with a populated .env.


def _parse_env_file(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # strip one matched pair of quotes — values with '#' or spaces are
        # commonly quoted; keys are validated below
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key.replace("_", "A").replace("-", "B").isalnum():
            out[key] = value
    return out


def _env_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("INTENTGRAPH_ENV_FILE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    here = Path.cwd()
    for depth, parent in enumerate([here, *here.parents]):
        if depth > 6:
            break
        candidates.append(parent / ".env")
    return candidates


def load_env_file() -> int:
    """Populate unset environment variables from the first .env found.

    Returns the number of variables applied. No-op when disabled
    (``INTENTGRAPH_DISABLE_DOTENV=1``) or when no candidate file exists.
    Keys already in the environment are never overwritten.
    """
    if os.environ.get("INTENTGRAPH_DISABLE_DOTENV", "").strip() == "1":
        return 0
    for candidate in _env_file_candidates():
        try:
            if not candidate.is_file():
                continue
            parsed = _parse_env_file(candidate.read_text(encoding="utf-8"))
        except OSError:
            continue
        applied = 0
        for key, value in parsed.items():
            if key not in os.environ:
                os.environ[key] = value
                applied += 1
        return applied
    return 0


def config_to_dict(config: Config) -> dict[str, Any]:
    return {
        "data_dir": config.data_dir,
        "embedding": dict(config.embedding),
        "llm": dict(config.llm),
        "reranker": dict(config.reranker),
        "web_search": dict(config.web_search),
        "graphify": dict(config.graphify),
        "retrieval": dict(config.retrieval),
        "api": dict(config.api),
        "privacy": dict(config.privacy),
    }


def load_config(
    *,
    data_dir: str | Path | None = None,
    config_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Resolve a :class:`Config` from defaults, file, and environment.

    ``data_dir`` (arg or ``INTENTGRAPH_HOME``) decides where the archive
    lives; the config file is looked up inside it unless an explicit path is
    given, so a memory archive carries its own settings when moved.
    """
    # .env is the FIRST resolution step: credentials and explicit provider
    # opt-ins live there (never in the tracked config file), and real
    # environment variables still win because load_env_file only fills
    # variables that are unset. One call site — every surface (CLI, SDK,
    # MCP tools, HTTP API) resolves configuration through load_config.
    load_env_file()

    cfg_dict = config_to_dict(Config())

    if data_dir is None:
        data_dir = os.environ.get("INTENTGRAPH_HOME", DEFAULT_DATA_DIR)
    cfg_dict["data_dir"] = str(data_dir)

    # config file: explicit path > $INTENTGRAPH_CONFIG > <data_dir>/intentgraph.json
    cf: Path | None = None
    if config_file is not None:
        cf = Path(config_file)
    elif os.environ.get("INTENTGRAPH_CONFIG"):
        cf = Path(os.environ["INTENTGRAPH_CONFIG"])
    else:
        candidate = Path(cfg_dict["data_dir"]) / "intentgraph.json"
        cf = candidate if candidate.exists() else None
    if cf is not None and cf.exists():
        try:
            loaded = json.loads(cf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read config file {cf}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"config file {cf} must contain a JSON object")
        # comment keys ("//") keep the example config self-documenting;
        # strip them so they never leak into provider constructors
        def _strip_comments(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: _strip_comments(v) for k, v in node.items()
                        if not str(k).startswith("//")}
            return node

        loaded = _strip_comments(loaded)
        for key, value in loaded.items():
            if key in cfg_dict and isinstance(cfg_dict[key], dict) and isinstance(value, dict):
                cfg_dict[key].update(value)
            else:
                cfg_dict[key] = value

    # environment overrides
    for env_name, dotted in ENV_OVERRIDES.items():
        if env_name in os.environ:
            raw = os.environ[env_name]
            if env_name in _ENV_BOOL:
                value = raw.strip().lower() in ("1", "true", "yes", "on")
            elif env_name in _ENV_INT:
                try:
                    value = int(raw)
                except ValueError:
                    continue  # malformed override ignored, never fatal on boot
            else:
                value = raw
            _set_dotted(cfg_dict, dotted, value)

    # caller overrides last (the SDK/API pass explicit options)
    valid_top = {"data_dir", "embedding", "llm", "reranker", "web_search",
                 "graphify", "retrieval", "api", "privacy"}
    for dotted, value in (overrides or {}).items():
        top = dotted.split(".")[0]
        if top not in valid_top:
            raise ValueError(
                f"unknown config override: {dotted!r} "
                f"(valid sections: {sorted(valid_top)})")
        _set_dotted(cfg_dict, dotted, value)

    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    cfg.enforce_privacy()
    return cfg
