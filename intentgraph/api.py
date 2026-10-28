"""IntentGraph local-first JSON HTTP API (stdlib http.server, no new deps).

Why stdlib http.server instead of a framework: the local deployment must
work with `pip install intentgraph` alone — no starlette, no uvicorn, no ASGI
server. A threaded stdlib server comfortably serves the local one-consumer /
small-team pattern IntentGraph targets, and a future cloud deployment can
front the SAME core (intentgraph.sdk.IntentGraph) with any framework without
touching this module's contract: the URL shapes below are the product.

Endpoints:

    GET    /health                      liveness + archive stats
    POST   /v1/memories                 ingest {content, type?, tags?, ...}
    GET    /v1/memories                 list (pagination, filters)
    GET    /v1/memories/{id}            original record (full content)
    GET    /v1/memories/{id}/versions   version history
    DELETE /v1/memories/{id}            soft delete
    POST   /v1/memories/{id}/restore    undo soft delete
    POST   /v1/memories/{id}/update     append a new version {content, note?}
    POST   /v1/links                    create a typed relationship
    POST   /v1/retrieve                 {query, mode?, top_k?} → candidates
    POST   /v1/context                   {query, budget_tokens?} → evidence packet
    POST   /v1/feedback                  {query, outcome, memory_ids?, note?}
    POST   /v1/query                   {query, top_k?, ...} → memories + code + links
    GET    /v1/code-links               list memory ↔ code links (filters)
    GET    /v1/status                   full archive status
    GET    /graph/memory-code           web UI: combined knowledge graph (default at /)
    GET    /graph/memory                web UI: memory graph
    GET    /graph/code                  web UI: code graph
    GET    /                            web UI: the memory-code knowledge graph

Security posture (untrusted content, untrusted callers):

* Body size capped (``MAX_BODY_BYTES``) — the DoS is reading the body, so
  the cap is enforced before the body is buffered.
* Auth: optional bearer token. REQUIRED automatically when binding to a
  non-loopback host — a LAN-exposed memory archive with no auth is a data
  breach waiting to happen, not a convenience.
* Errors are structured JSON with correct status codes; internal exceptions
  never leak stack traces to the client.
* Every response carries ``Cache-Control: no-store`` — memory content is
  not cacheable intermediary material.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from intentgraph.config import Config, load_config
from intentgraph.sdk import IntentGraph
from intentgraph.store import StoreError

__all__ = ["serve", "make_server", "MAX_BODY_BYTES", "IntentGraphAPIHandler"]

MAX_BODY_BYTES = 4 * 1024 * 1024  # 4 MiB: one record caps at 2MB of content
_RECORD_ID_RE = re.compile(r"^mem_[0-9a-f]{16}$")


class _API:
    """Holds the singleton IntentGraph instance behind the server."""

    def __init__(self, config: Config):
        self.ig = IntentGraph(config=config)
        self.lock = threading.Lock()
        # IntentGraph operations are not individually thread-safe (SQLite
        # connection sharing); the lock serializes write-heavy endpoints
        # while reads stay fast in practice for the local pattern.


def _clean_path_id(raw: str) -> str | None:
    raw = raw.strip("/")
    if _RECORD_ID_RE.match(raw):
        return raw
    return None


class IntentGraphAPIHandler(BaseHTTPRequestHandler):
    server_version = "IntentGraphAPI/1"
    protocol_version = "HTTP/1.1"

    # --- plumbing -------------------------------------------------------------

    @property
    def api(self) -> _API:
        return self.server.api  # type: ignore[attr-defined]

    # --- web UI: graph views (dynamically generated) --------------------------

    #: the view-switcher dropdown injected at the top of the graph sidebar
    _VIEW_SWITCHER = """<div style="padding: 12px; border-bottom: 1px solid #2a2a4e">
  <select onchange="location.href='/graph/'+this.value" style="width:100%; background:#0f0f1a; border:1px solid #3a3a5e; color:#e0e0e0; padding:7px 10px; border-radius:6px; font-size:13px; outline:none; cursor:pointer">
    <option value="memory-code"{s_mc}>Memory + Code graph</option>
    <option value="memory"{s_m}>Memory graph</option>
    <option value="code"{s_c}>Code graph</option>
  </select>
</div>"""

    @classmethod
    def _inject_view_switcher(cls, html: str, current: str) -> str:
        """Inject the view-switcher dropdown as the first sidebar element,
        plus the Evidence Replay engine (ask bar, playback controls, voice
        narration, zoom/glow animation)."""
        sel = {"memory-code": "", "memory": "", "code": ""}
        sel[current] = " selected"
        switcher = cls._VIEW_SWITCHER.format(
            s_mc=sel["memory-code"], s_m=sel["memory"], s_c=sel["code"])
        html = html.replace('<div id="sidebar">',
                            '<div id="sidebar">' + switcher, 1)
        # the Evidence Replay engine
        import pathlib
        replay_js = pathlib.Path(__file__).resolve().parent / "frontend" / "replay.js"
        if replay_js.exists():
            engine = "<script>\n" + replay_js.read_text(encoding="utf-8") + \
                     "\n</script>"
            html = html.replace("</body>", engine + "</body>", 1)
        return html

    def _graph_authorized(self) -> bool:
        """Graph views accept both the normal header auth and ``?key=`` —
        the view-switcher dropdown navigates via ``location.href``, which
        cannot set headers."""
        if self._authorized():
            return True
        required = getattr(self.server, "api_key", "")
        if not required:
            return True
        from urllib.parse import unquote_plus
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        return unquote_plus(params.get("key", "")) == required

    def _serve_graph(self, view: str) -> None:
        """Generate and serve the interactive knowledge graph for one view,
        with the view-switcher dropdown injected into the sidebar."""
        import tempfile
        ig = self.api.ig
        fd, tmp = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        try:
            if view == "memory":
                from intentgraph.viz import write_memory_html
                write_memory_html(ig, tmp)
            elif view in ("memory-code", ""):
                from intentgraph.viz import write_memory_html
                write_memory_html(ig, tmp, with_code=True)
                view = "memory-code"
            elif view == "code":
                graph_path = ig.config.graphify.get("graph_path") or ""
                if not graph_path or not Path(graph_path).exists():
                    self._error(404, "no code graph — build one with "
                                 "`intentgraph codegraph build .`")
                    return
                from intentgraph.codegraph.analyze import load_graph
                from intentgraph.codegraph.cli import _CODE_LEXICON
                from intentgraph.codegraph.export import \
                    communities_by_directory
                from intentgraph.renderer import to_html
                G, _, _ = load_graph(graph_path)
                communities, labels = communities_by_directory(G)
                to_html(G, communities, tmp, community_labels=labels,
                        lexicon=_CODE_LEXICON)
            else:
                self._error(404, f"unknown graph view: {view}")
                return
            html = Path(tmp).read_text(encoding="utf-8")
        finally:
            Path(tmp).unlink(missing_ok=True)
        html = self._inject_view_switcher(html, view)
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet default logs
        pass

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        # When the request body was never read (e.g. a 413), the client is
        # still mid-send: without draining the socket, it hits a broken pipe
        # and never receives the response at all. Drain bounded chunks (never
        # buffering more than 64k in memory) up to a sane window, then close.
        if not getattr(self, "_body_read", False):
            try:
                remaining = int(self.headers.get("Content-Length") or 0)
                drain_cap = 64 * 1024 * 1024  # 64 MiB drain window
                while remaining > 0 and drain_cap > 0:
                    chunk = self.rfile.read(min(65536, remaining, drain_cap))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    drain_cap -= len(chunk)
            except OSError:
                pass
            self.close_connection = True
        self._send_json(status, {"error": message})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            # Bodyless POST is legal: routes that need fields validate their
            # own required keys; routes like restore take none.
            self._body_read = True
            return {}
        if length > MAX_BODY_BYTES:
            # do NOT set _body_read: _error() must know the client is still
            # mid-send so it drains the socket before responding (else the
            # client hits a broken pipe and never sees the 413 at all)
            raise _ClientError(413, f"body too large (cap {MAX_BODY_BYTES} bytes)")
        raw = self.rfile.read(length)
        self._body_read = True
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _ClientError(400, "body must be valid UTF-8 JSON")
        if not isinstance(data, dict):
            raise _ClientError(400, "body must be a JSON object")
        return data

    def _authorized(self) -> bool:
        required = getattr(self.server, "api_key", "")
        if not required:
            return True
        header = self.headers.get("Authorization") or self.headers.get("X-API-Key") or ""
        token = header.removeprefix("Bearer ").strip()
        return token == required

    # --- routing ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 — stdlib naming
        self._body_read = False  # per-request state (keep-alive safe)
        path = self.path.split("?", 1)[0]

        # web UI: the knowledge graph (memory-code by default at /)
        m = re.match(r"^/graph/(memory-code|memory|code)$", path)
        if path == "/" or m:
            view = m.group(1) if m else "memory-code"
            if not self._graph_authorized():
                self._error(401, "missing or invalid API key")
                return
            with self.api.lock:
                self._serve_graph(view)
            return

        if not self._authorized():
            self._error(401, "missing or invalid API key")
            return
        if path == "/health":
            with self.api.lock:
                self._send_json(200, {"status": "ok", "service": "intentgraph",
                                      "archive": _safe_stats(self.api.ig)})
            return
        if path == "/v1/status":
            with self.api.lock:
                status = self.api.ig.status()
            status.pop("lexical_index", None)  # index internals, not API data
            self._send_json(200, status)
            return
        if path == "/v1/code-links":
            qs = dict(pair.split("=", 1) for pair in
                      (self.path.split("?", 1)[1].split("&") if "?" in self.path else [])
                      if "=" in pair)
            from urllib.parse import unquote_plus

            link_type = unquote_plus(qs.get("type", "")) or None
            if link_type not in (None, "explicit", "extracted", "candidate"):
                self._error(400, "type must be explicit, extracted, or candidate")
                return
            with self.api.lock:
                links = self.api.ig.store.iter_code_links(
                    record_id=unquote_plus(qs.get("record", "")) or None,
                    node_id=unquote_plus(qs.get("node", "")) or None,
                    link_type=link_type)
            self._send_json(200, {"count": len(links), "links": links})
            return
        m = re.match(r"^/v1/memories/([^/]+)$", path)
        if m:
            rid = _clean_path_id(m.group(1))
            if rid is None:
                self._error(400, "malformed record id (expected mem_<16hex>)")
                return
            rec = self.api.ig.get(rid)
            if rec is None:
                self._error(404, f"no such record: {rid}")
                return
            self._send_json(200, rec)
            return
        m = re.match(r"^/v1/memories/([^/]+)/versions$", path)
        if m:
            rid = _clean_path_id(m.group(1))
            if rid is None:
                self._error(400, "malformed record id")
                return
            self._send_json(200, {"id": rid, "versions": self.api.ig.get_versions(rid)})
            return
        if path == "/v1/memories":
            qs = dict(pair.split("=", 1) for pair in
                      (self.path.split("?", 1)[1].split("&") if "?" in self.path else [])
                      if "=" in pair)
            from urllib.parse import unquote_plus

            records, total = self.api.ig.list(
                type=unquote_plus(qs.get("type", "")) or None,
                tag=unquote_plus(qs.get("tag", "")) or None,
                status=unquote_plus(qs.get("status", "")) or None,
                limit=min(200, max(1, int(qs.get("limit", "20") or 20))),
                offset=max(0, int(qs.get("offset", "0") or 0)))
            self._send_json(200, {"total": total, "records": records})
            return
        self._error(404, f"no such endpoint: GET {path}")

    def do_DELETE(self) -> None:  # noqa: N802
        self._body_read = False
        if not self._authorized():
            self._error(401, "missing or invalid API key")
            return
        m = re.match(r"^/v1/memories/([^/]+)$", self.path.split("?")[0])
        if not m:
            self._error(404, f"no such endpoint: DELETE {self.path}")
            return
        rid = _clean_path_id(m.group(1))
        if rid is None:
            self._error(400, "malformed record id")
            return
        with self.api.lock:
            deleted = self.api.ig.delete(rid)
        if not deleted:
            self._error(404, f"no such record: {rid}")
            return
        self._send_json(200, deleted)

    def do_POST(self) -> None:  # noqa: N802
        self._body_read = False
        if not self._authorized():
            self._error(401, "missing or invalid API key")
            return
        path = self.path.split("?", 1)[0]
        try:
            data = self._body()
            with self.api.lock:
                payload = self._post_route(path, data)
            self._send_json(200, payload)
        except _ClientError as exc:
            self._error(exc.status, exc.message)
        except StoreError as exc:
            self._error(400, str(exc))
        except FileNotFoundError as exc:
            self._error(404, str(exc))
        except Exception as exc:  # noqa: BLE001 — never leak internals
            self._error(500, f"internal error: {type(exc).__name__}")

    def _post_route(self, path: str, data: dict[str, Any]) -> Any:
        ig = self.api.ig
        if path == "/v1/query":
            # cross-graph retrieval: memories + code entities + verified links
            query = str(data.get("query") or "").strip()
            if not query:
                raise _ClientError(400, "query is required")
            try:
                top_k = min(50, max(1, int(data.get("top_k") or 12)))
                code_top_k = min(50, max(0, int(data.get("code_top_k") or 8)))
                depth = min(3, max(0, int(data.get("depth") or 1)))
                budget = data.get("budget_tokens")
                budget_tokens = min(16_000, max(256, int(budget))) \
                    if budget else None
            except (TypeError, ValueError):
                raise _ClientError(400, "top_k/code_top_k/depth/budget_tokens "
                                        "must be integers")
            return ig.query(query, top_k=top_k, code_top_k=code_top_k,
                            depth=depth, budget_tokens=budget_tokens,
                            include_candidates=bool(data.get("include_candidates")))
        if path == "/v1/memories":
            content = data.get("content")
            if not isinstance(content, str) or not content.strip():
                raise _ClientError(400, "'content' must be a non-empty string")
            res = ig.ingest(
                content,
                type=data.get("type"),
                tags=data.get("tags") or None,
                source_type=data.get("source_type") or "api",
                source_id=data.get("source_id") or "",
                event_time=data.get("event_time"),
                metadata=data.get("metadata") or None,
                graph_anchors=data.get("graph_anchors") or None,
                provenance=data.get("provenance") or {"via": "http api"},
                allow_duplicate=bool(data.get("allow_duplicate")),
            )
            return res.to_dict()

        m = re.match(r"^/v1/memories/([^/]+)/update$", path)
        if m:
            rid = _clean_path_id(m.group(1))
            if rid is None:
                raise _ClientError(400, "malformed record id")
            content = data.get("content")
            if not isinstance(content, str):
                raise _ClientError(400, "'content' must be a string")
            return ig.update(rid, content=content, note=str(data.get("note") or ""))

        m = re.match(r"^/v1/memories/([^/]+)/restore$", path)
        if m:
            rid = _clean_path_id(m.group(1))
            if rid is None:
                raise _ClientError(400, "malformed record id")
            restored = ig.restore(rid)
            if not restored:
                raise _ClientError(404, f"no such record: {rid}")
            return restored

        if path == "/v1/links":
            source, target, relation = (data.get("source"), data.get("target"),
                                        data.get("relation"))
            if not source or not target or not relation:
                raise _ClientError(400, "'source', 'target', 'relation' required")
            return ig.link(str(source), str(target), str(relation),
                           explicit=not data.get("inferred", False),
                           basis=str(data.get("basis") or ""),
                           metadata=data.get("metadata") or None)

        if path == "/v1/retrieve":
            query = data.get("query")
            if not isinstance(query, str) or not query.strip():
                raise _ClientError(400, "'query' must be a non-empty string")
            mode = str(data.get("mode") or "hybrid")
            if mode not in ("lexical", "semantic", "graph", "hybrid"):
                raise _ClientError(400, f"unknown mode: {mode!r}")
            return ig.retrieve(query, mode=mode,
                               top_k=int(data.get("top_k") or 20),
                               include_inactive=bool(data.get("include_inactive")))

        if path == "/v1/context":
            query = data.get("query")
            if not isinstance(query, str) or not query.strip():
                raise _ClientError(400, "'query' must be a non-empty string")
            budget = data.get("budget_tokens")
            return ig.remember(query,
                               budget_tokens=int(budget) if budget else None)

        if path == "/v1/feedback":
            query = data.get("query") or ""
            outcome = str(data.get("outcome") or "")
            if outcome not in ("useful", "missing", "incorrect"):
                raise _ClientError(400, "'outcome' must be useful|missing|incorrect")
            return ig.feedback(query, outcome,
                               memory_ids=data.get("memory_ids") or None,
                               note=str(data.get("note") or ""))

        if path == "/v1/replay":
            # evidence replay: a step-by-step navigation script from a
            # REAL retrieval — the cinematic graph experience
            query = str(data.get("query") or "").strip()
            if not query:
                raise _ClientError(400, "query is required")
            from intentgraph.replay import generate_replay
            return generate_replay(
                ig, query,
                top_k=min(5, max(1, int(data.get("top_k") or 3))),
                code_top_k=min(3, max(0, int(data.get("code_top_k") or 2))))

        if path == "/v1/export":
            path_arg = data.get("path")
            if not isinstance(path_arg, str) or not path_arg:
                raise _ClientError(400, "'path' required")
            # An HTTP caller must not aim export/import at arbitrary
            # filesystem locations (overwrite a dotfile, exfiltrate /etc).
            # Constrain bundles to the archive's data directory — backups
            # live next to the memory they back up.
            target = _within_data_dir(path_arg, self.api.ig.config.data_dir)
            return ig.export(str(target))

        if path == "/v1/import":
            path_arg = data.get("path")
            if not isinstance(path_arg, str) or not path_arg:
                raise _ClientError(400, "'path' required")
            target = _within_data_dir(path_arg, self.api.ig.config.data_dir)
            if not target.exists():
                raise _ClientError(404, f"no such bundle: {target}")
            return ig.import_(str(target))

        raise _ClientError(404, f"no such endpoint: POST {path}")


class _ClientError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


def _safe_stats(ig: IntentGraph) -> dict[str, Any]:
    try:
        s = ig.status()
        return {k: s[k] for k in ("records", "records_active", "edges",
                                   "versions") if k in s}
    except Exception:  # noqa: BLE001 — health must never 500
        return {"records": -1}


class _IntentGraphServer(ThreadingHTTPServer):
    """Typed view of the dynamically-configured server attributes."""

    daemon_threads = True
    api_key: str = ""
    api: "_API"


def _within_data_dir(raw: str, data_dir: str) -> Path:
    """Constrain an export/import path to the archive's data directory.

    Relative paths resolve against the data dir; absolute or ``..``-escaping
    paths must already sit inside it. Anything else is a 400 — an HTTP caller
    must not be able to point the server at arbitrary filesystem locations
    (this is the API equivalent of the CLI trusting only the local shell user).
    """
    from pathlib import Path

    base = Path(data_dir).resolve()
    candidate = (base / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise _ClientError(400, "path must stay inside the data directory "
                                f"({data_dir}); use a relative name for backups")
    return candidate


def make_server(host: str, port: int, api_key: str, config: Config) -> _IntentGraphServer:
    """Build a configured, not-yet-serving HTTP server (used by serve + tests)."""
    server = _IntentGraphServer((host, port), IntentGraphAPIHandler)
    server.api_key = api_key
    server.api = _API(config)
    return server


def serve(*, overrides: dict[str, Any] | None = None, host_override: str | None = None,
          port_override: int | None = None, api_key_override: str | None = None,
          config: Config | None = None) -> int:
    """Start the blocking HTTP API server. Returns an exit code (never raises)."""
    cfg = config or load_config(overrides=overrides or None)
    host = host_override or cfg.api.get("host", "127.0.0.1")
    port = int(port_override or cfg.api.get("port", 8710))
    api_key = api_key_override or cfg.api.get("api_key") or ""

    is_loopback = host in ("127.0.0.1", "localhost", "::1")
    if not is_loopback and not api_key:
        print("error: refusing to bind a non-loopback host without an API key. "
              "Pass --api-key or set INTENTGRAPH_API_KEY.", file=sys.stderr)
        return 2

    server = make_server(host, port, api_key, cfg)
    print(f"IntentGraph API listening on http://{host}:{port} "
          f"(auth {'required' if api_key else 'off (loopback)'})")
    print(f"memory archive: {cfg.data_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        server.server_close()
        server.api.ig.close()
    return 0
