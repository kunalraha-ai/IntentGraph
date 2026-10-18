"""IntentGraph's own MCP server — the four memory tools over stdio.

Previously the `intentgraph-mcp` entry point targeted the platform graph
server (which registers IntentGraph's memory tools alongside its graph
tools). This module is the IntentGraph-native server: a slim, focused
process that registers exactly the four memory tools
(``remember``/``recall``/``context``/``feedback``) with the usage
instructions from :data:`intentgraph.mcp.SERVER_INSTRUCTIONS`, and speaks
the MCP stdio transport.

Design notes:

* **Same tool implementations** — handlers delegate to
  :mod:`intentgraph.mcp` exactly as the graph-server registration did, so
  tool behavior, validation, and project-scoped archives are identical and
  covered by the same tests.
* **mcp 1.x / 2.x dual-compat**, mirroring the registration strategy used
  elsewhere in this repo: the 1.x decorator API when present, otherwise the
  2.x ``on_*`` constructor callbacks.
* **A missing/corrupt archive never terminates the process** — tools
  report per-call errors (isError), keeping the server alive for other
  projects, the same policy the shared graph server follows.
"""
from __future__ import annotations

import sys

from intentgraph import mcp as _tools

__all__ = ["_build_server", "main", "serve"]


def _build_server():
    """Build the configured low-level MCP Server (used by main + tests)."""
    try:
        from mcp.server import Server
        from mcp import types
    except ImportError as e:
        raise ImportError(
            'mcp not installed. Run: pip install "graphifyy[mcp]"'
        ) from e
    server = Server("intentgraph", instructions=_tools.SERVER_INSTRUCTIONS)
    sessions = _tools.IntentGraphSessions()

    class _ToolError(Exception):
        """Tool-signalled failure → isError result (never a process crash)."""

    def _bind(fn):
        def handler(arguments: dict) -> str:
            try:
                return fn(sessions, arguments)
            except _tools.MCPToolError as exc:
                raise _ToolError(str(exc)) from exc
        return handler

    handlers = {
        "remember": _bind(_tools.tool_remember),
        "recall": _bind(_tools.tool_recall),
        "context": _bind(_tools.tool_context),
        "feedback": _bind(_tools.tool_feedback),
    }

    if hasattr(Server, "list_tools"):
        # mcp 1.x: decorator-based registration
        async def list_tools() -> list:
            from mcp import types as _t
            return [_t.Tool(name=s["name"], description=s["description"],
                            inputSchema=s["inputSchema"])
                    for s in _tools.tool_schemas()]

        async def call_tool(name: str, arguments: dict):
            handler = handlers.get(name)
            if not handler:
                # unknown tool is an isError result, never a silent text
                # answer that a client could mistake for success
                raise _ToolError(f"Unknown tool: {name}")
            try:
                return [types.TextContent(type="text", text=handler(arguments))]
            except _ToolError:
                raise
            except Exception as exc:  # noqa: BLE001 — never kill the server
                return [types.TextContent(
                    type="text", text=f"Error executing {name}: {exc}")]

        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
    else:
        # mcp 2.x: constructor callbacks with the (ctx, params) contract
        async def _on_list_tools(ctx, params):
            from mcp import types as _t
            return _t.ListToolsResult(tools=[
                _t.Tool(name=s["name"], description=s["description"],
                        inputSchema=s["inputSchema"])
                for s in _tools.tool_schemas()])

        async def _on_call_tool(ctx, params):
            name = params.name
            handler = handlers.get(name)
            if not handler:
                return types.CallToolResult(
                    content=[types.TextContent(
                        type="text", text=f"Unknown tool: {name}")],
                    isError=True)
            try:
                from typing import Any as _Any
                content: _Any = [types.TextContent(
                    type="text", text=handler(dict(params.arguments or {})))]
                return types.CallToolResult(content=content)
            except _ToolError as exc:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(exc))],
                    isError=True)
            except Exception as exc:  # noqa: BLE001 — never kill the server
                return types.CallToolResult(
                    content=[types.TextContent(
                        type="text", text=f"Error executing {name}: {exc}")],
                    isError=True)

        from typing import Any as _Any
        _server_cls: _Any = Server
        server = _server_cls("intentgraph",
                              instructions=_tools.SERVER_INSTRUCTIONS,
                              on_list_tools=_on_list_tools,
                              on_call_tool=_on_call_tool)

    # keep sessions reachable for tests
    server.intentgraph_sessions = sessions  # type: ignore[attr-defined]
    return server


def serve() -> None:
    """Start the MCP stdio server (blocking).

    The loop is the standard mcp-server shape: build the low-level Server,
    hand it the stdio streams, run until EOF. Blank-line stdin filtering is
    skipped — that guard exists for the graph server's CLI-invoked contexts;
    this entry point is only ever launched as an MCP stdio process.
    """
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as e:
        raise ImportError(
            'mcp not installed. Run: pip install "graphifyy[mcp]"') from e
    import asyncio

    server = _build_server()

    async def _main() -> None:
        async with stdio_server() as streams:
            await server.run(streams[0], streams[1],
                             server.create_initialization_options())

    asyncio.run(_main())


def main(argv=None) -> None:
    """Console entry point (``intentgraph-mcp``).

    No arguments are required. A positional path, if passed, is accepted
    and ignored for backward compatibility with configs that pointed the
    alias at the shared graph server with a graph path argument.
    """
    serve()


if __name__ == "__main__":  # pragma: no cover
    main()
