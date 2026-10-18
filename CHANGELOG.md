# Changelog

## 0.1.0 (2026-10-18) — first standalone release

IntentGraph ships as its own product: persistent memory and retrieval intelligence for AI agents. One package (`intentgraph/`), MIT-licensed, local-first — the default configuration makes zero network calls.

**What's in the box**

- **Memory archive** — SQLite store with append-only version history, soft delete, tombstones, export/import; typed edges (`derived_from`, `caused_by`, `validated_by`, `supersedes`, `contradicts`, …), each explicit or inferred with its recorded basis; MinHash near-duplicate linking (deterministic across processes); auto-links for record ids mentioned in content.
- **Retrieval** — exact-id, BM25 lexical (identifier-aware tokenization), optional semantic (remote embeddings), and graph expansion over typed memory edges, fused with weighted RRF; plus the adaptive engine: question → evidence-requirements analysis → iterative gap-targeted retrieval → verification → token-budgeted evidence packet with an explicit `sufficient`/`partial`/`insufficient` verdict. Contradictions surface both sides; superseded records keep their successor named; missing evidence is reported, never fabricated.
- **Four surfaces** — SDK, CLI (`intentgraph`), local HTTP API (stdlib-only), and MCP agent tools (`intentgraph-mcp`: `remember`, `recall`, `context`, `feedback` — with server-injected usage instructions, project-scoped archives, per-call errors that never kill the process, mcp 1.x/2.x dual-compat).
- **Providers (all opt-in, credentials via gitignored `.env`)** — Nebius Token Factory and NVIDIA NIM embeddings behind one client (model discovery, dimension validation, stale-model detection forcing reindex, vector spaces keyed by provider+model+endpoint and never mixed), an LLM planning-policy interface with deterministic fallback, a reranker interface, and selective Tavily web search whose results carry full provenance and are never user memories. `privacy.local_only: true` refuses every remote provider class at configuration load.
- **Visualization** — the real archive as a self-contained interactive `graph.html`: memory-type communities, evidence panel, retrieval highlight/dim/reset driven by a REAL retrieve() call (cyan = direct, violet = graph-expanded), honest empty states, `</`-escaped JSON and sanitized fields at the renderer boundary.
- **Code-graph bridge (data-only)** — records with `graph_anchors` link into a `graph.json` file when one exists, so recall can follow code structure; bounded, deterministic, degrades to "no code-adjacency" honestly.
- **Benchmarks** — deterministic, reproducible, honestly labeled (`intentgraph benchmark`, `--agent`, `--providers`): pipeline recall, fresh-session recovery, selectivity, and provider latency.

**Provenance**

IntentGraph was developed in five phases (memory core, MCP integration, remote providers, visualization, engine decoupling) inside a fork of Graphify, then decoupled to zero engine imports and split out. This repository's earlier revisions distributed that Apache-2.0-licensed graph engine; it has been removed from the tree — see NOTICE for the provenance statement and git history for the record.
