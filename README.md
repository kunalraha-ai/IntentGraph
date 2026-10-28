# IntentGraph

**Persistent memory and retrieval intelligence for AI agents.**

IntentGraph gives an AI system a memory that survives the session: it ingests what happened — decisions and their rationale, failed experiments, test outcomes, external evidence — connects those memories with typed, auditable relationships, and then, when asked, retrieves the *evidence* behind an answer instead of a plausible guess.

The centerpiece is the **evidence packet**: a token-budgeted, source-referenced context containing the retrieved memories with their provenance, the relationships between them, any contradictions or superseded records surfaced (never silently resolved), and an explicit verdict — `sufficient`, `partial`, or `insufficient`. When the archive does not answer the question, IntentGraph says so instead of inventing.

```
uv sync
.venv/bin/python -m intentgraph ingest --text "ADR-14: amounts are stored as integer minor units — the gateway API is integer-only." --type decision
.venv/bin/python -m intentgraph remember "what did we decide about order amounts, and why?"
.venv/bin/python -m intentgraph viz --query "checkout regression"   # evidence highlighted on the graph
```

---

## Why IntentGraph

An agent that cannot remember repeats its failures. A RAG index that only finds similar text cannot tell you *what was decided*, *what already failed and why*, or *which fact is superseded*. IntentGraph is built for exactly those questions:

- **Evidence, not vibes.** Every retrieved memory carries its stable record id, source, timestamps, and retrieval method. A similarity score is a retrieval fact, never a truth claim.
- **Honest failure.** Contradictions surface both sides with provenance; superseded records stay visible with their successor named; missing evidence is reported as a gap rather than fabricated.
- **Local-first.** The default configuration makes zero network calls. Lexical retrieval, graph traversal, evidence verification, and the adaptive engine are fully deterministic and need no model, no key, and no cloud. Remote providers are strictly opt-in.
- **Agent-native.** Four MCP tools (`intentgraph_remember`, `intentgraph_recall`, `intentgraph_context`, `intentgraph_feedback`) plug into coding agents; a local HTTP API serves everything else.

## How it works

```
ingest → archive (SQLite) → hybrid retrieval → verification → evidence packet
              │                   │                  │
         typed memory        exact + BM25        provenance, validity,
         graph edges      + optional semantic     temporal consistency,
         (explicit /      + graph expansion      contradictions, coverage,
          inferred)       fused with RRF          honest verdict
```

1. **Analyze** the question into evidence requirements (what kinds of evidence would a complete answer need?).
2. **Retrieve** a broad round — exact identifiers, BM25, optional semantic — and fuse with weighted Reciprocal Rank Fusion that never drops single-source evidence.
3. **Expand** through the memory graph's high-signal typed edges (`caused_by`, `failed_because_of`, `validated_by`, `supersedes`, …), bounded by depth and fan-out.
4. **Verify**: provenance, validity (supersession with successor pointer), temporal consistency, explicit-vs-asserted support, declared contradictions, near-duplicate grouping, topic-anchored requirement coverage.
5. **Target the gaps**: unmet requirements trigger follow-up retrieval sharpened toward exactly the missing evidence shape — never a blind neighborhood crawl. Stop on coverage, iteration cap, no-progress, or budget exhaustion.
6. **Compile** a token-budgeted packet: every evidence entry cited with id, provenance, retrieval methods, and status; contradictions, limitations, and unresolved questions included.

## The memory model

| Concept | What it is |
|---|---|
| **Record** | The original memory, stored verbatim: stable `mem_*` id, content hash, type (`decision`/`fact`/`experience`/`action`/`conversation`/…), source + provenance, event time and ingestion time, tags, validity window |
| **Versioning** | Updates append a new version and snapshot the previous — history is always readable; the store is append-only |
| **Typed edges** | `related_to`, `derived_from`, `caused_by`, `contradicts`, `supersedes`, `depends_on`, `occurred_before/after`, `validated_by`, `failed_because_of`, `belongs_to`, `duplicate_of`, `mentions` — each explicit or inferred **with its recorded basis** |
| **Auto-links** | Content referencing other record ids links automatically; MinHash near-duplicates are linked, never silently merged |
| **Deletion** | Soft delete (reversible) by default; explicit purge leaves a tombstone; `supersede` flips validity and records the edge in one atomic operation |

## Retrieval modes

| Mode | What runs |
|---|---|
| `lexical` | BM25 (identifier-aware tokenization, light stemming) — deterministic, always available |
| `semantic` | cosine over embeddings — opt-in through a provider; falls back with an honest note when unconfigured |
| `graph` | exact-id seeds + typed-edge expansion (bounded depth/fan-out, relation-weighted) |
| `hybrid` | exact + lexical + semantic fused with RRF, then graph expansion of the top seeds |
| `adaptive` | the full engine above: iterative, gap-targeted, verified, with explicit stopping conditions |

## Agent tools (MCP)

```jsonc
// .opencode/opencode.json — OpenCode setup (V2)
{
  "mcp": {
    "servers": {
      "intentgraph": {
        "type": "local",
        "command": ["/absolute/path/to/your/IntentGraph/clone/.venv/bin/intentgraph-mcp"],
        "codemode": false
      }
    }
  }
}
```

| Tool | What it does |
|---|---|
| `intentgraph_remember` | Save a decision (with rationale), failed approach, test result, or lesson — with source/event metadata and typed links to existing records |
| `intentgraph_recall` | Hybrid, graph-aware search: ranked candidates with provenance + excerpts |
| `intentgraph_context` | Full evidence packet for a task: provenance, conflicts, supersession, explicit gaps |
| `intentgraph_feedback` | Record whether evidence was `useful` / `missing` / `incorrect` |

The server ships usage instructions at initialization, so agents that honor MCP `instructions` get the memory loop (recall before substantial work; save rationale and failures; never store secrets) injected into their system prompt. Project-scoped archives (`<project>/intentgraph-data/`) mean one project's memories never leak into another's recalls.

## HTTP API

`intentgraph api` serves a dependency-free local JSON API (default `127.0.0.1:8710`; bearer auth required automatically on non-loopback binds) — **and the web UI at `http://127.0.0.1:8710/`**: the interactive knowledge graph rendered from your live archive, with a dropdown in the sidebar to switch between the combined memory+code graph (default), the memory graph, and the code graph. The **Evidence Replay** bar floats above the graph: ask a question, press Replay, and the camera zooms through the retrieval trail — glowing nodes, traveling edge-light, voice narration, and playback controls.

| Endpoint | Purpose |
|---|---|
| `GET /` or `GET /graph/memory-code` | web UI: the combined memory+code knowledge graph |
| `GET /graph/memory` | web UI: the memory graph |
| `GET /graph/code` | web UI: the code graph |
| `GET /health` | liveness + archive stats |
| `GET /v1/status` | full archive/providers/bridge/code-links status |
| `POST /v1/memories` | ingest |
| `GET /v1/memories` | list (pagination/filters) |
| `GET /v1/memories/{id}` | original record, full content |
| `GET /v1/memories/{id}/versions` | version history |
| `DELETE /v1/memories/{id}` · `POST …/restore` · `POST …/update` | lifecycle |
| `POST /v1/links` | typed relationship |
| `POST /v1/retrieve` | one retrieval round (any mode, full transparency) |
| `POST /v1/query` | cross-graph: memories + code entities + verified links |
| `POST /v1/context` | evidence packet |
| `GET /v1/code-links` | memory ↔ code links (filterable by record/node/tier) |
| `POST /v1/feedback` | retrieval feedback |
| `POST /v1/replay` | evidence replay: an ordered navigation script for the graph's cinematic playback |
| `POST /v1/export` · `POST /v1/import` | portable bundle |

## CLI

```bash
intentgraph ingest --text "…" --type decision --source-id "ADR-14"   # save
intentgraph get <mem_id>                    # fetch original record
intentgraph list --type decision            # filter/paginate
intentgraph link <a> <b> --relation derived_from --basis "…"   # typed edge
intentgraph supersede <old> <new>           # atomic validity + edge
intentgraph retrieve "query" --mode hybrid  # one round
intentgraph remember "question" --budget 4000  # adaptive → evidence packet
intentgraph query "question"              # cross-graph: memories + code + verified links
intentgraph codelink add|list|refresh|suggest  # memory ↔ code evidence wiring
intentgraph viz --query "question"          # interactive graph with real highlights
intentgraph viz --with-code                 # combined memory+code graph (verified links green)
intentgraph codegraph build .                # native codebase graph (Python)
intentgraph codegraph stats|show|path|render  # inspect and query the code graph
intentgraph api                             # local HTTP API
intentgraph benchmark [--agent|--providers] # reproducible measurements
intentgraph embed [--list-models]           # semantic index / model discovery
intentgraph websearch "query"               # explicit external search
intentgraph export bundle.jsonl            # versioned, provenance-complete backup
intentgraph import bundle.jsonl             # idempotent restore
intentgraph doctor                          # configuration + archive diagnosis
```

(`uv run python -m intentgraph …` works before `uv sync` puts the script on PATH.)

## External providers (all opt-in)

| Provider | Role | Notes |
|---|---|---|
| **Nebius Token Factory** | embeddings (default remote), optional LLM planning, reranker interface | OpenAI-compatible; `NEBIUS_API_KEY`; model discovery via `embed --list-models` |
| **NVIDIA NIM** | embeddings | Serves the NVIDIA-built embedding models (`nvidia/nemotron-3-embed-1b`, …); `NVIDIA_API_KEY` (nvapi-*); switch providers by editing two config lines |
| **Tavily** | selective web search | Triggered only for freshness-demanding questions; results carry full provenance and are never user memories; persistence is explicit opt-in |
| Any OpenAI-compatible endpoint | embeddings / planning | Ollama, vLLM, LiteLLM, … via `base_url` |

Credentials live in a gitignored `.env` (template: `.env.example`) loaded automatically; real environment variables always win; keys never appear in config files, logs, or errors. Model switching is safe: stored vectors are keyed by provider+model+endpoint, the index refuses to mix vector spaces (even identical dimensions), and `intentgraph embed` reindexes after a switch. A catalog entry is never assumed to serve — probe or smoke-test the exact model first.

## Visualization

```bash
intentgraph viz --query "what did we try for the checkout regression?"
```

The real memory archive rendered as an interactive `graph.html` — knowledge memories grouped by type, captured conversation events as diamonds explicitly labeled "not a verified fact", and the query's **actual retrieval results** highlighted (cyan = direct evidence, violet = graph-expanded supporting context, everything else dimmed, one-click reset). Contradictions amber on both endpoints, superseded records gray-dashed with their successor named. No fake connections, no fabricated matches: empty retrieval renders "No evidence returned — nothing highlighted."

## Codebase graphs (native, Python)

`intentgraph codegraph build .` scans a repository and writes a typed, evidence-backed `codegraph-out/graph.json` — written from first principles with Python's standard `ast` parser and NetworkX, with **no external code-graph engine installed, imported, or used**. The graph feeds both the retrieval and visualization layers: memory records can anchor to code nodes (`ingest --anchors` / `update --anchors`), and `intentgraph codegraph render` draws the code graph through the same interactive renderer.

- **Extracted (Python only today):** modules, classes, functions, methods; imports (in-repo, resolved), statically provable calls and references, inheritance, test→code links, entry points (`if __name__ == "__main__":`), file ownership, and `contains`/`defines` structure.
- **Evidence discipline:** every edge is `EXTRACTED` (proven from source) or `INFERRED` (documented heuristic: ambiguous base resolution, naming-convention test links). External dependencies (stdlib/site-packages) and anything unresolvable are recorded as metadata and counters — never invented as edges.
- **Honest limits:** lexical, best-effort static analysis — dynamic attributes, monkey-patching, and local-variable type tracking are out of scope and reported as unresolved counts. A `LanguageParser` protocol exists for adding other languages; none are implemented yet.
- **Deterministic:** sorted scans, content-derived stable ids (path-flattened, matching the bridge's documented anchor scheme), and byte-identical exports for identical trees. `.gitignore` (practical subset), built-in excludes (venvs, caches, build output, memory archives), secret-like files never read, symlinks never followed, size and count limits reported.

```bash
intentgraph codegraph build . --verbose     # scan → graph.json (+ stats in stdout)
intentgraph codegraph stats                 # types, relations, components, hubs, cycles
intentgraph codegraph show <node_id>        # attributes + typed edges (+ neighborhood)
intentgraph codegraph path <a> <b>           # deterministic shortest paths
intentgraph codegraph render                 # interactive graph.html
```

Output is schema-versioned (`graph.schema: intentgraph-codegraph`); legacy `graph.json` files without the marker stay valid, and a file from a newer schema version is refused with a clear error — never silently reinterpreted.

**Connected to memory.** The archive's `code_links` table wires memories to code nodes in three tiers: *explicit* (`codelink add`, validated against the graph), *extracted* (a deterministic matcher links a record's own backticked/dotted/file-path references at ingest, `codelink refresh` catches up and removes stale ones), and *candidate* (semantic suggestions via `codelink suggest` — labelled, inspectable, and never followed as retrieval evidence). Then one question reaches both graphs:

```bash
intentgraph query "why does Service.step exist"   # memories + code entities + verified links
intentgraph codelink list                        # inspect every link with its basis
intentgraph viz --with-code                      # combined graph: verified links green, candidates dashed
```

The MCP `recall` tool gains the same power through its optional `include_code` flag; without it, and without any code graph, memory retrieval behaves exactly as before.

## Benchmarks (measured, honestly labeled)

Reproducible with `intentgraph benchmark`, `--agent`, and `--providers`. The corpora are small and synthetic with hand-verified ground truth — these numbers validate wiring, selectivity, and honesty, not production-corpus quality:

| Measurement | Result |
|---|---|
| Pipeline (7 queries, known relevant ids): hybrid & adaptive evidence recall | **1.0** (lexical alone 0.94) |
| Fresh session with prior-session memory vs without: required-evidence recall | **1.0 vs 0.0** (by record id); failed-approach exposure true vs false |
| Local-history question → external requests | **0** (selectivity asserted) |
| Adaptive retrieval latency (local, in-memory) | milliseconds |
| Real remote embeddings (same corpus) | Nebius `Qwen/Qwen3-Embedding-8B` ~930 ms mean `remember()`; NVIDIA `nemotron-3-embed-1b` ~336 ms |

## Privacy & security

- **Zero network by default.** No provider, no key, no telemetry.
- `privacy.local_only: true` refuses **every** remote provider class at configuration load — before any request can be made.
- Memory content is untrusted input: packet bodies render inside untrusted-evidence fences; known chat-template injection sentinels are defanged; API body caps, strict id validation, path containment, and no-stack-trace errors.
- What leaves the machine is documented per provider class (embeddings send record text; search sends the query only) — never more.
- Never store secrets, credentials, or personal data; memory persists across sessions.

## Configuration

JSON config at `<data_dir>/intentgraph.json` (annotated template: `examples/intentgraph.config.json`), environment variables as overrides, `.env` for credentials. Key knobs: `retrieval.mode` (`adaptive`), `max_iterations`, `context_budget_tokens`, `timeout_s`, `embedding.provider` (`none`|`hash`|`nebius`|`nvidia`|`openai`), `llm.provider`, `reranker.provider`, `web_search.{provider, enabled, max_results, store_results}`, `privacy.local_only`.

## Development

```bash
uv sync --extra mcp --extra tokens        # env with optional extras
pytest tests/ -q                          # 531 tests pass offline; 5 live-provider smokes stay gated behind INTENTGRAPH_LIVE_TESTS=1
ruff check intentgraph/ && pyright intentgraph/   # clean / 0 errors
INTENTGRAPH_LIVE_TESTS=1 pytest tests/test_intentgraph_live_providers.py -v   # opt-in live smokes
examples/intentgraph-demo.sh                # runnable end-to-end demo
examples/intentgraph_phase3_demo.py [--live]     # provider-integration demo
```

The ordinary test suite is credential-hermetic — it disables `.env` loading and strips provider keys per-test, so it runs offline and green on any machine, regardless of what's configured.

## Documentation

- **[docs/intentgraph.md](./docs/intentgraph.md)** — the full guide: memory model, retrieval, verification, MCP/OpenCode setup, providers (Nebius/NVIDIA/Tavily), privacy, troubleshooting, limitations
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** — module map, symbol-verified against the code
- **[CHANGELOG.md](./CHANGELOG.md)** — release record

## Status & limitations

Working today: persistent archive, typed memory graph, hybrid/adaptive retrieval, verification, packets, MCP tools, HTTP API, dual embedding providers, selective Tavily, visualization, export/import, three benchmarks. Known boundaries: the requirement analyzer and relation weights are documented heuristics (not learned); the hash embedding provider is a lexical fallback, not semantics; reranker model availability is account-dependent (probe first); memory tooling has no automatic session hooks — the agent calls the tools.

## License

MIT — see [LICENSE](./LICENSE). Provenance and attribution for earlier revisions of this repository: [NOTICE](./NOTICE).
